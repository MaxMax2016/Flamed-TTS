import os
import re
import time
import torch
import librosa
import numpy as np
from g2p_en import G2p
from string import punctuation
from omegaconf import DictConfig
from oz2.text import text_to_sequence
from oz2.models.oz2_lightning import OZ2Lightning
from oz2.models.synthesizer import (
    PriorGenerator,
    ProbGenerator,
)
from oz2.models.facodec import (
    FACodecEncoder,
    FACodecDecoder,
)


class OZ2(OZ2Lightning):
    
    @classmethod
    def from_pretrained(cls, cfg, ckpt_path, device, weights_only=False, training_mode=False):
        cfg['prob_generator']['device'] = device
        cfg['prior_generator']['device'] = device
        model = OZ2(cfg)
        model.lexicon = model.read_lexicon()
        model.g2p = G2p()
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=weights_only)
        if not weights_only:
            model.load_state_dict(ckpt['state_dict'])
        else:
            model.load_state_dict(ckpt)
        del ckpt
        if not training_mode:
            model.eval()
        return model
    
    def __init__(self, cfg):
        super(OZ2, self).__init__()

        self.cfg = cfg
        self.prior_generator = PriorGenerator(cfg['prior_generator'])
        self.prob_generator = ProbGenerator(cfg['prob_generator'])
        
    def forward(
        self,
        phonemes, 
        x_len, 
        codes, 
        y_len, 
        phone_durations,
        sil_durations,
        embs,
        prompts,
        spks,
        ):
        
        # Forward & compute losses of codes generator
        (
            prior_embs,
            tgt_masks,
            ar_losses,
        ) = self.prior_generator.compute_loss(
            texts=phonemes,
            src_lens=x_len,
            max_src_len=phonemes.size(-1),
            codes=codes,
            tgt_lens=y_len,
            max_tgt_len=codes.size(-1),
            phone_durations=phone_durations,
            sil_durations=sil_durations,
            prompts=prompts,
            prompts_len=prompts.size(-1),
        )
        
        # Forward & compute losses of flow matching
        prob_losses = self.prob_generator.compute_loss(
            x1=embs,
            cond=prior_embs,
            spk=spks,
            mask=~tgt_masks.unsqueeze(-1),
        )
        
        return ar_losses | prob_losses
    
    @torch.inference_mode()
    def sample(
        self, 
        text: str = None,
        phonemes: torch.Tensor = None,
        prompt_raw: str | np.ndarray | torch.Tensor = None,
        prompt_processed: torch.Tensor = None,
        timbre: torch.Tensor = None,
        sr: int = 16000,
        codec_cfg: DictConfig = None,
        codec_encoder: torch.nn.Module = None,
        codec_decoder: torch.nn.Module = None,
        temp_durgen: float = 0.3,
        temp_denoiser: float = 0.3,
        nsteps_durgen: int = 64,
        nsteps_denoiser: int = 64,
        lexicon_path: str = None,
        cleaners: str = ['english_cleaners'],
        ):
        
        if codec_encoder is None or codec_decoder is None:
            if codec_cfg is None:
                raise ValueError('The codec_encoder or codec_decoder is set to None. To initialize the codec encoder or decoder, you need to provide a codec_cfg of type omegaconf.DictConfig.')
            codec_encoder, codec_decoder = self._get_codec_models(codec_cfg)

        text_provided = text is not None and phonemes is None
        phonemes_provided = text is None and phonemes is not None
        text_phoneme_exclusive_check =  text_provided or phonemes_provided
        if not text_phoneme_exclusive_check:
            raise ValueError('`text` and `phonemes` are mutually exclusive—only one should be provided, and the other must be None!')

        prompt_raw_provided = prompt_raw is not None and prompt_processed is None
        prompt_processed_provided = prompt_raw is None and prompt_processed is not None
        prompt_raw_processed_exclusive_check = prompt_raw_provided or prompt_processed_provided
        if not prompt_raw_processed_exclusive_check:
            raise ValueError('`prompt_raw` and `prompt_processed` are mutually exclusive—only one should be provided, and the other must be None!')

        # get starting timestamp of the progress
        start_time = time.time()
        
        # process phoneme
        if text_provided:
            phonemes, _, _ = self._preprocess_english(text, lexicon_path, cleaners)
        else:
            phonemes = phonemes.unsqueeze(0).to(self.device)
        
        # process acoustic prompt
        if prompt_raw_provided:
            acoustic_prompt = self._preprocess_acoustic_prompt(prompt_raw, sr)
            enc_out = codec_encoder(acoustic_prompt)
            _, prompts, _, _, timbre = codec_decoder(enc_out, eval_vq=False, vq=True)
            prompts = prompts.permute(1, 0, 2)
        else:
            if timbre is None:
                raise ValueError('`timbre` must be provided along with `prompt_processed`!')
            timbre = timbre.unsqueeze(0).to(self.device)           
            prompts = prompt_processed.unsqueeze(0).to(self.device)

        prior_emb_cond, prior_logits, tgt_mask = self.prior_generator.sample(
            texts=phonemes,
            src_lens=torch.zeros(phonemes.size(0), device=self.device) + phonemes.size(-1),
            max_src_len=phonemes.size(-1),
            prompts=prompts,
            prompts_len=prompts.size(-1),
            nfe=nsteps_durgen,
            temperature=temp_durgen,
        )
        
        # flow matching euler solving
        x1 = self.prob_generator.sample(
            cond=prior_emb_cond,
            spk=timbre,
            nfe=nsteps_denoiser,
            temperature=temp_denoiser,
            mask=~tgt_mask.unsqueeze(-1),
        )
        
        wav = codec_decoder.inference(x1, timbre)
        wav = wav[0][0].detach().cpu().numpy()
        
        end_time = time.time()
                
        return {
            'wav': wav,
            'time': end_time - start_time,
        }
    
    def _preprocess_acoustic_prompt(self, acoustic_prompt, sr=16000):
        if isinstance(acoustic_prompt, str):
            acoustic_prompt = librosa.load(acoustic_prompt, sr=sr)[0]
            acoustic_prompt = torch.from_numpy(acoustic_prompt).float()
            acoustic_prompt = acoustic_prompt.unsqueeze(0).unsqueeze(0).to(self.device)
        elif isinstance(acoustic_prompt, np.ndarray):
            acoustic_prompt = torch.from_numpy(acoustic_prompt).float()
            acoustic_prompt = acoustic_prompt.unsqueeze(0).unsqueeze(0).to(self.device)
        elif isinstance(acoustic_prompt, torch.Tensor):
            acoustic_prompt = acoustic_prompt.to(self.device)
        else:
            raise ValueError('Acoustic prompt must be one of [str, np.ndarray, torch.tensor]!')
        return acoustic_prompt
    
    def _get_codec_models(self, codec_cfg):
        codec_encoder = FACodecEncoder.from_pretrained(codec_cfg['encoder']).eval()
        codec_decoder = FACodecDecoder.from_pretrained(codec_cfg['decoder']).eval()
        return codec_encoder, codec_decoder
    
    def read_lexicon(self, lexicon_path=None):
        if not lexicon_path:
            lexicon_path = os.path.join(os.path.dirname(__file__), '..', 'lexicon', 'librispeech-lexicon.txt')
        lexicon = {}
        with open(lexicon_path) as f:
            for line in f:
                temp = re.split(r"\s+", line.strip("\n"))
                word = temp[0]
                phones = temp[1:]
                if word.lower() not in lexicon:
                    lexicon[word.lower()] = phones
        return lexicon
    
    def _preprocess_english(
        self, 
        text, 
        lexicon_path=None, 
        cleaners='english_cleaners'
        ):  
        text = text.rstrip(punctuation)
        phones = []
        words = re.split(r"([,;.\-\?\!\s+])", text)
        for w in words:
            if w.lower() in self.lexicon:
                phones += self.lexicon[w.lower()]
            else:
                phones += list(filter(lambda p: p != " ", self.g2p(w)))
        phones = "{sp " + " ".join(phones) + "}"
        phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
        phones = phones.replace("}{", " ")
        sequence = np.array(text_to_sequence(phones, cleaners))
        sequence = torch.from_numpy(sequence).unsqueeze(0).to(self.device)
        return sequence, text, phones
