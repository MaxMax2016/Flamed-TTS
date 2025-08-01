import os
import time
import torch
import librosa
import argparse
from oz2 import OZ2
from tqdm import tqdm
import soundfile as sf
from omegaconf import OmegaConf
from oz2.models.facodec import (
    FACodecEncoder, 
    FACodecDecoder
)


SR = 16000
CURDIR = os.path.dirname(__file__)
CONTENT_EXAMPLE = "The evolution usually starts from a population of randomly generated individuals, and is an iterative process, with the population in each iteration called a generation."


def load_audio(wav_path):
    wav = librosa.load(wav_path, sr=SR)[0]
    wav = torch.from_numpy(wav).float()
    wav = wav.unsqueeze(0).unsqueeze(0)
    return wav


def get_codec(accelerator):
    fa_encoder = FACodecEncoder(
        ngf=32,
        up_ratios=[2, 4, 5, 5],
        out_channels=256,
    )

    fa_decoder = FACodecDecoder(
        in_channels=256,
        upsample_initial_channel=1024,
        ngf=32,
        up_ratios=[5, 5, 4, 2],
        vq_num_q_c=2,
        vq_num_q_p=1,
        vq_num_q_r=3,
        vq_dim=256,
        codebook_dim=8,
        codebook_size_prosody=10,
        codebook_size_content=10,
        codebook_size_residual=10,
        use_gr_x_timbre=True,
        use_gr_residual_f0=True,
        use_gr_residual_phone=True,
    )

    encoder_ckpt = os.path.join(CURDIR, 'oz2', 'models', 'facodec', 'checkpoints', 'ns3_facodec_encoder.bin')
    decoder_ckpt = os.path.join(CURDIR, 'oz2', 'models', 'facodec', 'checkpoints', 'ns3_facodec_decoder.bin')
    fa_encoder.load_state_dict(torch.load(encoder_ckpt, weights_only=False))
    fa_decoder.load_state_dict(torch.load(decoder_ckpt, weights_only=False))
    fa_encoder.to(accelerator)
    fa_decoder.to(accelerator)
    fa_encoder.eval()
    fa_decoder.eval()

    return fa_encoder, fa_decoder


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', type=str, required=True)
    parser.add_argument('--cfg-path', type=str, required=True)
    parser.add_argument('--text', type=str, default=CONTENT_EXAMPLE)
    parser.add_argument('--prompt-dir', type=str, required=True)
    parser.add_argument('--prompt-list', type=str, nargs='+', required=True)
    parser.add_argument('--nsteps-durgen', type=int, default=64)
    parser.add_argument('--nsteps-denoiser', type=int, default=64)
    parser.add_argument('--temp-durgen', type=float, default=0.3)
    parser.add_argument('--temp-denoiser', type=float, default=0.3)
    parser.add_argument('--output-dir', type=str, default='.')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--weights-only', type=bool, default=True)
    args = parser.parse_args()

    ckpt_path = args.ckpt_path
    cfg_path = args.cfg_path
    text = args.text
    prompt_dir = args.prompt_dir
    prompt_list = args.prompt_list
    nsteps_durgen = args.nsteps_durgen
    nsteps_denoiser = args.nsteps_denoiser
    temp_durgen = args.temp_durgen
    temp_denoiser = args.temp_denoiser
    output_dir = args.output_dir
    accelerator = args.device
    weights_only = args.weights_only

    fa_encoder, fa_decoder = get_codec(accelerator)
    cfg = OmegaConf.load(cfg_path)
    cfg['prob_generator']['device'] = accelerator
    cfg['prior_generator']['device'] = accelerator

    model = OZ2.from_pretrained(
        cfg=cfg, 
        ckpt_path=ckpt_path,
        device=accelerator,
        weights_only=weights_only,
        training_mode=False
    )
    model.to(accelerator)
    
    infer_times, output_durations = [], []
    for prompt in tqdm(prompt_list):
        audio_prompt = load_audio(os.path.join(prompt_dir, prompt))

        results = model.sample(
            text=text,
            prompt_raw=audio_prompt,
            sr=SR,
            codec_encoder=fa_encoder,
            codec_decoder=fa_decoder,
            nsteps_durgen=nsteps_durgen,
            nsteps_denoiser=nsteps_denoiser,
            temp_durgen=temp_durgen,
            temp_denoiser=temp_denoiser,
        )

        infer_times.append(results['time'])
        output_durations.append(len(results['wav']) / SR)
        sf.write(f"{output_dir}/{prompt}-{nsteps_durgen}-{nsteps_denoiser}-{temp_durgen}-{temp_denoiser}.wav", results['wav'], SR)

    rtf = [t/d for t, d in zip(infer_times, output_durations)]
    print('='*20, 'Avg RTF', '='*20)
    print('>'*5, f'RTF: ', round(sum(rtf) / len(rtf), 3))
