import os
import tgt
import json
import torch
import random
import numpy as np
from typing import Any, Dict, Optional
from lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from oz2.text import text_to_sequence


class OZ2Dataset(LightningDataModule):
    def __init__(self, config):
        super().__init__()

        # this line allows to access init params with 'self' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        self.name = config['name']
        self.data_root = config['data_root']
        self.train_manifest = config['train_manifest']
        self.valid_manifest = config['valid_manifest']
        self.sampling_rate = config['sampling_rate']
        self.dur_min = config['dur_min']
        self.dur_max = config['dur_max']
        self.n_words_min = config['n_words_min']
        self.prompt_dur_max = config['prompt_dur_max']
        self.prompt_reduced_factor = config['prompt_reduced_factor']
        self.down_factors = config['down_factors']
        self.vocab_size = config['vocab_size']
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.pin_memory = config['pin_memory']
        self.cleaners = config['cleaners']
        self.add_blank = config['add_blank']
        self.seed = config['seed']
        self.device = config['device']
        self.sil_phones = config['sil_phones']

    def setup(self, stage: Optional[str] = None):  # pylint: disable=unused-argument
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        
        self.trainset = TextCodesDataset(  # pylint: disable=attribute-defined-outside-init
            self.data_root,
            self.train_manifest,
            self.cleaners,
            self.dur_min,
            self.dur_max,
            self.n_words_min,
            self.prompt_dur_max,
            self.sampling_rate,
            self.down_factors,
            self.sil_phones,
            self.add_blank,
            self.seed,
        )
        self.validset = TextCodesDataset(  # pylint: disable=attribute-defined-outside-init
            self.data_root,
            self.valid_manifest,
            self.cleaners,
            self.dur_min,
            self.dur_max,
            self.n_words_min,
            self.prompt_dur_max,
            self.sampling_rate,
            self.down_factors,
            self.sil_phones,
            self.add_blank,
            self.seed,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            collate_fn=TextCodesBatchCollate(
                device=self.device,
                prompt_max_len=self.prompt_dur_max * self.sampling_rate // np.prod(self.down_factors),
                prompt_reduced_factor=self.prompt_reduced_factor,
                vocab_size=self.vocab_size,
            ),
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=TextCodesBatchCollate(
                device=self.device,
                prompt_max_len=self.prompt_dur_max * self.sampling_rate // np.prod(self.down_factors),
                prompt_reduced_factor=self.prompt_reduced_factor,
                vocab_size=self.vocab_size,
            ),
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass  # pylint: disable=unnecessary-pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass  # pylint: disable=unnecessary-pass


class TextCodesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        manifest,
        cleaners,
        dur_min=0.3,
        dur_max=15,
        n_words_min=3,
        prompt_dur_max=3,
        sampling_rate=16000,
        down_factors=None,
        sil_phones=None,
        add_blank=True,
        seed=None,
    ):
        self.data_root = data_root
        self.manifest = manifest
        self.cleaners = cleaners
        self.dur_min = dur_min
        self.dur_max = dur_max
        self.prompt_dur_max = prompt_dur_max
        self.sampling_rate = sampling_rate
        self.sil_phones = sil_phones
        self.add_blank = add_blank

        if down_factors is None:
            self.down_factors = [2, 4, 5, 5]
        else:
            self.down_factors = down_factors
        self.down_factor = np.prod(self.down_factors)
        
        if sil_phones is None:
            self.sil_phones = ["sil", "sp", "spn", ""]
        else:
            self.sil_phones = sil_phones
            
        samples, filters, dur_total = [], [], 0
        with open(os.path.join(self.data_root, self.manifest), 'r', encoding='utf-8') as manifest:
            for line in manifest:
                sample = line.replace('\n', '')
                duration = float(sample.split('|')[1])
                n_words = len(sample.split('|')[2].split(' '))
                
                if duration < self.dur_min or duration > self.dur_max or n_words < n_words_min:
                    filters.append(sample)
                    continue
                samples.append(sample)        
                dur_total += duration
                
        dur_total = round(dur_total / 3600, 3)
        self.samples = samples

        print('+-'*50)
        print(f'>>>\t {self.manifest}: {dur_total} hours')
        print(f'>>>\t Valid utterances: {len(self.samples)}')
        print(f'>>>\t Filtered utterances: {len(filters)}')
        print('+-'*50)
                
        random.seed(seed)
        random.shuffle(self.samples)

    def get_datapoint(self, sample):
        (
            filename, 
            dur_in_sec, 
            transcript,
            style_prompt,
            textgrid_path,
            tgt_codes_path,
            cond_codes_path,
        ) = tuple(sample.split('|'))
        
        textgrid = tgt.io.read_textgrid(textgrid_path, include_empty_intervals=True)

        gt = json.load(open(tgt_codes_path))
        spk, codes, embs = gt['spkemb'], gt['quantizers'], gt['vqemb']
        spk = torch.FloatTensor(spk)
        codes = torch.stack([torch.IntTensor(quantizer) for quantizer in codes])
        embs = torch.stack([torch.FloatTensor(emb) for emb in embs])

        phones, phone_durations, sil_durations = self.get_alignment(textgrid.get_tier_by_name("phones"))
        phone_durations = torch.IntTensor(phone_durations)
        sil_durations = torch.IntTensor(sil_durations)
        phonemes = torch.IntTensor(text_to_sequence('{' + ' '.join(phones) + '}', self.cleaners))
        
        return {
            "phoneme": phonemes, 
            "code": codes,
            "emb": embs,
            "spk": spk,
            "phone_dur": phone_durations,
            "sil_dur": sil_durations,
        }
    
    def get_alignment(self, textgrid_tier):
        
        pre_phones, pre_durations = ['bos'], [0]
        for t in textgrid_tier._objects:
            s, e, p = t.start_time, t.end_time, t.text
            p = 'sp' if p == '' else p
            start_code = s * self.sampling_rate // self.down_factor
            end_code = e * self.sampling_rate // self.down_factor
            pre_phones.append(p)
            pre_durations.append(end_code - start_code)

        phones, phone_durations, sil_durations = [], [], []
        for idx in range(len(pre_phones)):
            if pre_phones[idx] in self.sil_phones:
                continue
            else:
                phones.append(pre_phones[idx])
                phone_durations.append(pre_durations[idx])
                if idx == len(pre_phones) - 1:
                    sil_durations.append(0)
                else:
                    if pre_phones[idx+1] in self.sil_phones:
                        sil_durations.append(pre_durations[idx+1])
                    else:
                        sil_durations.append(0)
        
        phones[0] = 'sp'
        del pre_phones, pre_durations           
        return phones, phone_durations, sil_durations

    def __getitem__(self, index):
        datapoint = self.get_datapoint(self.samples[index])
        return datapoint

    def __len__(self):
        return len(self.samples)


class TextCodesBatchCollate:
    def __init__(
        self, 
        device,
        prompt_max_len=800,
        prompt_reduced_factor=0.8,
        vocab_size=1024,
        ):
        
        self.device = device
        self.vocab_size = vocab_size
        self.prompt_max_len = prompt_max_len
        self.prompt_reduced_factor = prompt_reduced_factor
        
    def _process_acoustic_prompt(self, prompts):
        max_len = min([prompt.size(1) for prompt in prompts] + [self.prompt_max_len])
        max_len_reduced = int(self.prompt_reduced_factor * max_len)
        
        prompt_segments = []
        for prompt in prompts:
            start_idx = random.randint(0, prompt.size(1) - max_len_reduced)
            end_idx = start_idx + max_len_reduced
            prompt_segments.append(prompt[:,start_idx:end_idx])
            
        prompts = torch.stack(prompt_segments)
        # mask content quantizer
        prompts[:,1:3,:] = self.vocab_size
        # add eos
        # bs, qs, _ = prompts.shape
        # eos = torch.zeros((bs, qs, 1), dtype=prompts.dtype) + self.vocab_size
        # prompts = torch.cat([prompts, eos], dim=-1)
        return prompts
    
    def __call__(self, batch):
        B = len(batch)
        x_max_len = max([item["phoneme"].shape[-1] for item in batch])
        y_max_len = max([item["code"].shape[-1] for item in batch])
        n_codes = batch[0]["code"].shape[-2]
        emb_dim = batch[0]["emb"].shape[-1]

        phonemes = torch.zeros(
            (B, x_max_len), 
            dtype=torch.long, device=self.device
        )
        codes = torch.zeros(
            (B, n_codes, y_max_len), 
            dtype=torch.long, device=self.device
        ) + self.vocab_size
        embs = torch.zeros(
            (B, y_max_len, emb_dim), 
            dtype=torch.float, device=self.device
        )
        phone_durations = torch.zeros(
            (B, x_max_len), 
            dtype=torch.long, device=self.device
        )
        sil_durations = torch.zeros(
            (B, x_max_len), 
            dtype=torch.long, device=self.device
        )

        prompts, spks, x_len, y_len = [], [], [], []
        for i, item in enumerate(batch):
            p_i = item["phoneme"]
            c_i = item["code"]
            e_i = item["emb"]
            s_i = item["spk"]
            pd_i = item["phone_dur"]
            sd_i = item["sil_dur"]
            
            phonemes[i, : p_i.shape[-1]] = p_i
            codes[i, :, : c_i.shape[-1]] = c_i
            embs[i, : e_i.shape[0], :] = e_i
            phone_durations[i, : pd_i.shape[-1]] = pd_i
            sil_durations[i, : sd_i.shape[-1]] = sd_i

            prompts.append(c_i)
            spks.append(s_i)
            x_len.append(p_i.shape[-1])
            y_len.append(c_i.shape[-1])
            
        spks = torch.stack(spks)
        x_len = torch.tensor(x_len, dtype=torch.int, device=self.device)
        y_len = torch.tensor(y_len, dtype=torch.int, device=self.device)
        
        prompts = self._process_acoustic_prompt(prompts)
        del batch

        return (
            phonemes, 
            x_len, 
            codes, 
            y_len, 
            phone_durations,
            sil_durations,
            embs,
            prompts,
            spks,
        )