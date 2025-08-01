import torch
import wandb
from abc import ABC
from oz2.models.facodec import (
    FACodecEncoder,
    FACodecDecoder,
)
from omegaconf import DictConfig
from oz2.data import OZ2Dataset
from lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only
from transformers import get_cosine_schedule_with_warmup


class OZ2Lightning(LightningModule, ABC):
    def setup_dataset_optimizer(
        self, 
        dataset_cfg: DictConfig, 
        optimizer_cfg: DictConfig
        ):
        self.dataset_cfg = dataset_cfg
        self.optimizer_cfg = optimizer_cfg
        self.dataset = OZ2Dataset(dataset_cfg)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=optimizer_cfg['lr'],
            betas=optimizer_cfg["betas"],
            eps=optimizer_cfg["eps"],
            weight_decay=optimizer_cfg["weight_decay"],
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=optimizer_cfg['warmup_steps'],
            num_training_steps=optimizer_cfg['max_steps'],
        )
        
    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "total_loss_val",
            },
        }
    
    def get_dataset(self):
        self.dataset.setup()
        train_data=self.dataset.train_dataloader()
        val_data=self.dataset.val_dataloader()
        return train_data, val_data

    def training_step(self, batch, batch_idx):
        (
            phonemes, 
            x_len, 
            codes, 
            y_len, 
            phone_durations,
            sil_durations,
            embs,
            prompts,
            spks,
        ) = batch
        
        losses = self(
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
        
        total_loss, logging_data = 0, {}
        for item in losses:
            if '_loss' in item:
                total_loss += losses[item]
                logging_data[f'{item}_train'] = losses[item]
            else:
                logging_data[f'{item}'] = losses[item]
        logging_data['total_loss_train'] = total_loss
        logging_data['lr'] = self.scheduler.optimizer.param_groups[0]['lr']
        logging_data['step'] = float(self.global_step)
        self._logging(logging_data)

        return {"loss": total_loss, "log": losses}
    
    def validation_step(self, batch, batch_idx):
        (
            phonemes, 
            x_len, 
            codes, 
            y_len, 
            phone_durations,
            sil_durations,
            embs,
            prompts,
            spks,
        ) = batch
        
        losses = self(
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
        
        total_loss, logging_data = 0, {}
        for item in losses:
            if '_loss' in item:
                total_loss += losses[item]
                logging_data[f'{item}_val'] = losses[item]
        logging_data['total_loss_val'] = total_loss
        logging_data['step'] = float(self.global_step)
        self._logging(logging_data)

        return total_loss
    
    def test_step(self, batch, batch_idx):
        return
    
    def _logging(self, logs: dict):
        for key in logs:
            self.log(
                name=key,
                value=logs[key],
                on_step=True,
                on_epoch=True,
                logger=True,
                batch_size=self.optimizer_cfg['batch_size'],
                sync_dist=True,
            )

    @rank_zero_only
    def on_validation_batch_end(self, outputs, batch, batch_idx):
        if self.global_step < 1000:
            return
        
        phonemes, x_len, _, y_len, _, _, embs, prompts, spks = batch

        codec_encoder = FACodecEncoder.from_pretrained(self.cfg['codec_cfg']['encoder']).eval()
        codec_decoder = FACodecDecoder.from_pretrained(self.cfg['codec_cfg']['decoder']).eval()
        codec_encoder.to(self.cfg['codec_cfg']['device'])
        codec_decoder.to(self.cfg['codec_cfg']['device'])

        results = self.sample(
            prompt_processed=prompts[0],
            phonemes=phonemes[0, : x_len[0].item()],
            timbre=spks[0],
            codec_encoder=codec_encoder,
            codec_decoder=codec_decoder,
        )
        wav, prior_wav = results['wav'], results['prior_wav']

        gt_wav = codec_decoder.inference(
            embs[0, : y_len[0].item(), :].unsqueeze(0).permute(0, 2, 1), 
            spks[0].unsqueeze(0)
        )
        del codec_encoder, codec_decoder

        wandb.log({
            "synthesize/val_prior": wandb.Audio(prior_wav, sample_rate=self.cfg['codec_cfg']['sr'])
        }, step=self.global_step)

        wandb.log({
            "synthesize/val_synth": wandb.Audio(wav, sample_rate=self.cfg['codec_cfg']['sr'])
        }, step=self.global_step)
        
        wandb.log({
            "synthesize/val_gt": wandb.Audio(gt_wav[0][0].cpu().numpy(), sample_rate=self.cfg['codec_cfg']['sr'])
        }, step=self.global_step)