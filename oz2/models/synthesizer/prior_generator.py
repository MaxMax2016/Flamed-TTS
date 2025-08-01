import torch
import torch.nn as nn
from oz2.models.module import (
    Encoder, 
    Decoder,
)
import torch.nn.functional as F
from .pva import PVA
from oz2.utils.tools import get_mask_from_lengths


class PreEncoding(nn.Module):
    def __init__(self, hidden_dim, n_quantizer):
        super().__init__()

        self.prompt_emb = nn.Parameter(torch.rand(1, 1, hidden_dim))
        self.target_emb = nn.Parameter(torch.rand(1, 1, hidden_dim))
        self.quantizer_emb = nn.Embedding(n_quantizer, hidden_dim)

    def forward(self, x, prompt_len, q_idx):
        b, l, _ = x.shape
        x[:, :prompt_len, :] = x[:, :prompt_len, :] + self.prompt_emb.expand(b, prompt_len, -1).to(x.device)
        x[:, prompt_len:, :] = x[:, prompt_len:, :] + self.target_emb.expand(b, l - prompt_len, -1).to(x.device)
        q_emb = self.quantizer_emb(torch.tensor([q_idx], device=x.device))
        x =  x + q_emb.unsqueeze(0).expand(b, l, -1)
        return x


class PriorGenerator(nn.Module):
    def __init__(self, config):
        super(PriorGenerator, self).__init__()
        
        self.config = config
        encoder_hidden = config["transformer"]["encoder_hidden"]
        decoder_hidden = config["transformer"]["decoder_hidden"]
        vocab_size = config["codec"]["vocab_size"]
        n_quantizers = config["codec"]["n_quantizers"]

        self.encoder = Encoder(config)
        self.pva = PVA(config['variance_adaptor'])
        self.bridge = nn.Linear(encoder_hidden, decoder_hidden)

        self.code_embedding = nn.Embedding(
            num_embeddings=vocab_size + 1,
            embedding_dim=decoder_hidden,
            padding_idx=vocab_size,
        )
        self.shared_decoder = Decoder(config, config["transformer"]["decoder_shared_layers"])
        self.pre_encode = PreEncoding(decoder_hidden, n_quantizers)
        self.prior_decoder = nn.ModuleList()
        for ith in range(n_quantizers):
            self.prior_decoder.append(
                Decoder(
                    config, 
                    config["transformer"]["decoder_layers"][ith],
                )
            )
        
        self.head = nn.Linear(
            in_features=decoder_hidden,
            out_features=vocab_size + 1, 
        )
    
    def compute_loss(
        self,
        texts,
        src_lens,
        max_src_len,
        codes,
        tgt_lens,
        max_tgt_len,
        phone_durations,
        sil_durations,
        prompts,
        prompts_len,
        ):
        
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        tgt_masks = (
            get_mask_from_lengths(tgt_lens, max_tgt_len)
            if tgt_lens is not None
            else None
        )
        
        output = self.encoder(texts, src_masks)
        output, pva_losses = self.pva.compute_loss(
            output,
            src_lens,
            src_masks,
            max_tgt_len,
            phone_durations,
            sil_durations,
        )
        output = self.bridge(output)

        output, tgt_masks = self.shared_decoder(output, tgt_masks)
        tgt_masks_decode = get_mask_from_lengths(prompts_len + tgt_lens, prompts_len + max_tgt_len)
        prompt_embs = self.code_embedding(prompts)
                
        hiddens = []
        for ith, layer in enumerate(self.prior_decoder):
            prompt_emb = prompt_embs[:, ith, :, :]
            q_input = self.pre_encode(torch.cat([prompt_emb, output], dim=1), prompts_len, ith)
            output, tgt_masks_decode = layer(q_input, tgt_masks_decode)
            output = output[:, prompts_len:, :]
            hiddens.append(output.unsqueeze(1))
            
        output = torch.cat(hiddens, dim=1) # (b, n, l, d)

        logits = self.head(output)
        logits = logits * ~tgt_masks.unsqueeze(1).expand(-1, logits.size(1), -1).unsqueeze(3)
        logits = logits.permute(0, 3, 1, 2).contiguous() # (b, c, n, l)

        # Compute loss
        prior_loss = 0
        for idx in range(codes.size(1)):
            prior_loss = prior_loss + F.cross_entropy(logits[:, :, idx, :], codes[:, idx, :])
        prior_loss = prior_loss / codes.size(1)

        losses = pva_losses | {'prior_loss': prior_loss}

        del (
            texts,
            src_lens,
            src_masks,
            codes,
            tgt_lens,
            tgt_masks_decode,
            phone_durations,
            sil_durations,
            prompts, 
            logits,
            prompt_embs,
            prompt_emb,
            q_input,
            hiddens,
        )

        return output, tgt_masks, losses
    
    def sample(
        self,
        texts,
        src_lens,
        max_src_len,
        prompts,
        prompts_len,
        nfe=4,
        temperature=1.0,
        ):
        
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        
        output = self.encoder(texts, src_masks)
        output, tgt_lens = self.pva.sample(
            output,
            src_lens,
            src_masks,
            nfe=nfe,
            temperature=temperature,
        )
        output = self.bridge(output)

        tgt_masks = get_mask_from_lengths(tgt_lens, output.size(1))
        output, tgt_masks = self.shared_decoder(output, tgt_masks)
        tgt_masks_decode = get_mask_from_lengths(prompts_len + tgt_lens, prompts_len + output.size(1))
        prompt_embs = self.code_embedding(prompts)
                
        hiddens = []
        for ith, layer in enumerate(self.prior_decoder):
            prompt_emb = prompt_embs[:, ith, :, :]
            q_input = self.pre_encode(torch.cat([prompt_emb, output], dim=1), prompts_len, ith)
            output, tgt_masks_decode = layer(q_input, tgt_masks_decode)
            output = output[:, prompts_len:, :]
            hiddens.append(output.unsqueeze(1))
            
        output = torch.cat(hiddens, dim=1) # (b, n, l, d)

        logits = self.head(output)
        logits = logits * ~tgt_masks.unsqueeze(1).expand(-1, logits.size(1), -1).unsqueeze(3)
        logits = logits.permute(0, 3, 1, 2).contiguous() # (b, c, n, l)

        del (
            texts,
            src_lens,
            src_masks,
            tgt_lens,
            tgt_masks_decode,
            prompts, 
            prompt_embs,
            prompt_emb,
            q_input,
            hiddens,
        )

        return output, logits, tgt_masks