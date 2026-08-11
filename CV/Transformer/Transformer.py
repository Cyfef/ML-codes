import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from wandb_utils import *


@dataclass
class TransformerConfig:
    img_feat_dim: int = 512            # image feature dim
    wordvec_dim: int = 128        
    D: int = 128         # Transformer 隐层维度 TOKEN dim
    H: int = 4            # number of attention heads
    num_transformerlayers: int = 1           # Transformer 层数
    max_length: int = 30          # 最大序列长度（包含<START>）
    mlp_ratio: float = 4.0        # MLP 隐层缩放因子 (mlp_dim = mlp_ratio * hidden_dim)


def make_causal_mask(seq_len: int, 
                     device: Optional[torch.device] = None,
                     dtype: torch.dtype = torch.float32
                     ) -> torch.Tensor:
    """
    返回形状 (1, 1, seq_len, seq_len) 的张量，其中可被关注的位置值为 0，
    未来位置值为 -1e9。
    """
    mask = torch.triu(torch.ones((seq_len, seq_len), device=device, dtype=dtype), diagonal=1)
    mask = mask * -1e9
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, D: int, H: int):
        super().__init__()
        assert D % H == 0, "D must be divisible by H"
        self.D = D
        self.H = H
        self.D_H = D // H

        self.q_proj = nn.Linear(D, D)
        self.k_proj = nn.Linear(D, D)
        self.v_proj = nn.Linear(D, D)
        self.out_proj = nn.Linear(D, D)

    def forward(self, 
                X: torch.Tensor,                        # (B,N,D)
                mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        B, N, D = X.shape
        
        Q = self.q_proj(X).view(B, N, self.H, self.D_H).transpose(1, 2)     # (B,H,N,D_H)
        K = self.k_proj(X).view(B, N, self.H, self.D_H).transpose(1, 2)     # (B,H,N,D_H)
        V = self.v_proj(X).view(B, N, self.H, self.D_H).transpose(1, 2)     # (B,H,N,D_H)

        E = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)             # (B,H,N,N)
        if mask is not None:
            E += mask
        A = torch.softmax(E, dim=-1)        # (B,H,N,N)
        Y = torch.matmul(A, V)              # (B,H,N,D_H)

        Y = Y.transpose(1, 2).contiguous().view(B, N, D)    # (B,N,D)
        O = self.out_proj(Y)        # (B,N,D)
        return O, A


class TransformerBlock(nn.Module):
    def __init__(self, 
                 D: int, 
                 H: int, 
                 mlp_dim: Optional[int] = None):
        super().__init__()

        if mlp_dim is None:
            mlp_dim = 4 * D

        self.ln1 = nn.LayerNorm(D)
        self.attn = MultiHeadSelfAttention(D, H)
        self.ln2 = nn.LayerNorm(D)
        self.mlp = nn.Sequential(
            nn.Linear(D, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, D),
        )

    def forward(self, 
                x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        
        attn_out, _ = self.attn(self.ln1(x), mask)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))

        return x


class TransformerCaptioner(nn.Module):

    def __init__(self, 
                 word_to_idx: Dict[str, int], 
                 config: TransformerConfig):
        super().__init__()

        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}

        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx["<START>"]
        self._end = word_to_idx.get("<END>", None)

        self.max_length = config.max_length
        self.config = config

        vocab_size = len(word_to_idx)

        self.feature_proj = nn.Linear(config.img_feat_dim, config.D)

        self.word_embed = nn.Embedding(vocab_size, config.wordvec_dim)
        self.word_proj = nn.Linear(config.wordvec_dim, config.D)

        self.pos_embed = nn.Parameter(torch.empty(1, config.max_length + 1, config.D))      # (1, max_length + 1, D)
        nn.init.xavier_normal_(self.pos_embed)

        self.blocks = nn.ModuleList([
            TransformerBlock(config.D, 
                             config.H,
                             mlp_dim=int(config.mlp_ratio * config.D))
            for _ in range(config.num_transformerlayers)
        ])

        self.ln = nn.LayerNorm(config.D)
        self.vocab_proj = nn.Linear(config.D, vocab_size)

        # key: (seq_len, device, dtype), value:mask
        self.mask_cache: Dict[Tuple[int, torch.device, torch.dtype], torch.Tensor] = {}

    def _to_tensor(self, 
                   x, 
                   dtype=None):
        
        if torch.is_tensor(x):
            return x
        return torch.as_tensor(x, dtype=dtype)

    def _get_causal_mask(self, 
                         seq_len: int, 
                         device: torch.device, 
                         dtype: torch.dtype
                         ) -> torch.Tensor:
        
        key = (seq_len, device, dtype)
        if key not in self.mask_cache:
            self.mask_cache[key] = make_causal_mask(seq_len, device, dtype)
        return self.mask_cache[key]

    def forward(self, 
                img_features: torch.Tensor,     # (B,img_feat_dim)
                captions_in: torch.Tensor       # (B,T)
                ) -> torch.Tensor:
        """
        Args:
            img_features: (B,img_feat_dim)
            captions_in: (B, T) prompt index

        Returns:
            logits: (B, T, vocab_size)
        """

        img_features = self._to_tensor(img_features, torch.float32)
        captions_in = self._to_tensor(captions_in, torch.long)

        device = self.pos_embed.device
        img_features = img_features.to(device)
        captions_in = captions_in.to(device)

        B, T = captions_in.shape
        if T + 1 > self.pos_embed.shape[1]:
            raise ValueError(f"The length of the input sequence ({T+1}) exceeds the maximum length allowed by the model ({self.pos_embed.shape[1]-1})")

        # img token, word token cat
        image_token = self.feature_proj(img_features).unsqueeze(1)          # (B, 1, H)
        word_tokens = self.word_proj(self.word_embed(captions_in))          # (B, T, H)
        x = torch.cat([image_token, word_tokens], dim=1)                    # (B, T+1, H)

        # position embedding
        x = x + self.pos_embed[:, :T+1, :]

        # casual mask
        mask = self._get_causal_mask(T+1, device, x.dtype)

        for block in self.blocks:
            x = block(x, mask)

        x = self.ln(x)
        logits = self.vocab_proj(x[:, 1:, :])   # drop image token position
        return logits

    def loss(self, 
             img_features: torch.Tensor,    # (B,img_feat_dim)
             captions: torch.Tensor         # (B,T)
             ) -> torch.Tensor:
        """
        Cross Entropy Loss
        captions: (N, L) contains <START> and <END>
        """
        captions = self._to_tensor(captions, torch.long).to(self.pos_embed.device)

        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        logits = self.forward(img_features, captions_in)
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            captions_out.reshape(-1),
            ignore_index=self._null,
        )

    @torch.no_grad()
    def sample(self, 
               img_features: torch.Tensor,          # (B,img_feat_dim)
               max_length: Optional[int] = None,
               temperature: float = 1.0
               ) -> torch.Tensor:
        """
        Args:
            img_features: (N, input_dim)
            max_length: 
            temperature: >0
        """
        if max_length is None:
            max_length = self.max_length
        if max_length > self.max_length:
            raise ValueError(f"Request to generate a text of length {max_length} which exceeds the maximum length of the model, which is {self.max_length}.")

        img_features = self._to_tensor(img_features, torch.float32).to(self.pos_embed.device)
        B = img_features.shape[0]

        captions = torch.full((B, max_length), self._null, dtype=torch.long, device=img_features.device)
        prefix = torch.full((B, 1), self._start, dtype=torch.long, device=img_features.device)      # [<START>] (B,1)

        for t in range(max_length):
            logits = self.forward(img_features, prefix)           # (B, t+1, V)
            next_logits = logits[:, -1, :] / temperature      # (B, V)
            probs = F.softmax(next_logits, dim=-1)

            if temperature == 0.0:
                # greedy
                next_word = torch.argmax(probs, dim=-1)
            else:
                next_word = torch.multinomial(probs, 1).squeeze(1)  # (B,)
            captions[:, t] = next_word
            prefix = torch.cat([prefix, next_word[:, None]], dim=1)

        return captions.cpu()

class TransformerCaptionerTrainer():
    def __init__(self,
                 model,
                 optimizer,
                 device):
        
        self.model=model
        self.optimizer=optimizer
        self.device=device

    def train(self,
              train_dataloader,
              save_path:str,
              num_epochs: int = 50,
              log_interval: int = 10,
              ):

        wandb_init()
        self.model.train()
        iter_count=0

        for epoch in range(1,num_epochs+1):
            for (features, captions) in train_dataloader:
                
                features = features.to(self.device, dtype=torch.float32)
                captions = captions.to(self.device, dtype=torch.long)

                self.optimizer.zero_grad()

                loss = self.model.loss(features, captions)
                loss.backward()

                self.optimizer.step()

                wandb_log({
                            "train/loss": loss,
                            "train/iteration": iter_count,
                            })

                if iter_count % log_interval == 0:
                    print(f'Iter: {iter_count}, Loss: {loss.item():.4}')

                iter_count += 1

        torch.save(self.model.state_dict(), save_path)
        wandb_finish()
