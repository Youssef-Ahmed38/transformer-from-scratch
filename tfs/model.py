"""GPT-style Transformer. Attention, normalization and the KV cache are written by hand."""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    dropout: float = 0.1
    attn: str = "manual"  # "manual" spells out the math, "sdpa" uses PyTorch's fast kernel


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps) * self.gain + self.bias


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.attn = cfg.attn
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.proj.residual = True

    def forward(self, x, cache=None, key_mask=None):
        B, T, D = x.shape
        shape = (B, T, self.n_head, self.head_dim)
        q, k, v = (t.view(shape).transpose(1, 2) for t in self.qkv(x).split(D, dim=-1))

        offset = 0
        if cache is not None:
            offset = cache[0].size(2)
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)

        # A query at absolute position p may attend to keys 0..p.
        allowed = torch.ones(T, offset + T, dtype=torch.bool, device=x.device).tril(offset)
        if key_mask is not None:
            # Masked positions stop being context. Each query keeps itself so softmax stays defined.
            allowed = (allowed & key_mask[:, None, None, :]) | torch.eye(T, dtype=torch.bool, device=x.device)
        p_drop = self.dropout if self.training else 0.0
        if self.attn == "sdpa":
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=p_drop)
        else:
            scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
            weights = scores.masked_fill(~allowed, float("-inf")).softmax(dim=-1)
            out = F.dropout(weights, p_drop, self.training) @ v

        out = out.transpose(1, 2).reshape(B, T, D)
        return F.dropout(self.proj(out), p_drop, self.training), (k, v)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = LayerNorm(cfg.d_model)
        self.fc1 = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.fc2 = nn.Linear(4 * cfg.d_model, cfg.d_model)
        self.fc2.residual = True
        self.dropout = cfg.dropout

    def forward(self, x, cache=None, key_mask=None):
        out, cache = self.attn(self.norm1(x), cache, key_mask)
        x = x + out
        out = self.fc2(F.gelu(self.fc1(self.norm2(x))))
        return x + F.dropout(out, self.dropout, self.training), cache


class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = LayerNorm(cfg.d_model)
        self.apply(self._init)

    def _init(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            std = 0.02
            # Each block adds two residual branches, so shrink them to keep variance stable.
            if getattr(module, "residual", False):
                std /= math.sqrt(2 * self.cfg.n_layer)
            nn.init.normal_(module.weight, 0.0, std)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, caches=None):
        offset = caches[0][0].size(2) if caches else 0
        T = idx.size(1)
        assert offset + T <= self.cfg.block_size, "sequence longer than block_size"
        pos = torch.arange(offset, offset + T, device=idx.device)
        x = F.dropout(self.tok_emb(idx) + self.pos_emb(pos), self.cfg.dropout, self.training)

        new_caches = []
        for i, block in enumerate(self.blocks):
            x, cache = block(x, caches[i] if caches else None)
            new_caches.append(cache)

        # Output projection shares its weights with the token embedding.
        logits = self.norm(x) @ self.tok_emb.weight.T
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss, new_caches

    def optimizer(self, lr, weight_decay, fused=False):
        decay = [p for p in self.parameters() if p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.99), fused=fused)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, use_cache=True):
        was_training = self.training
        self.eval()
        caches = None
        for _ in range(max_new_tokens):
            if use_cache and caches is not None and caches[0][0].size(2) < self.cfg.block_size:
                logits, _, caches = self(idx[:, -1:], caches=caches)
            elif use_cache:
                # Positions are absolute, so a full cache cannot slide. Rebuild from half a
                # block, which keeps the rebuild rare instead of paying for it on every token.
                keep = idx[:, -self.cfg.block_size // 2 :] if caches else idx[:, -self.cfg.block_size :]
                logits, _, caches = self(keep)
            else:
                logits, _, _ = self(idx[:, -self.cfg.block_size :])

            # Sample in float32: multinomial measured about 16x slower on bfloat16 on the GPU.
            logits = logits[:, -1, :].float() / max(temperature, 1e-6)
            if top_k is not None:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            idx = torch.cat([idx, torch.multinomial(logits.softmax(-1), 1)], dim=1)
        self.train(was_training)
        return idx
