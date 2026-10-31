"""Read attention weights out of a trained model for visualization."""
import math

import torch


@torch.no_grad()
def attention_maps(model, idx):
    """Weights for the first sequence, shaped (layers, heads, tokens, tokens)."""
    model.eval()
    T = idx.size(1)
    h = model.tok_emb(idx) + model.pos_emb(torch.arange(T, device=idx.device))
    allowed = torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()
    maps = []
    for block in model.blocks:
        attn = block.attn
        q, k, _ = attn.qkv(block.norm1(h)).split(h.size(-1), dim=-1)
        shape = (*idx.shape, attn.n_head, attn.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(attn.head_dim)
        maps.append(scores.masked_fill(~allowed, float("-inf")).softmax(-1)[0])
        h = block(h)[0]
    return torch.stack(maps)
