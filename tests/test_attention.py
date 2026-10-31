import torch

from tfs.attention import attention_maps
from tfs.model import Config, Transformer


def small():
    torch.manual_seed(0)
    return Transformer(Config(vocab_size=50, block_size=32, n_layer=2, n_head=2, d_model=32, dropout=0.0))


def test_shape_and_rows_sum_to_one():
    weights = attention_maps(small(), torch.randint(50, (1, 10)))
    assert weights.shape == (2, 2, 10, 10)
    assert torch.allclose(weights.sum(-1), torch.ones(2, 2, 10), atol=1e-5)


def test_no_token_attends_to_the_future():
    weights = attention_maps(small(), torch.randint(50, (1, 10)))
    assert weights.triu(1).abs().sum() == 0


def test_maps_reproduce_the_real_attention_output():
    model = small()
    idx = torch.randint(50, (1, 10))
    weights = attention_maps(model, idx)[0]
    block = model.blocks[0]
    x = model.tok_emb(idx) + model.pos_emb(torch.arange(10))
    with torch.no_grad():
        v = block.attn.qkv(block.norm1(x)).split(32, dim=-1)[2]
        v = v.view(1, 10, 2, 16).transpose(1, 2)
        ours = (weights[None] @ v).transpose(1, 2).reshape(1, 10, 32)
        out, _ = block.attn(block.norm1(x))
    assert torch.allclose(block.attn.proj(ours), out, atol=1e-5)
