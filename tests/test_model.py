import math

import pytest
import torch
import torch.nn as nn

from tfs.model import Config, LayerNorm, Transformer


def small(attn="manual", dropout=0.0):
    torch.manual_seed(0)
    cfg = Config(vocab_size=50, block_size=32, n_layer=2, n_head=2, d_model=32, dropout=dropout, attn=attn)
    return Transformer(cfg)


def test_layernorm_matches_pytorch():
    x = torch.randn(4, 8, 16)
    ours, theirs = LayerNorm(16), nn.LayerNorm(16)
    assert torch.allclose(ours(x), theirs(x), atol=1e-6)


def test_logits_shape_and_initial_loss_near_uniform():
    model = small().eval()
    idx = torch.randint(50, (4, 16))
    logits, loss, _ = model(idx, targets=idx)
    assert logits.shape == (4, 16, 50)
    assert abs(loss.item() - math.log(50)) < 0.5


@pytest.mark.parametrize("attn", ["manual", "sdpa"])
def test_future_tokens_cannot_change_past_logits(attn):
    model = small(attn).eval()
    a = torch.randint(50, (1, 20))
    b = a.clone()
    b[0, 12:] = (b[0, 12:] + 1) % 50
    with torch.no_grad():
        la, lb = model(a)[0], model(b)[0]
    assert torch.allclose(la[0, :12], lb[0, :12], atol=1e-5)
    assert not torch.allclose(la[0, 12:], lb[0, 12:], atol=1e-5)


def test_gradient_only_flows_from_past():
    model = small().eval()
    x = model.tok_emb(torch.randint(50, (1, 10))).detach().requires_grad_()
    out = model.blocks[0](x)[0]
    out[0, 4].sum().backward()
    assert x.grad[0, :5].abs().sum() > 0
    assert x.grad[0, 5:].abs().sum() == 0


def test_manual_and_sdpa_attention_agree():
    manual, sdpa = small("manual").eval(), small("sdpa").eval()
    idx = torch.randint(50, (2, 16))
    with torch.no_grad():
        assert torch.allclose(manual(idx)[0], sdpa(idx)[0], atol=1e-5)


@pytest.mark.parametrize("attn", ["manual", "sdpa"])
def test_kv_cache_matches_full_forward(attn):
    model = small(attn).eval()
    idx = torch.randint(50, (2, 12))
    with torch.no_grad():
        full = model(idx)[0]
        logits, _, caches = model(idx[:, :5])
        assert torch.allclose(logits, full[:, :5], atol=1e-5)
        for t in range(5, 12):
            logits, _, caches = model(idx[:, t : t + 1], caches=caches)
            assert torch.allclose(logits[:, 0], full[:, t], atol=1e-4)


def test_cached_and_uncached_generation_agree():
    model = small()
    start = torch.randint(50, (1, 4))
    torch.manual_seed(1)
    cached = model.generate(start, 20, use_cache=True)
    torch.manual_seed(1)
    uncached = model.generate(start, 20, use_cache=False)
    assert torch.equal(cached, uncached)


def test_generation_runs_past_block_size():
    model = small()
    out = model.generate(torch.zeros(1, 1, dtype=torch.long), 100, use_cache=True)
    assert out.shape == (1, 101)


def test_too_long_input_is_rejected():
    with pytest.raises(AssertionError):
        small()(torch.zeros(1, 33, dtype=torch.long))


@pytest.mark.parametrize("attn", ["manual", "sdpa"])
def test_all_true_key_mask_changes_nothing(attn):
    model = small(attn).eval()
    x = model.tok_emb(torch.randint(50, (2, 10)))
    with torch.no_grad():
        plain = model.blocks[0](x)[0]
        masked = model.blocks[0](x, key_mask=torch.ones(2, 10, dtype=torch.bool))[0]
    assert torch.allclose(plain, masked, atol=1e-6)


@pytest.mark.parametrize("attn", ["manual", "sdpa"])
def test_masked_key_cannot_influence_later_tokens(attn):
    model = small(attn).eval()
    x = model.tok_emb(torch.randint(50, (1, 10)))
    changed = x.clone()
    changed[0, 3] += 1.0
    key_mask = torch.ones(1, 10, dtype=torch.bool)
    key_mask[0, 3] = False
    with torch.no_grad():
        a = model.blocks[0](x, key_mask=key_mask)[0]
        b = model.blocks[0](changed, key_mask=key_mask)[0]
    keep = [i for i in range(10) if i != 3]
    assert torch.allclose(a[0, keep], b[0, keep], atol=1e-5)


def test_accepts_non_contiguous_targets():
    # The GPU sampler slices one tensor into inputs and targets.
    batch = torch.randint(50, (2, 17))
    assert small()(batch[:, :-1], targets=batch[:, 1:])[1].isfinite()


def test_every_parameter_gets_a_gradient():
    model = small()
    idx = torch.randint(50, (2, 16))
    model(idx, targets=idx)[1].backward()
    assert all(p.grad is not None for p in model.parameters())


def test_overfits_one_batch():
    model = small()
    optimizer = model.optimizer(lr=3e-3, weight_decay=0.0)
    x = torch.randint(50, (4, 16))
    y = torch.roll(x, -1, dims=1)
    for _ in range(300):
        _, loss, _ = model(x, targets=y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.05
