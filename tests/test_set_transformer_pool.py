"""Set Transformer pooler (ISAB + PMA) invariants.

Covers the properties a permutation-invariant set pooler must satisfy so the
manuscript's §6 claim ("drop-in successor to AttentionPool") is testable on CPU
before any GPU training run.
"""
from __future__ import annotations

import math

import pytest
import torch

import model as model_mod


def _pool(dim=8, heads=2, inducing=4, seed=0):
    torch.manual_seed(seed)
    return model_mod.SetTransformerPool(dim, num_heads=heads, num_inducing=inducing)


def test_output_shape():
    """[B, P, D] + [B, P] mask -> [B, D]."""
    pool = _pool()
    x = torch.randn(4, 6, 8)
    mask = torch.ones(4, 6)
    assert pool(x, mask).shape == (4, 8)


def test_make_pooler_registration():
    pool = model_mod.make_pooler("set_transformer", input_dim=16, st_heads=4, st_inducing=8)
    assert isinstance(pool, model_mod.SetTransformerPool)


def test_dim_not_divisible_by_heads_raises():
    with pytest.raises(ValueError):
        model_mod.SetTransformerPool(dim=10, num_heads=3)


def test_permutation_invariance():
    """Reordering proteins must not change the pooled genome vector."""
    pool = _pool().eval()
    x = torch.randn(2, 7, 8)
    mask = torch.ones(2, 7)
    out = pool(x, mask)
    perm = torch.randperm(7)
    out_perm = pool(x[:, perm, :], mask[:, perm])
    assert torch.allclose(out, out_perm, atol=1e-5)


def test_mask_invariance_to_padding():
    """Appending padded (mask=0) proteins must not change the output."""
    pool = _pool().eval()
    x = torch.randn(3, 5, 8)
    mask = torch.ones(3, 5)
    base = pool(x, mask)

    pad = torch.randn(3, 4, 8)  # junk that must be ignored
    x_pad = torch.cat([x, pad], dim=1)
    mask_pad = torch.cat([mask, torch.zeros(3, 4)], dim=1)
    padded = pool(x_pad, mask_pad)
    assert torch.allclose(base, padded, atol=1e-5)


def test_variable_length_genomes_in_one_batch():
    """A batch mixing a 2-protein and a 6-protein genome must run and the short
    genome's output must match running it alone (padding ignored)."""
    pool = _pool().eval()
    g_short = torch.randn(1, 2, 8)
    alone = pool(g_short, torch.ones(1, 2))

    batch = torch.zeros(2, 6, 8)
    batch[0, :2] = g_short[0]
    batch[1] = torch.randn(6, 8)
    mask = torch.zeros(2, 6)
    mask[0, :2] = 1.0
    mask[1, :] = 1.0
    out = pool(batch, mask)
    assert torch.allclose(out[0:1], alone, atol=1e-5)


def test_gradient_flows_to_inducing_and_seed():
    pool = _pool()
    x = torch.randn(2, 5, 8, requires_grad=True)
    mask = torch.ones(2, 5)
    pool(x, mask).sum().backward()
    assert pool.isab.inducing.grad is not None
    assert pool.isab.inducing.grad.abs().sum() > 0
    assert pool.pma.seeds.grad is not None
    assert pool.pma.seeds.grad.abs().sum() > 0
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_attn_readout_shape_and_normalization():
    """store_attn exposes [B, P] seed->protein weights summing to 1 over reals."""
    pool = _pool().eval()
    pool.store_attn = True
    x = torch.randn(3, 5, 8)
    mask = torch.ones(3, 5)
    pool(x, mask)
    attn = pool.last_attn
    assert attn.shape == (3, 5)
    assert torch.allclose(attn.sum(dim=1), torch.ones(3), atol=1e-4)


def test_attn_zero_on_padding():
    """Padded positions must receive ~zero attention weight."""
    pool = _pool().eval()
    pool.store_attn = True
    x = torch.randn(2, 6, 8)
    mask = torch.ones(2, 6)
    mask[:, 4:] = 0.0  # last two are padding
    pool(x, mask)
    assert torch.allclose(pool.last_attn[:, 4:], torch.zeros(2, 2), atol=1e-5)


def test_full_model_with_set_transformer_forward():
    """End-to-end: per-protein input through a set_transformer model -> logits."""
    torch.manual_seed(0)
    specs = {"aerobic": {"head_type": "binary", "size": 1}}
    model = model_mod.MicrobeFoundationModel(
        input_dim=8, head_specs=specs, hidden=16, dropout=0.0,
        attention_pool=True, pooling="set_transformer", st_heads=2, st_inducing=4,
    )
    x = torch.randn(4, 6, 8)
    mask = torch.ones(4, 6)
    out = model(x, mask)
    assert out["aerobic"].shape == (4, 1)


def test_checkpoint_roundtrip_rebuilds():
    """A set_transformer state_dict must load into a model rebuilt from saved cfg."""
    torch.manual_seed(0)
    specs = {"aerobic": {"head_type": "binary", "size": 1}}
    m1 = model_mod.MicrobeFoundationModel(
        input_dim=8, head_specs=specs, hidden=16, dropout=0.0,
        attention_pool=True, pooling="set_transformer", st_heads=2, st_inducing=4,
    ).eval()
    sd = m1.state_dict()
    m2 = model_mod.MicrobeFoundationModel(
        input_dim=8, head_specs=specs, hidden=16, dropout=0.0,
        attention_pool=True, pooling="set_transformer", st_heads=2, st_inducing=4,
    ).eval()
    m2.load_state_dict(sd)  # must not raise
    x = torch.randn(2, 5, 8)
    mask = torch.ones(2, 5)
    assert torch.allclose(m1(x, mask)["aerobic"], m2(x, mask)["aerobic"], atol=1e-6)
