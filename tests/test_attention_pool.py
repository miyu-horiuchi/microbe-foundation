"""Attention pooling + per-protein loader invariants."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import model as model_mod


def test_attention_pool_output_shape():
    """[B, P, D] + mask -> [B, D]."""
    pool = model_mod.AttentionPool(dim=8)
    x = torch.randn(4, 6, 8)
    mask = torch.ones(4, 6)
    out = pool(x, mask)
    assert out.shape == (4, 8)


def test_attention_weights_sum_to_one():
    """Softmax over real proteins must produce weights summing to 1."""
    torch.manual_seed(0)
    pool = model_mod.AttentionPool(dim=8)
    x = torch.randn(3, 5, 8)
    mask = torch.ones(3, 5)
    scores = pool.score(x).squeeze(-1).masked_fill(mask == 0, float("-inf"))
    attn = torch.softmax(scores, dim=1)
    assert torch.allclose(attn.sum(dim=1), torch.ones(3), atol=1e-5)


def test_padding_is_ignored():
    """Padded protein slots must not affect the pooled output."""
    torch.manual_seed(0)
    pool = model_mod.AttentionPool(dim=8)
    real = torch.randn(1, 3, 8)
    mask_real = torch.ones(1, 3)
    out_real = pool(real, mask_real)

    # Same 3 real proteins + 2 garbage padded slots that the mask zeroes out.
    padded = torch.cat([real, torch.randn(1, 2, 8) * 1000], dim=1)
    mask_padded = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])
    out_padded = pool(padded, mask_padded)

    assert torch.allclose(out_real, out_padded, atol=1e-5)


def test_attention_pool_gradient_flows():
    """Backprop must reach the scorer weights."""
    pool = model_mod.AttentionPool(dim=8)
    x = torch.randn(2, 4, 8)
    mask = torch.ones(2, 4)
    out = pool(x, mask)
    out.sum().backward()
    assert pool.score[0].weight.grad is not None
    assert pool.score[0].weight.grad.abs().sum() > 0


def test_model_with_attention_pool_forward():
    """Full model in attention_pool mode: [B, P, D] + mask -> per-head logits."""
    specs = {
        "motility": {"head_type": "binary", "size": 1},
        "gram_stain": {"head_type": "multiclass", "size": 3},
    }
    model = model_mod.MicrobeFoundationModel(
        input_dim=8, head_specs=specs, hidden=16, dropout=0.0, attention_pool=True
    )
    x = torch.randn(4, 5, 8)
    mask = torch.ones(4, 5)
    out = model(x, mask)
    assert out["motility"].shape == (4, 1)
    assert out["gram_stain"].shape == (4, 3)


def test_mean_protein_pool_ignores_padding():
    pool = model_mod.MeanProteinPool()
    real = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    padded = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]]])
    assert torch.allclose(
        pool(padded, torch.tensor([[1.0, 1.0, 0.0]])),
        pool(real, torch.tensor([[1.0, 1.0]])),
    )


def test_max_protein_pool_ignores_padding():
    pool = model_mod.MaxProteinPool()
    x = torch.tensor([[[1.0, 7.0], [3.0, 4.0], [99.0, 99.0]]])
    out = pool(x, torch.tensor([[1.0, 1.0, 0.0]]))
    assert out.tolist() == [[3.0, 7.0]]


def test_topk_pool_output_shape_and_gradient():
    torch.manual_seed(0)
    pool = model_mod.TopKProteinPool(dim=8, k=2)
    x = torch.randn(3, 5, 8)
    mask = torch.tensor([
        [1.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
    ])
    out = pool(x, mask)
    assert out.shape == (3, 8)
    out.sum().backward()
    assert pool.score.weight.grad is not None


def test_gated_attention_weights_sum_to_one():
    torch.manual_seed(0)
    pool = model_mod.GatedAttentionPool(dim=8)
    pool.store_attn = True
    x = torch.randn(2, 4, 8)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    out = pool(x, mask)
    assert out.shape == (2, 8)
    assert torch.allclose(pool.last_attn.sum(dim=1), torch.ones(2), atol=1e-5)
    assert pool.last_attn[0, 2:].sum().item() == pytest.approx(0.0)


def test_model_with_gated_attention_forward():
    specs = {
        "motility": {"head_type": "binary", "size": 1},
        "gram_stain": {"head_type": "multiclass", "size": 3},
    }
    model = model_mod.MicrobeFoundationModel(
        input_dim=8, head_specs=specs, hidden=16, dropout=0.0, pooling="gated_attention"
    )
    out = model(torch.randn(4, 5, 8), torch.ones(4, 5))
    assert out["motility"].shape == (4, 1)
    assert out["gram_stain"].shape == (4, 3)


def test_collate_perprotein_pads_and_masks():
    """Ragged genomes pad to batch-max P; mask marks real proteins."""
    # Three "genomes" with 2, 5, 3 proteins of dim 4.
    batch = []
    for p in (2, 5, 3):
        x = torch.randn(p, 4)
        labels = {"motility": torch.tensor(1.0)}
        masks = {"motility": torch.tensor(1.0)}
        batch.append((x, labels, masks))
    (padded, pmask), label_dict, mask_dict = model_mod.collate_perprotein(batch)

    assert padded.shape == (3, 5, 4)        # padded to max P = 5
    assert pmask.shape == (3, 5)
    assert pmask[0].tolist() == [1, 1, 0, 0, 0]   # genome 0 had 2 proteins
    assert pmask[1].tolist() == [1, 1, 1, 1, 1]   # genome 1 had 5
    assert pmask[2].tolist() == [1, 1, 1, 0, 0]   # genome 2 had 3
    assert label_dict["motility"].shape == (3,)


def test_per_protein_dataset_lazy_load(tmp_path):
    """Dataset reads a genome's matrix from disk on access."""
    arr = (np.random.randn(7, 4)).astype(np.float16)
    p = tmp_path / "123.npy"
    np.save(p, arr)
    labels = {"motility": torch.tensor([1.0])}
    masks = {"motility": torch.tensor([1.0])}
    ds = model_mod.PerProteinDataset([p], labels, masks)
    x, lab, msk = ds[0]
    assert x.shape == (7, 4)
    assert x.dtype == torch.float32       # upcast from stored fp16
    assert lab["motility"].item() == 1.0


def test_per_protein_dataset_max_proteins_cap(tmp_path):
    """max_proteins subsamples genomes with too many proteins."""
    arr = (np.random.randn(20, 4)).astype(np.float16)
    p = tmp_path / "9.npy"
    np.save(p, arr)
    ds = model_mod.PerProteinDataset([p], {}, {}, max_proteins=5)
    x, _, _ = ds[0]
    assert x.shape == (5, 4)
