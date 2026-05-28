"""Model invariants — schema↔model alignment and loss correctness."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import model as model_mod

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tiny_schema():
    """A 3-head subset for fast testing — one of each head type."""
    return {
        "schema_version": "test",
        "n_traits": 3,
        "traits": [
            {"name": "motility", "head": "binary", "block": "morphology",
             "estimated_label_count": 100},
            {"name": "gram_stain", "head": "multiclass", "block": "morphology",
             "classes": ["positive", "negative", "variable"],
             "estimated_label_count": 100},
            {"name": "fatty_acid_profile", "head": "regression_vector", "block": "chemotaxonomy",
             "n_outputs": 3, "estimated_label_count": 50},
        ],
    }


@pytest.fixture
def tiny_vocab():
    return {
        "vocabularies": {
            "fatty_acid_profile": {
                "items": [{"value": "C16:0"}, {"value": "C18:0"}, {"value": "C14:0"}]
            },
        },
    }


@pytest.fixture
def tiny_df():
    """4 strains, mixed missingness across heads."""
    return pd.DataFrame([
        {"motility": True, "gram_stain": "positive",
         "fatty_acid_profile": {"C16:0": 30.0, "C18:0": 20.0}},
        {"motility": False, "gram_stain": "negative",
         "fatty_acid_profile": None},
        {"motility": None, "gram_stain": "negative",
         "fatty_acid_profile": {"C14:0": 10.0}},
        {"motility": True, "gram_stain": None,
         "fatty_acid_profile": None},
    ])


def test_prepare_labels_shapes(tiny_df, tiny_vocab, tiny_schema):
    labels, masks, specs = model_mod.prepare_labels(tiny_df, tiny_vocab, tiny_schema)

    assert labels["motility"].shape == (4,)
    assert labels["gram_stain"].shape == (4,)
    assert labels["fatty_acid_profile"].shape == (4, 3)

    # Masks marked where labeled
    assert masks["motility"].tolist() == [1, 1, 0, 1]
    assert masks["gram_stain"].tolist() == [1, 1, 1, 0]
    # Per-element mask for FAME — only reported FAMEs incur loss
    assert masks["fatty_acid_profile"][0].tolist() == [1, 1, 0]  # C16, C18 reported; C14 not
    assert masks["fatty_acid_profile"][2].tolist() == [0, 0, 1]  # only C14 reported


def test_fame_values_normalized(tiny_df, tiny_vocab, tiny_schema):
    """FAME percentages must be divided by 100 for scale-stable MSE."""
    labels, _, _ = model_mod.prepare_labels(tiny_df, tiny_vocab, tiny_schema)
    fa = labels["fatty_acid_profile"]
    assert fa[0, 0].item() == pytest.approx(0.30, abs=1e-6)  # 30% -> 0.30
    assert fa[0, 1].item() == pytest.approx(0.20, abs=1e-6)


def test_multiclass_uses_ignore_index(tiny_df, tiny_vocab, tiny_schema):
    """Missing-label rows must hold -1 in the multiclass target tensor."""
    labels, _, _ = model_mod.prepare_labels(tiny_df, tiny_vocab, tiny_schema)
    gs = labels["gram_stain"]
    assert gs[3].item() == -1  # missing
    assert gs[0].item() == 0   # "positive" -> index 0


def test_model_constructs_one_head_per_trait(tiny_df, tiny_vocab, tiny_schema):
    _, _, specs = model_mod.prepare_labels(tiny_df, tiny_vocab, tiny_schema)
    model = model_mod.MicrobeFoundationModel(input_dim=8, head_specs=specs, hidden=16, dropout=0.0)
    assert set(model.heads.keys()) == {"motility", "gram_stain", "fatty_acid_profile"}
    # Head output sizes match spec
    assert model.heads["motility"].out_features == 1
    assert model.heads["gram_stain"].out_features == 3
    assert model.heads["fatty_acid_profile"].out_features == 3


def test_forward_pass_shapes(tiny_df, tiny_vocab, tiny_schema):
    _, _, specs = model_mod.prepare_labels(tiny_df, tiny_vocab, tiny_schema)
    model = model_mod.MicrobeFoundationModel(input_dim=8, head_specs=specs, hidden=16, dropout=0.0)
    x = torch.randn(4, 8)
    out = model(x)
    assert out["motility"].shape == (4, 1)
    assert out["gram_stain"].shape == (4, 3)
    assert out["fatty_acid_profile"].shape == (4, 3)


def test_masked_loss_skips_empty_heads(tiny_df, tiny_vocab, tiny_schema):
    """A head with no labels in the batch contributes zero (and isn't averaged in)."""
    labels, masks, specs = model_mod.prepare_labels(tiny_df, tiny_vocab, tiny_schema)
    # Zero out fatty_acid_profile mask entirely
    masks["fatty_acid_profile"] = torch.zeros_like(masks["fatty_acid_profile"])
    model = model_mod.MicrobeFoundationModel(input_dim=8, head_specs=specs, hidden=16, dropout=0.0)
    x = torch.randn(4, 8)
    out = model(x)
    total, per_head = model_mod.masked_loss(out, labels, masks, specs)
    assert "fatty_acid_profile" not in per_head
    assert total.item() > 0


def test_masked_loss_gradient_flows(tiny_df, tiny_vocab, tiny_schema):
    """Backprop must update encoder weights."""
    labels, masks, specs = model_mod.prepare_labels(tiny_df, tiny_vocab, tiny_schema)
    model = model_mod.MicrobeFoundationModel(input_dim=8, head_specs=specs, hidden=16, dropout=0.0)
    x = torch.randn(4, 8)
    out = model(x)
    total, _ = model_mod.masked_loss(out, labels, masks, specs)
    total.backward()
    # First encoder layer must have a non-zero gradient
    assert model.encoder[0].weight.grad is not None
    assert model.encoder[0].weight.grad.abs().sum() > 0
