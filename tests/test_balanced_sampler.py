"""Family-balanced sampling weight helper."""
from __future__ import annotations

import numpy as np

import model as model_mod


def test_weights_sum_per_family_equalized():
    """Total sampling mass per family must be identical across families, even when
    family sizes differ wildly (the long-tail case the helper exists to fix)."""
    families = ["A"] * 100 + ["B"] * 1 + ["C"] * 10
    w = model_mod.family_sample_weights(families)
    fam = np.array(families)
    mass = {f: w[fam == f].sum() for f in ["A", "B", "C"]}
    assert np.allclose(mass["A"], mass["B"])
    assert np.allclose(mass["B"], mass["C"])


def test_weights_normalize_to_one():
    w = model_mod.family_sample_weights(["A", "A", "B", "C", "C", "C"])
    assert np.isclose(w.sum(), 1.0)


def test_singleton_family_outweighs_member_of_large_family():
    w = model_mod.family_sample_weights(["big"] * 50 + ["rare"])
    assert w[-1] > w[0]
    # the rare singleton should carry the same mass as the entire big family
    assert np.isclose(w[-1], w[:50].sum())


def test_none_and_nan_families_bucketed():
    w = model_mod.family_sample_weights(["A", None, float("nan"), "A"])
    assert len(w) == 4
    assert np.all(w > 0)
    # None and nan collapse into one "<NA>" group of size 2 -> equal per-item weight
    assert np.isclose(w[1], w[2])


def test_length_matches_input():
    fams = ["X", "Y", "Z", "X", "Y"]
    assert len(model_mod.family_sample_weights(fams)) == len(fams)
