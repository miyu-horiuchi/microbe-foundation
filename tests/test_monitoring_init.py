"""The monitoring package must be importable as a microbe_model submodule."""
from __future__ import annotations


def test_package_imports():
    import microbe_model.monitoring as m
    assert m.__doc__ is not None


def test_public_api_exports():
    from microbe_model.monitoring import (
        DriftReport,
        DiffusionBackend,
        EuclideanBackend,
        ReferenceManifold,
        assess_drift,
        mmd_permutation_test,
    )
    assert all(
        obj is not None
        for obj in (
            DriftReport, DiffusionBackend, EuclideanBackend,
            ReferenceManifold, assess_drift, mmd_permutation_test,
        )
    )
