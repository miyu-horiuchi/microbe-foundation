"""The monitoring package must be importable as a microbe_model submodule."""
from __future__ import annotations


def test_package_imports():
    import microbe_model.monitoring as m
    assert m.__doc__ is not None
