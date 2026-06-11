"""Embedding-based drift monitoring (Tier 0–1).

Tier 0: geometric, label-free OOD scoring of ESM2 genome embeddings.
Tier 1: drift classification combining per-genome OOD rate + population MMD test.

Concept drift (P(Y|X) change) is NOT observable here; this package detects
data drift (P(X) change) only.
"""
from __future__ import annotations
