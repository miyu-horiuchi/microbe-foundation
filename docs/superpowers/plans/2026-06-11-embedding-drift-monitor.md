# Embedding Drift Monitor (Tier 0–1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a label-free geometric drift monitor (Tier 0) and drift classifier (Tier 1) in `microbe_model/monitoring/`, then validate that diffusion-distance OOD scoring beats a Euclidean baseline on the family split.

**Architecture:** A `ReferenceManifold` standardizes ESM2 genome embeddings, subsamples anchors, and delegates per-genome OOD scoring to a pluggable distance backend (`EuclideanBackend` = k-NN distance in standardized space; `DiffusionBackend` = k-NN distance in diffusion-map coordinates with Nyström out-of-sample extension). A permutation MMD test gives a valid population-level p-value. `assess_drift` combines per-genome OOD rate + MMD p-value into a `DriftReport`. A validation harness measures AUROC on held-out families.

**Tech Stack:** numpy, scikit-learn (NearestNeighbors, rbf_kernel, roc_auc_score), scipy (eigsh, pdist, spearmanr), pandas/pyarrow. Python 3.10+ typing.

**Spec:** `docs/superpowers/specs/2026-06-11-embedding-drift-monitor-design.md`

---

### Task 1: Dependencies + package skeleton

**Files:**
- Modify: `requirements.txt`
- Create: `microbe_model/monitoring/__init__.py`
- Test: `tests/test_monitoring_init.py`

- [ ] **Step 1: Add scipy to requirements**

Add this line to `requirements.txt` (after the `scikit-learn>=1.4` line):

```
scipy>=1.11
```

- [ ] **Step 2: Create empty package init**

Create `microbe_model/monitoring/__init__.py` with:

```python
"""Embedding-based drift monitoring (Tier 0–1).

Tier 0: geometric, label-free OOD scoring of ESM2 genome embeddings.
Tier 1: drift classification combining per-genome OOD rate + population MMD test.

Concept drift (P(Y|X) change) is NOT observable here; this package detects
data drift (P(X) change) only.
"""
from __future__ import annotations
```

- [ ] **Step 3: Write the failing import test**

Create `tests/test_monitoring_init.py`:

```python
"""The monitoring package must be importable as a microbe_model submodule."""
from __future__ import annotations


def test_package_imports():
    import microbe_model.monitoring as m
    assert m.__doc__ is not None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_monitoring_init.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add requirements.txt microbe_model/monitoring/__init__.py tests/test_monitoring_init.py
git commit -m "feat(monitoring): add scipy dep and package skeleton"
```

---

### Task 2: MMD permutation test

**Files:**
- Create: `microbe_model/monitoring/mmd.py`
- Test: `tests/test_monitoring_mmd.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_monitoring_mmd.py`:

```python
"""MMD permutation test: valid type-I under the null, power under a real shift."""
from __future__ import annotations

import numpy as np

from microbe_model.monitoring.mmd import median_gamma, mmd_permutation_test


def test_median_gamma_positive():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    assert median_gamma(X) > 0


def test_null_gives_large_pvalue():
    """Same distribution -> not flagged as different (p well above 0.05)."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 5))
    Y = rng.normal(size=(80, 5))
    mmd2, p = mmd_permutation_test(X, Y, n_perm=200, seed=0)
    assert p > 0.05


def test_shift_gives_small_pvalue():
    """A clear mean shift -> low p-value (drift detected)."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 5))
    Y = rng.normal(size=(80, 5)) + 3.0
    mmd2, p = mmd_permutation_test(X, Y, n_perm=200, seed=0)
    assert p < 0.05
    assert mmd2 > 0


def test_deterministic_with_seed():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 3))
    Y = rng.normal(size=(40, 3)) + 1.0
    a = mmd_permutation_test(X, Y, n_perm=100, seed=7)
    b = mmd_permutation_test(X, Y, n_perm=100, seed=7)
    assert a == b
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_monitoring_mmd.py -v`
Expected: FAIL with `ModuleNotFoundError: microbe_model.monitoring.mmd`

- [ ] **Step 3: Implement mmd.py**

Create `microbe_model/monitoring/mmd.py`:

```python
"""Maximum Mean Discrepancy two-sample test with a permutation-based p-value.

The permutation test yields a valid p-value with controlled type-I error — the
population-level drift verifier. Kernel is RBF with a median-heuristic bandwidth.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import rbf_kernel


def median_gamma(X: np.ndarray) -> float:
    """RBF gamma from the median-heuristic: gamma = 1 / median(pairwise sq. dist)."""
    X = np.asarray(X, dtype=float)
    d2 = pdist(X, metric="sqeuclidean")
    med = float(np.median(d2)) if d2.size else 0.0
    if med <= 0:
        med = 1.0
    return 1.0 / med


def _mmd2_from_gram(Kxx: np.ndarray, Kyy: np.ndarray, Kxy: np.ndarray) -> float:
    """Unbiased MMD^2 estimate from precomputed kernel blocks."""
    m = Kxx.shape[0]
    n = Kyy.shape[0]
    sum_xx = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
    sum_yy = (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
    sum_xy = Kxy.mean()
    return float(sum_xx + sum_yy - 2.0 * sum_xy)


def mmd_permutation_test(
    X_ref: np.ndarray,
    X_test: np.ndarray,
    *,
    gamma: float | None = None,
    n_perm: int = 200,
    seed: int = 0,
) -> tuple[float, float]:
    """Return (observed MMD^2, permutation p-value) for X_ref vs X_test."""
    X_ref = np.asarray(X_ref, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    Z = np.vstack([X_ref, X_test])
    if gamma is None:
        gamma = median_gamma(Z)
    K = rbf_kernel(Z, Z, gamma=gamma)
    m = len(X_ref)
    N = len(Z)

    def mmd2(idx_x: np.ndarray, idx_y: np.ndarray) -> float:
        Kxx = K[np.ix_(idx_x, idx_x)]
        Kyy = K[np.ix_(idx_y, idx_y)]
        Kxy = K[np.ix_(idx_x, idx_y)]
        return _mmd2_from_gram(Kxx, Kyy, Kxy)

    observed = mmd2(np.arange(m), np.arange(m, N))
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(N)
        if mmd2(perm[:m], perm[m:]) >= observed:
            count += 1
    p_value = (1 + count) / (1 + n_perm)
    return observed, p_value
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_monitoring_mmd.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add microbe_model/monitoring/mmd.py tests/test_monitoring_mmd.py
git commit -m "feat(monitoring): MMD permutation test"
```

---

### Task 3: Euclidean distance backend

**Files:**
- Create: `microbe_model/monitoring/backends.py`
- Test: `tests/test_monitoring_backends.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_monitoring_backends.py`:

```python
"""Distance backends: OOD score rises with distance from the reference support."""
from __future__ import annotations

import numpy as np

from microbe_model.monitoring.backends import EuclideanBackend


def test_euclidean_scores_far_higher_than_near():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(200, 4))
    backend = EuclideanBackend(k=10).fit(ref)
    near = rng.normal(size=(20, 4))            # same distribution
    far = rng.normal(size=(20, 4)) + 8.0       # shifted far away
    assert backend.score(far).mean() > backend.score(near).mean()


def test_euclidean_score_shape():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(100, 3))
    backend = EuclideanBackend(k=5).fit(ref)
    X = rng.normal(size=(7, 3))
    s = backend.score(X)
    assert s.shape == (7,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_monitoring_backends.py -v`
Expected: FAIL with `ImportError` / `ModuleNotFoundError`

- [ ] **Step 3: Implement backends.py (Euclidean only for now)**

Create `microbe_model/monitoring/backends.py`:

```python
"""Pluggable distance backends for OOD scoring.

Each backend exposes the same interface so the validation harness can A/B them:
    fit(X_ref) -> self
    score(X)   -> np.ndarray   # per-row OOD score, higher = more out-of-distribution

EuclideanBackend scores in standardized embedding space (baseline).
DiffusionBackend scores in diffusion-map coordinates (curvature-aware contender).
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.neighbors import NearestNeighbors


class DistanceBackend(Protocol):
    def fit(self, X_ref: np.ndarray) -> "DistanceBackend": ...
    def score(self, X: np.ndarray) -> np.ndarray: ...


class EuclideanBackend:
    """Mean distance to the k nearest reference points in standardized space."""

    def __init__(self, k: int = 10) -> None:
        self.k = k

    def fit(self, X_ref: np.ndarray) -> "EuclideanBackend":
        X_ref = np.asarray(X_ref, dtype=float)
        k = min(self.k, len(X_ref))
        self._nn = NearestNeighbors(n_neighbors=k).fit(X_ref)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        dist, _ = self._nn.kneighbors(X)
        return dist.mean(axis=1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_monitoring_backends.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add microbe_model/monitoring/backends.py tests/test_monitoring_backends.py
git commit -m "feat(monitoring): Euclidean OOD backend"
```

---

### Task 4: Diffusion backend with Nyström extension

**Files:**
- Modify: `microbe_model/monitoring/backends.py`
- Test: `tests/test_monitoring_backends.py`

- [ ] **Step 1: Add failing tests for the diffusion backend**

Append to `tests/test_monitoring_backends.py`:

```python
from microbe_model.monitoring.backends import DiffusionBackend


def test_diffusion_nystrom_roundtrip():
    """transform() on the reference reproduces the fitted reference coordinates."""
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(150, 5))
    backend = DiffusionBackend(n_components=6, k=10).fit(ref)
    coords = backend.transform(ref)
    assert np.allclose(coords, backend.coords_ref_, atol=1e-8)


def test_diffusion_scores_far_higher_than_near():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(200, 5))
    backend = DiffusionBackend(n_components=6, k=10).fit(ref)
    near = rng.normal(size=(20, 5))
    far = rng.normal(size=(20, 5)) + 8.0
    assert backend.score(far).mean() > backend.score(near).mean()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_monitoring_backends.py -v`
Expected: FAIL with `ImportError: cannot import name 'DiffusionBackend'`

- [ ] **Step 3: Implement DiffusionBackend**

Append to `microbe_model/monitoring/backends.py` (also add the two imports shown at top):

Add these imports to the existing import block at the top of the file:

```python
from scipy.sparse.linalg import eigsh
from sklearn.metrics.pairwise import rbf_kernel

from .mmd import median_gamma
```

Append the class:

```python
class DiffusionBackend:
    """OOD score = mean k-NN distance in diffusion-map coordinates.

    Diffusion coordinates are the discrete heat-kernel embedding of the
    reference manifold. Out-of-sample points are placed via Nyström extension:
    for any X, coords(X) = row_normalize(rbf(X, X_ref)) @ phi, which reproduces
    the reference coordinates exactly when X == X_ref.
    """

    def __init__(self, n_components: int = 10, k: int = 10, gamma: float | None = None) -> None:
        self.n_components = n_components
        self.k = k
        self.gamma = gamma

    def fit(self, X_ref: np.ndarray) -> "DiffusionBackend":
        X_ref = np.asarray(X_ref, dtype=float)
        self._X_ref = X_ref
        self.gamma_ = self.gamma if self.gamma is not None else median_gamma(X_ref)

        W = rbf_kernel(X_ref, X_ref, gamma=self.gamma_)
        d = W.sum(axis=1)
        d_inv_sqrt = 1.0 / np.sqrt(d)
        A = d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :]  # symmetric normalized affinity

        ncomp = min(self.n_components + 1, len(X_ref) - 1)
        vals, vecs = eigsh(A, k=ncomp, which="LM")
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        # Drop the trivial leading (constant) component.
        self.lambdas_ = vals[1:]
        self.phi_ = vecs[:, 1:] * d_inv_sqrt[:, None]  # right eigenvectors of the random walk

        self.coords_ref_ = self._coords(X_ref)
        k = min(self.k, len(X_ref))
        self._nn = NearestNeighbors(n_neighbors=k).fit(self.coords_ref_)
        return self

    def _coords(self, X: np.ndarray) -> np.ndarray:
        Wx = rbf_kernel(X, self._X_ref, gamma=self.gamma_)
        Px = Wx / Wx.sum(axis=1, keepdims=True)
        return Px @ self.phi_

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._coords(np.asarray(X, dtype=float))

    def score(self, X: np.ndarray) -> np.ndarray:
        coords = self._coords(np.asarray(X, dtype=float))
        dist, _ = self._nn.kneighbors(coords)
        return dist.mean(axis=1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_monitoring_backends.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add microbe_model/monitoring/backends.py tests/test_monitoring_backends.py
git commit -m "feat(monitoring): diffusion backend with Nystrom extension"
```

---

### Task 5: ReferenceManifold

**Files:**
- Create: `microbe_model/monitoring/reference.py`
- Test: `tests/test_monitoring_reference.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_monitoring_reference.py`:

```python
"""ReferenceManifold: standardize, calibrate threshold, score, persist."""
from __future__ import annotations

import numpy as np

from microbe_model.monitoring.backends import EuclideanBackend
from microbe_model.monitoring.reference import ReferenceManifold


def _fit_simple(seed=0):
    rng = np.random.default_rng(seed)
    ref = rng.normal(size=(300, 4))
    return ReferenceManifold(backend=EuclideanBackend(k=10), max_reference=None).fit(ref), rng


def test_flags_far_points_as_ood():
    m, rng = _fit_simple()
    far = rng.normal(size=(15, 4)) + 10.0
    assert m.is_ood(far).all()


def test_does_not_flag_inliers_en_masse():
    m, rng = _fit_simple()
    inliers = rng.normal(size=(100, 4))
    # threshold is the 95th pct of reference scores -> roughly <=10% inliers flagged
    assert m.is_ood(inliers).mean() < 0.25


def test_anchor_subsampling_caps_reference():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(500, 4))
    m = ReferenceManifold(backend=EuclideanBackend(k=5), max_reference=100, seed=0).fit(ref)
    # backend was fit on at most max_reference anchors
    assert m.backend._nn.n_samples_fit_ == 100


def test_save_load_roundtrip(tmp_path):
    m, rng = _fit_simple()
    X = rng.normal(size=(10, 4))
    before = m.ood_score(X)
    path = tmp_path / "manifold.pkl"
    m.save(path)
    loaded = ReferenceManifold.load(path)
    after = loaded.ood_score(X)
    assert np.allclose(before, after)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_monitoring_reference.py -v`
Expected: FAIL with `ModuleNotFoundError: microbe_model.monitoring.reference`

- [ ] **Step 3: Implement reference.py**

Create `microbe_model/monitoring/reference.py`:

```python
"""ReferenceManifold: the fitted reference distribution for OOD scoring.

Standardizes embeddings, optionally subsamples anchor points (so diffusion
eigendecomposition stays tractable on ~19k genomes), delegates scoring to a
distance backend, and calibrates an OOD threshold from reference self-scores.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import numpy as np

from .backends import DistanceBackend, EuclideanBackend


@dataclass
class ReferenceManifold:
    backend: DistanceBackend = field(default_factory=EuclideanBackend)
    max_reference: int | None = 4000
    threshold_quantile: float = 0.95
    seed: int = 0
    mean_: np.ndarray | None = field(default=None, repr=False)
    scale_: np.ndarray | None = field(default=None, repr=False)
    threshold_: float | None = None

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    def fit(self, X_ref: np.ndarray) -> "ReferenceManifold":
        X_ref = np.asarray(X_ref, dtype=float)
        self.mean_ = X_ref.mean(axis=0)
        scale = X_ref.std(axis=0)
        scale[scale == 0] = 1.0
        self.scale_ = scale

        Xs = self._standardize(X_ref)
        anchors = Xs
        if self.max_reference is not None and len(Xs) > self.max_reference:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(Xs), self.max_reference, replace=False)
            anchors = Xs[idx]

        self.backend.fit(anchors)
        ref_scores = self.backend.score(anchors)
        self.threshold_ = float(np.quantile(ref_scores, self.threshold_quantile))
        return self

    def ood_score(self, X: np.ndarray) -> np.ndarray:
        return self.backend.score(self._standardize(X))

    def is_ood(self, X: np.ndarray) -> np.ndarray:
        return self.ood_score(X) > self.threshold_

    def save(self, path) -> None:
        # pickle: the fitted state holds sklearn NearestNeighbors objects with no
        # clean JSON form. These are the user's own locally-fitted artifacts, not
        # untrusted input — only load manifolds you produced yourself.
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path) -> "ReferenceManifold":
        # Trust boundary: only load a manifold file you created (see save()).
        with open(path, "rb") as fh:
            return pickle.load(fh)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_monitoring_reference.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add microbe_model/monitoring/reference.py tests/test_monitoring_reference.py
git commit -m "feat(monitoring): ReferenceManifold with threshold calibration and persistence"
```

---

### Task 6: Drift classifier (Tier 1)

**Files:**
- Create: `microbe_model/monitoring/drift.py`
- Test: `tests/test_monitoring_drift.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_monitoring_drift.py`:

```python
"""assess_drift: no_drift under the null, data_drift under a real shift."""
from __future__ import annotations

import numpy as np

from microbe_model.monitoring.backends import EuclideanBackend
from microbe_model.monitoring.drift import assess_drift, DriftReport
from microbe_model.monitoring.reference import ReferenceManifold


def _manifold(seed=0):
    rng = np.random.default_rng(seed)
    ref = rng.normal(size=(300, 4))
    m = ReferenceManifold(backend=EuclideanBackend(k=10), max_reference=None).fit(ref)
    return m, ref, rng


def test_no_drift_on_same_distribution():
    m, ref, rng = _manifold()
    test = rng.normal(size=(80, 4))
    report = assess_drift(m, ref, test, n_perm=200, seed=0)
    assert isinstance(report, DriftReport)
    assert report.classification == "no_drift"


def test_data_drift_on_shift():
    m, ref, rng = _manifold()
    test = rng.normal(size=(80, 4)) + 6.0
    report = assess_drift(m, ref, test, n_perm=200, seed=0)
    assert report.classification == "data_drift"
    assert report.ood_rate > 0.5
    assert "Concept drift" in report.recommendation
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_monitoring_drift.py -v`
Expected: FAIL with `ModuleNotFoundError: microbe_model.monitoring.drift`

- [ ] **Step 3: Implement drift.py**

Create `microbe_model/monitoring/drift.py`:

```python
"""Tier 1 drift classifier.

Combines a per-genome OOD rate (Tier 0 scores vs the calibrated threshold) with a
population-level MMD permutation p-value. Flags data drift only; concept drift
(P(Y|X)) is unobservable from embeddings and is surfaced as a recommendation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mmd import mmd_permutation_test
from .reference import ReferenceManifold

DATA_DRIFT_REC = (
    "Data drift detected (P(X) shift). Characterize the new distribution, measure "
    "OOD rates, and confirm downstream performance is affected before retraining. "
    "Concept drift (P(Y|X)) is NOT observable from embeddings; acquire fresh labels "
    "to assess it."
)
NO_DRIFT_REC = (
    "No distributional shift detected at the chosen alpha. Embedding monitoring "
    "cannot rule out concept drift (P(Y|X)); only fresh labels can."
)


@dataclass
class DriftReport:
    n_test: int
    ood_rate: float
    ood_threshold: float
    mmd2: float
    p_value: float
    classification: str
    recommendation: str


def assess_drift(
    manifold: ReferenceManifold,
    X_ref_sample: np.ndarray,
    X_test: np.ndarray,
    *,
    alpha: float = 0.05,
    ood_rate_threshold: float = 0.2,
    n_perm: int = 200,
    seed: int = 0,
) -> DriftReport:
    """Assess whether X_test has drifted from the reference distribution."""
    X_test = np.asarray(X_test, dtype=float)
    scores = manifold.ood_score(X_test)
    ood_rate = float((scores > manifold.threshold_).mean())

    mmd2, p_value = mmd_permutation_test(
        manifold._standardize(X_ref_sample),
        manifold._standardize(X_test),
        n_perm=n_perm,
        seed=seed,
    )

    drifted = (p_value < alpha) or (ood_rate > ood_rate_threshold)
    classification = "data_drift" if drifted else "no_drift"
    recommendation = DATA_DRIFT_REC if drifted else NO_DRIFT_REC
    return DriftReport(
        n_test=len(X_test),
        ood_rate=ood_rate,
        ood_threshold=float(manifold.threshold_),
        mmd2=mmd2,
        p_value=p_value,
        classification=classification,
        recommendation=recommendation,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_monitoring_drift.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add microbe_model/monitoring/drift.py tests/test_monitoring_drift.py
git commit -m "feat(monitoring): Tier 1 drift classifier"
```

---

### Task 7: Family-split validation harness

**Files:**
- Create: `microbe_model/monitoring/validate.py`
- Test: `tests/test_monitoring_validate.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_monitoring_validate.py`:

```python
"""Validation harness: OOD scores separate out-of-clade from in-clade genomes."""
from __future__ import annotations

import numpy as np

from microbe_model.monitoring.backends import EuclideanBackend
from microbe_model.monitoring.validate import error_correlation, evaluate_backend


def test_evaluate_backend_separates_clusters():
    rng = np.random.default_rng(0)
    train = rng.normal(size=(300, 5))
    heldin = rng.normal(size=(100, 5))          # same distribution -> low OOD
    heldout = rng.normal(size=(100, 5)) + 8.0   # novel clade -> high OOD
    auroc, _, _ = evaluate_backend(
        train, heldout, heldin, EuclideanBackend(k=10), max_reference=None
    )
    assert auroc > 0.9


def test_error_correlation_monotonic():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    errors = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert error_correlation(scores, errors) > 0.99
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_monitoring_validate.py -v`
Expected: FAIL with `ModuleNotFoundError: microbe_model.monitoring.validate`

- [ ] **Step 3: Implement validate.py**

Create `microbe_model/monitoring/validate.py`:

```python
"""Family-split validation: does diffusion-distance beat the Euclidean baseline?

Reference = train families (in-distribution). Positives = held-out families
(novel sequence space). Negatives = held-in validation genomes. Reports AUROC of
the OOD score at separating novel-clade from in-clade genomes, and (optionally)
the Spearman correlation between OOD score and per-genome model error.

Run on real data (integration):
    python -m microbe_model.monitoring.validate \
        --features data/esm2_features.npz --splits data/splits.parquet
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .backends import DiffusionBackend, EuclideanBackend
from .reference import ReferenceManifold


def evaluate_backend(X_train, X_heldout, X_heldin, backend, **manifold_kwargs):
    """Fit on X_train; return (AUROC, fitted manifold, held-out OOD scores)."""
    manifold = ReferenceManifold(backend=backend, **manifold_kwargs).fit(X_train)
    s_out = manifold.ood_score(X_heldout)
    s_in = manifold.ood_score(X_heldin)
    labels = np.r_[np.ones(len(s_out)), np.zeros(len(s_in))]
    scores = np.r_[s_out, s_in]
    return float(roc_auc_score(labels, scores)), manifold, s_out


def error_correlation(ood_scores, errors) -> float:
    """Spearman rho between per-genome OOD score and per-genome model error."""
    rho, _ = spearmanr(np.asarray(ood_scores), np.asarray(errors))
    return float(rho)


def load_family_split(features_path: str, splits_path: str):
    """Return (X_train, X_heldout, X_heldin) embeddings for the family split."""
    import pandas as pd

    # No allow_pickle: bacdive_ids/accessions are fixed-width unicode arrays.
    data = np.load(features_path)
    feats = data["features"]
    ids = data["bacdive_ids"]
    sp = pd.read_parquet(splits_path)[["bacdive_id", "family_split"]]
    id_to_split = dict(zip(sp["bacdive_id"].astype(str), sp["family_split"]))
    split_of = np.array([id_to_split.get(str(i), "unknown") for i in ids])
    return (
        feats[split_of == "train"],
        feats[split_of == "test"],
        feats[split_of == "val"],
    )


def run_family_split_validation(features_path: str, splits_path: str, **manifold_kwargs):
    """Compare Euclidean vs diffusion backends on the family split. Returns a dict."""
    X_train, X_heldout, X_heldin = load_family_split(features_path, splits_path)
    results = {}
    for name, backend in (("euclidean", EuclideanBackend()), ("diffusion", DiffusionBackend())):
        auroc, _, _ = evaluate_backend(X_train, X_heldout, X_heldin, backend, **manifold_kwargs)
        results[name] = {
            "auroc": auroc,
            "n_train": int(len(X_train)),
            "n_heldout": int(len(X_heldout)),
            "n_heldin": int(len(X_heldin)),
        }
    return results


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="data/esm2_features.npz")
    ap.add_argument("--splits", default="data/splits.parquet")
    args = ap.parse_args()
    results = run_family_split_validation(args.features, args.splits)
    for name, r in results.items():
        print(f"{name:10s} AUROC={r['auroc']:.4f}  "
              f"(train={r['n_train']}, heldout={r['n_heldout']}, heldin={r['n_heldin']})")


if __name__ == "__main__":
    _main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_monitoring_validate.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add microbe_model/monitoring/validate.py tests/test_monitoring_validate.py
git commit -m "feat(monitoring): family-split validation harness"
```

---

### Task 8: Public exports + full-suite verification

**Files:**
- Modify: `microbe_model/monitoring/__init__.py`
- Test: `tests/test_monitoring_init.py`

- [ ] **Step 1: Add a failing test for the public API**

Append to `tests/test_monitoring_init.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_monitoring_init.py::test_public_api_exports -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Add exports to __init__.py**

Append to `microbe_model/monitoring/__init__.py`:

```python
from .backends import DiffusionBackend, EuclideanBackend
from .drift import DriftReport, assess_drift
from .mmd import mmd_permutation_test
from .reference import ReferenceManifold

__all__ = [
    "DiffusionBackend",
    "EuclideanBackend",
    "DriftReport",
    "assess_drift",
    "mmd_permutation_test",
    "ReferenceManifold",
]
```

- [ ] **Step 4: Run the full monitoring suite**

Run: `pytest tests/test_monitoring_init.py tests/test_monitoring_mmd.py tests/test_monitoring_backends.py tests/test_monitoring_reference.py tests/test_monitoring_drift.py tests/test_monitoring_validate.py -v`
Expected: PASS (all monitoring tests green)

- [ ] **Step 5: Run the whole repo suite to confirm no regressions**

Run: `pytest`
Expected: PASS (existing tests still green; integration tests skipped by default)

- [ ] **Step 6: Commit**

```bash
git add microbe_model/monitoring/__init__.py tests/test_monitoring_init.py
git commit -m "feat(monitoring): export public API and verify full suite"
```

---

## Post-implementation: run the real validation

After the suite is green, run the family-split experiment on real embeddings (this is the go/no-go for the geometry, not part of the automated suite):

```bash
python -m microbe_model.monitoring.validate --features data/esm2_features.npz --splits data/splits.parquet
```

Record both AUROCs. **Success = diffusion AUROC > euclidean AUROC** at flagging held-out-family genomes. If diffusion does not beat Euclidean, that is a valid result: keep the simpler Euclidean backend and document it. This decides whether Tiers 2–4 build on the diffusion or Euclidean OOD signal.

## Self-review notes

- **Spec coverage:** Tier 0 (`reference.py` + `backends.py`), Tier 1 (`drift.py`), validation harness (`validate.py`), Nyström out-of-sample (Task 4), anchor subsampling for the 19k-genome scale (Task 5), concept-drift-out-of-scope recommendation (Task 6). Tiers 2–4 are deferred per the spec and intentionally absent.
- **Type consistency:** `fit`/`score` signatures match across `EuclideanBackend`, `DiffusionBackend`, and the `DistanceBackend` Protocol; `ReferenceManifold._standardize`/`ood_score`/`threshold_` are used consistently by `assess_drift`; `evaluate_backend` returns `(auroc, manifold, scores)` as consumed by the harness.
- **No placeholders:** every code step is complete and runnable.
