# Capacity Ladder, Fusion, and Stacking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three analyses to the baseline-regressor paper — a capacity-ladder plateau/escalation diagnostic, embedding+extra-data fusion, and learned stacking — as a new module that reuses `paper/baseline_regressors.py`.

**Architecture:** New module `paper/baseline_capacity_ladder.py` imports the existing loading/target/estimator/metrics helpers from `baseline_regressors.py` (Table 32 untouched). Pure functions (categorical encoders, plateau detection, escalation verdict, OOF stacking) are unit-tested; the three runners are verified by a 3-trait smoke run. Figures are added to `paper/figures/make_baseline_figures.py`. The paper's three new subsections are written last, from real outputs.

**Tech Stack:** Python 3.9, numpy, pandas, scikit-learn 1.6, matplotlib 3.9, pytest 6.2.

## Global Constraints

- CPU-only; no encoder fine-tuning; single seed 0.
- Do NOT modify `paper/baseline_regressors.py` or `paper/tables/32_baseline_regressors.*` — Table 32 must stay byte-identical.
- Primary metric per target = AUPRC when family-test positive rate < 0.20, else AUROC (reuse `baseline_regressors.primary_metric`).
- Family-held-out split from `data/splits.parquet` (`fsplit` ∈ {train, test}).
- Extra-data / second-embedding encoders are fit on TRAIN rows only; unseen or missing test categories map to an all-zero row (no leakage).
- Plateau EPS = 0.005; escalation MARGIN = 0.02; stacking GroupKFold n_splits = 5 (skip target if train has < 5 distinct families).
- Coverage ceiling = `knn_auroc` column of `paper/tables/33_cosine_family_collapse_summary.csv`, keyed by `trait`; missing trait ⇒ ceiling NaN ⇒ verdict `unknown`.
- Tests live in `tests/`, follow `tests/test_baseline_regressors.py` (insert `paper/` on `sys.path`, import module, small pandas frames).
- Run tests with `python -m pytest tests/test_baseline_capacity_ladder.py -v`.

---

### Task 1: Module scaffold + categorical extra-data encoders

**Files:**
- Create: `paper/baseline_capacity_ladder.py`
- Test: `tests/test_baseline_capacity_ladder.py`

**Interfaces:**
- Consumes: `baseline_regressors.load_aligned_frame`, `.Target`, `.build_estimator`, `.predict_probability`, `.metrics_row`, `.primary_metric`, `.collect_targets`.
- Produces:
  - `build_onehot(train_series: pd.Series, top_k: int | None = None) -> list[str]` — sorted category list from train (top-K by frequency if `top_k` set).
  - `apply_onehot(series: pd.Series, categories: list[str]) -> np.ndarray` — shape `(len(series), len(categories))`; missing/unseen → all-zero row.
  - `TAXONOMY_COLS = ["phylum", "class", "order"]`, `ISOLATION_COLS = ["isolation_source", "country"]`.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the capacity-ladder / fusion / stacking module."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper"))
import baseline_capacity_ladder as cl


def test_apply_onehot_maps_unseen_and_missing_to_zero_row():
    train = pd.Series(["a", "b", "a", "b"])
    cats = cl.build_onehot(train)
    assert cats == ["a", "b"]
    test = pd.Series(["a", "c", None])
    mat = cl.apply_onehot(test, cats)
    assert mat.shape == (3, 2)
    assert mat[0].tolist() == [1.0, 0.0]   # known "a"
    assert mat[1].tolist() == [0.0, 0.0]   # unseen "c" -> zero row
    assert mat[2].tolist() == [0.0, 0.0]   # missing -> zero row


def test_build_onehot_top_k_keeps_most_frequent():
    train = pd.Series(["a", "a", "a", "b", "b", "c"])
    assert cl.build_onehot(train, top_k=2) == ["a", "b"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_baseline_capacity_ladder.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'baseline_capacity_ladder'`.

- [ ] **Step 3: Write minimal implementation**

Create `paper/baseline_capacity_ladder.py`:

```python
#!/usr/bin/env python3
"""Capacity-ladder plateau/escalation diagnostic, extra-data fusion, and
learned stacking on frozen genome embeddings.

Companion to paper/baseline_regressors.py (Table 32). Reuses its loaders,
targets, estimators, and metrics; adds Tables 34 (capacity ladder), 35
(fusion), and 36 (stacking). CPU-only, single seed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_regressors as br  # noqa: E402

TAXONOMY_COLS = ["phylum", "class", "order"]
ISOLATION_COLS = ["isolation_source", "country"]


def build_onehot(train_series: pd.Series, top_k: int | None = None) -> list[str]:
    vals = train_series.dropna().astype(str)
    counts = vals.value_counts()
    cats = list(counts.index) if top_k is None else list(counts.index[:top_k])
    return sorted(cats) if top_k is None else list(cats)


def apply_onehot(series: pd.Series, categories: list[str]) -> np.ndarray:
    index = {c: i for i, c in enumerate(categories)}
    mat = np.zeros((len(series), len(categories)), dtype=np.float32)
    for row, v in enumerate(series.tolist()):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        j = index.get(str(v))
        if j is not None:
            mat[row, j] = 1.0
    return mat
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_baseline_capacity_ladder.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add paper/baseline_capacity_ladder.py tests/test_baseline_capacity_ladder.py
git commit -m "feat(ladder): scaffold module + train-fit one-hot encoders"
```

---

### Task 2: Plateau detection + escalation verdict

**Files:**
- Modify: `paper/baseline_capacity_ladder.py`
- Test: `tests/test_baseline_capacity_ladder.py`

**Interfaces:**
- Produces:
  - `LADDER: list[tuple[str, int | None]]` — ordered rungs `[("logistic_l2",6),("logistic_l2",10),("logistic_l2",25),("logistic_l2",50),("logistic_l2",100),("logistic_l2",None),("regularized_rf",None),("hist_gd",None)]`.
  - `detect_plateau(metrics: list[float], eps: float = 0.005) -> dict` → keys `lower_bound`, `plateau_index`, `plateau_value`, `plateaued`.
  - `escalation_verdict(plateau_value: float, ceiling: float, plateaued: bool, margin: float = 0.02) -> str` → one of `keep-scaling-simple`, `NO — coverage-limited`, `YES — capacity-limited`, `unknown`.

- [ ] **Step 1: Write the failing test**

```python
def test_detect_plateau_finds_first_flat_run():
    # rises then flattens: gains 0.05, 0.05, 0.003, 0.001, 0.000, 0.000
    metrics = [0.70, 0.75, 0.80, 0.803, 0.804, 0.804, 0.804, 0.804]
    out = cl.detect_plateau(metrics, eps=0.005)
    assert out["lower_bound"] == 0.70
    assert out["plateaued"] is True
    assert out["plateau_index"] == 4      # first rung of the 2-in-a-row flat run
    assert abs(out["plateau_value"] - 0.804) < 1e-9


def test_detect_plateau_reports_not_plateaued_when_still_climbing():
    metrics = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    out = cl.detect_plateau(metrics, eps=0.005)
    assert out["plateaued"] is False
    assert out["plateau_index"] == len(metrics) - 1


def test_escalation_verdict_three_cases_plus_unknown():
    assert cl.escalation_verdict(0.80, 0.95, plateaued=False) == "keep-scaling-simple"
    assert cl.escalation_verdict(0.80, 0.81, plateaued=True) == "NO — coverage-limited"
    assert cl.escalation_verdict(0.80, 0.95, plateaued=True) == "YES — capacity-limited"
    assert cl.escalation_verdict(0.80, float("nan"), plateaued=True) == "unknown"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_baseline_capacity_ladder.py -k "plateau or verdict" -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'detect_plateau'`.

- [ ] **Step 3: Write minimal implementation**

Append to `paper/baseline_capacity_ladder.py`:

```python
LADDER: list[tuple[str, int | None]] = [
    ("logistic_l2", 6),
    ("logistic_l2", 10),
    ("logistic_l2", 25),
    ("logistic_l2", 50),
    ("logistic_l2", 100),
    ("logistic_l2", None),
    ("regularized_rf", None),
    ("hist_gd", None),
]


def detect_plateau(metrics: list[float], eps: float = 0.005) -> dict:
    lower_bound = metrics[0]
    running = metrics[0]
    flat = 0
    for i in range(1, len(metrics)):
        gain = metrics[i] - running
        running = max(running, metrics[i])
        if gain < eps:
            flat += 1
            if flat >= 2:
                return {
                    "lower_bound": lower_bound,
                    "plateau_index": i - 1,
                    "plateau_value": metrics[i - 1],
                    "plateaued": True,
                }
        else:
            flat = 0
    return {
        "lower_bound": lower_bound,
        "plateau_index": len(metrics) - 1,
        "plateau_value": metrics[-1],
        "plateaued": False,
    }


def escalation_verdict(plateau_value: float, ceiling: float, plateaued: bool,
                       margin: float = 0.02) -> str:
    if not plateaued:
        return "keep-scaling-simple"
    if ceiling is None or np.isnan(ceiling):
        return "unknown"
    return "NO — coverage-limited" if (ceiling - plateau_value) <= margin else "YES — capacity-limited"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_baseline_capacity_ladder.py -k "plateau or verdict" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add paper/baseline_capacity_ladder.py tests/test_baseline_capacity_ladder.py
git commit -m "feat(ladder): plateau detection + escalation verdict rule"
```

---

### Task 3: Ceiling loader + capacity-ladder runner (Table 34)

**Files:**
- Modify: `paper/baseline_capacity_ladder.py`
- Test: `tests/test_baseline_capacity_ladder.py`

**Interfaces:**
- Produces:
  - `load_ceilings(path: str | Path) -> dict[str, float]` — maps `trait` → `knn_auroc`.
  - `run_capacity_ladder(feats, df, targets, ceilings, seed) -> tuple[list[dict], list[dict]]` — `(rung_rows, summary_rows)`. `rung_rows`: one per (target, rung) with `metric`. `summary_rows`: per target with `lower_bound`, `plateau_value`, `plateaued`, `ceiling`, `verdict`, `primary_metric`.

Ladder metric per rung reuses the Table-32 machinery: split train/test rows by `fsplit`, build the estimator with `br.build_estimator(model, rank, x_train, seed)` (chance excluded — ladder starts at rung 1), predict with `br.predict_probability`, score with `br.metrics_row`, pick metric via `br.primary_metric(row_with_test_pos_rate)`.

- [ ] **Step 1: Write the failing test**

```python
def test_load_ceilings_reads_knn_auroc(tmp_path):
    csv = tmp_path / "s.csv"
    csv.write_text("trait,knn_auroc,probe_auroc\nmotility,0.695,0.728\nsporulation,0.924,0.935\n")
    c = cl.load_ceilings(csv)
    assert abs(c["sporulation"] - 0.924) < 1e-9
    assert abs(c["motility"] - 0.695) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_baseline_capacity_ladder.py -k ceilings -v`
Expected: FAIL — `AttributeError: ... 'load_ceilings'`.

- [ ] **Step 3: Write minimal implementation**

Append to `paper/baseline_capacity_ladder.py`:

```python
def load_ceilings(path: str | Path) -> dict[str, float]:
    df = pd.read_csv(path)
    return {str(t): float(v) for t, v in zip(df["trait"], df["knn_auroc"])}


def _train_test_arrays(feats, df, target, seed):
    sub = df[target.mask].copy()
    y = target.y[target.mask]
    is_train = (sub["fsplit"] == "train").to_numpy()
    is_test = (sub["fsplit"] == "test").to_numpy()
    if is_train.sum() == 0 or is_test.sum() == 0:
        return None
    if len(np.unique(y[is_train])) < 2 or len(np.unique(y[is_test])) < 2:
        return None
    return (
        sub,
        feats[sub["row"].to_numpy()[is_train]], y[is_train].astype(int),
        feats[sub["row"].to_numpy()[is_test]], y[is_test].astype(int),
    )


def _score(x_train, y_train, x_test, y_test, model, rank, seed):
    est = br.build_estimator(model, rank, x_train, seed)
    prob = br.predict_probability(br.clone(est), x_train, y_train, x_test)
    row = br.metrics_row(y_test, prob)
    row["test_pos_rate"] = float(y_test.mean())
    metric = br.primary_metric(row)
    return metric, float(row[metric])


def run_capacity_ladder(feats, df, targets, ceilings, seed):
    rung_rows, summary_rows = [], []
    for target in targets:
        arr = _train_test_arrays(feats, df, target, seed)
        if arr is None:
            continue
        _, x_train, y_train, x_test, y_test = arr
        metrics, metric_name = [], None
        for step, (model, rank) in enumerate(LADDER):
            metric_name, value = _score(x_train, y_train, x_test, y_test, model, rank, seed)
            metrics.append(value)
            rung_rows.append({
                "target": target.name, "group": target.group, "step": step,
                "model": model, "rank": br.rank_label(rank),
                "primary_metric": metric_name, "score": value,
            })
        pl = detect_plateau(metrics)
        ceiling = ceilings.get(target.name, float("nan"))
        verdict = escalation_verdict(pl["plateau_value"], ceiling, pl["plateaued"])
        summary_rows.append({
            "target": target.name, "group": target.group,
            "primary_metric": metric_name,
            "lower_bound": pl["lower_bound"],
            "plateau_step": pl["plateau_index"],
            "plateau_value": pl["plateau_value"],
            "plateaued": pl["plateaued"],
            "ceiling_knn_auroc": ceiling,
            "verdict": verdict,
        })
    return rung_rows, summary_rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_baseline_capacity_ladder.py -k ceilings -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add paper/baseline_capacity_ladder.py tests/test_baseline_capacity_ladder.py
git commit -m "feat(ladder): ceiling loader + capacity-ladder runner"
```

---

### Task 4: Fusion runner (Table 35)

**Files:**
- Modify: `paper/baseline_capacity_ladder.py`
- Test: `tests/test_baseline_capacity_ladder.py`

**Interfaces:**
- Produces:
  - `build_extra_matrix(sub: pd.DataFrame, source: str, train_mask: np.ndarray) -> np.ndarray` — `source` ∈ {`taxonomy`, `isolation`}; one-hot built from train rows only.
  - `load_second_embedding(path, bacdive_ids, n_components=50, seed=0) -> np.ndarray` — aligns a second npz to the current row order by `bacdive_ids`, PCA-reduces to `n_components`; genomes absent in the second file get a NaN row (imputed downstream).
  - `run_fusion(feats, df, targets, sources, second_embed_paths, seed) -> list[dict]` — rows `(target, source, arm ∈ {embed, extra, embed+extra}, model, metric, score, lift)`.

Fusion uses a fixed representative model subset: `[("logistic_l2",25),("regularized_rf",None),("hist_gd",None)]`. `lift = score(embed+extra) − score(embed)` for the same model. eggNOG (`data/eggnog_features_6738.npz`, 24 854 dims) MUST be PCA-reduced to 50 components before concat so it does not swamp the 640-dim ESM-2 vector.

- [ ] **Step 1: Write the failing test**

```python
def test_build_extra_matrix_taxonomy_is_train_fit():
    sub = pd.DataFrame({
        "phylum": ["P1", "P1", "P2", "P3"],
        "class":  ["C1", "C1", "C2", "C3"],
        "order":  ["O1", "O1", "O2", "O3"],
    })
    train_mask = np.array([True, True, False, False])
    mat = cl.build_extra_matrix(sub, "taxonomy", train_mask)
    # only P1/C1/O1 seen in train -> 3 columns; unseen test rows are all-zero
    assert mat.shape == (4, 3)
    assert mat[0].tolist() == [1.0, 1.0, 1.0]
    assert mat[2].tolist() == [0.0, 0.0, 0.0]
    assert mat[3].tolist() == [0.0, 0.0, 0.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_baseline_capacity_ladder.py -k extra_matrix -v`
Expected: FAIL — `AttributeError: ... 'build_extra_matrix'`.

- [ ] **Step 3: Write minimal implementation**

Append to `paper/baseline_capacity_ladder.py`:

```python
from sklearn.decomposition import PCA  # noqa: E402

FUSION_MODELS: list[tuple[str, int | None]] = [
    ("logistic_l2", 25), ("regularized_rf", None), ("hist_gd", None),
]
EXTRA_COLS = {"taxonomy": TAXONOMY_COLS, "isolation": ISOLATION_COLS}


def build_extra_matrix(sub: pd.DataFrame, source: str, train_mask: np.ndarray) -> np.ndarray:
    cols = EXTRA_COLS[source]
    top_k = None if source == "taxonomy" else 30
    blocks = []
    for col in cols:
        cats = build_onehot(sub[col][train_mask], top_k=top_k)
        blocks.append(apply_onehot(sub[col], cats))
    return np.hstack(blocks).astype(np.float32) if blocks else np.zeros((len(sub), 0), np.float32)


def load_second_embedding(path, bacdive_ids, n_components=50, seed=0):
    d = np.load(path, allow_pickle=False)
    ids = [str(i) for i in d["bacdive_ids"]]
    feats = np.asarray(d["features"], dtype=np.float32)
    if feats.shape[1] > n_components:
        feats = PCA(n_components=n_components, random_state=seed).fit_transform(feats)
    id_to_row = {b: i for i, b in enumerate(ids)}
    out = np.full((len(bacdive_ids), feats.shape[1]), np.nan, dtype=np.float32)
    for r, b in enumerate(bacdive_ids):
        j = id_to_row.get(str(b))
        if j is not None:
            out[r] = feats[j]
    return out


def _fusion_score(x_train, y_train, x_test, y_test, model, seed):
    metric, value = _score(x_train, y_train, x_test, y_test, model[0], model[1], seed)
    return metric, value


def run_fusion(feats, df, targets, sources, second_embed_paths, seed):
    rows = []
    for target in targets:
        sub = df[target.mask].copy()
        y = target.y[target.mask]
        is_train = (sub["fsplit"] == "train").to_numpy()
        is_test = (sub["fsplit"] == "test").to_numpy()
        if is_train.sum() == 0 or is_test.sum() == 0:
            continue
        if len(np.unique(y[is_train])) < 2 or len(np.unique(y[is_test])) < 2:
            continue
        rowsel = sub["row"].to_numpy()
        embed = feats[rowsel]
        yt, yte = y[is_train].astype(int), y[is_test].astype(int)

        extra_mats = {s: build_extra_matrix(sub, s, is_train) for s in sources}
        for path in second_embed_paths:
            extra_mats[f"embed:{Path(path).stem}"] = load_second_embedding(
                path, sub["bid"].tolist(), seed=seed)

        for source, extra in extra_mats.items():
            combos = {
                "embed": embed,
                "extra": extra,
                "embed+extra": np.hstack([embed, extra]),
            }
            base = {}
            for arm, X in combos.items():
                if X.shape[1] == 0:
                    continue
                for model in FUSION_MODELS:
                    metric, value = _fusion_score(
                        X[is_train], yt, X[is_test], yte, model, seed)
                    key = (model[0], br.rank_label(model[1]))
                    if arm == "embed":
                        base[key] = value
                    lift = value - base.get(key, float("nan")) if arm == "embed+extra" else float("nan")
                    rows.append({
                        "target": target.name, "group": target.group,
                        "source": source, "arm": arm,
                        "model": model[0], "rank": br.rank_label(model[1]),
                        "primary_metric": metric, "score": value, "lift": lift,
                    })
    return rows
```

Note: `br.build_estimator`'s pipeline imputes NaNs (median) then scales, so the NaN rows from `load_second_embedding` are handled without extra code.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_baseline_capacity_ladder.py -k extra_matrix -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add paper/baseline_capacity_ladder.py tests/test_baseline_capacity_ladder.py
git commit -m "feat(fusion): embedding + taxonomy/isolation/second-embedding fusion runner"
```

---

### Task 5: Learned stacking with GroupKFold-on-family (Table 36)

**Files:**
- Modify: `paper/baseline_capacity_ladder.py`
- Test: `tests/test_baseline_capacity_ladder.py`

**Interfaces:**
- Produces:
  - `oof_predictions(x_train, y_train, families_train, base_models, seed, n_splits=5) -> np.ndarray | None` — `(n_train, n_base)` out-of-fold probabilities using `GroupKFold` on `families_train`; returns `None` if distinct families < `n_splits`.
  - `run_stack(feats, df, targets, seed, n_splits=5) -> list[dict]` — rows `(target, method ∈ {best_base, soft_vote, learned_stack}, metric, score)`.

Base models = the Table-32 ensemble set: `[("logistic_l2",25),("sgd_logistic",25),("poly2_logistic",10),("regularized_rf",25),("hist_gd",25)]`.

- [ ] **Step 1: Write the failing test**

```python
def test_oof_predictions_uses_grouped_folds_without_family_leak():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 5)).astype(np.float32)
    y = (x[:, 0] > 0).astype(int)
    families = np.repeat([f"fam{i}" for i in range(8)], 5)  # 8 families
    base = [("logistic_l2", None)]
    oof = cl.oof_predictions(x, y, families, base, seed=0, n_splits=4)
    assert oof is not None
    assert oof.shape == (40, 1)
    assert np.all((oof >= 0) & (oof <= 1))


def test_oof_predictions_returns_none_when_too_few_families():
    x = np.zeros((10, 3), np.float32)
    y = np.array([0, 1] * 5)
    families = np.array(["a", "b"] * 5)      # only 2 families
    assert cl.oof_predictions(x, y, families, [("logistic_l2", None)], seed=0, n_splits=5) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_baseline_capacity_ladder.py -k oof -v`
Expected: FAIL — `AttributeError: ... 'oof_predictions'`.

- [ ] **Step 3: Write minimal implementation**

Append to `paper/baseline_capacity_ladder.py`:

```python
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

STACK_BASE: list[tuple[str, int | None]] = [
    ("logistic_l2", 25), ("sgd_logistic", 25), ("poly2_logistic", 10),
    ("regularized_rf", 25), ("hist_gd", 25),
]


def oof_predictions(x_train, y_train, families_train, base_models, seed, n_splits=5):
    if len(set(families_train.tolist())) < n_splits:
        return None
    oof = np.zeros((len(y_train), len(base_models)), dtype=np.float32)
    gkf = GroupKFold(n_splits=n_splits)
    for tr_idx, va_idx in gkf.split(x_train, y_train, families_train):
        if len(np.unique(y_train[tr_idx])) < 2:
            continue
        for m, (model, rank) in enumerate(base_models):
            est = br.build_estimator(model, rank, x_train[tr_idx], seed)
            oof[va_idx, m] = br.predict_probability(
                br.clone(est), x_train[tr_idx], y_train[tr_idx], x_train[va_idx])
    return oof


def run_stack(feats, df, targets, seed, n_splits=5):
    rows = []
    for target in targets:
        arr = _train_test_arrays(feats, df, target, seed)
        if arr is None:
            continue
        sub, x_train, y_train, x_test, y_test = arr
        fam_train = sub["family"].to_numpy()[(sub["fsplit"] == "train").to_numpy()]
        oof = oof_predictions(x_train, y_train, fam_train, STACK_BASE, seed, n_splits)

        test_probs = []
        for model, rank in STACK_BASE:
            est = br.build_estimator(model, rank, x_train, seed)
            test_probs.append(br.predict_probability(br.clone(est), x_train, y_train, x_test))
        test_probs = np.vstack(test_probs).T  # (n_test, n_base)

        def _emit(method, prob):
            row = br.metrics_row(y_test, prob)
            row["test_pos_rate"] = float(y_test.mean())
            metric = br.primary_metric(row)
            rows.append({"target": target.name, "group": target.group,
                         "method": method, "primary_metric": metric, "score": float(row[metric])})

        best_base = max(range(len(STACK_BASE)),
                        key=lambda m: br.metrics_row(y_test, test_probs[:, m])[
                            br.primary_metric({"test_pos_rate": float(y_test.mean())})])
        _emit("best_base", test_probs[:, best_base])
        _emit("soft_vote", test_probs.mean(axis=1))
        if oof is not None:
            meta = LogisticRegression(max_iter=3000, class_weight="balanced")
            meta.fit(oof, y_train)
            _emit("learned_stack", meta.predict_proba(test_probs)[:, 1])
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_baseline_capacity_ladder.py -k oof -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add paper/baseline_capacity_ladder.py tests/test_baseline_capacity_ladder.py
git commit -m "feat(stack): GroupKFold-on-family OOF learned stacking vs soft-vote"
```

---

### Task 6: CLI wiring + table writers + smoke run

**Files:**
- Modify: `paper/baseline_capacity_ladder.py`
- Verify: writes `paper/tables/34_capacity_ladder.{csv,md}`, `35_fusion.{csv,md}`, `36_stack.{csv,md}`.

**Interfaces:**
- Produces: `write_tables(out_dir, rung_rows, summary_rows, fusion_rows, stack_rows) -> None` and `main()`.

- [ ] **Step 1: Add markdown writers + main**

Append to `paper/baseline_capacity_ladder.py`:

```python
def _mean_lift_by_source(fusion_rows):
    by = {}
    for r in fusion_rows:
        if r["arm"] == "embed+extra" and not np.isnan(r["lift"]):
            by.setdefault(r["source"], []).append(r["lift"])
    return {s: float(np.mean(v)) for s, v in sorted(by.items())}


def write_tables(out_dir, rung_rows, summary_rows, fusion_rows, stack_rows):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rung_rows).to_csv(out / "34_capacity_ladder_rungs.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out / "34_capacity_ladder.csv", index=False)
    pd.DataFrame(fusion_rows).to_csv(out / "35_fusion.csv", index=False)
    pd.DataFrame(stack_rows).to_csv(out / "36_stack.csv", index=False)

    lines = ["# Table 34 -- Capacity ladder: plateau and escalation verdict", "",
             "Ladder: L2 logistic at PCA rank 6/10/25/50/100/full, then random forest and "
             "hist gradient boosting at full rank. Plateau EPS=0.005; escalation MARGIN=0.02; "
             "ceiling = cosine-kNN AUROC from Table 33.", "",
             "| Target | Primary | Lower bound | Plateau | Plateaued | Ceiling (kNN) | Verdict |",
             "|---|---|---:|---:|:--:|---:|---|"]
    for r in summary_rows:
        lines.append(f"| `{r['target']}` | {r['primary_metric']} | {r['lower_bound']:.3f} | "
                     f"{r['plateau_value']:.3f} | {r['plateaued']} | {r['ceiling_knn_auroc']:.3f} | {r['verdict']} |")
    (out / "34_capacity_ladder.md").write_text("\n".join(lines) + "\n")

    ml = _mean_lift_by_source(fusion_rows)
    flines = ["# Table 35 -- Embedding + extra-data fusion", "",
              "Lift = primary metric(embed+extra) − primary metric(embed), averaged over "
              "logistic/RF/hist-GB and all targets. Extra encoders fit on train only; unseen "
              "test categories map to zero. Taxonomy raising clade-confounded traits is not a "
              "clean generalization gain.", "",
              "| Extra-data source | Mean lift |", "|---|---:|"]
    for s, v in ml.items():
        flines.append(f"| {s} | {v:+.3f} |")
    (out / "35_fusion.md").write_text("\n".join(flines) + "\n")

    slines = ["# Table 36 -- Learned stack vs soft-vote", "",
              "Base learners = Table-32 ensemble set. Out-of-fold predictions use GroupKFold on "
              "`family` (no family leakage); meta-learner = balanced logistic regression.", "",
              "| Target | best_base | soft_vote | learned_stack |", "|---|---:|---:|---:|"]
    by_t = {}
    for r in stack_rows:
        by_t.setdefault(r["target"], {})[r["method"]] = r["score"]
    for t, d in sorted(by_t.items()):
        slines.append(f"| `{t}` | {d.get('best_base', float('nan')):.3f} | "
                      f"{d.get('soft_vote', float('nan')):.3f} | {d.get('learned_stack', float('nan')):.3f} |")
    (out / "36_stack.md").write_text("\n".join(slines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="data/esm2_features.npz")
    ap.add_argument("--splits", default="data/splits.parquet")
    ap.add_argument("--traits-path", default="data/traits.parquet")
    ap.add_argument("--targets", nargs="+", default=br.DEFAULT_BINARY_TARGETS)
    ap.add_argument("--top-media", type=int, default=5)
    ap.add_argument("--min-positive", type=int, default=50)
    ap.add_argument("--cosine-summary", default="paper/tables/33_cosine_family_collapse_summary.csv")
    ap.add_argument("--fusion-sources", nargs="+", default=["taxonomy", "isolation"])
    ap.add_argument("--second-embeddings", nargs="*", default=["data/eggnog_features_6738.npz"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="paper/tables")
    args = ap.parse_args()

    feats, df = br.load_aligned_frame(args.features, args.splits, args.traits_path)
    targets = br.collect_targets(df, args.targets, top_media=args.top_media, min_positive=args.min_positive)
    ceilings = load_ceilings(args.cosine_summary)

    rung_rows, summary_rows = run_capacity_ladder(feats, df, targets, ceilings, args.seed)
    fusion_rows = run_fusion(feats, df, targets, args.fusion_sources, args.second_embeddings, args.seed)
    stack_rows = run_stack(feats, df, targets, args.seed)
    write_tables(args.out_dir, rung_rows, summary_rows, fusion_rows, stack_rows)
    print(f"wrote 34/35/36 tables to {args.out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the 3-trait smoke run**

Run: `python paper/baseline_capacity_ladder.py --targets motility sporulation catalase --top-media 0 --second-embeddings`
Expected: prints `wrote 34/35/36 tables to paper/tables`; creates the six table files. (`--second-embeddings` with no value skips the eggNOG arm for a fast smoke.)

- [ ] **Step 3: Sanity-check the smoke output**

Run: `cat paper/tables/34_capacity_ladder.md`
Expected: sporulation row shows ceiling ≈ 0.924 (from Table 33) and a `NO — coverage-limited` or `YES` verdict consistent with its plateau; catalase/motility present.

- [ ] **Step 4: Confirm Table 32 is untouched**

Run: `git status --porcelain paper/tables/32_baseline_regressors.csv paper/baseline_regressors.py`
Expected: no output (unchanged).

- [ ] **Step 5: Commit**

```bash
git add paper/baseline_capacity_ladder.py paper/tables/34_capacity_ladder* paper/tables/35_fusion* paper/tables/36_stack*
git commit -m "feat(ladder): CLI + table writers; smoke run for 34/35/36"
```

---

### Task 7: Figures

**Files:**
- Modify: `paper/figures/make_baseline_figures.py`
- Verify: writes `paper/figures/capacity_ladder.{pdf,png}`, `fusion_lift.{pdf,png}`.

- [ ] **Step 1: Inspect the existing figure module's style**

Run: `sed -n '1,40p' paper/figures/make_baseline_figures.py`
Expected: see the matplotlib style/helpers (fonts, `savefig` for pdf+png) used by the existing `baseline_regressor_*` figures.

- [ ] **Step 2: Add two figure functions**

Add to `paper/figures/make_baseline_figures.py`, following that module's existing style helpers (reuse its rc-params, color list, and dual pdf/png save helper — do not introduce a new style):

```python
def fig_capacity_ladder(tables_dir, out_dir):
    import pandas as pd
    rungs = pd.read_csv(tables_dir / "34_capacity_ladder_rungs.csv")
    summ = pd.read_csv(tables_dir / "34_capacity_ladder.csv").set_index("target")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for target, g in rungs.groupby("target"):
        g = g.sort_values("step")
        ax.plot(g["step"], g["score"], marker="o", label=target)
        ceil = summ.loc[target, "ceiling_knn_auroc"] if target in summ.index else float("nan")
        if pd.notna(ceil):
            ax.axhline(ceil, ls="--", lw=0.8, alpha=0.5)
    ax.set_xticks(range(8))
    ax.set_xticklabels(["lr6", "lr10", "lr25", "lr50", "lr100", "lrfull", "rf", "histgb"], rotation=30)
    ax.set_xlabel("capacity rung (simple → complex)")
    ax.set_ylabel("primary metric on held-out families")
    ax.set_title("Capacity ladder: where simple readouts plateau")
    ax.legend(fontsize=7)
    _save(fig, out_dir / "capacity_ladder")   # reuse existing dual pdf/png saver


def fig_fusion_lift(tables_dir, out_dir):
    import pandas as pd
    import numpy as np
    fusion = pd.read_csv(tables_dir / "35_fusion.csv")
    sub = fusion[(fusion["arm"] == "embed+extra") & fusion["lift"].notna()]
    means = sub.groupby("source")["lift"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(means.index, means.values)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("mean lift over embedding-only")
    ax.set_title("Does cheap side-data help under family shift?")
    _save(fig, out_dir / "fusion_lift")
```

If the existing module's saver has a different name than `_save`, use that name; match its signature.

- [ ] **Step 3: Wire both into the module's `main()` and run**

Run: `python paper/figures/make_baseline_figures.py`
Expected: creates `paper/figures/capacity_ladder.pdf/.png` and `paper/figures/fusion_lift.pdf/.png` with no errors.

- [ ] **Step 4: Commit**

```bash
git add paper/figures/make_baseline_figures.py paper/figures/capacity_ladder.* paper/figures/fusion_lift.*
git commit -m "feat(figures): capacity-ladder and fusion-lift figures"
```

---

### Task 8: Full run + paper subsections

**Files:**
- Modify: `paper/baseline_regressor_family_shift.md`
- Regenerate: tables 34/35/36 + figures on the full target set.

- [ ] **Step 1: Full run on the default target set**

Run: `python paper/baseline_capacity_ladder.py`
Expected: tables 34/35/36 regenerated over 7 binary traits + top-5 media + eggNOG second-embedding arm. Note any target skipped for stacking (too few families) — it will simply be absent from `learned_stack` in `36_stack.md`; that is expected, not an error.

- [ ] **Step 2: Regenerate figures**

Run: `python paper/figures/make_baseline_figures.py`
Expected: figures updated from the full tables.

- [ ] **Step 3: Read the real numbers**

Run: `cat paper/tables/34_capacity_ladder.md paper/tables/35_fusion.md paper/tables/36_stack.md`
Expected: concrete verdict distribution, mean lifts per source, stack-vs-soft-vote scores — these are the numbers to quote in prose (do not invent numbers; copy from the tables).

- [ ] **Step 4: Add three subsections to the paper**

Insert after the Results section in `paper/baseline_regressor_family_shift.md`, using the actual numbers from Step 3:

1. **"A capacity ladder tells you when to stop."** State the rule (grow the readout until the primary metric plateaus, EPS=0.005; escalate only if the cosine-kNN coverage ceiling exceeds the plateau by > 0.02). Report the verdict distribution from Table 34; connect the dominant `NO — coverage-limited` verdicts to the Table-33 finding and to papers 2–3 (attention pooling / LoRA already tried and did not help). Reference Figure `capacity_ladder`.
2. **"Does cheap side-data help under family shift?"** Report per-source mean lift from Table 35; call out that higher-rank taxonomy helps clade-confounded traits (flag the confound) while the sequence-independent isolation metadata and the second embedding add what they add. Reference Figure `fusion_lift`.
3. **"Learned stacking vs a fixed vote."** Report from Table 36 whether the GroupKFold-on-family learned stack beats the fixed soft-vote on average; note the leakage-safe protocol.

- [ ] **Step 5: Update Reproducibility + Limitations**

In the same file: add `paper/baseline_capacity_ladder.py` and tables 34/35/36 + the two figures to Reproducibility; add to Limitations the single-seed caveat and the taxonomy clade-confound.

- [ ] **Step 6: Commit**

```bash
git add paper/baseline_regressor_family_shift.md paper/tables/34_capacity_ladder* paper/tables/35_fusion* paper/tables/36_stack* paper/figures/capacity_ladder.* paper/figures/fusion_lift.*
git commit -m "paper(baseline): add capacity-ladder, fusion, and stacking sections"
```

---

## Self-Review

**Spec coverage:**
- Piece 1 (capacity ladder → plateau → escalate): Tasks 2, 3, 6, 7, 8. ✓ (ladder, plateau rule EPS=0.005, verdict MARGIN=0.02, Table-33 knn ceiling, figure, prose).
- Piece 2 (fusion: taxonomy + isolation + second embedding, ablated, lift, clade-confound flag): Tasks 1, 4, 6, 7, 8. ✓
- Piece 3 (learned stack, GroupKFold-on-family, vs soft-vote): Tasks 5, 6, 8. ✓
- Non-goals honored: no transformer trained; Table 32 untouched (Task 6 Step 4 asserts it); single seed. ✓
- Defaults (7 traits + top-5 media, seed 0, CPU): Task 6 `main()` + Task 8 full run. ✓
- Risks: ceiling NaN → verdict `unknown` (Task 2/3); eggNOG alignment loss + 50-comp PCA (Task 4); runtime cap via smoke-first then full (Tasks 6, 8). ✓

**Placeholder scan:** No TBD/TODO; all code steps carry real code; the only deferred detail is the existing figure module's saver name, which Task 7 Step 1 inspects and Step 2 instructs to match — not a placeholder.

**Type consistency:** `_score`/`_train_test_arrays` defined in Task 3 and reused in Tasks 4–5; `build_onehot`/`apply_onehot` (Task 1) used by `build_extra_matrix` (Task 4); `LADDER`/`detect_plateau`/`escalation_verdict` (Task 2) used by `run_capacity_ladder` (Task 3); `STACK_BASE`/`oof_predictions` (Task 5) consistent; table filenames (`34_capacity_ladder.csv`, `34_capacity_ladder_rungs.csv`, `35_fusion.csv`, `36_stack.csv`) consistent between writer (Task 6) and figures (Task 7).
