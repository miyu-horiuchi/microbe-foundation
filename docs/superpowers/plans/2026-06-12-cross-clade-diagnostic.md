# Cross-Clade Collapse Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CPU-only diagnostic that returns a verdict on *why* family-level trait transfer collapses — insufficient training-clade coverage vs a frozen-representation wall — via a train-family-diversity curve and cross-clade k-NN label transfer.

**Architecture:** A root-level analysis script `cross_clade_diagnostic.py` (mirrors `ood_error_analysis.py`) with pure, tested helpers, two diagnostic functions, an orchestrator that writes a results table + figure, and a CLI. Reuses a `LogisticRegression` probe, the family splits, and the existing `binary_trait_labels`.

**Tech Stack:** numpy, pandas, scikit-learn (LogisticRegression, NearestNeighbors, f1_score, roc_auc_score), matplotlib 3.4 (Agg). Python 3.9-safe (annotation-only typing under `from __future__ import annotations`). Tests: pytest plain-assert, run with `python -m pytest`.

**Spec:** `docs/superpowers/specs/2026-06-12-cross-clade-diagnostic-design.md`

---

### Task 1: Pure helpers — family sampling + k-NN majority

**Files:**
- Create: `cross_clade_diagnostic.py`
- Test: `tests/test_cross_clade_diagnostic.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cross_clade_diagnostic.py`:

```python
"""Cross-clade diagnostic: pure helpers and synthetic end-to-end plumbing."""
from __future__ import annotations

import numpy as np

import cross_clade_diagnostic as mod


def test_sample_families_deterministic_distinct_subset():
    fams = ["a", "b", "c", "d", "e", "a", "b"]  # duplicates collapse
    out = mod.sample_families(fams, k=3, seed=0)
    assert len(out) == 3
    assert len(set(out)) == 3
    assert set(out) <= {"a", "b", "c", "d", "e"}
    assert mod.sample_families(fams, k=3, seed=0) == out  # deterministic


def test_sample_families_caps_at_available():
    fams = ["a", "b"]
    assert sorted(mod.sample_families(fams, k=10, seed=1)) == ["a", "b"]


def test_knn_majority_predict_all_positive_and_all_negative():
    X_ref = np.array([[0.0], [0.1], [0.2], [10.0], [10.1], [10.2]])
    y_ref = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    # query near the positive cluster -> ~1.0; near the negative cluster -> ~0.0
    q = np.array([[0.05], [10.05]])
    probs = mod.knn_majority_predict(X_ref, y_ref, q, k=3)
    assert probs.shape == (2,)
    assert probs[0] == 1.0
    assert probs[1] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cross_clade_diagnostic.py -q`
Expected: FAIL with `ModuleNotFoundError: cross_clade_diagnostic`

- [ ] **Step 3: Create cross_clade_diagnostic.py with the helpers**

Create `cross_clade_diagnostic.py`:

```python
"""Cross-clade collapse diagnostic.

Returns a verdict on WHY family-level trait transfer collapses: insufficient
training-clade coverage (fixable without fine-tuning) vs a frozen-representation
wall (only encoder fine-tuning can help). Two diagnostics:

  A. Train-family-diversity curve: test macro-F1 vs number of training families.
  B. Cross-clade k-NN label transfer: do a novel genome's nearest training-family
     neighbours carry the label?

Reuses the 640-d ESM-2 features, the family split, and a LogisticRegression probe.
CPU-only; does not modify or fine-tune the encoder.

Usage:
    python cross_clade_diagnostic.py
    python cross_clade_diagnostic.py --out-dir paper/tables --fig-dir paper/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from ood_error_analysis import binary_trait_labels

TRAITS = ["sporulation", "motility", "catalase"]
DIVERSITY_KS = [5, 10, 20, 40, 80]   # plus "all" appended at run time
DIVERSITY_SEEDS = [0, 1, 2]
K_NN = 10


def sample_families(families, k: int, seed: int) -> list[str]:
    """Return k distinct families, drawn deterministically; all if k >= available."""
    uniq = sorted(set(families))
    if k >= len(uniq):
        return uniq
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(uniq), k, replace=False)
    return [uniq[i] for i in sorted(idx)]


def knn_majority_predict(X_ref, y_ref, X_query, k: int = K_NN) -> np.ndarray:
    """Predicted positive-probability = fraction of the k nearest X_ref that are positive."""
    X_ref = np.asarray(X_ref, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float)
    X_query = np.asarray(X_query, dtype=float)
    kk = min(k, len(X_ref))
    nn = NearestNeighbors(n_neighbors=kk).fit(X_ref)
    _, idx = nn.kneighbors(X_query)
    return y_ref[idx].mean(axis=1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cross_clade_diagnostic.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add cross_clade_diagnostic.py tests/test_cross_clade_diagnostic.py
git commit -m "feat(diagnostic): cross-clade helpers (family sampling, k-NN majority)"
```

---

### Task 2: The two diagnostics + trait data prep

**Files:**
- Modify: `cross_clade_diagnostic.py`
- Test: `tests/test_cross_clade_diagnostic.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_cross_clade_diagnostic.py`:

```python
def test_knn_transfer_separable_beats_chance():
    rng = np.random.default_rng(0)
    # two well-separated clusters, label = cluster
    X_train = np.vstack([rng.normal(0, 0.3, (60, 4)), rng.normal(6, 0.3, (60, 4))])
    y_train = np.r_[np.zeros(60), np.ones(60)]
    X_test = np.vstack([rng.normal(0, 0.3, (20, 4)), rng.normal(6, 0.3, (20, 4))])
    y_test = np.r_[np.zeros(20), np.ones(20)]
    res = mod.knn_transfer(X_train, y_train, X_test, y_test, k=5)
    assert set(res) >= {"knn_f1", "knn_auroc", "probe_f1", "probe_auroc", "chance_f1"}
    assert res["knn_auroc"] > 0.9
    assert res["knn_f1"] > res["chance_f1"]


def test_diversity_curve_returns_row_per_k_seed():
    rng = np.random.default_rng(1)
    n_fam = 12
    fams = np.repeat([f"fam{i}" for i in range(n_fam)], 10)
    X_train = rng.normal(size=(len(fams), 4))
    y_train = (X_train[:, 0] > 0).astype(float)
    X_test = rng.normal(size=(40, 4))
    y_test = (X_test[:, 0] > 0).astype(float)
    rows = mod.family_diversity_curve(X_train, y_train, list(fams), X_test, y_test,
                                      ks=[3, 6], seeds=[0, 1])
    # 2 ks x 2 seeds = 4 rows (no degenerate single-class subsets here)
    assert len(rows) == 4
    for r in rows:
        assert r["k_families"] in (3, 6)
        assert 0.0 <= r["test_f1"] <= 1.0
        assert r["n_train"] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cross_clade_diagnostic.py -q -k "transfer or diversity"`
Expected: FAIL (`AttributeError: module ... has no attribute 'knn_transfer'`)

- [ ] **Step 3: Implement the two diagnostics**

Append to `cross_clade_diagnostic.py`:

```python
def _probe(X_train, y_train, X_test):
    """Standardize on train, fit a balanced LogisticRegression, return P(test=1)."""
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scaler.transform(X_train), y_train)
    return clf.predict_proba(scaler.transform(X_test))[:, 1]


def _macro_f1(y_true, prob) -> float:
    return float(f1_score(y_true, (np.asarray(prob) > 0.5).astype(int), average="macro", zero_division=0))


def _chance_f1(y_true) -> float:
    """Macro-F1 of always predicting the training-majority class."""
    majority = 1 if np.mean(y_true) >= 0.5 else 0
    pred = np.full(len(y_true), majority)
    return float(f1_score(y_true, pred, average="macro", zero_division=0))


def knn_transfer(X_train, y_train, X_test, y_test, k: int = K_NN) -> dict:
    """Cross-clade k-NN label transfer vs the trained probe and a chance baseline.

    Standardizes embedding space on the training genomes; neighbours are drawn only
    from training-family genomes (no leakage).
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    y_test = np.asarray(y_test, dtype=float)

    scaler = StandardScaler().fit(X_train)
    knn_prob = knn_majority_predict(scaler.transform(X_train), y_train, scaler.transform(X_test), k=k)
    probe_prob = _probe(X_train, y_train, X_test)
    return {
        "knn_f1": _macro_f1(y_test, knn_prob),
        "knn_auroc": float(roc_auc_score(y_test, knn_prob)),
        "probe_f1": _macro_f1(y_test, probe_prob),
        "probe_auroc": float(roc_auc_score(y_test, probe_prob)),
        "chance_f1": _chance_f1(y_test),
        "n_test": int(len(y_test)),
        "pos_rate": float(y_test.mean()),
    }


def family_diversity_curve(X_train, y_train, fam_train, X_test, y_test, ks, seeds) -> list[dict]:
    """Test macro-F1 as a function of the number of distinct training families.

    For each (k, seed): sample k families, restrict training to genomes in them,
    train a probe, evaluate on the fixed test set. Single-class subsets are skipped
    (with a printed note).
    """
    fam_train = np.asarray(fam_train)
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    rows = []
    for k in ks:
        for seed in seeds:
            fams = sample_families(list(fam_train), k, seed)
            sel = np.isin(fam_train, list(fams))
            ytr = y_train[sel]
            if len(np.unique(ytr)) < 2:
                print(f"  skip k={k} seed={seed}: single-class subset")
                continue
            prob = _probe(X_train[sel], ytr, X_test)
            rows.append({
                "k_families": int(k),
                "seed": int(seed),
                "n_train": int(sel.sum()),
                "test_f1": _macro_f1(y_test, prob),
            })
    return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cross_clade_diagnostic.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add cross_clade_diagnostic.py tests/test_cross_clade_diagnostic.py
git commit -m "feat(diagnostic): k-NN transfer + family-diversity curve"
```

---

### Task 3: Orchestrator, outputs, CLI + real run

**Files:**
- Modify: `cross_clade_diagnostic.py`
- Test: `tests/test_cross_clade_diagnostic.py`
- Generates: `paper/tables/15_cross_clade_diagnostic.{md,csv}`, `paper/figures/cross_clade_diversity_curve.png`

- [ ] **Step 1: Add a failing test for the data-prep + verdict helpers**

Append to `tests/test_cross_clade_diagnostic.py`:

```python
def test_verdict_labels_match_signals():
    # diversity rising + knn good -> coverage; flat + poor -> wall
    assert mod.verdict(diversity_rising=True, knn_good=True) == "coverage-limited"
    assert mod.verdict(diversity_rising=False, knn_good=False) == "representation-wall"
    assert mod.verdict(diversity_rising=True, knn_good=False) == "mixed"


def test_is_rising_detects_monotone_gain():
    # mean test_f1 grows with k -> rising
    rows = [{"k_families": 5, "test_f1": 0.4}, {"k_families": 5, "test_f1": 0.42},
            {"k_families": 80, "test_f1": 0.6}, {"k_families": 80, "test_f1": 0.62}]
    assert mod.is_rising(rows) is True
    flat = [{"k_families": 5, "test_f1": 0.5}, {"k_families": 80, "test_f1": 0.5}]
    assert mod.is_rising(flat) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cross_clade_diagnostic.py -q -k "verdict or rising"`
Expected: FAIL (`AttributeError: ... 'verdict'`)

- [ ] **Step 3: Implement prep, verdict, orchestration, figure, CLI**

Append to `cross_clade_diagnostic.py`:

```python
def is_rising(diversity_rows, min_gain: float = 0.02) -> bool:
    """True if mean test_f1 at the largest k exceeds the smallest k by > min_gain."""
    if not diversity_rows:
        return False
    by_k: dict[int, list] = {}
    for r in diversity_rows:
        by_k.setdefault(r["k_families"], []).append(r["test_f1"])
    ks = sorted(by_k)
    lo = float(np.mean(by_k[ks[0]]))
    hi = float(np.mean(by_k[ks[-1]]))
    return (hi - lo) > min_gain


def verdict(diversity_rising: bool, knn_good: bool) -> str:
    if diversity_rising and knn_good:
        return "coverage-limited"
    if not diversity_rising and not knn_good:
        return "representation-wall"
    return "mixed"


def prepare_trait(feats, id_to_row, tr, trait):
    """Return (X_train, y_train, fam_train, X_test, y_test) for the family split."""
    y, mask = binary_trait_labels(tr[trait])
    sub = tr[mask]
    ys = y[mask]
    is_tr = (sub["fsplit"] == "train").to_numpy()
    is_te = (sub["fsplit"] == "test").to_numpy()
    rows_tr = sub["row"].to_numpy()[is_tr]
    rows_te = sub["row"].to_numpy()[is_te]
    return (feats[rows_tr], ys[is_tr], sub["family"].to_numpy()[is_tr],
            feats[rows_te], ys[is_te])


def run(features_path, splits_path, traits_path, traits=TRAITS):
    data = np.load(features_path)
    feats = data["features"]
    ids = np.array([str(i) for i in data["bacdive_ids"]])
    id_to_row = {b: i for i, b in enumerate(ids)}
    tr = pd.read_parquet(traits_path)
    tr["bid"] = tr["bacdive_id"].astype(str)
    tr = tr[tr["bid"].isin(id_to_row)].copy()
    sp = pd.read_parquet(splits_path)[["bacdive_id", "family_split"]]
    fam = dict(zip(sp["bacdive_id"].astype(str), sp["family_split"]))
    tr["fsplit"] = tr["bid"].map(lambda b: fam.get(b, "unknown"))
    tr["row"] = tr["bid"].map(id_to_row)

    diversity, transfer = {}, {}
    for trait in traits:
        if trait not in tr.columns:
            continue
        Xtr, ytr, fam_tr, Xte, yte = prepare_trait(feats, id_to_row, tr, trait)
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue
        n_fam = len(set(fam_tr))
        ks = [k for k in DIVERSITY_KS if k < n_fam] + [n_fam]
        diversity[trait] = family_diversity_curve(Xtr, ytr, fam_tr, Xte, yte, ks, DIVERSITY_SEEDS)
        transfer[trait] = knn_transfer(Xtr, ytr, Xte, yte)
    return diversity, transfer


def to_markdown(diversity, transfer) -> str:
    lines = ["# Table 15 — Cross-clade collapse diagnostic", "",
             "Why family-level transfer collapses: training-clade *coverage* vs a frozen-"
             "representation *wall*. Diagnostic A = test macro-F1 vs #training families; "
             "B = cross-clade k-NN label transfer vs the probe and a chance baseline.", "",
             "## B. Cross-clade k-NN transfer", "",
             "| Trait | Test n | Pos rate | k-NN F1 | k-NN AUROC | Probe F1 | Probe AUROC | Chance F1 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for trait, t in transfer.items():
        lines.append(f"| `{trait}` | {t['n_test']} | {t['pos_rate']:.3f} | {t['knn_f1']:.3f} | "
                     f"{t['knn_auroc']:.3f} | {t['probe_f1']:.3f} | {t['probe_auroc']:.3f} | {t['chance_f1']:.3f} |")
    lines += ["", "## A. Family-diversity curve (mean test macro-F1 over seeds)", "",
              "| Trait | k=min | k=max | rising? | verdict |", "|---|---:|---:|:--:|:--:|"]
    for trait, rows in diversity.items():
        if not rows:
            continue
        by_k: dict[int, list] = {}
        for r in rows:
            by_k.setdefault(r["k_families"], []).append(r["test_f1"])
        ks = sorted(by_k)
        rising = is_rising(rows)
        knn_good = transfer[trait]["knn_f1"] > transfer[trait]["chance_f1"] + 0.05
        lines.append(f"| `{trait}` | {np.mean(by_k[ks[0]]):.3f} | {np.mean(by_k[ks[-1]]):.3f} | "
                     f"{'yes' if rising else 'no'} | {verdict(rising, knn_good)} |")
    return "\n".join(lines) + "\n"


def _save_figure(diversity, fig_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for trait, rows in diversity.items():
        if not rows:
            continue
        by_k: dict[int, list] = {}
        for r in rows:
            by_k.setdefault(r["k_families"], []).append(r["test_f1"])
        ks = sorted(by_k)
        means = [float(np.mean(by_k[k])) for k in ks]
        ax.plot(ks, means, marker="o", label=trait)
    ax.set_xlabel("number of training families")
    ax.set_ylabel("family-test macro-F1")
    ax.set_title("Cross-clade transfer vs training-clade diversity")
    ax.legend()
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="data/esm2_features.npz")
    ap.add_argument("--splits", default="data/splits.parquet")
    ap.add_argument("--traits", default="data/traits.parquet")
    ap.add_argument("--out-dir", default="paper/tables")
    ap.add_argument("--fig-dir", default="paper/figures")
    args = ap.parse_args()

    diversity, transfer = run(args.features, args.splits, args.traits)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = to_markdown(diversity, transfer)
    (out_dir / "15_cross_clade_diagnostic.md").write_text(md)
    flat = [{"trait": t, **r} for t, rows in diversity.items() for r in rows]
    pd.DataFrame(flat).to_csv(out_dir / "15_cross_clade_diagnostic.csv", index=False)
    _save_figure(diversity, Path(args.fig_dir) / "cross_clade_diversity_curve.png")
    print(md)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cross_clade_diagnostic.py -q`
Expected: PASS (7 passed)

- [ ] **Step 5: Run on real data; generate table + figure**

Run: `python cross_clade_diagnostic.py --out-dir paper/tables --fig-dir paper/figures`
Expected: prints Table 15; writes `paper/tables/15_cross_clade_diagnostic.{md,csv}` and `paper/figures/cross_clade_diversity_curve.png`. Read the printed verdict per trait.

- [ ] **Step 6: Commit (script, tests, and generated artifacts)**

```bash
git add cross_clade_diagnostic.py tests/test_cross_clade_diagnostic.py \
        paper/tables/15_cross_clade_diagnostic.md paper/tables/15_cross_clade_diagnostic.csv \
        paper/figures/cross_clade_diversity_curve.png
git commit -m "feat(diagnostic): cross-clade verdict run (table + diversity-curve figure)"
```

- [ ] **Step 7: Full-suite regression check**

Run: `python -m pytest tests/ -q`
Expected: PASS — all pre-existing tests plus the new diagnostic tests green (integration deselected is fine).

---

## Self-review notes

- **Spec coverage:** Diagnostic A (`family_diversity_curve`, Task 2; figure in Task 3), Diagnostic B (`knn_transfer`, Task 2), triangulated verdict (`verdict`/`is_rising`, Task 3), traits sporulation/motility/catalase (`TRAITS`), binary-label reuse (`binary_trait_labels` imported from `ood_error_analysis`), single-class skip (in `family_diversity_curve`), chance + probe baselines (`knn_transfer`), committed table + figure (Task 3). All covered.
- **Type consistency:** `sample_families(families,k,seed)->list`, `knn_majority_predict(X_ref,y_ref,X_query,k)->ndarray`, `knn_transfer(...)->dict` with keys used verbatim in `to_markdown`, `family_diversity_curve(...)->list[dict]` with `k_families`/`seed`/`n_train`/`test_f1` used in `is_rising`/`to_markdown`/CSV, `verdict(bool,bool)->str`. Consistent across tasks.
- **No placeholders:** every code step is complete and runnable. `k = all` is realized as `n_fam` appended to `ks` in `run`. Figure uses Agg (matplotlib 3.4 confirmed present).
