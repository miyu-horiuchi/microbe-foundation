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


def sample_families(families, k: int, seed: int) -> list:
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


def family_diversity_curve(X_train, y_train, fam_train, X_test, y_test, ks, seeds) -> list:
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


def is_rising(diversity_rows, min_gain: float = 0.02) -> bool:
    """True if mean test_f1 at the largest k exceeds the smallest k by > min_gain."""
    if not diversity_rows:
        return False
    by_k: dict = {}
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
        by_k: dict = {}
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
        by_k: dict = {}
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
