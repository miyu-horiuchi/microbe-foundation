"""Reproducible family-split OOD->error analysis across the predictability gradient.

Question: does the Tier-0 embedding OOD score (microbe_model/monitoring) predict
per-genome model error on genuinely novel families?

For each binary trait we train a transparent linear probe (LogisticRegression on the
640-d ESM2 features) on family-train genomes, predict on family-test genomes (novel
families), and correlate per-genome |error| with the Euclidean OOD score (reference =
family-train genomes). A positive Spearman rho means "more novel -> more error".

Finding (see docs / project memory): the relationship is NOT general. It is positive
only for highly-learnable machinery traits (sporulation), null for most compositional
traits, and robustly NEGATIVE for imbalanced pathogenicity traits (novel genomes are
confidently-and-correctly non-pathogenic). The OOD score is a novelty detector, not a
per-genome error oracle.

Usage:
    python ood_error_analysis.py
    python ood_error_analysis.py --out-dir paper/tables
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from microbe_model.monitoring import EuclideanBackend, ReferenceManifold

# Binary-head traits spanning the predictability gradient.
MACHINERY = ["motility", "sporulation"]
COMPOSITIONAL = ["catalase", "cytochrome_oxidase", "pigmentation"]
OTHER = ["pathogenicity_animal", "pathogenicity_human"]
DEFAULT_TRAITS = MACHINERY + COMPOSITIONAL + OTHER


def binary_trait_labels(col: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Materialize a binary-head trait: (labels float 0/1, mask of labeled rows).

    Matches model.py prepare_labels: y = float(bool(value)); mask where not null.
    """
    mask = col.notna().to_numpy()
    labels = col.fillna(False).astype(bool).astype(float).to_numpy()
    return labels, mask


def evaluate_trait(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_ref: np.ndarray,
    *,
    k: int = 10,
) -> dict | None:
    """Linear-probe AUROC on novel families + Spearman(OOD score, |error|).

    Returns None if the trait is degenerate (a single class in train or test).
    """
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scaler.transform(X_train), y_train)
    proba = clf.predict_proba(scaler.transform(X_test))[:, 1]
    error = np.abs(y_test - proba)

    manifold = ReferenceManifold(backend=EuclideanBackend(k=k)).fit(X_ref)
    ood = manifold.ood_score(X_test)

    rho, p_value = spearmanr(ood, error)
    return {
        "n_test": int(len(y_test)),
        "pos_rate": float(y_test.mean()),
        "auroc": float(roc_auc_score(y_test, proba)),
        "spearman_ood_error": float(rho),
        "p_value": float(p_value),
    }


def _group(trait: str) -> str:
    if trait in MACHINERY:
        return "machinery"
    if trait in COMPOSITIONAL:
        return "compositional"
    return "other"


def run_gradient(features_path: str, splits_path: str, traits_path: str, traits=DEFAULT_TRAITS) -> list[dict]:
    """Compute the OOD->error correlation per trait on the family split."""
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

    X_ref = feats[tr.loc[tr["fsplit"] == "train", "row"].to_numpy()]

    rows = []
    for trait in traits:
        if trait not in tr.columns:
            continue
        y, mask = binary_trait_labels(tr[trait])
        sub = tr[mask]
        ys = y[mask]
        is_train = (sub["fsplit"] == "train").to_numpy()
        is_test = (sub["fsplit"] == "test").to_numpy()
        result = evaluate_trait(
            feats[sub["row"].to_numpy()[is_train]], ys[is_train],
            feats[sub["row"].to_numpy()[is_test]], ys[is_test],
            X_ref,
        )
        if result is None:
            continue
        rows.append({"trait": trait, "group": _group(trait), **result})
    return rows


def to_markdown(rows: list[dict]) -> str:
    head = "# Table 14 — Family-split OOD score vs per-genome error\n\n"
    intro = (
        "Linear-probe per-trait test of whether the embedding OOD score predicts per-genome "
        "error on novel families. Positive Spearman = more novel implies more error. The "
        "relationship is trait-specific: positive only for sporulation, null for most traits, "
        "negative for imbalanced pathogenicity. The OOD score is a novelty detector, not a "
        "per-genome error oracle.\n\n"
    )
    cols = "| Trait | Group | Test n | Pos rate | AUROC | Spearman(OOD,err) | p |\n"
    sep = "|---|---|---:|---:|---:|---:|---:|\n"
    body = ""
    for r in rows:
        body += (
            f"| `{r['trait']}` | {r['group']} | {r['n_test']} | {r['pos_rate']:.3f} | "
            f"{r['auroc']:.3f} | {r['spearman_ood_error']:+.4f} | {r['p_value']:.1e} |\n"
        )
    return head + intro + cols + sep + body


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="data/esm2_features.npz")
    ap.add_argument("--splits", default="data/splits.parquet")
    ap.add_argument("--traits", default="data/traits.parquet")
    ap.add_argument("--out-dir", default="paper/tables")
    args = ap.parse_args()

    rows = run_gradient(args.features, args.splits, args.traits)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "14_ood_error_gradient.md").write_text(to_markdown(rows))
    pd.DataFrame(rows).to_csv(out_dir / "14_ood_error_gradient.csv", index=False)
    print(to_markdown(rows))


if __name__ == "__main__":
    main()
