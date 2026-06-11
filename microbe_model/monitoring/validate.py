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
