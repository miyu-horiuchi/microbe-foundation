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


def load_family_split(features_path: str, splits_path: str, *, indist_frac: float = 0.2, seed: int = 0):
    """Return (X_ref, X_novel, X_indist) embeddings for an OOD-detection eval.

    Correct OOD framing on the family split: the reference and the in-distribution
    negatives must come from the SAME families, differing only by strain; the
    positives come from families never seen in training.

    - X_ref     : a (1 - indist_frac) slice of train-family genomes -> fits the manifold.
    - X_indist  : the held-out indist_frac of train-family genomes (seen families,
                  unseen strains) -> in-distribution NEGATIVES.
    - X_novel   : test-family genomes (novel families) -> OOD POSITIVES.

    Note: val/test families are themselves disjoint from train families, so val
    genomes are NOT in-distribution and must not be used as negatives.
    """
    import pandas as pd

    # No allow_pickle: bacdive_ids/accessions are fixed-width unicode arrays.
    data = np.load(features_path)
    feats = data["features"]
    ids = data["bacdive_ids"]
    sp = pd.read_parquet(splits_path)[["bacdive_id", "family_split"]]
    id_to_split = dict(zip(sp["bacdive_id"].astype(str), sp["family_split"]))
    split_of = np.array([id_to_split.get(str(i), "unknown") for i in ids])

    train = feats[split_of == "train"]
    novel = feats[split_of == "test"]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train))
    n_indist = int(round(len(train) * indist_frac))
    indist = train[perm[:n_indist]]
    ref = train[perm[n_indist:]]
    return ref, novel, indist


def run_family_split_validation(features_path: str, splits_path: str, **manifold_kwargs):
    """Compare Euclidean vs diffusion backends on the family split. Returns a dict.

    Positives = novel-family genomes (OOD); negatives = held-out train-family
    genomes (in-distribution). AUROC measures how well the OOD score separates
    genuinely novel families from unseen strains of known families.
    """
    X_ref, X_novel, X_indist = load_family_split(features_path, splits_path)
    results = {}
    for name, backend in (("euclidean", EuclideanBackend()), ("diffusion", DiffusionBackend())):
        auroc, _, _ = evaluate_backend(X_ref, X_novel, X_indist, backend, **manifold_kwargs)
        results[name] = {
            "auroc": auroc,
            "n_ref": int(len(X_ref)),
            "n_novel": int(len(X_novel)),
            "n_indist": int(len(X_indist)),
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
              f"(ref={r['n_ref']}, novel={r['n_novel']}, indist={r['n_indist']})")


if __name__ == "__main__":
    _main()
