#!/usr/bin/env python3
"""Table 32: frozen embeddings + simple downstream baseline regressors.

This experiment answers a concrete reviewer/interview critique:

    "Before fine-tuning ESM-2, have you tried a family of simple regressors /
    classifiers on the frozen embeddings? Maybe the downstream task only needs a
    low-dimensional signal, and a regularized model will plateau before a bigger
    encoder helps."

We keep the encoder fixed and vary only the downstream model capacity:

* chance / majority-class floor
* L2 logistic probe
* elastic-net SGD logistic probe ("gd")
* polynomial logistic probe on a low-rank PCA projection
* regularized random forest
* histogram gradient boosting
* a fixed soft-vote ensemble over representative simple models

The rank sweep (6, 10, 25, 50, 100, full by default) explicitly tests whether the
signal lives in a tiny subspace, as in the perturbation-data critique, or whether
performance continues improving with more embedding dimensions.

CPU-only; no encoder fine-tuning.

Usage:
    python paper/baseline_regressors.py
    python paper/baseline_regressors.py --features data/bacformer_features_all.npz
    python paper/baseline_regressors.py --targets motility sporulation catalase --ranks 6,10,25,full
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ood_error_analysis import binary_trait_labels  # noqa: E402


DEFAULT_BINARY_TARGETS = [
    "catalase",
    "cytochrome_oxidase",
    "sporulation",
    "pigmentation",
    "pathogenicity_human",
    "pathogenicity_animal",
    "motility",
]
DEFAULT_RANKS: tuple[int | None, ...] = (6, 10, 25, 50, 100, None)
ENSEMBLE_MEMBERS = {
    ("logistic_l2", 25),
    ("sgd_logistic", 25),
    ("poly2_logistic", 10),
    ("regularized_rf", 25),
    ("hist_gd", 25),
}


@dataclass(frozen=True)
class Target:
    """A binary one-vs-rest target aligned to the traits dataframe."""

    name: str
    group: str
    y: np.ndarray
    mask: np.ndarray


def parse_ranks(raw: str) -> tuple[int | None, ...]:
    """Parse comma-separated PCA ranks; `full` means no PCA bottleneck."""
    out: list[int | None] = []
    for part in raw.split(","):
        item = part.strip().lower()
        if not item:
            continue
        if item in {"full", "none", "all"}:
            out.append(None)
        else:
            rank = int(item)
            if rank <= 0:
                raise ValueError("ranks must be positive integers or 'full'")
            out.append(rank)
    if not out:
        raise ValueError("at least one rank is required")
    return tuple(out)


def rank_label(rank: int | None) -> str:
    return "full" if rank is None else str(rank)


def load_features(path: str | Path) -> tuple[np.ndarray, list[str]]:
    d = np.load(path, allow_pickle=False)
    return np.asarray(d["features"], dtype=np.float32), [str(i) for i in d["bacdive_ids"]]


def load_aligned_frame(
    features_path: str | Path,
    splits_path: str | Path,
    traits_path: str | Path,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return (features, traits frame) with `row` and `fsplit` columns attached."""
    feats, ids = load_features(features_path)
    id_to_row = {b: i for i, b in enumerate(ids)}

    tr = pd.read_parquet(traits_path)
    tr["bid"] = tr["bacdive_id"].astype(str)
    tr = tr[tr["bid"].isin(id_to_row)].copy()

    sp = pd.read_parquet(splits_path)[["bacdive_id", "family_split"]]
    split_by_id = dict(zip(sp["bacdive_id"].astype(str), sp["family_split"]))
    tr["fsplit"] = tr["bid"].map(lambda b: split_by_id.get(b, "unknown"))
    tr["row"] = tr["bid"].map(id_to_row).astype(int)
    return feats, tr


def _is_missing(v) -> bool:
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def multilabel_values(v) -> set[str]:
    """Normalize a list-like multilabel cell to a set of positive labels."""
    if _is_missing(v):
        return set()
    if isinstance(v, dict):
        return {str(k) for k, val in v.items() if bool(val)}
    if isinstance(v, str):
        return {v}
    if isinstance(v, Iterable):
        return {str(x) for x in v if x is not None}
    return set()


def top_multilabel_targets(
    df: pd.DataFrame,
    trait: str,
    *,
    top_k: int,
    min_positive: int,
) -> list[tuple[str, np.ndarray, np.ndarray, int]]:
    """Build one-vs-rest targets for the most frequent labels in a multilabel trait."""
    if trait not in df.columns or top_k <= 0:
        return []

    observed = df[trait].apply(lambda v: not _is_missing(v) and len(multilabel_values(v)) > 0).to_numpy()
    counts: dict[str, int] = {}
    for vals in df.loc[observed, trait].apply(multilabel_values):
        for val in vals:
            counts[val] = counts.get(val, 0) + 1

    labels = [
        label for label, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if count >= min_positive
    ][:top_k]

    out = []
    values = df[trait].apply(multilabel_values)
    for label in labels:
        y = values.apply(lambda vals: label in vals).astype(float).to_numpy()
        out.append((f"{trait}:{label}", y, observed.copy(), counts[label]))
    return out


def collect_targets(
    df: pd.DataFrame,
    binary_targets: list[str],
    *,
    top_media: int,
    min_positive: int,
) -> list[Target]:
    targets: list[Target] = []
    for trait in binary_targets:
        if trait not in df.columns:
            continue
        y, mask = binary_trait_labels(df[trait])
        targets.append(Target(trait, "binary_trait", y.astype(float), mask.astype(bool)))

    for name, y, mask, _count in top_multilabel_targets(
        df, "cultivation_medium", top_k=top_media, min_positive=min_positive
    ):
        targets.append(Target(name, "cultivation_medium_one_vs_rest", y.astype(float), mask.astype(bool)))
    return targets


def _cap_rank(rank: int | None, x_train: np.ndarray) -> int | None:
    if rank is None:
        return None
    max_rank = max(1, min(x_train.shape[0] - 1, x_train.shape[1]))
    return min(rank, max_rank)


def _preprocess(rank: int | None, x_train: np.ndarray, random_state: int) -> list[tuple[str, object]]:
    steps: list[tuple[str, object]] = [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
    capped = _cap_rank(rank, x_train)
    if capped is not None and capped < min(x_train.shape):
        steps.append(("pca", PCA(n_components=capped, random_state=random_state)))
    return steps


def build_estimator(model_name: str, rank: int | None, x_train: np.ndarray, random_state: int) -> Pipeline:
    steps = _preprocess(rank, x_train, random_state)
    if model_name == "logistic_l2":
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)
    elif model_name == "sgd_logistic":
        clf = SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=1e-4,
            l1_ratio=0.15,
            class_weight="balanced",
            max_iter=3000,
            tol=1e-4,
            random_state=random_state,
        )
    elif model_name == "poly2_logistic":
        steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
        steps.append(("poly_scale", StandardScaler()))
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.25)
    elif model_name == "regularized_rf":
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )
    elif model_name == "hist_gd":
        clf = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=random_state,
        )
    else:
        raise ValueError(f"unknown model: {model_name}")
    return Pipeline([*steps, ("clf", clf)])


def model_grid(ranks: tuple[int | None, ...], max_poly_rank: int) -> list[tuple[str, int | None]]:
    grid: list[tuple[str, int | None]] = [("chance", None)]
    for model_name in ("logistic_l2", "sgd_logistic", "regularized_rf", "hist_gd"):
        for rank in ranks:
            grid.append((model_name, rank))
    for rank in ranks:
        if rank is not None and rank <= max_poly_rank:
            grid.append(("poly2_logistic", rank))
    return grid


def predict_probability(estimator, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        estimator.fit(x_train, y_train)
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(x_test)[:, 1]
    score = estimator.decision_function(x_test)
    return 1.0 / (1.0 + np.exp(-score))


def safe_roc_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, prob))
    except ValueError:
        return float("nan")


def metrics_row(y_test: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    pred = (prob >= 0.5).astype(int)
    return {
        "acc": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "auroc": safe_roc_auc(y_test, prob),
        "auprc": float(average_precision_score(y_test, prob)),
    }


def evaluate_target(
    feats: np.ndarray,
    df: pd.DataFrame,
    target: Target,
    grid: list[tuple[str, int | None]],
    *,
    random_state: int,
) -> list[dict]:
    sub = df[target.mask].copy()
    y = target.y[target.mask]
    is_train = (sub["fsplit"] == "train").to_numpy()
    is_test = (sub["fsplit"] == "test").to_numpy()

    if is_train.sum() == 0 or is_test.sum() == 0:
        return []
    if len(np.unique(y[is_train])) < 2 or len(np.unique(y[is_test])) < 2:
        return []

    x_train = feats[sub["row"].to_numpy()[is_train]]
    y_train = y[is_train].astype(int)
    x_test = feats[sub["row"].to_numpy()[is_test]]
    y_test = y[is_test].astype(int)

    rows: list[dict] = []
    probs_for_ensemble: dict[tuple[str, int | None], np.ndarray] = {}
    for model_name, rank in grid:
        try:
            if model_name == "chance":
                est = DummyClassifier(strategy="most_frequent")
            else:
                est = build_estimator(model_name, rank, x_train, random_state)
            prob = predict_probability(clone(est), x_train, y_train, x_test)
        except Exception as e:
            rows.append({
                "target": target.name,
                "group": target.group,
                "model": model_name,
                "rank": rank_label(rank),
                "status": f"failed: {type(e).__name__}: {e}",
            })
            continue

        if (model_name, rank) in ENSEMBLE_MEMBERS:
            probs_for_ensemble[(model_name, rank)] = prob
        row = {
            "target": target.name,
            "group": target.group,
            "model": model_name,
            "rank": rank_label(rank),
            "status": "ok",
            "n_train": int(is_train.sum()),
            "n_test": int(is_test.sum()),
            "train_pos_rate": float(y_train.mean()),
            "test_pos_rate": float(y_test.mean()),
            **metrics_row(y_test, prob),
        }
        rows.append(row)

    if len(probs_for_ensemble) >= 2:
        prob = np.mean(np.vstack(list(probs_for_ensemble.values())), axis=0)
        rows.append({
            "target": target.name,
            "group": target.group,
            "model": "soft_vote_ensemble",
            "rank": "fixed",
            "status": "ok",
            "n_train": int(is_train.sum()),
            "n_test": int(is_test.sum()),
            "train_pos_rate": float(y_train.mean()),
            "test_pos_rate": float(y_test.mean()),
            **metrics_row(y_test, prob),
        })
    return rows


def primary_metric(row: dict) -> str:
    """Prefer AUPRC for rare positives; AUROC otherwise."""
    return "auprc" if float(row.get("test_pos_rate", 1.0)) < 0.20 else "auroc"


def metric_value(row: dict) -> float:
    metric = primary_metric(row)
    val = float(row[metric])
    return -float("inf") if math.isnan(val) else val


def best_rows(rows: list[dict]) -> list[dict]:
    ok = [r for r in rows if r.get("status") == "ok"]
    by_target: dict[str, list[dict]] = {}
    for r in ok:
        by_target.setdefault(r["target"], []).append(r)
    out = []
    for target, trs in sorted(by_target.items()):
        best = max(trs, key=metric_value)
        chance = next((r for r in trs if r["model"] == "chance"), None)
        metric = primary_metric(best)
        lift = best[metric] - (chance[metric] if chance is not None else float("nan"))
        out.append({**best, "primary_metric": metric, "lift_vs_chance": lift})
    return out


def model_means(rows: list[dict]) -> list[dict]:
    ok = [r for r in rows if r.get("status") == "ok" and r["model"] != "chance"]
    grouped: dict[tuple[str, str], list[dict]] = {}
    for r in ok:
        grouped.setdefault((r["model"], r["rank"]), []).append(r)
    out = []
    for (model, rank), vals in sorted(grouped.items()):
        out.append({
            "model": model,
            "rank": rank,
            "n_targets": len(vals),
            "mean_macro_f1": float(np.mean([r["macro_f1"] for r in vals])),
            "mean_auroc": float(np.nanmean([r["auroc"] for r in vals])),
            "mean_auprc": float(np.mean([r["auprc"] for r in vals])),
        })
    return out


def to_markdown(rows: list[dict], features_path: str, ranks: tuple[int | None, ...]) -> str:
    best = best_rows(rows)
    means = model_means(rows)
    lines = [
        "# Table 32 -- Baseline regressor family on frozen genome embeddings",
        "",
        f"Features: `{features_path}`. Family-held-out split. Rank sweep: "
        + ", ".join(rank_label(r) for r in ranks)
        + ". Polynomial models use PCA first, so degree-2 expansion tests low-rank "
        "interactions without exploding the original embedding dimension.",
        "",
        "Primary metric = AUPRC when the family-test positive rate is <20%, otherwise AUROC. "
        "This makes rare pathogenicity/media targets less misleading than accuracy.",
        "",
        "## Best simple downstream model per target",
        "",
        "| Target | Group | Test n | Pos rate | Best model | Rank | Primary | Score | Lift vs chance | Macro-F1 | AUROC | AUPRC |",
        "|---|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in best:
        metric = r["primary_metric"]
        lines.append(
            f"| `{r['target']}` | {r['group']} | {r['n_test']} | {r['test_pos_rate']:.3f} | "
            f"{r['model']} | {r['rank']} | {metric} | {r[metric]:.3f} | "
            f"{r['lift_vs_chance']:+.3f} | {r['macro_f1']:.3f} | {r['auroc']:.3f} | {r['auprc']:.3f} |"
        )

    lines += [
        "",
        "## Model-family means",
        "",
        "| Model | Rank | Targets | Mean macro-F1 | Mean AUROC | Mean AUPRC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in means:
        lines.append(
            f"| {r['model']} | {r['rank']} | {r['n_targets']} | "
            f"{r['mean_macro_f1']:.3f} | {r['mean_auroc']:.3f} | {r['mean_auprc']:.3f} |"
        )

    failed = [r for r in rows if r.get("status") != "ok"]
    if failed:
        lines += ["", f"Skipped/failed fits: {len(failed)}. See CSV for details."]
    return "\n".join(lines) + "\n"


def run(args) -> list[dict]:
    feats, df = load_aligned_frame(args.features, args.splits, args.traits_path)
    targets = collect_targets(
        df,
        args.targets,
        top_media=args.top_media,
        min_positive=args.min_positive,
    )
    grid = model_grid(parse_ranks(args.ranks), args.max_poly_rank)

    all_rows: list[dict] = []
    for target in targets:
        all_rows.extend(evaluate_target(feats, df, target, grid, random_state=args.seed))
    return all_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="data/esm2_features.npz")
    ap.add_argument("--splits", default="data/splits.parquet")
    ap.add_argument("--traits-path", default="data/traits.parquet")
    ap.add_argument("--targets", nargs="+", default=DEFAULT_BINARY_TARGETS)
    ap.add_argument("--ranks", default="6,10,25,50,100,full")
    ap.add_argument("--top-media", type=int, default=5,
                    help="Also evaluate the top-K cultivation_medium one-vs-rest labels.")
    ap.add_argument("--min-positive", type=int, default=50,
                    help="Minimum positives required for a multilabel target.")
    ap.add_argument("--max-poly-rank", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="paper/tables")
    args = ap.parse_args()

    ranks = parse_ranks(args.ranks)
    rows = run(args)
    if not any(r.get("status") == "ok" for r in rows):
        raise SystemExit("no targets evaluated successfully")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "32_baseline_regressors.csv", index=False)
    md = to_markdown(rows, args.features, ranks)
    (out_dir / "32_baseline_regressors.md").write_text(md)
    (out_dir / "32_baseline_regressors.json").write_text(json.dumps(rows, indent=2, allow_nan=True))
    print(md)
    print(f"wrote {out_dir / '32_baseline_regressors.md'} / .csv / .json")


if __name__ == "__main__":
    main()
