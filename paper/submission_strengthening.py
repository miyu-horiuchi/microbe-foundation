"""
submission_strengthening.py

Generate submission-readiness analyses requested by reviewer-style feedback:

1. Full per-trait mean-vs-attention absolute results across species/genus/family
   splits and available seeds.
2. A quantitative trait-localization proxy from eggNOG gene-family features.
3. A taxonomy-majority baseline to expose clade-confounding risk.
4. A compact SVG figure linking measured localization to attention gain.

The localization analysis is intentionally an audit subset: the current eggNOG
feature matrix covers fewer genomes than the full ESM-2 benchmark. The output
tables label that scope explicitly.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"
RUNS = ROOT / "runs"
TRAITS_PATH = ROOT / "data" / "traits.parquet"
SPLITS_PATH = ROOT / "data" / "splits.parquet"
EGGNOG_FEATURES = ROOT / "data" / "eggnog_features_6738.npz"
BENCHMARK_FEATURES = ROOT / "data" / "esm2_features.npz"


COMPOSITIONAL = {
    "gram_stain",
    "cell_shape",
    "motility",
    "sporulation",
    "oxygen_tolerance",
    "catalase",
    "cytochrome_oxidase",
    "temperature_class",
    "ph_class",
    "halophily",
    "pigmentation",
}

MACHINERY = {
    "pathogenicity_human",
    "pathogenicity_animal",
    "cultivation_medium",
    "carbon_utilization",
    "metabolite_production",
    "amr_phenotype",
    "biosafety_level",
    "fatty_acid_profile",
}

METADATA = {"isolation_source", "country"}


def trait_class(name: str) -> str:
    if name in COMPOSITIONAL:
        return "compositional"
    if name in MACHINERY:
        return "machinery"
    if name in METADATA:
        return "metadata"
    return "other"


def fmt(x: float | None, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{digits}f}"


def metric_for_head(head: dict) -> str:
    metrics = head.get("metrics") or {}
    for key in ("f1", "f1_macro", "acc", "rmse"):
        if key in metrics:
            return key
    return head.get("metric_kind", "score")


def metric_value(head: dict, metric: str) -> float | None:
    metrics = head.get("metrics") or {head.get("metric_kind"): head.get("score")}
    val = metrics.get(metric)
    return None if val is None else float(val)


def load_pooling_runs() -> dict[tuple[str, str], list[dict]]:
    out: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for path in sorted(RUNS.glob("*.json")):
        name = path.stem
        if not (name.startswith("meanpool-") or name.startswith("attnpool-")):
            continue
        data = json.loads(path.read_text())
        model = "attention" if name.startswith("attnpool-") else "mean"
        split = data.get("split_level")
        if split in {"species", "genus", "family"}:
            out[(model, split)].append(data)
    return out


def aggregate(values: list[float]) -> tuple[float | None, float | None, int]:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return None, None, 0
    return float(np.mean(vals)), float(np.std(vals, ddof=0)), len(vals)


def pooling_results_table() -> tuple[pd.DataFrame, str]:
    runs = load_pooling_runs()
    traits = sorted(
        {
            trait
            for run_list in runs.values()
            for run in run_list
            for trait in run.get("per_head", {})
        }
    )
    rows = []
    for split in ("species", "genus", "family"):
        for trait in traits:
            mean_heads = [r["per_head"].get(trait) for r in runs.get(("mean", split), [])]
            attn_heads = [r["per_head"].get(trait) for r in runs.get(("attention", split), [])]
            heads = [h for h in mean_heads + attn_heads if h]
            if not heads:
                continue
            metric = metric_for_head(heads[0])
            mean_vals = [metric_value(h, metric) for h in mean_heads if h]
            attn_vals = [metric_value(h, metric) for h in attn_heads if h]
            mean_mu, mean_sd, mean_n = aggregate(mean_vals)
            attn_mu, attn_sd, attn_n = aggregate(attn_vals)
            delta = None if mean_mu is None or attn_mu is None else attn_mu - mean_mu
            head_type = heads[0].get("head_type", "")
            rows.append(
                {
                    "split": split,
                    "trait": trait,
                    "trait_class": trait_class(trait),
                    "head_type": head_type,
                    "metric": metric,
                    "mean_pool": mean_mu,
                    "mean_pool_sd": mean_sd,
                    "mean_n": mean_n,
                    "attention_pool": attn_mu,
                    "attention_pool_sd": attn_sd,
                    "attn_n": attn_n,
                    "attention_gain": delta,
                }
            )
    df = pd.DataFrame(rows)
    out = [
        "# Table 10 — Absolute mean-pool vs attention-pool results",
        "",
        "Mean and standard deviation across available seeds in `runs/`. "
        "The reported metric is F1 when available, then macro-F1, then accuracy; "
        "RMSE heads are listed but should not be mixed with classification gains.",
        "",
        "| Split | Trait | Class | Metric | Mean-pool | Attention-pool | Δ | Seeds |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for _, r in df.sort_values(["split", "trait"]).iterrows():
        seeds = min(int(r["mean_n"]), int(r["attn_n"]))
        out.append(
            f"| {r['split']} | `{r['trait']}` | {r['trait_class']} | {r['metric']} | "
            f"{fmt(r['mean_pool'])} ± {fmt(r['mean_pool_sd'])} | "
            f"{fmt(r['attention_pool'])} ± {fmt(r['attention_pool_sd'])} | "
            f"{fmt(r['attention_gain'], 3)} | {seeds} |"
        )
    return df, "\n".join(out) + "\n"


def concentration_metrics(scores: np.ndarray) -> dict[str, float | int]:
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    scores = scores[scores > 0]
    if len(scores) == 0:
        return {
            "top10_share": math.nan,
            "gini": math.nan,
            "n80": 0,
            "n_nonzero_features": 0,
            "normalized_entropy": math.nan,
        }
    scores.sort()
    total = float(scores.sum())
    desc = scores[::-1]
    top10 = float(desc[:10].sum() / total)
    cum = np.cumsum(desc)
    n80 = int(np.searchsorted(cum, 0.80 * total) + 1)
    n = len(scores)
    gini = float((2 * np.arange(1, n + 1) - n - 1).dot(scores) / (n * total))
    p = desc / total
    entropy = float(-(p * np.log(p + 1e-12)).sum() / max(math.log(n), 1e-12))
    return {
        "top10_share": top10,
        "gini": gini,
        "n80": n80,
        "n_nonzero_features": n,
        "normalized_entropy": entropy,
    }


def diff_scores_binary(X: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    pos = y == 1
    neg = y == 0
    if pos.sum() < 20 or neg.sum() < 20:
        return None
    pos_mean = X[pos].mean(axis=0)
    neg_mean = X[neg].mean(axis=0)
    return np.abs(pos_mean - neg_mean)


def diff_scores_multiclass(X: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    classes, counts = np.unique(y, return_counts=True)
    keep = counts >= 20
    classes = classes[keep]
    counts = counts[keep]
    if len(classes) < 2:
        return None
    global_mean = X[np.isin(y, classes)].mean(axis=0)
    score = np.zeros(X.shape[1], dtype=np.float64)
    total = counts.sum()
    for cls, count in zip(classes, counts):
        cls_mean = X[y == cls].mean(axis=0)
        score += (count / total) * np.abs(cls_mean - global_mean)
    return score


def sparse_linear_localization(
    X: np.ndarray,
    y: np.ndarray,
    split_values: np.ndarray,
    *,
    random_state: int = 0,
) -> dict[str, float | int] | None:
    """Fit an L1 sparse gene-family classifier on species-train genomes.

    This is deliberately a diagnostic, not a production predictor: it asks
    whether a trait can be explained by a small set of orthologous groups. The
    coefficient concentration is the measured localization score.
    """
    train = split_values == "train"
    test = split_values == "test"
    if train.sum() < 50 or test.sum() < 20:
        return None
    train_classes, train_counts = np.unique(y[train], return_counts=True)
    test_classes = np.unique(y[test])
    if len(train_classes) < 2 or len(test_classes) < 2:
        return None
    # Very tiny classes create unstable sparse coefficients and noisy macro-F1.
    if train_counts.min() < 10:
        return None

    Xs = sparse.csr_matrix(X)
    clf = SGDClassifier(
        loss="log",
        penalty="l1",
        alpha=1e-3,
        class_weight="balanced",
        max_iter=1000,
        tol=1e-3,
        random_state=random_state,
        n_jobs=1,
    )
    clf.fit(Xs[train], y[train])
    pred = clf.predict(Xs[test])
    macro_f1 = f1_score(y[test], pred, average="macro", zero_division=0)
    coefs = np.asarray(clf.coef_, dtype=np.float64)
    scores = np.abs(coefs).sum(axis=0)
    metrics = concentration_metrics(scores)
    metrics.update(
        {
            "sparse_macro_f1": float(macro_f1),
            "sparse_n_train": int(train.sum()),
            "sparse_n_test": int(test.sum()),
            "sparse_n_nonzero_coefficients": int((scores > 0).sum()),
        }
    )
    return metrics


def localization_table(pooling_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if not EGGNOG_FEATURES.exists():
        return pd.DataFrame(), "# Table 11 — Gene-family localization audit\n\n_(missing eggNOG feature matrix)_\n"

    traits_df = pd.read_parquet(TRAITS_PATH)
    splits_df = pd.read_parquet(SPLITS_PATH)
    traits_df = traits_df.merge(
        splits_df[["bacdive_id", "species_split"]], on="bacdive_id", how="left"
    )
    schema = json.loads((ROOT / "trait_schema.json").read_text())

    feature_npz = np.load(EGGNOG_FEATURES, allow_pickle=True)
    feature_ids = feature_npz["bacdive_ids"].astype(int)
    X = feature_npz["features"]
    row_by_id = {int(bid): i for i, bid in enumerate(traits_df["bacdive_id"].to_numpy())}
    trait_rows = np.array([row_by_id[int(bid)] for bid in feature_ids if int(bid) in row_by_id], dtype=int)
    keep_features = np.array([int(bid) in row_by_id for bid in feature_ids], dtype=bool)
    if not keep_features.all():
        X = X[keep_features]
        feature_ids = feature_ids[keep_features]

    rows = []
    scalar_specs = {
        t["name"]: t
        for t in schema["traits"]
        if t["head"] in {"binary", "multiclass"}
    }

    for trait, spec in scalar_specs.items():
        h = spec["head"]
        col = traits_df.iloc[trait_rows][trait]
        scores = None
        sparse_metrics = None

        if h == "binary":
            valid = col.notna().to_numpy()
            y = col[valid].astype(bool).astype(int).to_numpy()
            scores = diff_scores_binary(X[valid], y)
            n_labeled = int(valid.sum())
            sparse_metrics = sparse_linear_localization(
                X[valid],
                y,
                traits_df.iloc[trait_rows]["species_split"].to_numpy()[valid],
            )
        elif h == "multiclass":
            valid = col.notna().to_numpy()
            y = pd.Categorical(col[valid].astype(str)).codes
            scores = diff_scores_multiclass(X[valid], y)
            n_labeled = int(valid.sum())
            sparse_metrics = sparse_linear_localization(
                X[valid],
                y,
                traits_df.iloc[trait_rows]["species_split"].to_numpy()[valid],
            )

        if scores is None:
            continue
        metrics = concentration_metrics(scores)
        if sparse_metrics:
            for key in ("top10_share", "gini", "n80", "n_nonzero_features", "normalized_entropy"):
                metrics[f"sparse_{key}"] = sparse_metrics[key]
            for key in ("sparse_macro_f1", "sparse_n_train", "sparse_n_test", "sparse_n_nonzero_coefficients"):
                metrics[key] = sparse_metrics[key]
            metrics["localization_score"] = sparse_metrics["top10_share"]
            metrics["localization_n80"] = sparse_metrics["n80"]
            metrics["localization_source"] = "sparse_linear"
        else:
            metrics["localization_score"] = metrics["top10_share"]
            metrics["localization_n80"] = metrics["n80"]
            metrics["localization_source"] = "univariate"
        species_gain = pooling_df[
            (pooling_df["split"] == "species") & (pooling_df["trait"] == trait)
        ]["attention_gain"]
        genus_gain = pooling_df[
            (pooling_df["split"] == "genus") & (pooling_df["trait"] == trait)
        ]["attention_gain"]
        family_gain = pooling_df[
            (pooling_df["split"] == "family") & (pooling_df["trait"] == trait)
        ]["attention_gain"]
        rows.append(
            {
                "trait": trait,
                "trait_class": trait_class(trait),
                "head_type": h,
                "audit_labeled_genomes": n_labeled,
                **metrics,
                "species_attention_gain": float(species_gain.iloc[0]) if len(species_gain) else math.nan,
                "genus_attention_gain": float(genus_gain.iloc[0]) if len(genus_gain) else math.nan,
                "family_attention_gain": float(family_gain.iloc[0]) if len(family_gain) else math.nan,
            }
        )

    df = pd.DataFrame(rows).sort_values("top10_share", ascending=False)
    corr_df = df.dropna(subset=["localization_score", "species_attention_gain"])
    rho, p = (math.nan, math.nan)
    if len(corr_df) >= 3:
        rho, p = spearmanr(corr_df["localization_score"], corr_df["species_attention_gain"])

    out = [
        "# Table 11 — Sparse gene-family localization audit",
        "",
        f"Audit feature matrix: `{EGGNOG_FEATURES.relative_to(ROOT)}` "
        f"({X.shape[0]:,} genomes × {X.shape[1]:,} eggNOG orthologous groups).",
        "",
        "Localization proxy: for each scalar trait, fit an L1-regularized "
        "gene-family classifier on the species-train split, evaluate it on the "
        "species-test split, then measure how concentrated the absolute "
        "coefficient mass is. `localization_score` is the top-10 coefficient-mass "
        "share when the sparse fit is stable; otherwise it falls back to the "
        "univariate class-conditional association share. Multilabel and "
        "regression-vector heads are excluded from this scalar audit.",
        "",
        f"Spearman correlation between `localization_score` and species-level attention gain: "
        f"rho = **{fmt(rho)}**, p = **{fmt(p)}** (n={len(corr_df)} traits).",
        "",
        "| Trait | Class | Type | Source | Audit labels | Sparse train/test | Sparse macro-F1 | Localization | Sparse nonzero | n80 | Species Δ | Genus Δ | Family Δ |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.sort_values("localization_score", ascending=False).iterrows():
        sparse_train = r.get("sparse_n_train")
        sparse_test = r.get("sparse_n_test")
        train_test = "—" if pd.isna(sparse_train) or pd.isna(sparse_test) else f"{int(sparse_train):,}/{int(sparse_test):,}"
        out.append(
            f"| `{r['trait']}` | {r['trait_class']} | {r['head_type']} | "
            f"{r['localization_source']} | {int(r['audit_labeled_genomes']):,} | "
            f"{train_test} | {fmt(r.get('sparse_macro_f1'))} | "
            f"{fmt(r['localization_score'])} | "
            f"{fmt(r.get('sparse_n_nonzero_coefficients'), 0)} | "
            f"{int(r['localization_n80'])} | "
            f"{fmt(r['species_attention_gain'])} | {fmt(r['genus_attention_gain'])} | "
            f"{fmt(r['family_attention_gain'])} |"
        )
    return df, "\n".join(out) + "\n"


def majority(values: list) -> object:
    return Counter(values).most_common(1)[0][0]


def taxonomy_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    trait: str,
    levels: list[str],
) -> tuple[list, list]:
    train_valid = train[train[trait].notna()].copy()
    test_valid = test[test[trait].notna()].copy()
    if len(train_valid) == 0 or len(test_valid) == 0:
        return [], []
    global_majority = majority(train_valid[trait].tolist())
    lookup: dict[tuple[str, object], object] = {}
    for level in levels:
        if level not in train_valid.columns:
            continue
        for val, group in train_valid.dropna(subset=[level]).groupby(level):
            lookup[(level, val)] = majority(group[trait].tolist())
    y_true, y_pred = [], []
    for _, row in test_valid.iterrows():
        pred = global_majority
        for level in levels:
            val = row.get(level)
            if pd.notna(val) and (level, val) in lookup:
                pred = lookup[(level, val)]
                break
        y_true.append(row[trait])
        y_pred.append(pred)
    return y_true, y_pred


def taxonomy_baseline_table(pooling_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    traits_df = pd.read_parquet(TRAITS_PATH)
    splits_df = pd.read_parquet(SPLITS_PATH)
    df = traits_df.merge(splits_df[["bacdive_id", "family_split", "genus_split", "species_split"]], on="bacdive_id")
    schema = json.loads((ROOT / "trait_schema.json").read_text())
    scalar_traits = [t for t in schema["traits"] if t["head"] in {"binary", "multiclass"}]
    levels = ["species", "genus", "family", "order", "class", "phylum", "domain"]
    rows = []
    for split in ("species", "genus", "family"):
        split_col = f"{split}_split"
        train = df[df[split_col] == "train"]
        test = df[df[split_col] == "test"]
        for t in scalar_traits:
            trait = t["name"]
            y_true, y_pred = taxonomy_predictions(train, test, trait, levels)
            if len(y_true) < 20 or len(set(y_true)) < 2:
                continue
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            attn_row = pooling_df[(pooling_df["split"] == split) & (pooling_df["trait"] == trait)]
            attn_score = float(attn_row["attention_pool"].iloc[0]) if len(attn_row) else math.nan
            metric = str(attn_row["metric"].iloc[0]) if len(attn_row) else "f1"
            rows.append(
                {
                    "split": split,
                    "trait": trait,
                    "trait_class": trait_class(trait),
                    "n_test_labeled": len(y_true),
                    "taxonomy_acc": acc,
                    "taxonomy_f1": f1,
                    "attention_metric": metric,
                    "attention_score": attn_score,
                }
            )
    out_df = pd.DataFrame(rows)
    out = [
        "# Table 12 — Taxonomy-majority baseline",
        "",
        "Baseline: for each test genome, predict using the most specific taxonomy "
        "level observed in training (`species`, then `genus`, `family`, `order`, "
        "`class`, `phylum`, `domain`), falling back to the train-set majority. "
        "This is a confounding diagnostic, not a proposed model.",
        "",
        "| Split | Trait | Class | Test labels | Taxonomy acc | Taxonomy macro-F1 | Attention metric | Attention score |",
        "|---|---|---|---:|---:|---:|---|---:|",
    ]
    for _, r in out_df.sort_values(["split", "trait"]).iterrows():
        out.append(
            f"| {r['split']} | `{r['trait']}` | {r['trait_class']} | "
            f"{int(r['n_test_labeled']):,} | {fmt(r['taxonomy_acc'])} | "
            f"{fmt(r['taxonomy_f1'])} | {r['attention_metric']} | {fmt(r['attention_score'])} |"
        )
    return out_df, "\n".join(out) + "\n"


def matched_clade_control_table() -> tuple[pd.DataFrame, str]:
    traits_df = pd.read_parquet(TRAITS_PATH)
    splits_df = pd.read_parquet(SPLITS_PATH)
    df = traits_df.merge(
        splits_df[["bacdive_id", "family_split", "genus_split", "species_split"]],
        on="bacdive_id",
        how="inner",
    )
    if BENCHMARK_FEATURES.exists():
        feature_ids = set(np.load(BENCHMARK_FEATURES, allow_pickle=True)["bacdive_ids"].astype(int).tolist())
        df = df[df["bacdive_id"].astype(int).isin(feature_ids)].copy()

    rows = []
    target_traits = ["pathogenicity_animal", "pathogenicity_human"]
    for trait in target_traits:
        for split in ("species", "genus", "family"):
            split_col = f"{split}_split"
            train = df[df[split_col] == "train"]
            test = df[df[split_col] == "test"]
            test_valid = test[test[trait].notna()].copy()
            n_test_total = len(test_valid)
            for level in ("genus", "family"):
                train_valid = train[train[trait].notna()].dropna(subset=[level]).copy()
                if len(train_valid) == 0 or n_test_total == 0:
                    continue
                mixed = []
                for val, group in train_valid.groupby(level):
                    if group[trait].astype(bool).nunique() >= 2:
                        mixed.append(val)
                matched_test = test_valid[test_valid[level].isin(mixed)].copy()
                y_true, y_pred = taxonomy_predictions(train, matched_test, trait, [level])
                macro_f1 = (
                    f1_score(y_true, y_pred, average="macro", zero_division=0)
                    if len(y_true) >= 2 and len(set(y_true)) >= 2
                    else math.nan
                )
                acc = accuracy_score(y_true, y_pred) if len(y_true) else math.nan
                pos_rate = (
                    float(pd.Series(y_true).astype(bool).mean()) if len(y_true) else math.nan
                )
                rows.append(
                    {
                        "trait": trait,
                        "split": split,
                        "matched_level": level,
                        "n_test_labeled": n_test_total,
                        "n_matched_test": len(matched_test),
                        "matched_coverage": len(matched_test) / n_test_total if n_test_total else math.nan,
                        "n_mixed_train_clades": len(mixed),
                        "matched_positive_rate": pos_rate,
                        "matched_taxonomy_acc": acc,
                        "matched_taxonomy_macro_f1": macro_f1,
                    }
                )
    out_df = pd.DataFrame(rows)
    out = [
        "# Table 13 — Within-clade matched pathogenicity control",
        "",
        "This diagnostic restricts pathogenicity evaluation to test genomes whose "
        "genus or family is represented in training with both pathogenic and "
        "non-pathogenic labeled examples. It asks how much matched evaluation "
        "coverage exists after removing pure-clade shortcuts, and how strong a "
        "same-clade majority baseline remains on that matched subset.",
        "",
        f"Benchmark alignment: `{BENCHMARK_FEATURES.relative_to(ROOT)}` feature IDs when available.",
        "",
        "| Trait | Split | Matched level | Test labels | Matched test | Coverage | Mixed train clades | Matched positive rate | Same-clade acc | Same-clade macro-F1 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in out_df.sort_values(["trait", "split", "matched_level"]).iterrows():
        out.append(
            f"| `{r['trait']}` | {r['split']} | {r['matched_level']} | "
            f"{int(r['n_test_labeled']):,} | {int(r['n_matched_test']):,} | "
            f"{fmt(r['matched_coverage'])} | {int(r['n_mixed_train_clades']):,} | "
            f"{fmt(r['matched_positive_rate'])} | {fmt(r['matched_taxonomy_acc'])} | "
            f"{fmt(r['matched_taxonomy_macro_f1'])} |"
        )
    return out_df, "\n".join(out) + "\n"


def write_localization_svg(df: pd.DataFrame) -> None:
    plot = df.dropna(subset=["localization_score", "species_attention_gain"]).copy()
    plot = plot[plot["head_type"] != "regression_vector"]
    width, height = 880, 430
    x0, y0, x1, y1 = 95, 345, 790, 65
    if len(plot) == 0:
        return
    xmin, xmax = 0, float(plot["localization_score"].max()) * 1.18
    ymin = min(-0.03, float(plot["species_attention_gain"].min()) * 1.1)
    ymax = max(0.10, float(plot["species_attention_gain"].max()) * 1.1)

    def sx(x):
        return x0 + (float(x) - xmin) / (xmax - xmin) * (x1 - x0)

    def sy(y):
        return y0 - (float(y) - ymin) / (ymax - ymin) * (y0 - y1)

    colors = {
        "machinery": "#e8763a",
        "compositional": "#2ca6a4",
        "metadata": "#999999",
        "other": "#7c6dc7",
    }
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="Localization score predicts attention gain">',
        '<rect width="880" height="430" fill="#fff"/>',
        '<g stroke="#eeeeee" stroke-width="0.7">',
    ]
    for i in range(6):
        gy = y0 - i * (y0 - y1) / 5
        gx = x0 + i * (x1 - x0) / 5
        lines.append(f'<line x1="{x0}" y1="{gy:.1f}" x2="{x1}" y2="{gy:.1f}"/>')
        lines.append(f'<line x1="{gx:.1f}" y1="{y1}" x2="{gx:.1f}" y2="{y0}"/>')
    lines.extend(
        [
            "</g>",
            f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#c4c4c4"/>',
            f'<line x1="{x0}" y1="{y1}" x2="{x0}" y2="{y0}" stroke="#c4c4c4"/>',
            '<text x="440" y="32" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="15" fill="#555" font-style="italic">sparse gene-family localization audit tracks adaptive pooling gain</text>',
            '<text x="440" y="397" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="11.5" fill="#888" letter-spacing=".04em">gene-family localization: top-10 coefficient share</text>',
            '<text x="30" y="205" transform="rotate(-90,30,205)" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="11.5" fill="#888" letter-spacing=".04em">species-level attention gain</text>',
        ]
    )
    for _, r in plot.iterrows():
        cls = r["trait_class"]
        color = colors.get(cls, "#777")
        radius = 5.5 if cls == "machinery" else 4.5
        lines.append(
            f'<circle cx="{sx(r["localization_score"]):.1f}" cy="{sy(r["species_attention_gain"]):.1f}" '
            f'r="{radius}" fill="{color}" stroke="#fff" stroke-width="1.2"/>'
        )
        if r["trait"] in {"pathogenicity_animal", "pathogenicity_human"}:
            label = str(r["trait"]).replace("_", " ")
            dx = -128 if r["trait"] == "pathogenicity_animal" else 8
            dy = 22 if r["trait"] == "pathogenicity_animal" else -6
            lines.append(
                f'<text x="{sx(r["localization_score"]) + dx:.1f}" y="{sy(r["species_attention_gain"]) + dy:.1f}" '
                f'font-family="Helvetica,Arial,sans-serif" font-size="10.5" fill="{color}">{label}</text>'
            )
    rho, p = spearmanr(plot["localization_score"], plot["species_attention_gain"]) if len(plot) >= 3 else (math.nan, math.nan)
    lines.extend(
        [
            '<g font-family="Helvetica,Arial,sans-serif" font-size="11" fill="#555">',
            '<circle cx="145" cy="82" r="4.5" fill="#e8763a" stroke="#fff"/>',
            '<text x="165" y="86">machinery</text>',
            '<circle cx="145" cy="104" r="4.5" fill="#2ca6a4" stroke="#fff"/>',
            '<text x="165" y="108">compositional</text>',
            f'<text x="145" y="134" font-style="italic">Spearman rho = {fmt(float(rho))}, p = {fmt(float(p))}</text>',
            "</g>",
        ]
    )
    for tick in np.linspace(xmin, xmax, 5):
        lines.append(f'<text x="{sx(tick):.1f}" y="364" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="10" fill="#aaa">{tick:.4f}</text>')
    for tick in np.linspace(ymin, ymax, 5):
        lines.append(f'<text x="84" y="{sy(tick)+3:.1f}" text-anchor="end" font-family="Helvetica,Arial,sans-serif" font-size="10" fill="#aaa">{tick:+.2f}</text>')
    lines.append("</svg>")
    (FIGURES / "figure5_localization_gain.svg").write_text("\n".join(lines))


def main() -> None:
    TABLES.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    pooling_df, pooling_md = pooling_results_table()
    (TABLES / "10_pooling_absolute_results.md").write_text(pooling_md)
    pooling_df.to_csv(TABLES / "10_pooling_absolute_results.csv", index=False)

    localization_df, localization_md = localization_table(pooling_df)
    (TABLES / "11_localization_gain.md").write_text(localization_md)
    localization_df.to_csv(TABLES / "11_localization_gain.csv", index=False)
    write_localization_svg(localization_df)

    taxonomy_df, taxonomy_md = taxonomy_baseline_table(pooling_df)
    (TABLES / "12_taxonomy_baseline.md").write_text(taxonomy_md)
    taxonomy_df.to_csv(TABLES / "12_taxonomy_baseline.csv", index=False)

    matched_df, matched_md = matched_clade_control_table()
    (TABLES / "13_matched_clade_controls.md").write_text(matched_md)
    matched_df.to_csv(TABLES / "13_matched_clade_controls.csv", index=False)

    print("wrote:")
    for path in [
        TABLES / "10_pooling_absolute_results.md",
        TABLES / "10_pooling_absolute_results.csv",
        TABLES / "11_localization_gain.md",
        TABLES / "11_localization_gain.csv",
        TABLES / "12_taxonomy_baseline.md",
        TABLES / "12_taxonomy_baseline.csv",
        TABLES / "13_matched_clade_controls.md",
        TABLES / "13_matched_clade_controls.csv",
        FIGURES / "figure5_localization_gain.svg",
    ]:
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
