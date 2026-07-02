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
                    "plateau_index": i,
                    "plateau_value": running,
                    "plateaued": True,
                }
        else:
            flat = 0
    return {
        "lower_bound": lower_bound,
        "plateau_index": len(metrics) - 1,
        "plateau_value": running,
        "plateaued": False,
    }


def escalation_verdict(plateau_value: float, ceiling: float, plateaued: bool,
                       margin: float = 0.02) -> str:
    if not plateaued:
        return "keep-scaling-simple"
    if ceiling is None or np.isnan(ceiling):
        return "unknown"
    return "NO — coverage-limited" if (ceiling - plateau_value) <= margin else "YES — capacity-limited"


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
        # The ceiling is always a kNN AUROC (Table 33). If this target's primary
        # metric is AUPRC (rare-positive targets), comparing plateau vs. ceiling
        # would silently mix two different metric scales, so force "unknown"
        # instead of a meaningless AUROC-minus-AUPRC verdict.
        if metric_name != "auroc":
            ceiling = float("nan")
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
              "`family` (no family leakage); meta-learner = balanced logistic regression. "
              "The `best_base` column selects the strongest single base learner on the test set "
              "(an optimistic oracle), shown for reference only.", "",
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
