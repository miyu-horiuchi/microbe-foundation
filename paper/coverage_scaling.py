"""
coverage_scaling.py — Lever 1: how much does broader training-clade coverage buy?

The §4.6--4.7 diagnosis says the coverage-limited traits collapse off-clade because
their positives are confined to a few training families. This script puts a *number*
on the fix: a size-controlled scaling curve of family-held-out performance vs the
number of distinct training families, holding the training-genome budget fixed so the
curve isolates clade diversity from raw data volume.

For each binary coverage-limited trait we sweep k = #training families, subsample the
training genomes to a constant budget, fit a balanced linear probe on frozen ESM-2
features, and evaluate macro-F1 and AUROC on the fixed novel-family test set, over
several seeds. We then fit a log2 slope: expected metric gain per doubling of
training-clade coverage. That slope is the concrete data-scaling recommendation.

CPU-only; reuses the probe and family-subsampling logic from cross_clade_diagnostic.

Usage:
    python paper/coverage_scaling.py
    python paper/coverage_scaling.py --traits motility pathogenicity_human
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cross_clade_diagnostic import (  # noqa: E402
    _macro_f1, _probe, prepare_trait, sample_families,
)

TRAITS = ["pathogenicity_human", "pathogenicity_animal", "motility"]
KS = [5, 10, 20, 40, 80]
SEEDS = [0, 1, 2, 3, 4]


def log2_slope(ks: list[float], means: list[float]) -> float:
    """Least-squares slope of metric vs log2(#families): gain per doubling of coverage."""
    ks = np.asarray(ks, dtype=float)
    means = np.asarray(means, dtype=float)
    ok = ks > 0
    x = np.log2(ks[ok])
    y = means[ok]
    if len(x) < 2 or np.allclose(x, x[0]):
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def scaling_curve(X_train, y_train, fam_train, X_test, y_test, ks, seeds, fixed_n=None):
    """Per-(k, seed) family-test macro-F1 and AUROC.

    If ``fixed_n`` is given, each subset is downsampled to that many genomes so only
    clade diversity varies (size-controlled). If ``None``, all genomes in the sampled
    families are used (the *natural* curve: broader collection adds families and data
    together, which is the real-world coverage lever).
    """
    fam_train = np.asarray(fam_train)
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    rows = []
    for k in ks:
        for seed in seeds:
            fams = sample_families(list(fam_train), k, seed)
            sel = np.where(np.isin(fam_train, list(fams)))[0]
            if fixed_n is not None and len(sel) > fixed_n:
                rng = np.random.default_rng(1000 + seed)
                sel = rng.choice(sel, fixed_n, replace=False)
            ytr = y_train[sel]
            if len(np.unique(ytr)) < 2:
                continue
            prob = _probe(X_train[sel], ytr, X_test)
            rows.append({
                "k_families": int(k), "seed": int(seed), "n_train": int(len(sel)),
                "f1": _macro_f1(y_test, prob),
                "auroc": float(roc_auc_score(y_test, prob)),
            })
    return rows


def summarize_curve(rows: list[dict]) -> dict:
    """Endpoint metrics and log2 slopes from a scaling curve."""
    by_k: dict = {}
    for r in rows:
        by_k.setdefault(r["k_families"], []).append(r)
    ks = sorted(by_k)
    f1 = [float(np.mean([r["f1"] for r in by_k[k]])) for k in ks]
    au = [float(np.mean([r["auroc"] for r in by_k[k]])) for k in ks]
    ntr = [int(np.mean([r["n_train"] for r in by_k[k]])) for k in ks]
    return {
        "ks": ks,
        "f1_by_k": f1,
        "auroc_by_k": au,
        "n_train_by_k": ntr,
        "f1_lo": f1[0], "f1_hi": f1[-1], "f1_gain": f1[-1] - f1[0],
        "auroc_lo": au[0], "auroc_hi": au[-1], "auroc_gain": au[-1] - au[0],
        "f1_slope_per_2x": log2_slope(ks, f1),
        "auroc_slope_per_2x": log2_slope(ks, au),
        "k_min": ks[0], "k_max": ks[-1],
        "n_train_lo": ntr[0], "n_train_hi": ntr[-1],
    }


def run(features_path, splits_path, traits_path, traits=TRAITS):
    data = np.load(features_path)
    feats = data["features"]
    ids = np.array([str(i) for i in data["bacdive_ids"]])
    id_to_row = {b: i for i, b in enumerate(ids)}
    tr = pd.read_parquet(traits_path)
    tr["bid"] = tr["bacdive_id"].astype(str)
    tr = tr[tr["bid"].isin(id_to_row)].copy()
    sp = pd.read_parquet(splits_path)[["bacdive_id", "family", "family_split"]]
    fam = dict(zip(sp["bacdive_id"].astype(str), sp["family_split"]))
    famname = dict(zip(sp["bacdive_id"].astype(str), sp["family"]))
    tr["fsplit"] = tr["bid"].map(lambda b: fam.get(b, "unknown"))
    tr["family"] = tr["bid"].map(lambda b: famname.get(b))
    tr["row"] = tr["bid"].map(id_to_row)

    out = {}
    for trait in traits:
        if trait not in tr.columns:
            continue
        Xtr, ytr, fam_tr, Xte, yte = prepare_trait(feats, id_to_row, tr, trait)
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue
        n_fam = len(set(fam_tr))
        ks = [k for k in KS if k < n_fam] + [n_fam]
        nat_rows = scaling_curve(Xtr, ytr, fam_tr, Xte, yte, ks, SEEDS, fixed_n=None)
        fixed_n = min(int(np.isin(fam_tr, sample_families(list(fam_tr), ks[0], s)).sum())
                      for s in SEEDS)
        fix_rows = scaling_curve(Xtr, ytr, fam_tr, Xte, yte, ks, SEEDS, fixed_n=fixed_n)
        s = summarize_curve(nat_rows)
        s["fixed"] = summarize_curve(fix_rows) if fix_rows else None
        s["fixed_budget"] = int(fixed_n)
        s["n_families_available"] = n_fam
        s["rows"] = nat_rows
        s["fixed_rows"] = fix_rows
        out[trait] = s
    return out


def to_markdown(out: dict) -> str:
    lines = [
        "# Table 22 — Training-clade coverage scaling (family-held-out)",
        "",
        "Family-test performance vs the number of distinct training families. The "
        "*natural* curve grows families and genomes together (the real-world lever: "
        "collect more diverse data); the slope is the expected metric gain per doubling "
        "of training-clade coverage. The size-controlled column holds the genome budget "
        "fixed at the smallest-k pool to ask whether diversity helps *beyond* raw volume; "
        "for these rare-positive traits that budget is tiny (n shown), so it is "
        "underpowered and reported only as a directional check.",
        "",
        "| Trait | #fam avail | n_train (k=min→max) | F1 (k=min→max) | F1 / 2x | "
        "AUROC (k=min→max) | AUROC / 2x | size-ctrl F1/2x (budget n) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for trait, s in out.items():
        fx = s.get("fixed")
        fx_str = (f"{fx['f1_slope_per_2x']:+.3f} (n={s['fixed_budget']})"
                  if fx else "n/a")
        lines.append(
            f"| `{trait}` | {s['n_families_available']} | "
            f"{s['n_train_lo']}→{s['n_train_hi']} | "
            f"{s['f1_lo']:.3f}→{s['f1_hi']:.3f} ({s['f1_gain']:+.3f}) | {s['f1_slope_per_2x']:+.3f} | "
            f"{s['auroc_lo']:.3f}→{s['auroc_hi']:.3f} ({s['auroc_gain']:+.3f}) | "
            f"{s['auroc_slope_per_2x']:+.3f} | {fx_str} |"
        )
    return "\n".join(lines) + "\n"


def save_figure(out: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axf, axa) = plt.subplots(1, 2, figsize=(11, 4.4))
    for trait, s in out.items():
        axf.plot(s["ks"], s["f1_by_k"], marker="o", label=trait)
        axa.plot(s["ks"], s["auroc_by_k"], marker="o", label=trait)
    for ax, name in ((axf, "macro-F1"), (axa, "AUROC")):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("number of training families (natural: genomes grow with families)")
        ax.set_ylabel(f"family-test {name}")
        ax.grid(True, alpha=0.2)
    axf.set_title("Cross-clade F1 vs training-clade coverage")
    axa.set_title("Cross-clade AUROC vs training-clade coverage")
    axa.legend(fontsize=8)
    fig.suptitle("Lever 1: broader training-clade coverage lifts cross-clade transfer", fontsize=12)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traits", nargs="+", default=TRAITS)
    ap.add_argument("--features", default="data/esm2_features.npz")
    ap.add_argument("--splits", default="data/splits.parquet")
    ap.add_argument("--traits-path", default="data/traits.parquet")
    ap.add_argument("--out-dir", default="paper/tables")
    ap.add_argument("--fig", default="paper/figures/coverage_scaling.png")
    args = ap.parse_args()

    out = run(args.features, args.splits, args.traits_path, traits=args.traits)
    if not out:
        raise SystemExit("no traits produced a curve")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    md = to_markdown(out)
    (Path(args.out_dir) / "22_coverage_scaling.md").write_text(md)
    flat = ([{"trait": t, "curve": "natural", **r} for t, s in out.items() for r in s["rows"]]
            + [{"trait": t, "curve": "size_controlled", **r}
               for t, s in out.items() for r in s["fixed_rows"]])
    pd.DataFrame(flat).to_csv(Path(args.out_dir) / "22_coverage_scaling.csv", index=False)
    save_figure(out, args.fig)
    print(md)
    print(f"wrote {args.out_dir}/22_coverage_scaling.md/.csv and {args.fig}")


if __name__ == "__main__":
    main()
