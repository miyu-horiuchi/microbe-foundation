"""
vfdb_enrichment.py — Phase 3 of pathogenicity interpretability.

Tests whether the proteins the attention-pool up-weights in pathogenic genomes
are enriched for known virulence factors (VFDB), under two controls:

  Control 1 (within-genome): per pathogenic genome, VFDB-hit rate of its top-k
      attended proteins vs k randomly drawn proteins from the SAME genome.
      Paired across genomes (Wilcoxon) — controls for genome composition.
  Control 2 (pathogenic vs not): VFDB-hit rate of top-k attended proteins in
      pathogenic vs non-pathogenic genomes (Fisher exact) — does attention shift
      to virulence genes specifically when the trait is present?

Inputs: the attention parquet (extract_attention.py), a directory of pulled
proteins/<bid>.txt.gz (Modal volume, line i == .npy/attention index i), and a
diamond DB built from VFDB. Run:

    python vfdb_enrichment.py --attn runs/attn-pathogenicity_animal-species.parquet \
        --prot-dir /tmp/prot --vfdb-dmnd /tmp/vfdb/vfdb.dmnd --top-k 5 --out runs/vfdb_animal
"""
from __future__ import annotations

import argparse
import gzip
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

EVALUE = 1e-5  # diamond hit threshold for calling a protein a virulence factor


def read_proteins(path: Path) -> list[str]:
    with gzip.open(path, "rt") as fh:
        return [ln for ln in fh.read().split("\n") if ln]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--attn", type=Path, required=True)
    p.add_argument("--prot-dir", type=Path, required=True)
    p.add_argument("--vfdb-dmnd", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--out", type=Path, required=True, help="output prefix")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    d = pd.read_parquet(args.attn).dropna(subset=["true_label"])
    d = d[(d.true_label == 0) | (d.pred > 0.5)]   # pathogenic = confident-correct; controls = all neg
    have = {int(f.name[:-7]) for f in args.prot_dir.glob("*.txt.gz")}
    d = d[d.bacdive_id.isin(have)].reset_index(drop=True)

    # build candidate FASTA: top-k attended for all genomes; random-k for pathogenic (control 1)
    fasta = args.out.with_suffix(".faa")
    n_pos = n_neg = 0
    with open(fasta, "w") as fh:
        for _, r in d.iterrows():
            bid = int(r.bacdive_id); lab = int(r.true_label)
            seqs = read_proteins(args.prot_dir / f"{bid}.txt.gz")
            if len(seqs) < args.top_k * 3:
                continue
            top = list(r.top_idx[:args.top_k])
            for rank, idx in enumerate(top):
                if idx < len(seqs):
                    fh.write(f">{bid}|top|{rank}|{lab}\n{seqs[idx]}\n")
            if lab == 1:
                n_pos += 1
                pool = [i for i in range(len(seqs)) if i not in set(top)]
                for rank, idx in enumerate(rng.choice(pool, size=args.top_k, replace=False)):
                    fh.write(f">{bid}|rand|{rank}|{lab}\n{seqs[idx]}\n")
            else:
                n_neg += 1
    print(f"genomes: {n_pos} pathogenic, {n_neg} non-pathogenic; wrote candidates -> {fasta}")

    # diamond blastp vs VFDB
    hits_tsv = args.out.with_suffix(".tsv")
    subprocess.run([
        "diamond", "blastp", "--db", str(args.vfdb_dmnd), "--query", str(fasta),
        "--out", str(hits_tsv), "--outfmt", "6", "qseqid", "sseqid", "pident", "evalue",
        "--evalue", str(EVALUE), "--max-target-seqs", "1", "--quiet", "--threads", "4",
    ], check=True)
    hit = set()
    if hits_tsv.exists() and hits_tsv.stat().st_size:
        for line in hits_tsv.read_text().splitlines():
            q, s, pid, ev = line.split("\t")
            if float(ev) < EVALUE:
                hit.add(q)

    # reload candidate ids by role/label
    cand = []
    for line in fasta.read_text().splitlines():
        if line.startswith(">"):
            bid, role, rank, lab = line[1:].split("|")
            cand.append((line[1:], int(bid), role, int(lab)))
    cdf = pd.DataFrame(cand, columns=["qid", "bid", "role", "lab"])
    cdf["vf"] = cdf.qid.isin(hit).astype(int)

    def rate(sub): return sub.vf.mean() if len(sub) else float("nan")
    pos_top = cdf[(cdf.lab == 1) & (cdf.role == "top")]
    pos_rand = cdf[(cdf.lab == 1) & (cdf.role == "rand")]
    neg_top = cdf[(cdf.lab == 0) & (cdf.role == "top")]

    print("\n=== VFDB-hit rates (fraction of proteins that are virulence factors) ===")
    print(f"  pathogenic, TOP-attended : {rate(pos_top):.3f}  ({pos_top.vf.sum()}/{len(pos_top)})")
    print(f"  pathogenic, RANDOM       : {rate(pos_rand):.3f}  ({pos_rand.vf.sum()}/{len(pos_rand)})")
    print(f"  non-path,   TOP-attended : {rate(neg_top):.3f}  ({neg_top.vf.sum()}/{len(neg_top)})")

    # Control 1: paired within-genome (top vs random), Wilcoxon over genomes
    g = cdf[cdf.lab == 1].groupby(["bid", "role"]).vf.mean().unstack()
    g = g.dropna()
    if len(g) > 5:
        w = stats.wilcoxon(g["top"], g["rand"], alternative="greater", zero_method="zsplit")
        print(f"\n  Control 1 (within-genome, n={len(g)} genomes): top {g['top'].mean():.3f} vs random "
              f"{g['rand'].mean():.3f}  Wilcoxon p={w.pvalue:.2e}")

    # Control 2: top-attended VF rate, pathogenic vs non-pathogenic (Fisher)
    a, b = pos_top.vf.sum(), len(pos_top) - pos_top.vf.sum()
    c, e = neg_top.vf.sum(), len(neg_top) - neg_top.vf.sum()
    odr, fp = stats.fisher_exact([[a, b], [c, e]], alternative="greater")
    print(f"  Control 2 (pathogenic vs non-path top-attended): OR={odr:.2f}  Fisher p={fp:.2e}")

    cdf.to_parquet(args.out.with_suffix(".candidates.parquet"), index=False)
    print(f"\nwrote per-candidate VF calls -> {args.out.with_suffix('.candidates.parquet')}")


if __name__ == "__main__":
    main()
