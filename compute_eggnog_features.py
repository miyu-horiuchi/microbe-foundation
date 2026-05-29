"""
compute_eggnog_features.py — eggNOG orthologous-group features per genome.

Two-phase pipeline:

Phase A (--phase fetch):
    ProcessPoolExecutor with N workers does NCBI fetch + pyrodigal predict.
    Per genome, writes ./eggnog_work/proteins/<bid>.faa with the proteins,
    each header tagged ">bid_<bid>_p<idx>" so we can split back per-genome.

Phase B (--phase annotate):
    Concatenate all .faa files → mega.faa, run emapper.py once (diamond
    mode) over the whole file. Output: ./eggnog_work/eggnog.emapper.annotations
    (one row per protein with eggnog_OGs and COG_category fields).

Phase C (--phase vectorize):
    Parse emapper output, group by bid, build per-genome OG presence vector
    using a vocab of OGs that appear in >= --min-freq fraction of genomes.
    Output: data/eggnog_features.npz with the standard schema.

Usage:
    # Phase A: fetch + predict all proteins (parallel)
    python compute_eggnog_features.py --phase fetch --workers 16

    # Phase B: run emapper on concatenated FASTA
    python compute_eggnog_features.py --phase annotate \
        --eggnog-data /home/ubuntu/eggnog_data --cpus 30

    # Phase C: parse + vectorize
    python compute_eggnog_features.py --phase vectorize --min-freq 0.01
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
WORK_DIR = ROOT / "eggnog_work"
PROT_DIR = WORK_DIR / "proteins"
MEGA_FAA = WORK_DIR / "mega.faa"
EMAPPER_PREFIX = WORK_DIR / "eggnog"
EMAPPER_ANN = WORK_DIR / "eggnog.emapper.annotations"
DEFAULT_ACCESSIONS = DATA_DIR / "genome_accessions.tsv"
DEFAULT_OUT = DATA_DIR / "eggnog_features.npz"
DEFAULT_VOCAB = DATA_DIR / "eggnog_vocab.json"
ACC_DTYPE = "<U24"


# =============================================================================
# Phase A: fetch + predict
# =============================================================================


def _fetch_predict_one(args_tuple):
    """Worker: fetch FASTA + pyrodigal predict, write proteins to file."""
    bid, acc, out_path = args_tuple
    from microbe_model.features.genome import predict_genes
    from microbe_model.pipeline import _fetch_fasta_bytes
    p = Path(out_path)
    if p.exists() and p.stat().st_size > 0:
        return (bid, acc, "SKIP_EXISTS")
    try:
        contigs = _fetch_fasta_bytes(acc)
    except Exception:
        return (bid, acc, "FETCH_FAIL")
    if not contigs:
        return (bid, acc, "FETCH_FAIL")
    try:
        proteins, _cds, _total_nt = predict_genes(contigs)
    except Exception as e:
        return (bid, acc, f"PRED_FAIL:{type(e).__name__}")
    if not proteins:
        return (bid, acc, "PRED_FAIL:empty")
    # Write FASTA with tagged headers
    with open(p, "w") as fh:
        for i, seq in enumerate(proteins):
            if not seq:
                continue
            fh.write(f">bid_{bid}_p{i}\n{seq}\n")
    return (bid, acc, len(proteins))


def phase_fetch(args):
    PROT_DIR.mkdir(parents=True, exist_ok=True)
    acc_df = pd.read_csv(args.accessions, sep="\t")
    if args.limit:
        acc_df = acc_df.head(args.limit)
    pairs = [(int(r.bacdive_id), str(r.accession), str(PROT_DIR / f"{int(r.bacdive_id)}.faa"))
             for r in acc_df.itertuples()]

    # Skip ones already done (resumable)
    todo = [t for t in pairs if not Path(t[2]).exists() or Path(t[2]).stat().st_size == 0]
    skipped = len(pairs) - len(todo)
    print(f"phase=fetch  total={len(pairs):,}  skip_existing={skipped:,}  todo={len(todo):,}", flush=True)
    if not todo:
        return

    ok = ff = pf = 0
    start = time.time()
    max_inflight = args.workers * 4
    ex = ProcessPoolExecutor(max_workers=args.workers)
    try:
        in_flight = set()
        idx = 0
        while idx < len(todo) and len(in_flight) < max_inflight:
            in_flight.add(ex.submit(_fetch_predict_one, todo[idx])); idx += 1
        i = 0
        while in_flight:
            done_fut = next(as_completed(in_flight))
            in_flight.remove(done_fut)
            if idx < len(todo):
                in_flight.add(ex.submit(_fetch_predict_one, todo[idx])); idx += 1
            bid, acc, payload = done_fut.result()
            i += 1
            if isinstance(payload, int):
                ok += 1
            elif payload == "FETCH_FAIL":
                ff += 1
            elif isinstance(payload, str) and payload.startswith("PRED_FAIL"):
                pf += 1
            if i % 100 == 0:
                el = time.time() - start
                rate = i / max(el, 1e-6)
                eta_min = (len(todo) - i) / max(rate, 1e-6) / 60
                print(f"  [{i:>6,}/{len(todo):,}] ok={ok:,} ff={ff:,} pf={pf:,} "
                      f"rate={rate:.2f}/s eta={eta_min:.1f}min", flush=True)
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    print(f"\nfetch done. ok={ok:,} ff={ff:,} pf={pf:,}", flush=True)


# =============================================================================
# Phase B: concat + emapper
# =============================================================================


def phase_annotate(args):
    print(f"phase=annotate. concatenating .faa files ...", flush=True)
    files = sorted(PROT_DIR.glob("*.faa"))
    print(f"  found {len(files):,} protein files", flush=True)
    if not files:
        sys.exit("No protein files found. Run --phase fetch first.")

    with open(MEGA_FAA, "wb") as out:
        for f in files:
            with open(f, "rb") as r:
                out.write(r.read())
    sz_mb = MEGA_FAA.stat().st_size // 1024 // 1024
    nprot = subprocess.check_output(["grep", "-c", "^>", str(MEGA_FAA)]).decode().strip()
    print(f"  mega.faa: {sz_mb} MB, {int(nprot):,} proteins", flush=True)

    cmd = [
        "emapper.py",
        "-i", str(MEGA_FAA),
        "--itype", "proteins",
        "-m", "diamond",
        "--data_dir", args.eggnog_data,
        "--cpu", str(args.cpus),
        "--no_file_comments",
        "--output_dir", str(WORK_DIR),
        "-o", "eggnog",
        "--override",
    ]
    print(f"  running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    print(f"\nannotate done. output: {EMAPPER_ANN}", flush=True)


# =============================================================================
# Phase C: parse + vectorize
# =============================================================================


def phase_vectorize(args):
    if not EMAPPER_ANN.exists():
        sys.exit(f"Annotations file not found: {EMAPPER_ANN}. Run --phase annotate first.")

    print(f"phase=vectorize. reading {EMAPPER_ANN} ...", flush=True)
    # The emapper annotations file is TSV with these columns (no comment lines because --no_file_comments):
    # #query  seed_ortholog  evalue  score  eggNOG_OGs  max_annot_lvl  COG_category  Description ...
    # We want #query and eggNOG_OGs columns.
    # Format of eggNOG_OGs: comma-separated list like "COG0001@1|root,COG0001@2|Bacteria,..."

    # Stream and group by bid (extracted from query name "bid_<bid>_p<idx>")
    per_genome_ogs: dict[int, set[str]] = {}
    n_rows = 0
    t0 = time.time()
    with open(EMAPPER_ANN) as fh:
        header_line = fh.readline().strip()
        cols = header_line.lstrip("#").split("\t")
        try:
            col_query = cols.index("query")
            col_ogs = cols.index("eggNOG_OGs")
        except ValueError:
            # fall back to fixed positions
            col_query, col_ogs = 0, 4
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= col_ogs:
                continue
            query = parts[col_query]
            ogs_str = parts[col_ogs]
            if not query.startswith("bid_") or not ogs_str or ogs_str == "-":
                continue
            try:
                bid = int(query.split("_")[1])
            except (ValueError, IndexError):
                continue
            # Each OG entry has form "<og_id>@<taxid>|<taxname>"; we only keep og_id.
            ogs = set()
            for ent in ogs_str.split(","):
                og_id = ent.split("@")[0]
                if og_id:
                    ogs.add(og_id)
            per_genome_ogs.setdefault(bid, set()).update(ogs)
            n_rows += 1
            if n_rows % 1_000_000 == 0:
                dt = time.time() - t0
                print(f"  {n_rows:>10,} rows  {len(per_genome_ogs):,} genomes  rate={n_rows/dt:.0f}/s", flush=True)

    print(f"\nparsed {n_rows:,} rows across {len(per_genome_ogs):,} genomes", flush=True)

    # Build vocab: OGs that appear in >= min_freq of genomes
    n = len(per_genome_ogs)
    og_counts: dict[str, int] = {}
    for ogs in per_genome_ogs.values():
        for og in ogs:
            og_counts[og] = og_counts.get(og, 0) + 1
    threshold = max(1, int(n * args.min_freq))
    vocab = sorted([og for og, c in og_counts.items() if c >= threshold])
    print(f"vocab: {len(vocab):,} OGs at min_freq={args.min_freq} ({threshold} of {n} genomes)", flush=True)

    args.vocab.write_text(json.dumps({"min_freq": args.min_freq, "n_genomes": n, "vocab": vocab}, indent=2))
    print(f"  wrote vocab to {args.vocab}", flush=True)

    # Build per-genome binary vector
    idx = {og: i for i, og in enumerate(vocab)}
    acc_df = pd.read_csv(args.accessions, sep="\t")
    bid_to_acc = dict(zip(acc_df.bacdive_id.astype(int), acc_df.accession.astype(str)))
    bids_out = []
    accs_out = []
    feats_out = []
    for bid, ogs in per_genome_ogs.items():
        if bid not in bid_to_acc:
            continue
        v = np.zeros(len(vocab), dtype=np.float32)
        for og in ogs:
            j = idx.get(og)
            if j is not None:
                v[j] = 1.0
        bids_out.append(bid)
        accs_out.append(bid_to_acc[bid])
        feats_out.append(v)
    print(f"  built {len(bids_out):,} per-genome vectors", flush=True)

    np.savez(
        args.out,
        bacdive_ids=np.array(bids_out, dtype=np.int64),
        features=np.array(feats_out, dtype=np.float32),
        accessions=np.array(accs_out, dtype=ACC_DTYPE),
    )
    print(f"  wrote {args.out}  shape=({len(bids_out)}, {len(vocab)})", flush=True)


# =============================================================================
# CLI
# =============================================================================


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=["fetch", "annotate", "vectorize"], required=True)
    p.add_argument("--accessions", type=Path, default=DEFAULT_ACCESSIONS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    p.add_argument("--workers", type=int, default=16, help="parallel fetch+predict workers")
    p.add_argument("--cpus", type=int, default=30, help="cpus for emapper.py")
    p.add_argument("--eggnog-data", type=str, default="/home/ubuntu/eggnog_data")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--min-freq", type=float, default=0.01,
                   help="OG must appear in this fraction of genomes to enter vocab")
    args = p.parse_args()

    {
        "fetch": phase_fetch,
        "annotate": phase_annotate,
        "vectorize": phase_vectorize,
    }[args.phase](args)


if __name__ == "__main__":
    main()
