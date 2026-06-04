"""
modal_eggnog.py — eggNOG-mapper feature extraction on Modal.

Three-stage Modal app:

  1) `download_db()`         (run once) — downloads eggNOG DB into a Modal Volume.
  2) `process_batch(pairs)`  (parallel) — for a batch of (bacdive_id, accession)
                                          pairs: fetch FASTA, predict proteins
                                          with pyrodigal, run emapper.py once
                                          on the batch, return per-genome OG sets.
  3) local driver            (local)    — read genome_accessions.tsv, split into
                                          batches, .map() across many containers,
                                          aggregate into per-genome OG vectors,
                                          write data/eggnog_features.npz.

Run:
    modal run modal_eggnog.py::download_db          # one-time, ~10 min
    modal run modal_eggnog.py --limit 200           # smoke test on 200 genomes
    modal run modal_eggnog.py                       # full corpus
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).parent

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("diamond-aligner", "hmmer", "wget", "git", "build-essential")
    # Python 3.10 because eggnog-mapper 2.1.13's setup.py pins biopython==1.76,
    # which has C extensions that don't compile on Python 3.11+.
    .pip_install(
        "eggnog-mapper==2.1.13",
        "pyrodigal",
        "requests",
        "pandas",
        "numpy<2",
    )
    # ship the microbe_model package into the image so workers can import it
    .add_local_dir(str(ROOT / "microbe_model"), "/root/microbe_model", copy=True)
)

# Persistent volume for eggNOG DB (~59 GB decompressed) — downloaded once,
# mounted into every container.
db_volume = modal.Volume.from_name("microbe-eggnog-db", create_if_missing=True)
DB_DIR = "/eggnog_data"

app = modal.App("microbe-eggnog", image=image)

EGGNOG_BASE = "http://eggnog6.embl.de/download/emapperdb-5.0.2"
DB_FILES = ["eggnog.db.gz", "eggnog_proteins.dmnd.gz", "eggnog.taxa.tar.gz"]


# ---------------------------------------------------------------------------
# Stage 1: one-time DB download
# ---------------------------------------------------------------------------

EXPECTED_SIZES = {
    "eggnog.db.gz": 6776977123,
    "eggnog_proteins.dmnd.gz": 5208806170,
    "eggnog.taxa.tar.gz": 72797584,
}


@app.function(
    volumes={DB_DIR: db_volume},
    cpu=4,
    memory=8192,
    timeout=60 * 60,  # 1 hour
)
def download_db():
    """Download eggNOG-mapper diamond database into the volume. Idempotent +
    self-healing: detects truncated .gz files from prior preempted downloads
    and re-fetches them."""
    os.makedirs(DB_DIR, exist_ok=True)

    expected_decompressed = ["eggnog.db", "eggnog_proteins.dmnd", "eggnog.taxa.db"]
    if all(os.path.exists(f"{DB_DIR}/{f}") for f in expected_decompressed):
        print("DB already populated. Skipping download.")
        return

    for fname in DB_FILES:
        target_gz = f"{DB_DIR}/{fname}"
        expected_size = EXPECTED_SIZES.get(fname)
        if os.path.exists(target_gz):
            actual_size = os.path.getsize(target_gz)
            if expected_size is None or actual_size == expected_size:
                print(f"  already downloaded ({actual_size} bytes): {fname}")
                continue
            else:
                print(f"  TRUNCATED ({actual_size} of expected {expected_size}): {fname} — re-downloading")
                os.remove(target_gz)
        url = f"{EGGNOG_BASE}/{fname}"
        print(f"  downloading {url} -> {target_gz}", flush=True)
        subprocess.run(["wget", "-q", "-O", target_gz, url], check=True)
        actual_size = os.path.getsize(target_gz)
        if expected_size and actual_size != expected_size:
            raise RuntimeError(f"download of {fname} ended at {actual_size} != expected {expected_size}")

    print("decompressing ...", flush=True)
    for fname in ["eggnog.db.gz", "eggnog_proteins.dmnd.gz"]:
        gz = f"{DB_DIR}/{fname}"
        if os.path.exists(gz):
            subprocess.run(["gunzip", "-f", gz], check=True)
    tarball = f"{DB_DIR}/eggnog.taxa.tar.gz"
    if os.path.exists(tarball):
        subprocess.run(["tar", "xzf", tarball, "-C", DB_DIR], check=True)
        os.remove(tarball)

    db_volume.commit()
    print("DB ready in volume", flush=True)
    print(subprocess.check_output(["ls", "-lh", DB_DIR]).decode())


# ---------------------------------------------------------------------------
# Stage 2: per-batch worker — fetch + predict + emapper
# ---------------------------------------------------------------------------

def _fetch_predict_one(bid_acc):
    """Worker function — must be module-level so ProcessPoolExecutor can pickle it."""
    import sys
    sys.path.insert(0, "/root")
    from microbe_model.features.genome import predict_genes
    from microbe_model.pipeline import _fetch_fasta_bytes
    bid, acc = bid_acc
    try:
        contigs = _fetch_fasta_bytes(acc)
    except Exception:
        return (bid, None, "FETCH_FAIL")
    if not contigs:
        return (bid, None, "FETCH_FAIL")
    try:
        proteins, _cds, _nt = predict_genes(contigs)
    except Exception as e:
        return (bid, None, f"PRED_FAIL:{type(e).__name__}")
    if not proteins:
        return (bid, None, "PRED_FAIL:empty")
    return (bid, proteins, "OK")


@app.function(
    volumes={DB_DIR: db_volume},
    secrets=[modal.Secret.from_name("ncbi-api-key")],  # exports NCBI_API_KEY in container env
    cpu=16,
    memory=24576,
    timeout=60 * 60,        # 1 hr per batch
    retries=2,
    max_containers=300,     # fan out as wide as Modal will let us
)
def process_batch(pairs: list[tuple[int, str]], eval_cutoff: float = 1e-5) -> dict:
    """One worker processes a batch of (bid, accession) genomes end-to-end:
       parallel fetch + pyrodigal + emapper. Returns {bid -> list of OG IDs}.
    """
    import os, subprocess, tempfile, time, sys
    from concurrent.futures import ProcessPoolExecutor
    sys.path.insert(0, "/root")
    if not os.environ.get("NCBI_API_KEY"):
        print("WARNING: NCBI_API_KEY not in container env (Modal Secret missing?)", flush=True)

    t0 = time.time()
    work = tempfile.mkdtemp()
    faa_path = os.path.join(work, "batch.faa")

    # Parallel fetch + pyrodigal (12 worker processes per container).
    n_ok = n_ff = n_pf = 0
    with open(faa_path, "w") as out_fh, ProcessPoolExecutor(max_workers=12) as ex:
        for bid, proteins, status in ex.map(_fetch_predict_one, pairs):
            if status == "FETCH_FAIL":
                n_ff += 1
            elif status.startswith("PRED_FAIL"):
                n_pf += 1
            else:
                for i, seq in enumerate(proteins):
                    if seq:
                        out_fh.write(f">bid_{bid}_p{i}\n{seq}\n")
                n_ok += 1
    fetch_dt = time.time() - t0
    print(f"  fetch+predict (parallel): {n_ok} ok, {n_ff} ff, {n_pf} pf, in {fetch_dt:.0f}s", flush=True)

    if n_ok == 0:
        return {}

    # emapper.py
    out_prefix = "eggnog"
    cmd = [
        "emapper.py",
        "-i", faa_path,
        "--itype", "proteins",
        "-m", "diamond",
        "--data_dir", DB_DIR,
        "--cpu", str(os.cpu_count() or 8),
        "--no_file_comments",
        "--override",
        "--output_dir", work,
        "-o", out_prefix,
    ]
    print(f"  running emapper on {n_ok} genomes ...", flush=True)
    e0 = time.time()
    subprocess.run(cmd, check=True, cwd=work)
    emapper_dt = time.time() - e0

    ann_path = os.path.join(work, f"{out_prefix}.emapper.annotations")

    # Parse annotations → per-genome OG sets
    per_genome: dict[int, set[str]] = {}
    n_rows = 0
    with open(ann_path) as fh:
        header_line = fh.readline().lstrip("#").strip()
        cols = header_line.split("\t")
        try:
            col_query = cols.index("query")
            col_ogs = cols.index("eggNOG_OGs")
        except ValueError:
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
            for ent in ogs_str.split(","):
                og_id = ent.split("@")[0]
                if og_id:
                    per_genome.setdefault(bid, set()).add(og_id)
            n_rows += 1
    total_dt = time.time() - t0
    print(f"  emapper {emapper_dt:.0f}s, parsed {n_rows} hits across {len(per_genome)} genomes, "
          f"total batch {total_dt:.0f}s", flush=True)
    return {bid: sorted(ogs) for bid, ogs in per_genome.items()}


# ---------------------------------------------------------------------------
# Stage 3: local driver
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(limit: int = 0, batch_size: int = 100, min_freq: float = 0.01):
    """Drive the eggNOG run from local. Reads data/genome_accessions.tsv,
    splits into batches, fans out via .map(), aggregates into npz.
    """
    import pandas as pd
    import numpy as np

    # NCBI_API_KEY is read inside each container from the Modal Secret
    # named 'ncbi-api-key' (see process_batch decorator). No need to forward.

    acc_path = ROOT / "data" / "genome_accessions.tsv"
    df = pd.read_csv(acc_path, sep="\t")
    if limit:
        df = df.head(limit)
    pairs = [(int(r.bacdive_id), str(r.accession)) for r in df.itertuples()]

    # If a checkpoint exists, skip genomes already done so we don't redo work.
    per_genome: dict[int, list[str]] = {}
    ckpt_path = ROOT / "data" / "eggnog_per_genome_partial.json"
    if ckpt_path.exists():
        try:
            cached = json.loads(ckpt_path.read_text())
            per_genome.update({int(k): v for k, v in cached.items()})
            done_bids = set(per_genome.keys())
            before = len(pairs)
            pairs = [(b, a) for (b, a) in pairs if b not in done_bids]
            print(f"  resume: {len(per_genome):,} genomes already in checkpoint, "
                  f"{len(pairs):,} of {before:,} remaining", flush=True)
        except Exception as e:
            print(f"  checkpoint exists but couldn't load: {e}", flush=True)

    batches = [pairs[i : i + batch_size] for i in range(0, len(pairs), batch_size)]
    print(f"dispatching {len(batches)} batches of up to {batch_size} genomes "
          f"(total {len(pairs):,} genomes)", flush=True)

    t0 = time.time()
    n_done = 0
    # Checkpoint per-genome OG sets to disk every CKPT_EVERY batches so a
    # killed driver doesn't lose all in-memory work (which is what bit us on
    # the previous Modal eggNOG attempts). The checkpoint was already loaded
    # above to skip done genomes.
    CKPT_EVERY = 5

    for batch_result in process_batch.map(
        batches,
        order_outputs=False,
        return_exceptions=True,  # keep loop alive on Modal-side blob fetch errors
    ):
        if isinstance(batch_result, Exception):
            print(f"  [WARN] batch returned exception: {type(batch_result).__name__}: {batch_result}",
                  flush=True)
            continue
        per_genome.update(batch_result)
        n_done += 1
        dt = time.time() - t0
        rate = n_done / max(dt, 1e-6) * batch_size
        eta_min = (len(batches) - n_done) * dt / max(n_done, 1) / 60
        print(f"  [{n_done}/{len(batches)}] {len(per_genome):,} genomes done, "
              f"rate={rate:.1f} genomes/sec, eta={eta_min:.1f}min", flush=True)
        if n_done % CKPT_EVERY == 0:
            ckpt_path.write_text(json.dumps({str(k): v for k, v in per_genome.items()}))

    # final checkpoint
    ckpt_path.write_text(json.dumps({str(k): v for k, v in per_genome.items()}))
    print(f"\nall batches done. {len(per_genome):,} genomes with OGs.", flush=True)

    # Build vocab and per-genome vectors
    n = len(per_genome)
    og_counts: dict[str, int] = {}
    for ogs in per_genome.values():
        for og in ogs:
            og_counts[og] = og_counts.get(og, 0) + 1
    threshold = max(1, int(n * min_freq))
    vocab = sorted([og for og, c in og_counts.items() if c >= threshold])
    print(f"vocab: {len(vocab):,} OGs at min_freq={min_freq} ({threshold} of {n} genomes)",
          flush=True)

    idx = {og: i for i, og in enumerate(vocab)}
    bid_to_acc = dict(zip(df.bacdive_id.astype(int), df.accession.astype(str)))
    bids_out, accs_out, feats_out = [], [], []
    for bid, ogs in per_genome.items():
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

    out_npz = ROOT / "data" / "eggnog_features.npz"
    out_vocab = ROOT / "data" / "eggnog_vocab.json"
    np.savez(
        out_npz,
        bacdive_ids=np.array(bids_out, dtype=np.int64),
        features=np.array(feats_out, dtype=np.float32),
        accessions=np.array(accs_out, dtype="<U24"),
    )
    out_vocab.write_text(json.dumps({"min_freq": min_freq, "n_genomes": n, "vocab": vocab}, indent=2))
    print(f"wrote {out_npz} shape=({len(bids_out)}, {len(vocab)})", flush=True)
    print(f"wrote {out_vocab}", flush=True)
