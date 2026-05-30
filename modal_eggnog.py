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
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("diamond-aligner", "hmmer", "wget", "git", "build-essential")
    # Install eggnog-mapper from its GitHub master branch — the PyPI version
    # is stuck on 2.0.6 which pins biopython==1.76 (incompatible with everything
    # else). Master is 2.1.x and works with modern biopython.
    .pip_install(
        "eggnog-mapper @ git+https://github.com/eggnogdb/eggnog-mapper.git@2.1.13",
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

@app.function(
    volumes={DB_DIR: db_volume},
    cpu=4,
    memory=8192,
    timeout=60 * 60,  # 1 hour
)
def download_db():
    """Download eggNOG-mapper diamond database into the volume. Idempotent."""
    os.makedirs(DB_DIR, exist_ok=True)

    # Skip if all 3 final files already exist
    expected = ["eggnog.db", "eggnog_proteins.dmnd", "eggnog.taxa.db"]
    if all(os.path.exists(f"{DB_DIR}/{f}") for f in expected):
        print("DB already populated. Skipping download.")
        return

    for fname in DB_FILES:
        target_gz = f"{DB_DIR}/{fname}"
        if os.path.exists(target_gz):
            print(f"  already downloaded: {fname}")
            continue
        url = f"{EGGNOG_BASE}/{fname}"
        print(f"  downloading {url} -> {target_gz}", flush=True)
        subprocess.run(["wget", "-q", "-O", target_gz, url], check=True)

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

@app.function(
    volumes={DB_DIR: db_volume},
    cpu=8,
    memory=16384,
    timeout=60 * 60,        # 1 hr per batch
    retries=2,
)
def process_batch(pairs: list[tuple[int, str]], eval_cutoff: float = 1e-5,
                  ncbi_api_key: str = "") -> dict:
    """One worker processes a batch of (bid, accession) genomes end-to-end:
       fetch + pyrodigal + emapper. Returns dict {bid -> sorted list of OG IDs}.
    """
    import os, subprocess, tempfile, time, sys
    sys.path.insert(0, "/root")
    if ncbi_api_key:
        os.environ["NCBI_API_KEY"] = ncbi_api_key
    from microbe_model.features.genome import predict_genes
    from microbe_model.pipeline import _fetch_fasta_bytes

    t0 = time.time()
    work = tempfile.mkdtemp()
    faa_path = os.path.join(work, "batch.faa")
    n_ok = n_ff = n_pf = 0
    with open(faa_path, "w") as out_fh:
        for bid, acc in pairs:
            try:
                contigs = _fetch_fasta_bytes(acc)
            except Exception:
                n_ff += 1
                continue
            if not contigs:
                n_ff += 1
                continue
            try:
                proteins, _cds, _nt = predict_genes(contigs)
            except Exception:
                n_pf += 1
                continue
            if not proteins:
                n_pf += 1
                continue
            for i, seq in enumerate(proteins):
                if seq:
                    out_fh.write(f">bid_{bid}_p{i}\n{seq}\n")
            n_ok += 1
    fetch_dt = time.time() - t0
    print(f"  fetch+predict: {n_ok} ok, {n_ff} ff, {n_pf} pf, in {fetch_dt:.0f}s", flush=True)

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

    ncbi_key = os.environ.get("NCBI_API_KEY", "ca856b6c99014a67eff3c662db107a46ed08")

    acc_path = ROOT / "data" / "genome_accessions.tsv"
    df = pd.read_csv(acc_path, sep="\t")
    if limit:
        df = df.head(limit)
    pairs = [(int(r.bacdive_id), str(r.accession)) for r in df.itertuples()]
    batches = [pairs[i : i + batch_size] for i in range(0, len(pairs), batch_size)]
    print(f"dispatching {len(batches)} batches of up to {batch_size} genomes "
          f"(total {len(pairs):,} genomes)", flush=True)

    t0 = time.time()
    per_genome: dict[int, list[str]] = {}
    n_done = 0
    for batch_result in process_batch.map(
        batches,
        kwargs={"ncbi_api_key": ncbi_key},
        order_outputs=False,
        return_exceptions=False,
    ):
        per_genome.update(batch_result)
        n_done += 1
        dt = time.time() - t0
        rate = n_done / max(dt, 1e-6) * batch_size
        eta_min = (len(batches) - n_done) * dt / max(n_done, 1) / 60
        print(f"  [{n_done}/{len(batches)}] {len(per_genome):,} genomes done, "
              f"rate={rate:.1f} genomes/sec, eta={eta_min:.1f}min", flush=True)

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
