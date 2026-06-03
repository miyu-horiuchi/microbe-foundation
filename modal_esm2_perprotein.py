"""
modal_esm2_perprotein.py — un-pooled, full-proteome ESM-2 embeddings on Modal.

GPU analogue of compute_esm2_perprotein_mp.py. Instead of one local GPU, fan out
across many Modal GPU containers: each container loads ESM-2 once, then for a
batch of genomes does NCBI fetch + pyrodigal (CPU) and ESM-2 embedding (GPU),
writing one [n_proteins, 640] fp16 .npy per genome into a persistent Volume.

Two-stage Modal app:

  1) Embedder.process_batch(pairs)  (parallel GPU) — for a batch of
        (bacdive_id, accession): fetch FASTA, pyrodigal-predict proteins, embed
        every protein with ESM-2 (residue-mean-pooled, NOT pooled across
        proteins), write <bid>.npy to the output Volume. Returns manifest rows.
  2) local driver                   (local)       — read genome_accessions.tsv,
        skip genomes already in the Volume, .map() across containers, then merge
        + persist manifest.parquet into the Volume.

Output Volume layout (name: microbe-esm2-perprotein):
    <bacdive_id>.npy     float16 [n_proteins, embed_dim]
    manifest.parquet     bacdive_id, accession, n_proteins, status, path

Run:
    # smoke: 20 genomes on the tiny 8M model (proves the pipeline, ~cents)
    modal run modal_esm2_perprotein.py --limit 20 \
        --model facebook/esm2_t6_8M_UR50D --batch-size 10

    # full corpus on the 150M model (the locked spec config)
    modal run modal_esm2_perprotein.py

    # pull results down for local training (model.py --per-protein)
    modal volume get microbe-esm2-perprotein / ./data/esm2_perprotein

GPU type defaults to A100 (ample for 150M/640, cheaper than H100). Override:
    ESM2_GPU=H100 modal run modal_esm2_perprotein.py
"""
# NB: no `from __future__ import annotations` — it stringifies the class-body
# type hints and breaks modal.parameter's type resolution (Modal 1.2.x).
import os
import time
from pathlib import Path

import modal

ROOT = Path(__file__).parent

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "pyrodigal",
        "biopython",       # microbe_model.features.genome imports Bio.SeqIO
        "requests",
        "pandas",
        "pyarrow",
        "numpy<2",
    )
    # ship the microbe_model package so containers can import the existing
    # fetch / predict / embed helpers (single source of truth with the local path)
    .add_local_dir(str(ROOT / "microbe_model"), "/root/microbe_model", copy=True)
)

# Persistent output: one .npy per genome (~100 GB at full scale, fp16).
out_volume = modal.Volume.from_name("microbe-esm2-perprotein", create_if_missing=True)
OUT_DIR = "/out"

# Persistent HuggingFace cache so the ESM-2 weights download once, not per container.
hf_volume = modal.Volume.from_name("microbe-hf-cache", create_if_missing=True)
HF_DIR = "/hf_cache"

# A100 is plenty for 150M/640 and ~2x cheaper than H100. Override with ESM2_GPU.
GPU = os.environ.get("ESM2_GPU", "A100")

app = modal.App("microbe-esm2-perprotein", image=image)


def _genome_filename(bid: int) -> str:
    return f"{bid}.npy"


# ---------------------------------------------------------------------------
# Stage 1: per-batch GPU worker
# ---------------------------------------------------------------------------


def _fetch_predict_one(bid_acc):
    """Worker (module-level so ProcessPoolExecutor can pickle it): fetch + predict.
    Returns (bid, acc, proteins | None, status)."""
    import sys
    sys.path.insert(0, "/root")
    from microbe_model.features.genome import predict_genes
    from microbe_model.pipeline import _fetch_fasta_bytes
    bid, acc = bid_acc
    try:
        contigs = _fetch_fasta_bytes(acc)
    except Exception:
        return (bid, acc, None, "FETCH_FAIL")
    if not contigs:
        return (bid, acc, None, "FETCH_FAIL")
    try:
        proteins, _cds, _nt = predict_genes(contigs)
    except Exception as e:
        return (bid, acc, None, f"PRED_FAIL:{type(e).__name__}")
    if not proteins:
        return (bid, acc, None, "PRED_FAIL:empty")
    return (bid, acc, proteins, "OK")


@app.cls(
    gpu=GPU,
    volumes={OUT_DIR: out_volume, HF_DIR: hf_volume},
    secrets=[modal.Secret.from_name("ncbi-api-key")],  # exports NCBI_API_KEY
    cpu=16,
    memory=32768,
    timeout=60 * 60 * 2,   # 2 hr per batch
    retries=2,
    max_containers=int(os.environ.get("ESM2_MAX_CONTAINERS", "20")),
)
class Embedder:
    model_name: str = modal.parameter(default="facebook/esm2_t30_150M_UR50D")
    prot_batch_size: int = modal.parameter(default=32)

    @modal.enter()
    def load(self):
        """Load ESM-2 once per container (amortized across every batch it handles)."""
        import os as _os
        _os.environ["HF_HOME"] = HF_DIR
        import sys
        sys.path.insert(0, "/root")
        import torch
        from microbe_model.features.embeddings import load_esm2
        self._torch = torch
        t0 = time.time()
        self.tokenizer, self.model, self.device = load_esm2(self.model_name, device=torch.device("cuda"))
        self.embed_dim = int(self.model.config.hidden_size)
        print(f"loaded {self.model_name} (dim={self.embed_dim}) in {time.time()-t0:.0f}s", flush=True)

    @modal.method()
    def process_batch(self, pairs: list[tuple[int, str]]) -> list[dict]:
        """Fetch+predict (CPU, parallel) then ESM-2 embed (GPU) each genome; write
        one <bid>.npy per genome to the Volume. Returns one manifest row per genome."""
        import sys
        sys.path.insert(0, "/root")
        import numpy as np
        from concurrent.futures import ProcessPoolExecutor
        from microbe_model.features.embeddings import embed_proteins

        if not os.environ.get("NCBI_API_KEY"):
            print("WARNING: NCBI_API_KEY not in container env (Modal Secret missing?)", flush=True)

        out_volume.reload()
        t0 = time.time()
        rows: list[dict] = []
        n_ok = n_ff = n_pf = n_ef = n_skip = 0

        # CPU stage: fetch + pyrodigal across all genomes in the batch in parallel.
        with ProcessPoolExecutor(max_workers=12) as ex:
            fetched = list(ex.map(_fetch_predict_one, pairs))

        # GPU stage: embed each genome's full proteome, write its matrix.
        for bid, acc, proteins, status in fetched:
            fname = _genome_filename(bid)
            fpath = os.path.join(OUT_DIR, fname)
            if os.path.exists(fpath):
                n_skip += 1
                rows.append({"bacdive_id": bid, "accession": acc,
                             "n_proteins": int(np.load(fpath).shape[0]),
                             "status": "ok", "path": fname})
                continue
            if status == "FETCH_FAIL":
                n_ff += 1
                rows.append({"bacdive_id": bid, "accession": acc, "n_proteins": 0,
                             "status": "FETCH_FAIL", "path": ""})
                continue
            if status.startswith("PRED_FAIL"):
                n_pf += 1
                rows.append({"bacdive_id": bid, "accession": acc, "n_proteins": 0,
                             "status": status, "path": ""})
                continue
            try:
                # [n_proteins, embed_dim] — each protein residue-mean-pooled, NOT pooled across proteins.
                matrix = embed_proteins(proteins, self.tokenizer, self.model, self.device,
                                        batch_size=self.prot_batch_size)
                np.save(fpath, matrix.astype(np.float16))
                n_ok += 1
                rows.append({"bacdive_id": bid, "accession": acc,
                             "n_proteins": int(matrix.shape[0]), "status": "ok", "path": fname})
            except Exception as e:
                n_ef += 1
                print(f"  [warn] {bid} embed: {type(e).__name__}: {e}", flush=True)
                rows.append({"bacdive_id": bid, "accession": acc, "n_proteins": 0,
                             "status": f"EMBED_FAIL:{type(e).__name__}", "path": ""})

        out_volume.commit()
        dt = time.time() - t0
        tot_prot = sum(r["n_proteins"] for r in rows if r["status"] == "ok")
        print(f"  batch: ok={n_ok} skip={n_skip} ff={n_ff} pf={n_pf} ef={n_ef} "
              f"proteins={tot_prot:,} in {dt:.0f}s", flush=True)
        return rows


# ---------------------------------------------------------------------------
# Volume helpers (run remotely so they see the Volume)
# ---------------------------------------------------------------------------


@app.function(volumes={OUT_DIR: out_volume})
def scan_done() -> list[int]:
    """bacdive_ids whose .npy already exists and is non-empty (resume support)."""
    out_volume.reload()
    done = []
    for name in os.listdir(OUT_DIR):
        if name.endswith(".npy"):
            try:
                if os.path.getsize(os.path.join(OUT_DIR, name)) > 0:
                    done.append(int(name[:-4]))
            except (ValueError, OSError):
                continue
    return done


@app.function(volumes={OUT_DIR: out_volume})
def write_manifest(rows: list[dict]) -> int:
    """Merge this run's manifest rows with any existing manifest in the Volume
    (keyed by bacdive_id) and persist manifest.parquet. Returns total row count."""
    import pandas as pd
    out_volume.reload()
    mpath = os.path.join(OUT_DIR, "manifest.parquet")
    frames = []
    if os.path.exists(mpath):
        frames.append(pd.read_parquet(mpath))
    if rows:
        frames.append(pd.DataFrame(rows))
    if not frames:
        return 0
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset="bacdive_id", keep="last").reset_index(drop=True)
    merged.to_parquet(mpath, index=False)
    out_volume.commit()
    return len(merged)


# ---------------------------------------------------------------------------
# Stage 2: local driver
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(limit: int = 0, batch_size: int = 20,
         model: str = "facebook/esm2_t30_150M_UR50D", prot_batch_size: int = 32):
    """Drive the per-protein extraction. Reads data/genome_accessions.tsv, skips
    genomes already in the Volume, fans out over GPU containers, persists manifest."""
    import pandas as pd

    acc_path = ROOT / "data" / "genome_accessions.tsv"
    df = pd.read_csv(acc_path, sep="\t")
    if limit:
        df = df.head(limit)
    pairs = [(int(r.bacdive_id), str(r.accession)) for r in df.itertuples()]

    done = set(scan_done.remote())
    before = len(pairs)
    pairs = [(b, a) for (b, a) in pairs if b not in done]
    print(f"resume: {len(done):,} genomes already in Volume, "
          f"{len(pairs):,} of {before:,} to compute", flush=True)
    if not pairs:
        print("nothing to do.", flush=True)
        return

    batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]
    print(f"model={model} gpu={GPU}  dispatching {len(batches)} batches "
          f"of up to {batch_size} genomes", flush=True)

    embedder = Embedder(model_name=model, prot_batch_size=prot_batch_size)
    t0 = time.time()
    all_rows: list[dict] = []
    n_done = 0
    n_ok = 0
    for rows in embedder.process_batch.map(batches, order_outputs=False, return_exceptions=True):
        if isinstance(rows, Exception):
            print(f"  [WARN] batch raised: {type(rows).__name__}: {rows}", flush=True)
            continue
        all_rows.extend(rows)
        n_ok += sum(1 for r in rows if r["status"] == "ok")
        n_done += 1
        dt = time.time() - t0
        rate = n_done / max(dt, 1e-6) * batch_size
        eta_min = (len(batches) - n_done) * dt / max(n_done, 1) / 60
        # Persist manifest periodically so a killed driver doesn't lose the index.
        if n_done % 10 == 0:
            total = write_manifest.remote(all_rows)
            all_rows = []
            print(f"  [{n_done}/{len(batches)}] ok={n_ok:,} manifest={total:,} "
                  f"rate={rate:.1f} genomes/s eta={eta_min:.1f}min", flush=True)

    total = write_manifest.remote(all_rows)
    print(f"\ndone in {(time.time()-t0)/60:.1f}min. ok={n_ok:,}  manifest rows={total:,}", flush=True)
    print("pull results:  modal volume get microbe-esm2-perprotein / ./data/esm2_perprotein", flush=True)
