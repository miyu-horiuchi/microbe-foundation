"""
modal_esm2_perprotein.py — un-pooled, full-proteome ESM-2 embeddings on Modal.

GPU analogue of compute_esm2_perprotein_mp.py, split into two phases because
NCBI rate-limits hard under wide fan-out (a one-phase run lost ~44% of genomes
to FETCH_FAIL when ~10 GPU containers fetched in parallel on one API key):

  Phase A — fetch+predict (CPU, ONE container, <=16 workers)
      Reproduces the proven single-process config (19,608/19,637 = 99.8% on the
      prior pooled run). Fetches FASTA + pyrodigal-predicts proteins and caches
      each genome's protein sequences to the Volume as proteins/<bid>.txt.gz.
      This is the rate-limited stage — kept deliberately narrow.

  Phase B — embed (GPU, fan out WIDE, no NCBI)
      Each container reads cached proteins from the Volume and runs ESM-2
      (residue-mean-pooled per protein, NOT pooled across proteins), writing one
      [n_proteins, embed_dim] fp16 <bid>.npy per genome. No network in the hot
      loop, so it scales to however many GPUs Modal will give us.

Caching proteins also makes a later 650M re-embed free of any re-fetch.

Output Volume layout (name: microbe-esm2-perprotein):
    proteins/<bid>.txt.gz   gzipped, one protein AA sequence per line (Phase A)
    <bid>.npy               float16 [n_proteins, embed_dim]          (Phase B)
    manifest.parquet        bacdive_id, accession, n_proteins, status, path

Run:
    # smoke: 20 genomes, tiny 8M model, both phases
    modal run modal_esm2_perprotein.py --limit 20 \
        --model facebook/esm2_t6_8M_UR50D --batch-size 10

    # full corpus on 150M (locked spec config) — runs Phase A then Phase B
    modal run modal_esm2_perprotein.py

    # run only one phase
    modal run modal_esm2_perprotein.py --skip-embed     # fetch+cache only
    modal run modal_esm2_perprotein.py --skip-fetch     # embed cached proteins

    # pull results down for local training (model.py --per-protein)
    modal volume get microbe-esm2-perprotein "*.npy" ./data/esm2_perprotein
    modal volume get microbe-esm2-perprotein manifest.parquet ./data/esm2_perprotein

GPU type defaults to A100 (ample for 150M/640, cheaper than H100). Override:
    ESM2_GPU=H100 modal run modal_esm2_perprotein.py
"""
# NB: no `from __future__ import annotations` — it stringifies the class-body
# type hints and breaks modal.parameter's type resolution (Modal 1.2.x).
import gzip
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

# Persistent output: cached proteins + one .npy per genome (~100 GB at full scale).
out_volume = modal.Volume.from_name("microbe-esm2-perprotein", create_if_missing=True)
OUT_DIR = "/out"
PROT_DIR = "/out/proteins"

# Persistent HuggingFace cache so the ESM-2 weights download once, not per container.
hf_volume = modal.Volume.from_name("microbe-hf-cache", create_if_missing=True)
HF_DIR = "/hf_cache"

# A100 is plenty for 150M/640 and ~2x cheaper than H100. Override with ESM2_GPU.
GPU = os.environ.get("ESM2_GPU", "A100")

app = modal.App("microbe-esm2-perprotein", image=image)


def _npy_name(bid: int) -> str:
    return f"{bid}.npy"


def _prot_path(bid: int) -> str:
    return os.path.join(PROT_DIR, f"{bid}.txt.gz")


def _write_proteins(bid: int, proteins: list[str]) -> None:
    with gzip.open(_prot_path(bid), "wt") as fh:
        fh.write("\n".join(proteins))


def _read_proteins(bid: int) -> list[str]:
    with gzip.open(_prot_path(bid), "rt") as fh:
        return [ln for ln in fh.read().split("\n") if ln]


# ---------------------------------------------------------------------------
# Phase A: fetch + predict (CPU, single container, controlled concurrency)
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


@app.function(
    volumes={OUT_DIR: out_volume},
    secrets=[modal.Secret.from_name("ncbi-api-key")],  # exports NCBI_API_KEY
    cpu=16,
    memory=32768,
    timeout=60 * 60 * 6,   # 6 hr — whole corpus in one narrow container
    max_containers=1,      # ONE container: keeps global NCBI concurrency ~= workers
)
def fetch_predict_all(pairs: list[tuple[int, str]], workers: int = 16,
                      checkpoint_every: int = 200) -> list[dict]:
    """Fetch FASTA + pyrodigal-predict for every pair, caching proteins to the
    Volume. Resumable: skips genomes whose proteins/<bid>.txt.gz already exists.
    Returns one manifest-ish row per genome (status, n_proteins; no path yet)."""
    import sys
    sys.path.insert(0, "/root")
    from concurrent.futures import ProcessPoolExecutor

    os.makedirs(PROT_DIR, exist_ok=True)
    out_volume.reload()
    if not os.environ.get("NCBI_API_KEY"):
        print("WARNING: NCBI_API_KEY not in container env (Modal Secret missing?)", flush=True)

    todo = [(b, a) for (b, a) in pairs if not os.path.exists(_prot_path(b))]
    print(f"Phase A: {len(pairs)-len(todo):,} already cached, {len(todo):,} to fetch "
          f"({workers} workers)", flush=True)

    rows: list[dict] = []
    n_ok = n_ff = n_pf = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (bid, acc, proteins, status) in enumerate(ex.map(_fetch_predict_one, todo), 1):
            if status == "OK":
                _write_proteins(bid, proteins)
                n_ok += 1
                rows.append({"bacdive_id": bid, "accession": acc,
                             "n_proteins": len(proteins), "status": "fetched", "path": ""})
            elif status == "FETCH_FAIL":
                n_ff += 1
                rows.append({"bacdive_id": bid, "accession": acc,
                             "n_proteins": 0, "status": "FETCH_FAIL", "path": ""})
            else:
                n_pf += 1
                rows.append({"bacdive_id": bid, "accession": acc,
                             "n_proteins": 0, "status": status, "path": ""})
            if i % checkpoint_every == 0:
                out_volume.commit()
                rate = i / max(time.time() - t0, 1e-6)
                eta = (len(todo) - i) / max(rate, 1e-6) / 60
                print(f"  [{i:,}/{len(todo):,}] ok={n_ok:,} ff={n_ff:,} pf={n_pf:,} "
                      f"rate={rate:.1f}/s eta={eta:.1f}min", flush=True)
    out_volume.commit()
    print(f"Phase A done: ok={n_ok:,} ff={n_ff:,} pf={n_pf:,} in {(time.time()-t0)/60:.1f}min",
          flush=True)
    return rows


# ---------------------------------------------------------------------------
# Phase B: embed (GPU, wide fan-out, reads cached proteins — no NCBI)
# ---------------------------------------------------------------------------


@app.cls(
    gpu=GPU,
    volumes={OUT_DIR: out_volume, HF_DIR: hf_volume},
    cpu=8,
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
        os.environ["HF_HOME"] = HF_DIR
        import sys
        sys.path.insert(0, "/root")
        import torch
        from microbe_model.features.embeddings import load_esm2
        t0 = time.time()
        self.tokenizer, self.model, self.device = load_esm2(self.model_name, device=torch.device("cuda"))
        self.embed_dim = int(self.model.config.hidden_size)
        print(f"loaded {self.model_name} (dim={self.embed_dim}) in {time.time()-t0:.0f}s", flush=True)

    @modal.method()
    def embed_batch(self, pairs: list[tuple[int, str]]) -> list[dict]:
        """Read each genome's cached proteins from the Volume and embed its full
        proteome to <bid>.npy. Returns one manifest row per genome."""
        import sys
        sys.path.insert(0, "/root")
        import numpy as np
        from microbe_model.features.embeddings import embed_proteins

        out_volume.reload()
        t0 = time.time()
        rows: list[dict] = []
        n_ok = n_skip = n_miss = n_ef = 0
        for bid, acc in pairs:
            fname = _npy_name(bid)
            fpath = os.path.join(OUT_DIR, fname)
            if os.path.exists(fpath):
                n_skip += 1
                rows.append({"bacdive_id": bid, "accession": acc,
                             "n_proteins": int(np.load(fpath).shape[0]),
                             "status": "ok", "path": fname})
                continue
            if not os.path.exists(_prot_path(bid)):
                n_miss += 1   # not fetched (Phase A failure) — nothing to embed
                continue
            try:
                proteins = _read_proteins(bid)
                # [n_proteins, embed_dim] — residue-mean-pooled per protein, NOT pooled across proteins.
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
        tot = sum(r["n_proteins"] for r in rows if r["status"] == "ok")
        print(f"  batch: ok={n_ok} skip={n_skip} miss={n_miss} ef={n_ef} "
              f"proteins={tot:,} in {time.time()-t0:.0f}s", flush=True)
        return rows


# ---------------------------------------------------------------------------
# Volume helpers (run remotely so they see the Volume)
# ---------------------------------------------------------------------------


@app.function(volumes={OUT_DIR: out_volume})
def scan_done() -> list[int]:
    """bacdive_ids whose .npy already exists and is non-empty (Phase B resume)."""
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
def scan_fetched() -> list[int]:
    """bacdive_ids whose proteins/<bid>.txt.gz exists (Phase A output)."""
    out_volume.reload()
    if not os.path.isdir(PROT_DIR):
        return []
    out = []
    for name in os.listdir(PROT_DIR):
        if name.endswith(".txt.gz"):
            try:
                out.append(int(name[:-7]))
            except ValueError:
                continue
    return out


@app.function(volumes={OUT_DIR: out_volume})
def write_manifest(rows: list[dict]) -> int:
    """Merge rows into manifest.parquet (keyed by bacdive_id, keep last) and persist."""
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
# Driver
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(limit: int = 0, batch_size: int = 20,
         model: str = "facebook/esm2_t30_150M_UR50D", prot_batch_size: int = 32,
         skip_fetch: bool = False, skip_embed: bool = False, fetch_workers: int = 16):
    """Two-phase driver. Phase A caches proteins (narrow CPU), Phase B embeds
    (wide GPU). Both resumable from the Volume."""
    import pandas as pd

    df = pd.read_csv(ROOT / "data" / "genome_accessions.tsv", sep="\t")
    if limit:
        df = df.head(limit)
    pairs = [(int(r.bacdive_id), str(r.accession)) for r in df.itertuples()]
    print(f"corpus: {len(pairs):,} genomes  model={model}  gpu={GPU}", flush=True)

    # ---- Phase A: fetch + predict ----
    if not skip_fetch:
        fetched = set(scan_fetched.remote())
        to_fetch = [(b, a) for (b, a) in pairs if b not in fetched]
        print(f"Phase A: {len(fetched):,} cached, {len(to_fetch):,} to fetch", flush=True)
        if to_fetch:
            rows = fetch_predict_all.remote(to_fetch, workers=fetch_workers)
            n = write_manifest.remote(rows)
            print(f"Phase A manifest rows: {n:,}", flush=True)

    if skip_embed:
        print("skip-embed set; stopping after Phase A.", flush=True)
        return

    # ---- Phase B: embed cached proteins ----
    fetched = set(scan_fetched.remote())
    embedded = set(scan_done.remote())
    to_embed = [(b, a) for (b, a) in pairs if b in fetched and b not in embedded]
    print(f"Phase B: {len(embedded):,} embedded, {len(fetched):,} fetched, "
          f"{len(to_embed):,} to embed", flush=True)
    if not to_embed:
        print("nothing to embed.", flush=True)
        return

    batches = [to_embed[i:i + batch_size] for i in range(0, len(to_embed), batch_size)]
    print(f"dispatching {len(batches)} batches of up to {batch_size} genomes", flush=True)

    embedder = Embedder(model_name=model, prot_batch_size=prot_batch_size)
    t0 = time.time()
    pending: list[dict] = []
    n_done = n_ok = 0
    for rows in embedder.embed_batch.map(batches, order_outputs=False, return_exceptions=True):
        if isinstance(rows, Exception):
            print(f"  [WARN] batch raised: {type(rows).__name__}: {rows}", flush=True)
            continue
        pending.extend(rows)
        n_ok += sum(1 for r in rows if r["status"] == "ok")
        n_done += 1
        if n_done % 10 == 0:
            total = write_manifest.remote(pending)
            pending = []
            dt = time.time() - t0
            rate = n_done / max(dt, 1e-6) * batch_size
            eta = (len(batches) - n_done) * dt / max(n_done, 1) / 60
            print(f"  [{n_done}/{len(batches)}] ok={n_ok:,} manifest={total:,} "
                  f"rate={rate:.1f} genomes/s eta={eta:.1f}min", flush=True)

    total = write_manifest.remote(pending)
    print(f"\ndone in {(time.time()-t0)/60:.1f}min. embedded ok={n_ok:,}  manifest={total:,}",
          flush=True)
    print('pull:  modal volume get microbe-esm2-perprotein "*.npy" ./data/esm2_perprotein',
          flush=True)
