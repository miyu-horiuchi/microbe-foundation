# Per-Protein ESM-2 Extraction on Lambda (multi-GPU)

Runbook for `compute_esm2_perprotein_mp.py` — the un-pooled, full-proteome
extractor (one `[n_proteins, 640]` fp16 `.npy` per genome). This is ~80× the
compute of the pooled run, so a single GPU is impractical (~130 GPU-hr ≈ 5 days
on 1×H100). Use a **multi-GPU box and shard** one process per GPU.

Unlike the Modal path, each process fetches from NCBI itself — but a single box
with ≤16 fetch workers/process stays near NCBI's limit, so there's no rate-limit
collapse (the same config that got 99.8% on the pooled run).

## Instance sizing

| Instance | GPUs | Per-protein full run | Cost/hr | Est. wall |
|---|---|---|---:|---:|
| 8×H100 SXM | 8 | shard 0/8 … 7/8 | ~$20 | ~16–20 hr |
| 8×A100 80GB | 8 | shard 0/8 … 7/8 | ~$10–14 | ~24–30 hr |
| 1×H100 | 1 | `--shard 0/1` | ~$2 | ~5 days (avoid) |

**Recommendation: 8×H100.** ~$320–400 total, results in a day.

## 1. Provision + bootstrap

```bash
ssh ubuntu@<lambda-host>
git clone https://github.com/miyu-horiuchi/microbe-foundation
cd microbe-foundation
bash scripts/lambda_install.sh
export NCBI_API_KEY=<your-key>          # 3× faster fetches; see NCBI_API_KEY memo
# genome_accessions.tsv must be present (scp it up if not):
#   scp data/genome_accessions.tsv ubuntu@<host>:~/microbe-foundation/data/
```

## 2. Smoke test (cheap, proves the box)

```bash
python compute_esm2_perprotein_mp.py --model facebook/esm2_t6_8M_UR50D \
    --batch-size 4 --workers 4 --limit 5 --out-dir data/esm2_perprotein_smoke
ls data/esm2_perprotein_smoke/   # expect 5 .npy + manifest.parquet
```

## 3. Full run — one process per GPU, sharded

Launch N processes (N = #GPUs), each pinned to one GPU and owning a disjoint
1/N slice of the corpus. All write to the **same** `--out-dir`; shards never
collide (round-robin row slice + per-genome files). Resumable: re-running skips
any `<bid>.npy` already on disk.

```bash
# 8×H100 — paste as one block; logs to per-shard files
mkdir -p logs
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i nohup python compute_esm2_perprotein_mp.py \
      --model facebook/esm2_t30_150M_UR50D \
      --batch-size 64 --workers 12 \
      --shard $i/8 \
      --out-dir data/esm2_perprotein \
      > logs/shard_$i.log 2>&1 &
done
wait
```

Watch progress:
```bash
tail -f logs/shard_0.log
ls data/esm2_perprotein/*.npy | wc -l        # climbs toward ~19,600
```

Each shard maintains its own `manifest.parquet` write — after all shards finish,
rebuild a unified manifest from the files on disk:
```bash
python - <<'PY'
import glob, numpy as np, pandas as pd, os
rows=[]
for p in glob.glob("data/esm2_perprotein/*.npy"):
    bid=int(os.path.basename(p)[:-4])
    rows.append({"bacdive_id":bid,"n_proteins":int(np.load(p,mmap_mode='r').shape[0]),
                 "status":"ok","path":os.path.basename(p)})
pd.DataFrame(rows).to_parquet("data/esm2_perprotein/manifest.parquet", index=False)
print("manifest:", len(rows), "genomes")
PY
```

## 4. Transfer back (~100 GB) and train

```bash
# from your laptop
rsync -avz --progress \
    ubuntu@<lambda-host>:~/microbe-foundation/data/esm2_perprotein/ \
    ./data/esm2_perprotein/

python model.py --per-protein data/esm2_perprotein --split-level family --epochs 30 \
    --save-metrics runs/esm2-150M-attnpool-family.json --run-name esm2-150M-attnpool
python leaderboard.py
```

## If a run dies / spot preemption

Fully resumable — re-run the exact same shard loop. Each process scans
`data/esm2_perprotein/` for existing `<bid>.npy` and skips them, so it picks up
where it left off. No double work across shards.
```
