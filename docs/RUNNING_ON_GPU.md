# Running ESM-2 Feature Extraction on Lambda Labs

End-to-end runbook for computing `data/esm2_features.npz` on a rented Lambda GPU. Estimated total time: **3–6 hours wall clock** for ~20k genomes. Estimated cost: **$3–$10** depending on instance.

## TL;DR

```bash
# 1. SSH into your Lambda instance
ssh ubuntu@<your-lambda-host>

# 2. Bootstrap the repo + deps + accession TSV (one command)
git clone https://github.com/miyu-horiuchi/microbe-foundation
cd microbe-foundation
bash scripts/lambda_install.sh

# 3. Get a free NCBI API key (5 min) for 3x faster genome fetches
#    https://www.ncbi.nlm.nih.gov/account/settings/  (API Key Management)
export NCBI_API_KEY=<your-key>

# 4. Run the feature extraction (you'll need genome_accessions.tsv first)
#    If you ran fetch+parse locally, scp data/genome_accessions.tsv up.
#    Otherwise, regenerate by running steps 1-5 of the main pipeline.
python compute_esm2_features.py \
    --model facebook/esm2_t30_150M_UR50D \
    --sample-n 50 --batch-size 32

# 5. Transfer the result back
scp ubuntu@<lambda-host>:~/microbe-foundation/data/esm2_features.npz ./data/
```

## Instance sizing

| Instance | VRAM | ESM-2 model size | Batch | Cost/hr | Est. wall time |
|---|---|---|---|---:|---:|
| 1x A10 | 24 GB | t30 150M (640-d) | 32 | $0.75 | ~6 hr |
| 1x A100 40GB | 40 GB | t30 150M (640-d) | 64 | $1.10 | ~3 hr |
| 1x A100 80GB | 80 GB | t33 650M (1280-d) | 32 | $1.29 | ~5 hr |
| 1x H100 | 80 GB | t33 650M (1280-d) | 64 | $1.99 | ~3 hr |

**Recommendation: 1x A100 40GB with t30 150M.** Best price/performance for a benchmark baseline. Use t33 650M only when you have a paper-final run and want the strongest possible ESM-2 features.

GPU utilization will be modest (≤50%) because the NCBI genome fetch is rate-limited. The bottleneck is network I/O, not GPU compute.

## Why you want an NCBI API key

Without one, NCBI Datasets caps you at ~3 requests/sec. With a (free) API key set as `NCBI_API_KEY` env var, you get ~10 requests/sec. The microbe-model `pipeline._fetch_one_accession()` already reads this env var.

For 19,637 genomes:
- No key: ~6,500 seconds = **~1.8 hr just for fetching**
- With key: ~2,000 seconds = **~33 min**

Get one at https://www.ncbi.nlm.nih.gov/account/settings/ — under "API Key Management" → "Create an API Key". Takes 30 seconds.

## What if the run dies partway

The script is fully resumable. `data/esm2_features.npz` is rewritten every 25 genomes. If your spot instance gets preempted, just re-run the same command — it'll skip the already-embedded genomes and pick up where it left off.

```bash
# resumed run shows:
resumed: 5230 embeddings already in data/esm2_features.npz
to compute: 14407 genomes
```

## Sanity-checking before the big run

Smoke-test with 10 genomes first to confirm the model loads and ESM-2 runs:

```bash
python compute_esm2_features.py \
    --model facebook/esm2_t6_8M_UR50D \
    --sample-n 20 --batch-size 4 --limit 10
ls -la data/esm2_features.npz   # should exist, ~50 KB
```

Then delete that test file before the real run (or use a different `--out` path).

## What to transfer back

After the run completes you only need ONE file:

```bash
scp ubuntu@<lambda-host>:~/microbe-foundation/data/esm2_features.npz ./data/
```

Size: ~50 MB for 20k genomes × 640 floats. Trivial to download.

## Then on your laptop

```bash
python model.py \
    --features data/esm2_features.npz \
    --split-level family --epochs 30 \
    --save-metrics runs/esm2-150M-family.json \
    --run-name esm2-150M
python leaderboard.py
python compare_to_priors.py --our runs/esm2-150M-family.json
python paper/generate_tables.py
```

Five commands, total ~15 minutes on CPU. You'll have populated benchmark results, leaderboard, and updated paper tables.

## Cost ceiling

Rough worst case: 1x A100 40GB at $1.10/hr × 6 hours = **$6.60**. With the NCBI API key and smooth run, realistically $3–4. Cheaper than dinner.
