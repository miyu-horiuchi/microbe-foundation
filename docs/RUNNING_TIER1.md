# Running the tier-1 training matrix (Lambda + S3 sync)

The per-protein ESM-2 embeddings (~105 GB) already exist on S3 — **no
re-extraction needed**. This runbook provisions a Lambda GPU box, syncs the
embeddings, smoke-tests, then runs the full matrix. Budget: ~$9 S3 egress +
~45 min sync + ~$40 GPU (full 60-run matrix on 1×A100). See `COST=1` below.

Data source:
- `s3://microbe-foundation-esm2-perprotein/esm2_perprotein/` (AWS us-east-1)
- 19,592 × `<bacdive_id>.npy` `[n_proteins,640]` fp16 + `manifest.parquet`

## 0. From your laptop (one-time)

```bash
# push today's scripts so the box clones the latest (guard, COST, SMOKE):
git push origin feat/set-transformer-tier1

# sanity-check the embeddings are on S3 (needs your aws [default] creds):
aws s3 ls s3://microbe-foundation-esm2-perprotein/esm2_perprotein/ | wc -l   # ~19,593
```

## 1. Provision a GPU (laptop terminal — needs network + Lambda key)

```bash
export LAMBDA_API_KEY=secret_...
bash scripts/lambda_launch.sh list                       # see live capacity/price
GPU_KIND=gpu_1x_a100 bash scripts/lambda_launch.sh launch # prints ssh + steps
```

## 2. On the box: bootstrap + sync embeddings

```bash
ssh ubuntu@<ip>
git clone https://github.com/miyu-horiuchi/microbe-foundation && cd microbe-foundation
git checkout feat/set-transformer-tier1
bash scripts/lambda_install.sh

# AWS read key (scoped, read-only to the bucket):
aws configure                                            # paste key + region us-east-1
aws s3 sync s3://microbe-foundation-esm2-perprotein/esm2_perprotein/ \
    data/esm2_perprotein/                                # ~105 GB, ~45 min
ls data/esm2_perprotein/*.npy | wc -l                    # -> ~19,592
```

## 3. Budget check, smoke, then full matrix

```bash
COST=1 bash scripts/tier1_runs.sh        # dry-run GPU-hours/$ for the matrix
SMOKE=1 bash scripts/tier1_runs.sh       # 1 quick run (~5 min) — proves data loads + trains
bash scripts/tier1_runs.sh               # full: poolings x splits x 5 seeds + balanced + analysis
```

Outputs land in `runs/tier1/` (per-run metrics JSON + checkpoints). Aggregate
into mean ± 95% CI tables for the paper:

```bash
python3 paper/aggregate_seeds.py --runs-dir runs/tier1 --out paper/tables
```

## 4. Pull results back + tear down

```bash
# from your laptop:
rsync -avz ubuntu@<ip>:~/microbe-foundation/runs/tier1/ ./runs/tier1/
bash scripts/lambda_launch.sh terminate <instance-id>    # STOP billing
```

## Notes
- The interpreter guard in `tier1_runs.sh` auto-selects a working python; on the
  Lambda box the default (torch) `python3` is used. Set `MICROBE_PY` only if needed.
- Resumable: each `train_one` skips a config whose metrics JSON already exists, so
  a re-run continues where it left off.
- 650M stronger-encoder lever is a separate (cheaper-than-extraction) job — see
  `docs/dev-environment-notes.md` (needs the model-tagged-output fix first).
