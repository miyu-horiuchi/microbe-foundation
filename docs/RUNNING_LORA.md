# Running the encoder-adaptation experiment (LoRA fine-tune of ESM-2)

This is the **decisive lever test**: every frozen-embedding analysis showed the
family-split collapse is an out-of-distribution coverage limit that pooling,
retrieval, and self-training cannot fix. The one untested genuine lever is the
encoder. `finetune_lora.py` unfreezes ESM-2 with LoRA adapters (at the low rank
the profiler measured) and asks whether adapting the representation closes the
family-split gap a frozen encoder cannot.

It runs three modes on identical data/splits/seeds:

| mode    | ESM-2 weights | trainable |
|---------|---------------|-----------|
| `frozen`| frozen        | pooler + encoder MLP + heads (the cached-embedding baseline, computed live) |
| `lora`  | frozen        | LoRA adapters on attention projections + pooler + encoder MLP + heads |
| `full`  | trainable     | everything (expensive upper bound) |

**This is GPU-only.** Backprop through 150M/650M ESM-2 over hundreds of proteins
per genome needs real VRAM. A CPU plumbing smoke (no GPU/network/peft) is:

```bash
# generate ~120 mock genomes, then run the mock encoder end-to-end
python3 - <<'PY'
import pandas as pd, numpy as np, gzip; from pathlib import Path
tr=pd.read_parquet('data/traits.parquet'); sp=pd.read_parquet('data/splits.parquet')
m=tr[['bacdive_id']].merge(sp[['bacdive_id','family_split']],on='bacdive_id'); rng=np.random.default_rng(0)
out=Path('data/_mock_proteins'); out.mkdir(parents=True,exist_ok=True); AA=list('ACDEFGHIKLMNPQRSTVWY')
ids=[b for s in ['train','val','test'] for b in m[m.family_split==s].bacdive_id.head(40)]
for b in ids:
    seqs=[''.join(rng.choice(AA,int(rng.integers(30,120)))) for _ in range(int(rng.integers(5,20)))]
    gzip.open(out/f'{b}.txt.gz','wt').write('\n'.join(seqs))
print('wrote',len(ids))
PY
python3 finetune_lora.py --model-name mock --mode lora --smoke \
    --proteins-dir data/_mock_proteins --pooling set_transformer
```

## 0. Scoped pilot, end to end (run from YOUR laptop)

`scripts/lora_box_bootstrap.sh` is a single entrypoint that does everything on the
box: creates an **isolated venv** (so system/conda packages can't break the
`EsmModel` import -- the bug that cost us a run), runs a **fail-fast preflight**
(torch<->numpy interop + CUDA + `from transformers import EsmModel`) BEFORE any
data download, then syncs raw sequences from Modal, runs the pilot, saves results
to the Modal volume, and (optionally) self-terminates the instance.

```bash
# --- on your laptop ---
# 0) Mint a FRESH Lambda API key first (dashboard -> Settings -> API Keys).
#    The old key is dead (403) -- a stale key silently breaks auto-terminate.
export LAMBDA_API_KEY=secret_...NEW...

# 1) Launch (note the printed instance id + ip)
GPU_KIND=gpu_1x_a100_sxm4 bash scripts/lambda_launch.sh launch
export IP=<ip-from-output>
export IID=<instance-id-from-output>

# 2) Ship Modal creds so the box can pull raw protein sequences
scp ~/.modal.toml ubuntu@$IP:~/.modal.toml

# 3) Clone the repo on the box (first time only)
ssh ubuntu@$IP 'git clone https://github.com/miyu-horiuchi/microbe-foundation \
  && cd microbe-foundation && git checkout feat/set-transformer-tier1'

# 4) Run the pilot in a detached tmux. The bootstrap does venv+preflight+sync+run
#    +save+auto-terminate. Watch for "PREFLIGHT OK" in the first ~minute.
ssh ubuntu@$IP "cd microbe-foundation && git pull --ff-only && \
  tmux new-session -d -s pilot \
  'AUTO_TERMINATE=1 INSTANCE_ID=$IID LAMBDA_API_KEY=$LAMBDA_API_KEY \
   bash scripts/lora_box_bootstrap.sh 2>&1 | tee ~/pilot.log'"

# 5) Arm the hard watchdog as a cost backstop (force-terminate after 6h no matter what)
ssh ubuntu@$IP "INSTANCE_ID=$IID LAMBDA_API_KEY=$LAMBDA_API_KEY \
  nohup bash microbe-foundation/scripts/lambda_watchdog.sh 6 >/tmp/watchdog.log 2>&1 &"

# 6) Monitor (preflight first, then per-epoch training)
ssh ubuntu@$IP 'tail -f ~/pilot.log'
```

When done, the box saves `runs/lora/` (JSONs + Table 31) to the Modal volume and
terminates itself. Pull results to your laptop:

```bash
modal volume get microbe-esm2-perprotein lora_pilot ./runs/lora_pilot --force
```

Then the agent reads `runs/lora_pilot/*.json`, finalizes Table 31, and writes the
manuscript section.

## 1. Provision a GPU box

Lambda Cloud, one command (see `scripts/lambda_launch.sh`):

```bash
LAMBDA_API_KEY=... bash scripts/lambda_launch.sh launch     # default gpu_1x_a100
# for the 650M encoder or full fine-tune, prefer an 80GB card (A100/H100 80GB)
```

SSH in, clone the repo, then `bash scripts/lambda_install.sh` and
`pip install -r requirements.txt` (pulls `peft` + `accelerate`).

## 2. Get the raw protein sequences

ESM-2 needs the **raw AA sequences** (not the cached `.npy` embeddings). They are
cached on the Modal volume `microbe-esm2-perprotein` under `proteins/<bid>.txt.gz`
(one sequence per line; ORF order matches the embeddings). Sync them to
`$PROTEINS`:

```bash
mkdir -p data/esm2_proteins
modal volume get microbe-esm2-perprotein "proteins/*" data/esm2_proteins/
# (alternatively, regenerate from genomes with microbe_model/features/genome.py
#  predict_genes + NCBI fetch -- slower, needs NCBI_API_KEY)
```

`finetune_lora.py` accepts `<bid>.txt.gz`, `<bid>.txt`, or `proteins/<bid>.txt.gz`.

## 3. Run

```bash
# frozen + lora, family split, 3 seeds, 150M encoder
bash scripts/lora_runs.sh

# add the full-fine-tune upper bound, single seed
MODES="frozen lora full" SEEDS="0" bash scripts/lora_runs.sh

# stronger encoder
MODEL=facebook/esm2_t33_650M_UR50D bash scripts/lora_runs.sh

# sync results off the box + auto-shutdown
S3_DEST=s3://microbe-foundation-esm2-perprotein/lora_results/ bash scripts/lora_runs.sh
```

Knobs (env vars): `MODEL POOLING SPLIT MODES SEEDS EPOCHS BATCH MAX_PROTEINS
LORA_R LORA_ALPHA LR EXTRA OUTDIR S3_DEST`. Defaults run frozen+lora, 3 seeds,
`set_transformer` pooling, `--grad-checkpoint --balanced-families --class-weights`.

Memory levers if you OOM: lower `BATCH`, lower `MAX_PROTEINS`, lower
`--enc-microbatch` (proteins encoded per ESM-2 forward), keep `--grad-checkpoint`.

## 4. Read the result

`scripts/lora_runs.sh` calls `paper/lora_finetune_compare.py`, which writes
**Table 31** (`paper/tables/31_lora_finetune.md`) and a figure: overall
`avg_primary` and per-trait family-test scores for frozen vs LoRA vs full, plus
the **LoRA-minus-frozen delta** -- the headline number of the encoder lever.

- **If LoRA >> frozen on family-test** -> the frozen representation *was* the
  wall for novel clades; encoder adaptation is the lever, and the profiler says
  it's cheap (r~8-16).
- **If LoRA ~= frozen** -> the gap is information-limited (need more labelled
  clades), a strong negative result that closes the architecture/representation
  question.

## 5. (Optional) re-profile ranks on the unfrozen encoder

Once ESM-2 is in the graph, point the rank profiler at its attention Linears to
confirm the measured adaptation rank in situ:

```bash
python3 paper/lora_rank_profile.py --per-protein data/esm2_perprotein \
    --split-level family --pooling set_transformer \
    --include 'encoder.layer.*(query|key|value)' --batch 8 --ft-steps 300
```
