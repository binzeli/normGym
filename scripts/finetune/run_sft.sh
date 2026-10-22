#!/bin/bash
# LoRA SFT cold start for one norm group: builds the SFT dataset, then trains.
# Needs one GPU; an 80 GB card is recommended (48 GB is likely too tight for
# bf16 training with 8k-token samples). Settings are read from the environment, e.g.
#   NORM_GROUP=region EPOCHS=3 bash scripts/finetune/run_sft.sh

set -euo pipefail

# Run from the repo root
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Norm group (see experiment/train/config.py:NORM_GROUPS): queuing | region |
# trajectory | allocation. The SFT corpus and the output are keyed
# by it.
export NORM_GROUP="${NORM_GROUP:-queuing}"

# Base model (default = config.BASE_MODEL, Qwen3-VL-4B). For Ministral 3 point
# it at the local BF16 copy and set SFT_NAME so the Qwen adapter is not touched.
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-1e-4}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
CONVERT="${CONVERT:-1}"          # 1 = (re)build the SFT dataset first
# 4B lineage dir — the 8B SFT adapter stays untouched at checkpoints/sft.
SFT_NAME="${SFT_NAME:-sft}"
SFT_OUT="${SFT_OUT:-experiment/train/checkpoints/sft/$NORM_GROUP/$SFT_NAME}"   # point checkpoints/sft/<group>/current at it to make it the RL init
# Dataset override (sft_convert.py --out/--val_out); default = config.SFT_DATASET.
DATASET="${DATASET:-experiment/train/data/$NORM_GROUP/sft_train.jsonl}"
VAL_DATASET="${VAL_DATASET:-experiment/train/data/$NORM_GROUP/sft_val.jsonl}"

# Activate virtual environment
source .venv/bin/activate

# Reduce fragmentation on long, variable-length multimodal sequences.
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=== START: $(date) ==="
echo "base=$BASE_MODEL epochs=$EPOCHS lr=$LR grad_accum=$GRAD_ACCUM convert=$CONVERT out=$SFT_OUT dataset=$DATASET"
echo "host=$(hostname)"

# Fail fast if the GPU is unusable (a broken driver otherwise makes torch fall
# back to CPU and the run OOM an hour later).
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable on this machine'; print('GPU:', torch.cuda.get_device_name(0))"

# Step 1: build the SFT dataset from the existing API-model runs (CPU-only,
# cheap; skip with CONVERT=0 once data/<group>/sft_train.jsonl exists).
if [ "$CONVERT" = "1" ]; then
    python -u experiment/train/sft_convert.py
fi

# Step 2: LoRA SFT cold start.
python -u experiment/train/sft_train.py \
    --base_model "$BASE_MODEL" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --grad_accum "$GRAD_ACCUM" \
    --dataset "$DATASET" --val_dataset "$VAL_DATASET" \
    --out "$SFT_OUT"

echo "=== END: $(date) ==="
echo "Adapter for PPO:     $SFT_OUT/adapter_final"
echo "Merged for eval:     $SFT_OUT/merged"

# Deactivate the virtual environment when done
deactivate
