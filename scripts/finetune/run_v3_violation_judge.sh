#!/bin/bash
# VERSION 3 — violation judge (experiment/train/vigal/v3_violation_judge/), fully
# independent of v1/v2: its own harvest (oracle-only episodes, no model calls),
# task, trainer, data dir and checkpoints. The first GPU serves vLLM (base model +
# LoRA hot-swap); every other GPU is a trainer rank. Phases:
#   0. harvest (CPU, once; HARVEST=1 forces a fresh set)
#   1. vLLM up; ZERO_SHOT=N: sample GROUP_SIZE answers on N held-out records
#      with the base model -> [zero-shot] line (parse / balanced acc /
#      in-group disagreement), the "does RLOO from base have signal" check
#   2. RLOO training from INIT_ADAPTER (empty = base model + fresh LoRA)
# vLLM venv (one-time): python3 -m venv experiment/train/.venv-vllm && experiment/train/.venv-vllm/bin/pip install vllm

# Runs on one machine with 2+ GPUs: the first visible GPU serves vLLM and every
# other one is a torchrun trainer rank (choose them with CUDA_VISIBLE_DEVICES).
# Every setting below can be overridden from the environment, e.g.
#   NORM_GROUP=region RUN_NAME=run1 bash scripts/finetune/run_v3_violation_judge.sh

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Norm group (see experiment/train/config.py:NORM_GROUPS): queuing | region |
# trajectory | allocation. Everything below is keyed by it:
# data/$NORM_GROUP/…, checkpoints/…/$NORM_GROUP/$RUN_NAME. The trainer and
# every helper read TRAIN_NORMS/TEST_NORMS for this group. Override it
# from the environment.
export NORM_GROUP="${NORM_GROUP:-queuing}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"   # names this run's scratch dir

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
INIT_ADAPTER="${INIT_ADAPTER:-}"                  # empty = base model + fresh LoRA (ViGaL)
RUN_NAME="${RUN_NAME:-run1}"                            # -> checkpoints/v3_violation_judge/$NORM_GROUP/$RUN_NAME
V3_OUT="${V3_OUT:-experiment/train/checkpoints/v3_violation_judge/$NORM_GROUP/$RUN_NAME}"
# Full-memory redesign (2026-09-05): records carry the queried agent's FULL
# move history + ALL red-cross feedback (HISTORY_LEN=0), harvested at each
# norm's benchmark capacity; the output schema dropped the belief field. The
# default set name changed to train_fullmem so old sets are never silently
# reused.
RECORD_SET="${RECORD_SET:-train_fullmem}"              # data/$NORM_GROUP/violation_judge/<set>/ (train_fullmem = TRAIN_NORMS; test sets are eval only)
RECORDS="${RECORDS:-experiment/train/data/$NORM_GROUP/violation_judge/${NORMS:-$RECORD_SET}}"
RECORDS="${RECORDS// /_}"
NORMS="${NORMS:-}"; NORMS_ARG=""; [ -n "$NORMS" ] && NORMS_ARG="--norms $NORMS"
HARVEST="${HARVEST:-0}"
NUM_HARVEST_EPISODES="${NUM_HARVEST_EPISODES:-176}"   # 16 per train norm
HARVEST_WORKERS="${HARVEST_WORKERS:-24}"
HARVEST_MAX_STEPS="${HARVEST_MAX_STEPS:-120}"
CAPACITY="${CAPACITY:-0}"                # 0 = each norm's benchmark capacity from experiment_config.json
HISTORY_LEN="${HISTORY_LEN:-0}"          # 0 = full move history + all feedback in the prompt (was 5 moves / 8 crosses)
CATEGORY="${CATEGORY:-}"                 # norm category wording in the observer prompt (queue | region | trajectory | allocation | generic);
CATEGORY_ARG=""                          # empty = the $NORM_GROUP's own tag from config.py:NORM_GROUPS — only set to override it
[ -n "$CATEGORY" ] && CATEGORY_ARG="--category $CATEGORY"
NUM_NO_NORM_AGENTS="${NUM_NO_NORM_AGENTS:-1}"      # + the norm-blind subject = 2 blind of 10
COMPLIANT_FRAC="${COMPLIANT_FRAC:-0.25}"
NUM_UPDATES="${NUM_UPDATES:-200}"
PROMPTS="${PROMPTS:-64}"
GROUP_SIZE="${GROUP_SIZE:-4}"
POS_FRAC="${POS_FRAC:-0.4}"
MIN_STEP="${MIN_STEP:-50}"               # drop records before this env step (too little evidence to judge; matches v2)
SHOW_FEEDBACK="${SHOW_FEEDBACK:-1}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
MAX_TURN_TOKENS="${MAX_TURN_TOKENS:-24000}"       # full-memory prompts (every agent's position per step) exceed the old 8192 default
RLOO_EPOCHS="${RLOO_EPOCHS:-1}"
MINIBATCHES="${MINIBATCHES:-4}"
KL_BETA="${KL_BETA:-0.0}"
EVAL_EVERY="${EVAL_EVERY:-5}"
HARD_FRAC="${HARD_FRAC:-0.25}"                    # quarter of each batch from recently-wrong records (class-balanced, 40-update age cap)
EPISODE_EVAL_EVERY="${EPISODE_EVAL_EVERY:-20}"    # transfer test: the judge ACTS as the pipeline subject (0 = off)
EPISODE_EVAL_N="${EPISODE_EVAL_N:-16}"                # 16 full episodes (round-robin over the train norms), like v1/v2 re-harvest
EVAL_MAX="${EVAL_MAX:-400}"
ZERO_SHOT="${ZERO_SHOT:-200}"                      # 0 = skip the check
SAVE_EVERY="${SAVE_EVERY:-0}"
NUM_WORKERS="${NUM_WORKERS:-64}"
RESUME="${RESUME:-0}"; START_UPDATE="${START_UPDATE:-}"   # empty + RESUME=1 -> derived from training_log.jsonl below
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-20000}"
GRAD_CKPT="${GRAD_CKPT:-1}"; GRAD_CKPT_ARG=""; [ "$GRAD_CKPT" = "1" ] && GRAD_CKPT_ARG="--gradient_checkpointing"
SCRATCH="${SCRATCH:-experiment/train/rollouts/v3_violation_judge_${RUN_ID}}"
PORT="${PORT:-$((20000 + $$ % 20000))}"
HOST_IP="${HOST_IP:-127.0.0.1}"
ROLLOUT_URL="http://${HOST_IP}:${PORT}/v1"
VLLM_BIN="${VLLM_BIN:-experiment/train/.venv-vllm/bin/vllm}"
LORA_NAME="${LORA_NAME:-policy}"

source .venv/bin/activate
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
if [ "$RESUME" = "1" ]; then
    INIT_ADAPTER="$V3_OUT/adapter_latest"
    if [ -z "$START_UPDATE" ]; then
        # Resume point = last logged update + 1.
        START_UPDATE=$(python - "$V3_OUT/training_log.jsonl" <<'PY'
import json, sys
last = -1
for line in open(sys.argv[1]):
    line = line.strip()
    if line:
        last = json.loads(line)['update']
print(last + 1)
PY
        ) || { echo "[FAIL] RESUME=1 but could not read $V3_OUT/training_log.jsonl"; exit 1; }
        echo "[INFO] resume: START_UPDATE derived from training_log.jsonl -> $START_UPDATE"
    fi
fi
START_UPDATE="${START_UPDATE:-0}"
INIT_ARG=""; [ -n "$INIT_ADAPTER" ] && INIT_ARG="--init_adapter $INIT_ADAPTER"

echo "=== START: $(date) ==="
echo "host=$(hostname) base=$BASE_MODEL init_adapter=${INIT_ADAPTER:-<base model + fresh LoRA>} out=$V3_OUT"
echo "records=$RECORDS harvest=$HARVEST episodes=$NUM_HARVEST_EPISODES capacity=${CAPACITY}(0=benchmark) history_len=${HISTORY_LEN}(0=full) no_norm=$NUM_NO_NORM_AGENTS(+subject) compliant_frac=$COMPLIANT_FRAC"
echo "updates=$NUM_UPDATES prompts=$PROMPTS group_size=$GROUP_SIZE pos_frac=$POS_FRAC show_feedback=$SHOW_FEEDBACK temperature=$TEMPERATURE max_new_tokens=$MAX_NEW_TOKENS zero_shot=$ZERO_SHOT port=$PORT norms=${NORMS:-<all train norms>}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# GPUs: the first visible one serves vLLM, every other one is a trainer rank.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=, read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
    mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
[ "${#GPUS[@]}" -ge 2 ] || { echo "[FAIL] need 2+ GPUs (one serves vLLM, the rest train); found ${#GPUS[@]}"; exit 1; }
VLLM_GPU="${GPUS[0]}"
TRAIN_GPUS="$(IFS=,; echo "${GPUS[*]:1}")"
NPROC=$(( ${#GPUS[@]} - 1 ))

# --- phase 0: harvest (CPU) ---------------------------------------------------
if [ "$HARVEST" = "1" ] || [ ! -f "$RECORDS/records.jsonl" ]; then
    if [ "$HARVEST" = "1" ] && [ -d "$RECORDS" ]; then RECORDS="${RECORDS}_${RUN_ID}"; fi
    echo "[INFO] harvesting $NUM_HARVEST_EPISODES oracle episodes -> $RECORDS"
    python -u experiment/train/vigal/v3_violation_judge/harvest_judge.py \
        --out "$RECORDS" --scratch "$SCRATCH/harvest" --num_episodes "$NUM_HARVEST_EPISODES" \
        --num_workers "$HARVEST_WORKERS" --capacity "$CAPACITY" --max_steps "$HARVEST_MAX_STEPS" \
        --history_len "$HISTORY_LEN" $CATEGORY_ARG \
        --num_no_norm_agents "$NUM_NO_NORM_AGENTS" --compliant_frac "$COMPLIANT_FRAC" $NORMS_ARG
    echo "[OK] harvest done: $(wc -l < "$RECORDS/records.jsonl") records"
else
    echo "[INFO] reusing records: $RECORDS ($(wc -l < "$RECORDS/records.jsonl") records)"
fi

# --- vLLM server on the first GPU (base model; the LoRA is loaded at runtime) --------
mkdir -p "$SCRATCH"
VLLM_LOG="$SCRATCH/vllm.log"
echo "[INFO] starting vLLM (log: $VLLM_LOG)"
[ -x "$VLLM_BIN" ] || { echo "[FAIL] $VLLM_BIN not found — create the vLLM venv (see this script's header)"; exit 1; }
CUDA_VISIBLE_DEVICES=$VLLM_GPU VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 \
    "$VLLM_BIN" serve "$BASE_MODEL" --host "$HOST_IP" --served-model-name base \
        --enable-lora --max-lora-rank 32 --max-model-len 32768 \
        --gpu-memory-utilization 0.90 --port "$PORT" > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
cleanup() { echo "[INFO] stopping vLLM ($VLLM_PID)"; kill $VLLM_PID 2>/dev/null || true; }
trap cleanup EXIT
echo "[INFO] waiting for vLLM to come up..."
for i in $(seq 1 180); do
    if ! kill -0 $VLLM_PID 2>/dev/null; then echo "[FAIL] vLLM died during startup; last 40 lines:"; tail -40 "$VLLM_LOG"; exit 1; fi
    if curl -sf "http://${HOST_IP}:${PORT}/health" > /dev/null; then echo "[OK] vLLM ready after ${i} check(s)"; break; fi
    sleep 10
done

COMMON_ARGS="--records $RECORDS --base_model $BASE_MODEL $INIT_ARG --out $V3_OUT \
    --rollout_url $ROLLOUT_URL --lora_name $LORA_NAME --num_workers $NUM_WORKERS \
    --prompts_per_update $PROMPTS --group_size $GROUP_SIZE --pos_frac $POS_FRAC \
    --min_step $MIN_STEP \
    --show_feedback $SHOW_FEEDBACK --temperature $TEMPERATURE --max_new_tokens $MAX_NEW_TOKENS \
    --max_turn_tokens $MAX_TURN_TOKENS \
    --rloo_epochs $RLOO_EPOCHS --minibatches $MINIBATCHES --kl_beta $KL_BETA \
    --eval_every $EVAL_EVERY --eval_max $EVAL_MAX --save_every $SAVE_EVERY \
    --hard_frac $HARD_FRAC --episode_eval_every $EPISODE_EVAL_EVERY --episode_eval_n $EPISODE_EVAL_N \
    --scratch $SCRATCH/episode_eval \
    --num_updates $NUM_UPDATES --start_update $START_UPDATE \
    --train_batch_tokens $TRAIN_BATCH_TOKENS $GRAD_CKPT_ARG $NORMS_ARG"

# --- phase 1: zero-shot signal check (base model, CPU-side client) -----------
if [ "$ZERO_SHOT" -gt 0 ]; then
    CUDA_VISIBLE_DEVICES="${GPUS[1]}" python -u experiment/train/vigal/v3_violation_judge/step_rloo_judge_train.py \
        $COMMON_ARGS --zero_shot "$ZERO_SHOT" || echo "[WARN] zero-shot check failed (continuing)"
fi

# --- phase 2: trainer on every GPU but vLLM's ------------------------------
TRAIN_ARGS="$COMMON_ARGS"
echo "[INFO] $NPROC trainer rank(s) on GPU(s) $TRAIN_GPUS"
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" python -u -m torch.distributed.run --standalone \
    --nproc-per-node="$NPROC" \
    experiment/train/vigal/v3_violation_judge/step_rloo_judge_train.py $TRAIN_ARGS

echo "=== END: $(date) ==="
echo "Training log: $V3_OUT/training_log.jsonl"
