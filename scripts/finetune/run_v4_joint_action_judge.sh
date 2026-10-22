#!/bin/bash
# VERSION 4 — joint action + violation-judge RLOO
# (experiment/train/vigal/v4_joint_action_judge/step_rloo_joint_train.py):
# one LoRA trained on MIXED batches — ACTION_FRAC of every update's prompts are
# v1-style action snapshots (rule verifier on the chosen move), the rest are
# v3-style violation-judge records (env verdict on a yes/no). Self-contained:
# its own copies of the v1 harvester/verifier and the v3 judge task/harvester.
# Phases:
#   0. judge-record harvest (CPU, oracle episodes; only if the set is missing)
#   1. vLLM up (base model + the SFT init adapter as LoRA "policy")
#   2. action-snapshot harvest with the SFT policy (only if the set is missing;
#      mkdir-lock shared with v1/v2 on the same set)
#   3. ZERO_SHOT=N: v3's signal check for the judge task against the served
#      init adapter -> zero_shot.json (non-fatal: "can the SFT adapter judge?")
#   4. joint RLOO training
# The default data sets are the SAME train_fullmem sets v1 and v3 use, so a
# fresh group needs no harvest at all if v1 and v3 already ran there.

# Runs on one machine with 2+ GPUs: the first visible GPU serves vLLM and every
# other one is a torchrun trainer rank (choose them with CUDA_VISIBLE_DEVICES).
# Every setting below can be overridden from the environment, e.g.
#   NORM_GROUP=region RUN_NAME=run1 bash scripts/finetune/run_v4_joint_action_judge.sh

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Norm group (config.py:NORM_GROUPS): queuing | region | trajectory | allocation.
# Override it from the environment.
export NORM_GROUP="${NORM_GROUP:-queuing}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"   # names this run's scratch dir

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
INIT_ADAPTER="${INIT_ADAPTER:-experiment/train/checkpoints/sft/$NORM_GROUP/current/adapter_final}"   # SFT cold start (the actor format is needed to play re-harvest episodes)
NUM_UPDATES="${NUM_UPDATES:-200}"
PROMPTS="${PROMPTS:-64}"
GROUP_SIZE="${GROUP_SIZE:-4}"
# ---- mixing ------------------------------------------------------------------
ACTION_FRAC="${ACTION_FRAC:-0.5}"            # 32 of 64 prompts are action snapshots, 32 judge records
JUDGE_ADV_WEIGHT="${JUDGE_ADV_WEIGHT:-1.0}"  # scales judge-sample advantages (1 = weight by sample count)
SELECT_METRIC="${SELECT_METRIC:-eval_acc}"   # adapter_best on: eval_acc (actor) | eval_balanced_acc (judge) | mean
NUM_WORKERS="${NUM_WORKERS:-64}"
TEMPERATURE="${TEMPERATURE:-1.0}"
NORMS="${NORMS:-}"
NORMS_ARG=""
[ -n "$NORMS" ] && NORMS_ARG="--norms $NORMS"
# ---- action side (v1) ---------------------------------------------------------
SUBJECT_KIND="${SUBJECT_KIND:-full_memory_llm}"
SNAPSHOT_SET="${SNAPSHOT_SET:-train_fullmem}"          # data/$NORM_GROUP/snapshots_<set>/ (shared with v1/v2)
SNAPSHOTS="${SNAPSHOTS:-experiment/train/data/$NORM_GROUP/snapshots_${NORMS:-$SNAPSHOT_SET}}"
SNAPSHOTS="${SNAPSHOTS// /_}"
HARVEST="${HARVEST:-0}"                  # 1 = force a fresh action-snapshot set
NUM_HARVEST_EPISODES="${NUM_HARVEST_EPISODES:-176}"
HARVEST_WORKERS="${HARVEST_WORKERS:-32}"
CAPACITY="${CAPACITY:-0}"                # 0 = each norm's benchmark capacity
MIN_STEP="${MIN_STEP:-0}"                # action snapshots: all steps train
export NORMGYM_MEMORY_MAX_REASONING_CHARS="${NORMGYM_MEMORY_MAX_REASONING_CHARS:-250}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
REHARVEST_EVERY="${REHARVEST_EVERY:-20}"
REHARVEST_EPISODES="${REHARVEST_EPISODES:-16}"
HARD_FRAC="${HARD_FRAC:-0.25}"
VIOLATION_REWARD="${VIOLATION_REWARD:-0.0}"
COLLISION_REWARD="${COLLISION_REWARD:-0.0}"
UNPARSED_REWARD="${UNPARSED_REWARD:--0.1}"
EXTRA_SNAPSHOTS="${EXTRA_SNAPSHOTS:-}"
EXTRA_ARG=""
[ -n "$EXTRA_SNAPSHOTS" ] && EXTRA_ARG="--extra_snapshots $EXTRA_SNAPSHOTS"
# ---- judge side (v3) ----------------------------------------------------------
RECORD_SET="${RECORD_SET:-train_fullmem}"              # data/$NORM_GROUP/violation_judge/<set>/ (shared with v3)
RECORDS="${RECORDS:-experiment/train/data/$NORM_GROUP/violation_judge/${NORMS:-$RECORD_SET}}"
RECORDS="${RECORDS// /_}"
JUDGE_HARVEST="${JUDGE_HARVEST:-0}"      # 1 = force a fresh judge-record set
JUDGE_HARVEST_EPISODES="${JUDGE_HARVEST_EPISODES:-176}"
JUDGE_HARVEST_WORKERS="${JUDGE_HARVEST_WORKERS:-24}"
JUDGE_HARVEST_MAX_STEPS="${JUDGE_HARVEST_MAX_STEPS:-120}"
HISTORY_LEN="${HISTORY_LEN:-0}"
CATEGORY="${CATEGORY:-}"; CATEGORY_ARG=""; [ -n "$CATEGORY" ] && CATEGORY_ARG="--category $CATEGORY"
NUM_NO_NORM_AGENTS="${NUM_NO_NORM_AGENTS:-1}"
COMPLIANT_FRAC="${COMPLIANT_FRAC:-0.25}"
JUDGE_POS_FRAC="${JUDGE_POS_FRAC:-0.4}"
JUDGE_MIN_STEP="${JUDGE_MIN_STEP:-50}"
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-1024}"
SHOW_FEEDBACK="${SHOW_FEEDBACK:-1}"
JUDGE_HARD_FRAC="${JUDGE_HARD_FRAC:-0.25}"
JUDGE_EVAL_MAX="${JUDGE_EVAL_MAX:-400}"
ZERO_SHOT="${ZERO_SHOT:-200}"            # 0 = skip the judge signal check
# ---- shared ---------------------------------------------------------------------
MAX_TURN_TOKENS="${MAX_TURN_TOKENS:-24000}"
EVAL_EVERY="${EVAL_EVERY:-5}"
SAVE_EVERY="${SAVE_EVERY:-0}"
RUN_NAME="${RUN_NAME:-run1}"
RLOO_OUT="${RLOO_OUT:-experiment/train/checkpoints/v4_joint_action_judge/$NORM_GROUP/$RUN_NAME}"
RLOO_EPOCHS="${RLOO_EPOCHS:-1}"
MINIBATCHES="${MINIBATCHES:-4}"
KL_BETA="${KL_BETA:-0.0}"
SCRATCH="${SCRATCH:-experiment/train/rollouts/v4_joint_action_judge_${RUN_ID}}"
PORT="${PORT:-$((20000 + $$ % 20000))}"
HOST_IP="${HOST_IP:-127.0.0.1}"
ROLLOUT_URL="http://${HOST_IP}:${PORT}/v1"
VLLM_BIN="${VLLM_BIN:-experiment/train/.venv-vllm/bin/vllm}"
LORA_NAME="${LORA_NAME:-policy}"
RESUME="${RESUME:-0}"
START_UPDATE="${START_UPDATE:-}"
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-20000}"
GRAD_CKPT="${GRAD_CKPT:-1}"
GRAD_CKPT_ARG=""
[ "$GRAD_CKPT" = "1" ] && GRAD_CKPT_ARG="--gradient_checkpointing"

source .venv/bin/activate
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1

if [ "$RESUME" = "1" ]; then
    INIT_ADAPTER="$RLOO_OUT/adapter_latest"
    if [ -z "$START_UPDATE" ]; then
        START_UPDATE=$(python - "$RLOO_OUT/training_log.jsonl" <<'PY'
import json, sys
last = -1
for line in open(sys.argv[1]):
    line = line.strip()
    if line:
        last = json.loads(line)['update']
print(last + 1)
PY
        ) || { echo "[FAIL] RESUME=1 but could not read $RLOO_OUT/training_log.jsonl"; exit 1; }
        echo "[INFO] resume: START_UPDATE derived from training_log.jsonl -> $START_UPDATE"
    fi
fi
START_UPDATE="${START_UPDATE:-0}"

echo "=== START: $(date) ==="
echo "host=$(hostname) base=$BASE_MODEL init_adapter=$INIT_ADAPTER out=$RLOO_OUT"
echo "mix: action_frac=$ACTION_FRAC judge_adv_weight=$JUDGE_ADV_WEIGHT select_metric=$SELECT_METRIC"
echo "action: snapshots=$SNAPSHOTS min_step=$MIN_STEP rewards viol=$VIOLATION_REWARD coll=$COLLISION_REWARD unparsed=$UNPARSED_REWARD hard_frac=$HARD_FRAC reharvest=$REHARVEST_EPISODES/$REHARVEST_EVERY"
echo "judge: records=$RECORDS min_step=$JUDGE_MIN_STEP pos_frac=$JUDGE_POS_FRAC show_feedback=$SHOW_FEEDBACK zero_shot=$ZERO_SHOT"
echo "updates=$NUM_UPDATES prompts=$PROMPTS group_size=$GROUP_SIZE temperature=$TEMPERATURE rloo_epochs=$RLOO_EPOCHS minibatches=$MINIBATCHES kl_beta=$KL_BETA workers=$NUM_WORKERS port=$PORT norms=${NORMS:-<all train norms>}"
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

# --- phase 0: judge-record harvest (CPU, no model calls) -------------------------
if [ "$JUDGE_HARVEST" = "1" ] || [ ! -f "$RECORDS/records.jsonl" ]; then
    if [ "$JUDGE_HARVEST" = "1" ] && [ -d "$RECORDS" ]; then RECORDS="${RECORDS}_${RUN_ID}"; fi
    echo "[INFO] harvesting $JUDGE_HARVEST_EPISODES oracle episodes -> $RECORDS"
    python -u experiment/train/vigal/v4_joint_action_judge/harvest_judge.py \
        --out "$RECORDS" --scratch "$SCRATCH/judge_harvest" --num_episodes "$JUDGE_HARVEST_EPISODES" \
        --num_workers "$JUDGE_HARVEST_WORKERS" --capacity "$CAPACITY" --max_steps "$JUDGE_HARVEST_MAX_STEPS" \
        --history_len "$HISTORY_LEN" $CATEGORY_ARG \
        --num_no_norm_agents "$NUM_NO_NORM_AGENTS" --compliant_frac "$COMPLIANT_FRAC" $NORMS_ARG
    echo "[OK] judge harvest done: $(wc -l < "$RECORDS/records.jsonl") records"
else
    echo "[INFO] reusing judge records: $RECORDS ($(wc -l < "$RECORDS/records.jsonl") records)"
fi

# --- phase 1: vLLM server on the first GPU ---------------------------------------------
mkdir -p "$SCRATCH"
VLLM_LOG="$SCRATCH/vllm.log"
echo "[INFO] starting vLLM (log: $VLLM_LOG)"
[ -x "$VLLM_BIN" ] || { echo "[FAIL] $VLLM_BIN not found — create the vLLM venv"; exit 1; }
CUDA_VISIBLE_DEVICES=$VLLM_GPU VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 \
    "$VLLM_BIN" serve "$BASE_MODEL" \
        --host "$HOST_IP" \
        --served-model-name base \
        --enable-lora \
        --lora-modules "$LORA_NAME=$(realpath "$INIT_ADAPTER")" \
        --max-lora-rank 32 \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.90 \
        --port "$PORT" > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

cleanup() { echo "[INFO] stopping vLLM ($VLLM_PID)"; kill $VLLM_PID 2>/dev/null || true; }
trap cleanup EXIT

echo "[INFO] waiting for vLLM to come up..."
for i in $(seq 1 180); do
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "[FAIL] vLLM died during startup; last 40 lines:"; tail -40 "$VLLM_LOG"; exit 1
    fi
    if curl -sf "http://${HOST_IP}:${PORT}/health" > /dev/null; then
        echo "[OK] vLLM ready after ${i} check(s)"; break
    fi
    sleep 10
done

# --- phase 2: action-snapshot harvest (shared mkdir lock with v1/v2) -----------
HARVEST_LOCK="${SNAPSHOTS%/}.harvestlock"
if [ "$HARVEST" = "1" ]; then
    if [ -d "$SNAPSHOTS" ]; then
        SNAPSHOTS="${SNAPSHOTS}_${RUN_ID}"
    fi
    DO_HARVEST=1
else
    if [ -d "$HARVEST_LOCK" ]; then
        owner=$(cat "$HARVEST_LOCK/owner" 2>/dev/null)
        if [ -z "$owner" ] || ! kill -0 "$owner" 2>/dev/null; then
            case "$SNAPSHOTS" in
                experiment/train/data/*) echo "[WARN] stale harvest lock (owner process ${owner:-unknown} not running) -> removing lock and partial set $SNAPSHOTS"
                                         rm -rf "$HARVEST_LOCK" "$SNAPSHOTS" ;;
                *) echo "[FAIL] stale harvest lock but refusing to delete unexpected path $SNAPSHOTS"; exit 1 ;;
            esac
        fi
    fi
    if [ ! -f "$SNAPSHOTS/snapshots.jsonl" ] && mkdir "$HARVEST_LOCK" 2>/dev/null; then
        echo "$$" > "$HARVEST_LOCK/owner"
        DO_HARVEST=1
    else
        DO_HARVEST=0
    fi
fi
if [ "$DO_HARVEST" = "1" ]; then
    echo "[INFO] harvesting $NUM_HARVEST_EPISODES episodes -> $SNAPSHOTS"
    python -u experiment/train/vigal/v4_joint_action_judge/harvest_snapshots.py \
        --rollout_url "$ROLLOUT_URL" --model_name "$LORA_NAME" \
        --num_episodes "$NUM_HARVEST_EPISODES" --num_workers "$HARVEST_WORKERS" \
        --capacity "$CAPACITY" --subject_kind "$SUBJECT_KIND" \
        --scratch "$SCRATCH/harvest" \
        --out "$SNAPSHOTS" $NORMS_ARG
    echo "[OK] harvest done: $(wc -l < "$SNAPSHOTS/snapshots.jsonl") snapshots"
    rm -rf "$HARVEST_LOCK"
else
    prev=-1; stable=0; waited=0
    while :; do
        n=$(wc -l < "$SNAPSHOTS/snapshots.jsonl" 2>/dev/null || echo 0)
        if [ "$n" -gt 0 ] && [ "$n" -eq "$prev" ] && [ ! -d "$HARVEST_LOCK" ]; then
            stable=$((stable + 1)); [ "$stable" -ge 2 ] && break
        else
            stable=0
        fi
        if [ -d "$HARVEST_LOCK" ]; then
            owner=$(cat "$HARVEST_LOCK/owner" 2>/dev/null)
            if [ -n "$owner" ] && ! kill -0 "$owner" 2>/dev/null; then
                echo "[FAIL] harvesting process $owner died mid-harvest; rerun."; exit 1
            fi
        fi
        waited=$((waited + 150))
        if [ "$waited" -gt 21600 ]; then
            echo "[FAIL] snapshot set never completed: $SNAPSHOTS ($n states after ${waited}s)."; exit 1
        fi
        echo "[INFO] waiting for snapshot set to finish growing: $SNAPSHOTS ($n states)"
        prev=$n
        sleep 150
    done
    echo "[INFO] reusing snapshots: $SNAPSHOTS ($n states)"
fi

# --- trainer args -----------------------------------------------------------------
TRAIN_ARGS="--base_model $BASE_MODEL --init_adapter $INIT_ADAPTER \
    --snapshots $SNAPSHOTS --records $RECORDS \
    --num_updates $NUM_UPDATES --prompts_per_update $PROMPTS \
    --group_size $GROUP_SIZE --temperature $TEMPERATURE \
    --action_frac $ACTION_FRAC --judge_adv_weight $JUDGE_ADV_WEIGHT --select_metric $SELECT_METRIC \
    --rloo_epochs $RLOO_EPOCHS --minibatches $MINIBATCHES \
    --kl_beta $KL_BETA --eval_every $EVAL_EVERY --save_every $SAVE_EVERY \
    --reharvest_every $REHARVEST_EVERY --reharvest_episodes $REHARVEST_EPISODES \
    --hard_frac $HARD_FRAC $EXTRA_ARG \
    --violation_reward $VIOLATION_REWARD --collision_reward $COLLISION_REWARD \
    --unparsed_reward $UNPARSED_REWARD --max_new_tokens $MAX_NEW_TOKENS \
    --capacity $CAPACITY --subject_kind $SUBJECT_KIND --min_step $MIN_STEP \
    --judge_pos_frac $JUDGE_POS_FRAC --judge_min_step $JUDGE_MIN_STEP \
    --judge_max_new_tokens $JUDGE_MAX_NEW_TOKENS --judge_show_feedback $SHOW_FEEDBACK \
    --judge_hard_frac $JUDGE_HARD_FRAC --judge_eval_max $JUDGE_EVAL_MAX \
    --max_turn_tokens $MAX_TURN_TOKENS --scratch $SCRATCH/train \
    --start_update $START_UPDATE --rollout_url $ROLLOUT_URL \
    --num_workers $NUM_WORKERS --lora_name $LORA_NAME \
    --train_batch_tokens $TRAIN_BATCH_TOKENS $GRAD_CKPT_ARG $NORMS_ARG \
    --out $RLOO_OUT"

# --- phase 3: zero-shot judge signal check with the served init adapter ----------
# (client-side only; needs a visible GPU just to pass the trainer's CUDA guard)
if [ "$ZERO_SHOT" -gt 0 ] && [ "$START_UPDATE" = "0" ]; then
    CUDA_VISIBLE_DEVICES="${GPUS[1]}" python -u experiment/train/vigal/v4_joint_action_judge/step_rloo_joint_train.py \
        $TRAIN_ARGS --zero_shot "$ZERO_SHOT" --zero_shot_model_name "$LORA_NAME" \
        || echo "[WARN] zero-shot check failed (continuing)"
fi

# --- phase 4: trainer on every GPU but vLLM's ------------------------------
echo "[INFO] $NPROC trainer rank(s) on GPU(s) $TRAIN_GPUS"
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" python -u -m torch.distributed.run --standalone \
    --nproc-per-node="$NPROC" \
    experiment/train/vigal/v4_joint_action_judge/step_rloo_joint_train.py $TRAIN_ARGS

echo "=== END: $(date) ==="
echo "Training log: $RLOO_OUT/training_log.jsonl"
deactivate
