#!/bin/bash
# VERSION 2 — belief-rewarded RLOO (experiment/train/vigal/v2_belief_rloo/step_rloo_belief_train.py).

# vLLM venv (one-time):
#     python3 -m venv experiment/train/.venv-vllm
#     experiment/train/.venv-vllm/bin/pip install vllm

# Runs on one machine with 2+ GPUs: the first visible GPU serves vLLM and every
# other one is a torchrun trainer rank (choose them with CUDA_VISIBLE_DEVICES).
# Every setting below can be overridden from the environment, e.g.
#   NORM_GROUP=region RUN_NAME=run1 bash scripts/finetune/run_v2_belief_rloo.sh

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
INIT_ADAPTER="${INIT_ADAPTER:-experiment/train/checkpoints/sft/$NORM_GROUP/current/adapter_final}"   # the group's current SFT cold start (checkpoints/sft/<group>/current -> …)
NUM_UPDATES="${NUM_UPDATES:-200}"
# PROMPTS snapshots x GROUP_SIZE completions per update, 1 epoch. (ViGaL
# train_snake.sh: 120 x 6 with 6 optimizer steps; 64 x 4 kept by choice.)
PROMPTS="${PROMPTS:-64}"
GROUP_SIZE="${GROUP_SIZE:-4}"
# Concurrent generation requests (threads; vLLM batches them).
NUM_WORKERS="${NUM_WORKERS:-64}"
TEMPERATURE="${TEMPERATURE:-1.0}"
# Space-separated norm list to train on; empty = TRAIN_NORMS of $NORM_GROUP from config.py
# (for queuing: 11 of the 16 line norms; the other 5 are TEST_NORMS, never seen in
# training). Override it from the
# environment. Harvest AND training use it.
NORMS="${NORMS:-}"
NORMS_ARG=""
[ -n "$NORMS" ] && NORMS_ARG="--norms $NORMS"
# Snapshot set (harvest_snapshots.py output). Harvest settings apply only
# when it does not exist yet (or HARVEST=1 forces a fresh one). The default
# path is keyed by the norm list so a bend-only harvest is never silently
# reused by an all-norm run (or vice versa).
# Full-memory redesign (2026-09-05): snapshots are harvested by the
# FullMemoryLLMAgent (whole episode history in the prompt, no carried belief
# line) at each norm's benchmark capacity. The default set name changed to
# train_fullmem so the old windowed-memory sets are never silently reused.
SUBJECT_KIND="${SUBJECT_KIND:-full_memory_llm}"
SNAPSHOT_SET="${SNAPSHOT_SET:-train_fullmem}"          # data/$NORM_GROUP/snapshots_<set>/ (train_fullmem = TRAIN_NORMS, full-memory prompts)
SNAPSHOTS="${SNAPSHOTS:-experiment/train/data/$NORM_GROUP/snapshots_${NORMS:-$SNAPSHOT_SET}}"
SNAPSHOTS="${SNAPSHOTS// /_}"
HARVEST="${HARVEST:-0}"
# Harvest episodes are round-robin over the norm list: 176 = 16 per train norm.
NUM_HARVEST_EPISODES="${NUM_HARVEST_EPISODES:-176}"
HARVEST_WORKERS="${HARVEST_WORKERS:-32}"
CAPACITY="${CAPACITY:-0}"                # 0 = each norm's benchmark capacity from experiment_config.json
# Cap the per-step reasoning replayed in the full-memory prompt; without it a
# late-episode prompt (up to ~90 remembered steps) can blow the context.
export NORMGYM_MEMORY_MAX_REASONING_CHARS="${NORMGYM_MEMORY_MAX_REASONING_CHARS:-250}"
# Full-memory prompts run long: skip-threshold for encoded turns (was 8192).
MAX_TURN_TOKENS="${MAX_TURN_TOKENS:-24000}"
EVAL_EVERY="${EVAL_EVERY:-5}"            # greedy held-out snapshot eval (cheap)
EPISODE_EVAL_EVERY="${EPISODE_EVAL_EVERY:-0}"    # n=1 fixed-seed episode eval: off since 2026-08-29 (re-harvest = game-play validation)
EPISODE_EVAL_N="${EPISODE_EVAL_N:-1}"
# On-policy re-harvest: every N updates add M episodes played by the current
# adapter to the snapshot pool (0 = off). See step_rloo_train.py.
REHARVEST_EVERY="${REHARVEST_EVERY:-20}"
REHARVEST_EPISODES="${REHARVEST_EPISODES:-16}"
# Change 1 (after 108444): half of every batch from on-policy states the
# policy got wrong; EXTRA_SNAPSHOTS = space-separated re-harvest dirs of an
# earlier run to seed that pool.
HARD_FRAC="${HARD_FRAC:-0.25}"
# Reward = the paper's (snapshot_verifier.REWARD_DEFAULTS): correct 1, every
# wrong answer 0, +0.1 format. The lineage's negative charges (-1 / -0.25 /
# -1) are opt-in again via these knobs.
VIOLATION_REWARD="${VIOLATION_REWARD:-0.0}"
COLLISION_REWARD="${COLLISION_REWARD:-0.0}"
UNPARSED_REWARD="${UNPARSED_REWARD:--0.1}"
# v2 reward (identification, since 2026-09-05; see v2_belief_rloo/identification.py):
# the snapshot's action question is replaced by the lettered descriptions of
# ALL training norms; r = CORRECT_REWARD * (chosen letter == true norm)
# + FORMAT_REWARD if the {"reasoning", "choice"} JSON parsed. No LLM judge.
CORRECT_REWARD="${CORRECT_REWARD:-1.0}"
FORMAT_REWARD="${FORMAT_REWARD:-0.1}"         # the paper's +0.1 format term (0 to drop)
MIN_STEP="${MIN_STEP:-50}"                   # snapshots before this env step carry too little evidence to identify the norm
CATEGORY="${CATEGORY:-}"                     # norm category wording in the identification question (queue | region | trajectory | allocation | generic);
CATEGORY_ARG=""                              # empty = the $NORM_GROUP's own tag from config.py:NORM_GROUPS — only set to override it
[ -n "$CATEGORY" ] && CATEGORY_ARG="--category $CATEGORY"
EXTRA_SNAPSHOTS="${EXTRA_SNAPSHOTS:-}"
EXTRA_ARG=""
[ -n "$EXTRA_SNAPSHOTS" ] && EXTRA_ARG="--extra_snapshots $EXTRA_SNAPSHOTS"
SAVE_EVERY="${SAVE_EVERY:-0}"
# Two runs must never share an output dir (see RUN_LOG.md 2026-08-19).
# Every v2 run lives under checkpoints/v2_belief_rloo/<RUN_NAME>.
RUN_NAME="${RUN_NAME:-run1}"                            # -> checkpoints/<version>/$NORM_GROUP/$RUN_NAME
RLOO_OUT="${RLOO_OUT:-experiment/train/checkpoints/v2_belief_rloo/$NORM_GROUP/$RUN_NAME}"
RLOO_EPOCHS="${RLOO_EPOCHS:-1}"
MINIBATCHES="${MINIBATCHES:-4}"
# ViGaL default: no KL anchor. Set KL_BETA=0.05 to restore the SFT anchor if
# the run degenerates (cf. run 99152 in RUN_LOG.md).
KL_BETA="${KL_BETA:-0.0}"
# Per-run scratch: two runs sharing one rollouts dir rmtree each other's
# episode frames mid-run.
SCRATCH="${SCRATCH:-experiment/train/rollouts/v2_belief_rloo_${RUN_ID}}"
# Run-derived port, so two runs on one machine don't collide.
PORT="${PORT:-$((20000 + $$ % 20000))}"
HOST_IP="${HOST_IP:-127.0.0.1}"
ROLLOUT_URL="http://${HOST_IP}:${PORT}/v1"
VLLM_BIN="${VLLM_BIN:-experiment/train/.venv-vllm/bin/vllm}"
LORA_NAME="${LORA_NAME:-policy}"
RESUME="${RESUME:-0}"
START_UPDATE="${START_UPDATE:-0}"
# Gradient checkpointing is REQUIRED on one rank; token batching amortizes it.
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-20000}"
GRAD_CKPT="${GRAD_CKPT:-1}"
GRAD_CKPT_ARG=""
[ "$GRAD_CKPT" = "1" ] && GRAD_CKPT_ARG="--gradient_checkpointing"

source .venv/bin/activate
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Cache-only model resolution, so a transient Hub error can't kill the run.
export HF_HUB_OFFLINE=1

if [ "$RESUME" = "1" ]; then
    INIT_ADAPTER="$RLOO_OUT/adapter_latest"
fi

echo "=== START: $(date) ==="
echo "host=$(hostname) base=$BASE_MODEL init_adapter=$INIT_ADAPTER"
echo "identification reward: correct=$CORRECT_REWARD format=$FORMAT_REWARD min_step=$MIN_STEP hard_frac=$HARD_FRAC subject_kind=$SUBJECT_KIND capacity=${CAPACITY}(0=benchmark)"
echo "updates=$NUM_UPDATES prompts=$PROMPTS group_size=$GROUP_SIZE temperature=$TEMPERATURE rloo_epochs=$RLOO_EPOCHS minibatches=$MINIBATCHES kl_beta=$KL_BETA workers=$NUM_WORKERS port=$PORT snapshots=$SNAPSHOTS norms=${NORMS:-<all train norms>}"
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

# --- vLLM server on the first GPU -------------------------------------------------
mkdir -p "$SCRATCH"
VLLM_LOG="$SCRATCH/vllm.log"
echo "[INFO] starting vLLM (log: $VLLM_LOG)"
[ -x "$VLLM_BIN" ] || { echo "[FAIL] $VLLM_BIN not found — create the vLLM venv (see this script's header)"; exit 1; }
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
    # Liveness first: a health probe can succeed against a stranger's server.
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "[FAIL] vLLM died during startup; last 40 lines:"; tail -40 "$VLLM_LOG"; exit 1
    fi
    if curl -sf "http://${HOST_IP}:${PORT}/health" > /dev/null; then
        echo "[OK] vLLM ready after ${i} check(s)"; break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "[FAIL] vLLM died during startup; last 40 lines:"; tail -40 "$VLLM_LOG"; exit 1
    fi
    sleep 10
done

# --- phase 0: harvest snapshots (CPU workers + the vLLM server) -----------
# Safe to run alongside v1 on the same set: an atomic mkdir lock elects ONE
# harvester; everyone else waits for the finished set.
HARVEST_LOCK="${SNAPSHOTS%/}.harvestlock"
if [ "$HARVEST" = "1" ]; then
    if [ -d "$SNAPSHOTS" ]; then
        # Never overwrite a set another run may be training on.
        SNAPSHOTS="${SNAPSHOTS}_${RUN_ID}"
    fi
    DO_HARVEST=1
else
    # Stale lock: the lock dir records its owner's process id. If that process is gone
    # (crashed / cancelled mid-harvest), its partial set is garbage by design ->
    # remove both and harvest ourselves.
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
        DO_HARVEST=1        # won the lock: this run harvests the shared set
        # (lock is released only after a COMPLETED harvest — a crash leaves it in
        # place; the next run sees the dead owner above and re-harvests)
    else
        DO_HARVEST=0
    fi
fi
if [ "$DO_HARVEST" = "1" ]; then
    echo "[INFO] harvesting $NUM_HARVEST_EPISODES episodes -> $SNAPSHOTS"
    python -u experiment/train/vigal/v2_belief_rloo/harvest_snapshots.py \
        --rollout_url "$ROLLOUT_URL" --model_name "$LORA_NAME" \
        --num_episodes "$NUM_HARVEST_EPISODES" --num_workers "$HARVEST_WORKERS" \
        --capacity "$CAPACITY" --subject_kind "$SUBJECT_KIND" \
        --scratch "$SCRATCH/harvest" \
        --out "$SNAPSHOTS" $NORMS_ARG
    echo "[OK] harvest done: $(wc -l < "$SNAPSHOTS/snapshots.jsonl") snapshots"
    rm -rf "$HARVEST_LOCK"
else
    # Another run may still be WRITING this set. Wait until snapshots.jsonl is
    # non-empty, its harvester's lock is gone, and the count is stable across
    # two 150s checks (~5 min quiescent). Give up after ~6h.
    prev=-1; stable=0; waited=0
    while :; do
        n=$(wc -l < "$SNAPSHOTS/snapshots.jsonl" 2>/dev/null || echo 0)
        if [ "$n" -gt 0 ] && [ "$n" -eq "$prev" ] && [ ! -d "$HARVEST_LOCK" ]; then
            stable=$((stable + 1)); [ "$stable" -ge 2 ] && break
        else
            stable=0
        fi
        if [ -d "$HARVEST_LOCK" ]; then   # harvester died while we waited -> fail fast, not after 6 h
            owner=$(cat "$HARVEST_LOCK/owner" 2>/dev/null)
            if [ -n "$owner" ] && ! kill -0 "$owner" 2>/dev/null; then
                echo "[FAIL] harvesting process $owner died mid-harvest; rerun (the next run removes the stale lock and re-harvests)."; exit 1
            fi
        fi
        waited=$((waited + 150))
        if [ "$waited" -gt 21600 ]; then
            echo "[FAIL] snapshot set never completed: $SNAPSHOTS ($n states after ${waited}s)."
            echo "       If the harvesting run crashed, delete $HARVEST_LOCK and the partial set, then rerun."
            exit 1
        fi
        echo "[INFO] waiting for snapshot set to finish growing: $SNAPSHOTS ($n states)"
        prev=$n
        sleep 150
    done
    echo "[INFO] reusing snapshots: $SNAPSHOTS ($n states)"
fi

# --- trainer on every GPU but vLLM's ------------------------------
TRAIN_ARGS="--base_model $BASE_MODEL --init_adapter $INIT_ADAPTER \
    --snapshots $SNAPSHOTS \
    --num_updates $NUM_UPDATES --prompts_per_update $PROMPTS \
    --group_size $GROUP_SIZE --temperature $TEMPERATURE \
    --rloo_epochs $RLOO_EPOCHS --minibatches $MINIBATCHES \
    --kl_beta $KL_BETA --eval_every $EVAL_EVERY --save_every $SAVE_EVERY \
    --episode_eval_every $EPISODE_EVAL_EVERY --episode_eval_n $EPISODE_EVAL_N \
    --reharvest_every $REHARVEST_EVERY --reharvest_episodes $REHARVEST_EPISODES \
    --hard_frac $HARD_FRAC $EXTRA_ARG \
    --violation_reward $VIOLATION_REWARD --collision_reward $COLLISION_REWARD \
    --unparsed_reward $UNPARSED_REWARD \
    --correct_reward $CORRECT_REWARD --format_reward $FORMAT_REWARD \
    --min_step $MIN_STEP $CATEGORY_ARG \
    --capacity $CAPACITY --subject_kind $SUBJECT_KIND \
    --max_turn_tokens $MAX_TURN_TOKENS --scratch $SCRATCH/eval \
    --start_update $START_UPDATE --rollout_url $ROLLOUT_URL \
    --num_workers $NUM_WORKERS --lora_name $LORA_NAME \
    --train_batch_tokens $TRAIN_BATCH_TOKENS $GRAD_CKPT_ARG $NORMS_ARG \
    --out $RLOO_OUT"
echo "[INFO] $NPROC trainer rank(s) on GPU(s) $TRAIN_GPUS"
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" python -u -m torch.distributed.run --standalone \
    --nproc-per-node="$NPROC" \
    experiment/train/vigal/v2_belief_rloo/step_rloo_belief_train.py $TRAIN_ARGS

echo "=== END: $(date) ==="
echo "Training log: $RLOO_OUT/training_log.jsonl"
deactivate
