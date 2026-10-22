# RL fine-tuning of Qwen3-VL-4B on norm inference

The policy plays the pipeline's existing `memory_llm` subject — same prompts,
same `{reasoning, action, belief, confidence}` JSON, same env rewards.
normGym and `experiment/run` are **not modified**; training only drives
them.

## Norm groups × versions

The 38 norms form groups (`config.py:NORM_GROUPS`): **`queuing`** (16 line
norms, split 11 train / 5 test — all runs so far), `navigation`, `occupation`
(splits still TODO). Every version is trained separately per group, and
everything on disk is keyed `<group>` first:

```
config.py                NORM_GROUPS registry; NORM_GROUP=<group> (env, default queuing) picks TRAIN_NORMS/TEST_NORMS
data/<group>/            sft_train.jsonl, sft_val.jsonl, snapshots_<set>/ (v1/v2), violation_judge/<set>/ (v3); set = train | test
checkpoints/sft/<group>/<name>/           SFT cold starts; `current` -> the one new RL runs start from
checkpoints/<version>/<group>/<RUN_NAME>/ RL adapters (see checkpoints/README.md)
```

Launch a version on another group with `NORM_GROUP=<group> bash scripts/finetune/run_v<N>_….sh` once
its split is filled in; all launchers/scripts derive their paths from it.
Nothing under `data/` or `checkpoints/` is shared across groups.

## Versions

| Version | Method | Code | Launch | Weights |
|---|---|---|---|---|
| **v1** | ViGaL single-step RLOO: independent snapshots, K samples each graded by the rule verifier, leave-one-out baseline, paper reward (1/0 + 0.1 format), no critic, no KL | [`vigal/v1_action_rloo/`](vigal/v1_action_rloo/README.md) | `bash scripts/finetune/run_v1_action_rloo.sh` | `checkpoints/v1_action_rloo/<group>/<RUN_NAME>/` |
| **v2** | same RLOO recipe as v1, reward = judged belief only: `r = r_action + belief_coef·judge(belief)`, one cheap LLM call (gpt-5.4-nano) per RLOO group rating the K beliefs on the 7-point norm rubric | [`vigal/v2_belief_rloo/`](vigal/v2_belief_rloo/README.md) (self-contained) | `bash scripts/finetune/run_v2_belief_rloo.sh` | `checkpoints/v2_belief_rloo/<group>/<RUN_NAME>/` |
| **v3** | violation judge: observe 10 oracle planners (2 norm-blind) and answer yes/no "does this move violate the norm?"; reward = the env's own verdict (1/0 + 0.1 format); reasoning trained by RLOO from the base model; independent of v1/v2 | [`vigal/v3_violation_judge/`](vigal/v3_violation_judge/README.md) | `bash scripts/finetune/run_v3_violation_judge.sh` | `checkpoints/v3_violation_judge/<group>/<RUN_NAME>/` |

Layout: one folder per **training method family**, one sub-folder per version:
`experiment/train/vigal/` holds the three ViGaL-style single-step RLOO versions
([overview](vigal/README.md)); a different method later (e.g. GRPO, PPO, DPO)
gets its own family folder beside it. Each version has its own README (paper →
normGym mapping + what deviates), one launcher `scripts/finetune/run_v<N>_<name>.sh`
whose output defaults to `checkpoints/v<N>_<name>/$NORM_GROUP/$RUN_NAME`, and
all version-agnostic code (`config`, `modeling`, `policy_client`, `rollout`,
`rollout_pool`, SFT, eval) stays at this level. Until v2 job 119037 ends, the
old paths `vigal/*.py` and `v2_belief_rloo/` are symlinks into the new layout
(`finalize_layout.sh` removes them).

## Shared infrastructure (this level)

| File | Purpose |
|---|---|
| `config.py` | `NORM_GROUPS` registry + active `NORM_GROUP` → `TRAIN_NORMS` / `TEST_NORMS`, paths, env settings, LoRA defaults |
| `modeling.py` | Qwen3-VL + LoRA loading, prompt/response encoding, log-probs, merge-and-save |
| `policy_client.py` | `RecordingClient`: the LLM-client shim the pipeline calls; HF and OpenAI-compatible (vLLM) backends |
| `rollout.py` | One RL episode = one shortened interactive stage through `run_interactive_stage` |
| `rollout_pool.py` | Multi-process episode driver + vLLM LoRA hot-swap |
| `sft_convert.py`, `sft_train.py` | SFT cold start from the API-model runs → `checkpoints/sft/<group>/<name>/` (queuing: `line5`, the 5 train norms with traces = `current`, the launchers' default `INIT_ADAPTER`) |
| `eval_checkpoint.py` | Evaluates a merged checkpoint with the unchanged `run_experiment.py --provider hf` + `eval.py` on `TRAIN_NORMS` / `TEST_NORMS` — the final test (step 4 below, after `merge_adapter.py`): plays the test norms one after another (full pipeline: interactive stage, 3 cafe games, 3 new-map games) as the `memory_llm` subject → `results_test/<group>/<NAME>/` |
| `data/<group>/` (git-ignored) | SFT JSONL, snapshot sets (`snapshots_<set>/`), judge records (`violation_judge/<set>/`); `legacy/` = superseded sets |
| `checkpoints/` (git-ignored) | see [checkpoints/README.md](checkpoints/README.md) |
| `results_test/<group>/<NAME>/` (git-ignored) | final game-play test results of `eval_checkpoint.py` (same layout as `experiment/results/`, + `eval_summary.json`, `logs/<norm>.log`) |
| `pilot/<group>/` | archived pilot runs: training logs, plots, test results, slurm logs — weights deleted ([pilot/queuing/README.md](pilot/queuing/README.md)) |
| `finalize_layout.sh` | one-off layout migration of 2026-08-28 (applied; kept for reference) |

## Order of operations

```bash
# 0. one-time: pip install -r requirements.txt; vLLM venv — see the launcher header
export NORM_GROUP=queuing                        # every step below is keyed by this
bash scripts/finetune/run_sft.sh                   # 1. SFT dataset + cold start -> checkpoints/sft/$NORM_GROUP/sft (v1/v2 only)
ln -sfn sft experiment/train/checkpoints/sft/$NORM_GROUP/current   # 2. make it the RL init
bash scripts/finetune/run_v1_action_rloo.sh    # 3. RL (v1 shown; v2/v3 likewise) -> checkpoints/v1_action_rloo/$NORM_GROUP/run1
# 4. merge + evaluate on the held-out TEST_NORMS with the unchanged eval pipeline
python experiment/train/merge_adapter.py --adapter experiment/train/checkpoints/v1_action_rloo/$NORM_GROUP/run1/adapter_best \
    --out experiment/train/checkpoints/v1_action_rloo/$NORM_GROUP/run1/merged_best
python experiment/train/eval_checkpoint.py --model_path experiment/train/checkpoints/v1_action_rloo/$NORM_GROUP/run1/merged_best \
    --norms test --output_dir $PWD/experiment/train/results_test/$NORM_GROUP/v1_run1_best
```

## Cluster notes

- `h100` partition; `gh104` (CUDA init failure 08-26), `gh106`, `gh108` are excluded in the launchers.
- Walltime 72 h. v1 on 2 GPUs ≈ 10 min/update → 300 updates ≈ 55 h + evals; `RESUME=1 START_UPDATE=N` continues from `adapter_latest`.
- Slurm freezes the batch script at submit time, but the trainer's worker
  processes re-import `config.py` / `vigal/*.py` at every re-harvest and
  episode eval — do not edit those while a job is running.
- Never share an output dir or a rollout scratch dir between two jobs.
