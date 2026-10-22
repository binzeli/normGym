# v4 — joint action + violation-judge RLOO

One LoRA trained on **both** the v1 and the v3 objective at once. Every
update draws a mixed batch:

| half | data | prompt | output | reward |
|---|---|---|---|---|
| action (v1) | `data/<group>/snapshots_train_fullmem/` — states the SFT policy visited as the full-memory actor | the actor's own prompt (frame + full episode memory) | `{"reasoning","action","belief","confidence"}` | rule verifier on the move: good 1, else 0, unparsed −0.1, +0.1 format |
| judge (v3) | `data/<group>/violation_judge/train_fullmem/` — oracle-crowd episodes, one record per questioned move | the observer prompt (frame + memory + "Agent X is about to …") | `{"reasoning","violation"}` | env verdict on the yes/no: 1/0, +0.1 format |

Default 32 + 32 prompts × 4 samples (`--action_frac 0.5`), 200 updates.
Each sample is graded by its own verifier; the leave-one-out advantage is
computed within its own prompt's group, so an action answer is only ever
compared with action answers. The clipped policy loss then runs over all 256
samples and never sees which task a sample came from — the objectives meet
only in the shared weights. `--judge_adv_weight` rescales judge advantages
if one side needs more pull.

Everything else is what each version does on its own: the action half keeps
v1's weighted hard buckets and the on-policy re-harvest (16 episodes every
20 updates, DAgger-style aggregation); the judge half keeps v3's
class-balanced sampler (`--judge_pos_frac 0.4`, `--judge_min_step 50`) and
its wrong-sample hard pool. Init = the group's SFT adapter (needed to play
the re-harvest episodes). Held-out evals every 5 updates for both tasks:
`eval_acc` (actor, v1's names) and `eval_judge_*` + the alias
`eval_balanced_acc` (judge). `adapter_best` on `--select_metric`
(`eval_acc` by default — the final test plays the model as the actor).

## Files

| file | role |
|---|---|
| `step_rloo_joint_train.py` | the trainer (both samplers, per-task dispatch, shared RLOO loop, re-harvest, zero-shot gate) |
| `snapshot_verifier.py` | copy of v1's rule verifier (`label_state`, `grade_response`) |
| `harvest_snapshots.py` | copy of v1's snapshot harvester (imports this folder's verifier); used by the launcher's harvest and the in-training re-harvest |
| `judge_task.py` | copy of v3's observer prompt / parser / grader |
| `harvest_judge.py` | copy of v3's oracle-episode harvester (imports this folder's judge task) |

Self-contained like v3: nothing is imported from `v1_action_rloo/` or
`v3_violation_judge/`, only the version-neutral layer one level up.

## Run

```bash
export NORM_GROUP=queuing RUN_NAME=run1
bash scripts/finetune/run_v4_joint_action_judge.sh   # or _2gpu.sh
```

Phases: judge-record harvest (CPU; skipped when the set exists) → vLLM →
action-snapshot harvest (skipped when the set exists; shares v1/v2's mkdir
lock) → `ZERO_SHOT=200` judge signal check with the served SFT adapter
(`zero_shot.json`; a WEAK verdict is logged, not fatal) → training. Both
default data sets are the ones v1 and v3 train on, so a group where they
already ran needs no harvest. Output:
`checkpoints/v4_joint_action_judge/<group>/<RUN_NAME>/` (`adapter_latest`,
`adapter_best`, `best_metric.json`, `training_log.jsonl`, `zero_shot.json`,
`eval_judge_uNNNN.jsonl`, `reharvest/`). Resume: `RESUME=1` (START_UPDATE
derived from the log). Final test: merge with `experiment/train/merge_adapter.py`, then `experiment/train/eval_checkpoint.py` (see [../../README.md](../../README.md)).

Log line per update: `act: acc/viol/wait/parse/degen | judge: bacc/yes/parse/degen`
plus the shared `ent/kl/clip` and, at eval updates, `eval_acc` and `eval_bacc`.
