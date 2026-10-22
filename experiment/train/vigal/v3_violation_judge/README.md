# Version 3 — violation judge

> **Defaults changed 2026-08-28** (after run 125557 collapsed to "no"): lr 1e-5, group size 4, hard_frac 0.25 + 40-update age cap, hard pool class-balanced at pos_frac. See RUN_LOG.md.

> **v3** of the RL lineage ([../README.md](../README.md)); independent of
> v1/v2 (own harvest, task, trainer, data, checkpoints — shares only the
> version-neutral `config.py` / `modeling.py` / `policy_client.py` /
> `rollout_pool.py`). Launcher `scripts/finetune/run_v3_violation_judge.sh`;
> data `data/<group>/violation_judge/<set>/` (set = train | test); weights `checkpoints/v3_violation_judge/<group>/<RUN_NAME>/`.

**Task.** The LLM is an *observer*. A crowd of 10 `OracleAgent`s plays the
cafe — 8 follow the norm, 2 do not know it (the `oracle_without_norm`
subject + 1 no-norm background agent). Given the frame right before a move,
the agent's name and its move ("Finn is at (2, 7) and is about to move left,
which would take it to (2, 6)"), its last 5 moves, and the red-cross feedback
seen so far, the model answers `{"reasoning", "belief", "violation": yes|no}`.
Only the yes/no is rewarded: **1 if it matches the env's own verdict
(`episode_log[s+1]['norm_violations']`), else 0, +0.1 if the JSON parsed** —
the paper's reward with a rule verifier that is the environment itself.
Questions cover every move of the two norm-blind agents plus 25% of the
compliant agents' moves (so "which agents are blind" is not the cue).

**Same algorithm as v1/v2** (re-implemented in `step_rloo_judge_train.py`):
64 prompts × 4 samples at temperature 1.0, leave-one-out advantage, OpenRLHF
clipped per-token loss (clamp 3, ε 0.2), 1 epoch, 4 minibatches, lr 3e-5,
LoRA r32 on the LM projections, no KL. Differences forced by the task: no
episode eval / re-harvest (the data does not depend on the policy), batches
class-balanced (`POS_FRAC=0.4`; violations are ~2% of the harvest), init =
**base model + fresh LoRA** (ViGaL) — the launcher first runs a zero-shot
check (`ZERO_SHOT=200` records × 4 samples: parse rate, balanced accuracy,
in-group disagreement) and prints whether RLOO from base has signal.

**Frame/label timing** (verified on a rendered frame): `step_{s}.png` shows
positions *before* step s's moves and the crosses of step s−1's moves; the
move at step s is charged at step s+1. Records use frame s + label from s+1,
so the judged move's outcome is never visible. Smoke test: all positives came
from norm-blind agents, none from compliant ones.

**v3 defaults added 2026-08-27 (after run 120409 plateaued at 0.94–0.96 with
~60/64 all-correct groups; 120409 itself runs without them — every later
run has them).** (1) `HARD_FRAC=0.5`: half of each batch is drawn
from records where a sample was wrong in a recent update (FIFO pool of 4,000),
the rest class-balanced as before. (2) Repetition guard: a response that
repeats any 8-word sequence ≥ 4 times counts as unparsed (reward 0) — the
loops 120409 produced on hard states (0/389 false positives on normal
answers). (3) **Transfer episode eval** (`EPISODE_EVAL_EVERY=10`,
`EPISODE_EVAL_N=2`): the judge-trained LoRA is put in as the pipeline's
*acting* subject (its normal memory_llm prompt) for full 90-step episodes on
fixed scenarios — true env return, violations, success — to see whether
learning to judge violations makes a better actor. Logged as `[eval uN]
transfer: …`.

**Evals.** Every 5 updates, greedy on a balanced 400-record slice of held-out
episodes: `eval_acc`, `eval_bacc` (balanced), `eval_yes` (yes-rate — the
"always no" floor is bacc 0.5), `eval_parse`, per-norm accuracy;
`adapter_best` = best balanced accuracy; responses saved to
`eval_uNNNN.jsonl` (with the unrewarded `belief` for later judging).
Test norms: harvest them with `NORMS="<5 test norms>"` into their own set and
run `--zero_shot`-style eval with `INIT_ADAPTER=<adapter_best>`.

```bash
bash scripts/finetune/run_v3_violation_judge.sh      # harvest 176 episodes (~15 min CPU) -> zero-shot -> RLOO
#   SHOW_FEEDBACK=0  (no red-cross line in the prompt; frames still show older crosses)
#   COMPLIANT_FRAC / POS_FRAC / GROUP_SIZE / MAX_NEW_TOKENS as usual
```
