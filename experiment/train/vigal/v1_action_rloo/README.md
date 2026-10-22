# Version 1 — ViGaL single-step RLOO

> **Defaults changed 2026-08-28** (after run 115799 collapsed): lr 1e-5, group size 4, hard_frac 0.25 + 40-update hard-pool age cap, unparsed −0.1, format bonus requires a stated belief. See RUN_LOG.md.
> **2026-08-29:** `adapter_best` is now selected on held-out `eval_acc` (400 snapshots, every 5 updates; first eval always saves); the 1-episode game eval is logged only. `adapter_latest` is rewritten every update (all three versions).

> **v1** of the RL lineage (see [../README.md](../README.md) for the version
> index). Trainer: `step_rloo_train.py`; launchers:
> `scripts/finetune/run_v1_action_rloo.sh` (1 node, 2 GPUs) and
> `scripts/finetune/run_v1_action_rloo.sh`; outputs:
> `checkpoints/v1_action_rloo/<group>/<RUN_NAME>/`.

RL fine-tuning of Qwen3-VL-4B on norm inference following the recipe of
**"Play to Generalize: Learning to Reason Through Game Play"** (ViGaL,
[arXiv:2506.08011](https://arxiv.org/abs/2506.08011), code:
[yunfeixie233/ViGaL](https://github.com/yunfeixie233/ViGaL), built on
MM-EUREKA/OpenRLHF). The paper trains a 7B MLLM with RL on arcade games
(Snake, Rotation) and shows the gameplay-trained reasoning transfers to math
/ multi-discipline / spatial benchmarks.

Shares ALL rollout/model/eval infrastructure with the PPO trainer (the
modules at `experiment/train/`); only the algorithm lives here.

## Paper -> normGym mapping

| ViGaL (paper) | Here |
|---|---|
| RLOO (REINFORCE Leave-One-Out), no critic | `rloo_train.py`: `group_size` episodes per (norm, world, seed) scenario; baseline of an episode = mean return of its group siblings. No value head, no GAE. |
| No KL regularisation ("explore freely") | `kl_beta=0.0` default (flag kept — see below) |
| `<think>...</think><answer>...</answer>` + 0.1 format reward | The pipeline's `{reasoning, action, belief, confidence}` JSON: `reasoning` is the think block; the existing `format_bonus` / `parse_penalty` / `truncation_penalty` are the format reward. Prompts and pipeline untouched. |
| Accuracy reward (correct move = 1) | The env's shaped episode return (`train_return`: goals, violation/collision budgets, progress shaping) — normGym's ground-truth signal |
| Rollout temperature 1.0 | `temperature=0.7` (deliberately lowered to match the pipeline/PPO setting; the paper's 1.0 maximises intra-group diversity — raise it back if `degenerate_groups` climbs) |
| Group of N samples per game state | Group of `group_size=4` episodes per scenario (episode-level credit: all turns of an episode share its LOO advantage) |
| Clipped surrogate, multi-epoch (OpenRLHF RLOO) | PPO-style clipped loss and minibatches, minus everything critic-related |

The main deliberate deviation: ViGaL scores single decisions against a
ground-truth move, which normGym doesn't have — the verifiable signal here is
the episode outcome, so groups are whole episodes replaying one scenario
(same seed → same world and other-agent behaviour; return differences are
pure policy stochasticity).

## Norm split (2026-08-26)

The bend-only overfit setting is retired. The one setting now is the 16
queuing (line) norms of `src/norms/lining.py`, split in `config.py` into 11
`TRAIN_NORMS` and 5 `TEST_NORMS` (one held-out variant per line family,
mixed handedness: `zigzag_left2`, `bend_right2`, `square_left`,
`staircase_right`, `lshape_left`). Harvest, training, re-harvest and the
in-training evals use `TRAIN_NORMS` only; `eval_checkpoint.py` runs both
lists. Verifier smoke test on all 16 (oracle-mimicking fake policy, 60-step
episodes): every prompted state labelled, oracle move graded `good` in
744/746 decisions, 0 env violations — the 2 misses are caveat (2) below
(oracle move flagged `colliding` against current positions).

## Single-step (snapshot) RLOO — the recipe as the paper runs it

`rloo_train.py` deviates from ViGaL in one structural way: its groups are
whole episodes and every turn inherits the episode's scalar return. Run
105919 (RUN_LOG 08-24) showed where that leads — by update 21 the policy had
learned that never moving gives zero violations (return −8, best of the run)
with zero success, and nothing in the signal could say *which* move was wrong.

ViGaL never plays a continuous game during training. Its batch is 128
**independent snapshots**; the model answers each one once per sample, a
rule-based verifier grades every answer **immediately**, and RLOO's baseline
is the other samples of the *same* snapshot. normGym has the ground truth to
do exactly this — `check_norm_violation` says which moves the env would
penalise from a state, and the norm-following `OracleAgent` planner says
which move a compliant agent takes — so this directory now carries the
faithful version too:

| Piece | File |
|---|---|
| Verifier: `legal` / `violating` / `blocked` / `oracle` / `good` (oracle move + A*-tied compliant moves) per state, and `r = r_accuracy + r_format` per completion | `snapshot_verifier.py` |
| Snapshot harvest: play episodes with a policy (the SFT adapter) through vLLM, record every prompted state (system + user prompt + frame) with its labels; auto-wait and collision-forced steps skipped | `harvest_snapshots.py` -> `experiment/train/data/snapshots/` |
| Trainer: `prompts_per_update` snapshots x `group_size` completions, immediate grading, per-prompt LOO advantage, same clipped multi-epoch update as `rloo_train.py`; greedy `eval_acc` on held-out episodes every `eval_every` updates | `step_rloo_train.py` |
| Launcher (harvests first if the snapshot set is missing) | `scripts/finetune/run_v1_action_rloo.sh` / `..._2gpu.sh` |

Checked line by line against the paper (Sec. 2.2/2.3, App. A.3) and the
released OpenRLHF scripts (`examples/scripts/train_snake.sh`,
`reward_func_qwen_snake.py`, `advantage_estimator=rloo`, `PolicyLoss`):

| ViGaL | Here | Same? |
|---|---|---|
| Isolated game states, 36K per game; one dataset pass per episode | Harvested snapshots, iterated without replacement (`PromptSampler`) | yes (set size is what you harvest) |
| `rollout_batch_size 120`, `n_samples_per_prompt 6` | `prompts_per_update=64`, `group_size=4` (256 completions/update) | no — kept the lineage's batch (user choice); `PROMPTS=120 GROUP_SIZE=6 MINIBATCHES=6` restores the paper's |
| `max_epochs 1`, `train_batch_size 120` -> 6 optimizer steps | `rloo_epochs=1`, `minibatches=4` (64 samples/step) | epochs yes; step size scaled with the batch |
| r = r_accuracy (1 correct / 0 else) + r_format (0.1) | same since 2026-08-26 (`REWARD_DEFAULTS`: correct +1, everything else 0, +0.1 format). The lineage's earlier negative charges (−1 / −0.25 / −1) ranked the wrong answers and let RLOO promote `wait` (run 108906); they remain opt-in via `VIOLATION_REWARD` / `COLLISION_REWARD` / `UNPARSED_REWARD` | yes |
| RLOO baseline (sum - r_i)/(n-1), scalar per response, no std-norm | `loo_advantages`, `adv_norm=0` | yes |
| PolicyLoss: per-token ratio `exp(logp-old).clamp(max=3)`, clip 0.2, masked mean per sequence then mean | `policy_loss` (`ratio_clamp=3`, `clip_eps=0.2`) | yes |
| `init_kl_coef 0` (no KL), temperature 1.0 | `kl_beta=0`, `temperature=1.0` | yes |
| `freeze_prefix visual` | LoRA targets exist only in the LLM; trainer asserts no trainable `.visual.` params | yes |
| "Correct" = best move (closest apple, no death); Snake also asks for worst moves | `good` = oracle move + A*-tied compliant moves; no worst-move field (the pipeline's JSON schema is fixed) | close — the contrastive worst-move output is not reproducible without changing the eval prompt |
| Qwen2.5-VL-7B, full fine-tune, lr 1e-6, no SFT warm-up | Qwen3-VL-4B, LoRA r32, lr 3e-5, starts from the SFT adapter | no — deliberate (compute; JSON format already learned) |
| `generate_max_len 3000` | `max_new_tokens=1024` | no — deliberate (answer is a ~300-token JSON; run 102005 drifted into digit loops with long budgets) |
| Reasoning in `<think>` + answer tags | `reasoning` field of the pipeline's JSON | equivalent |

`wait` earns +1 only when the oracle also waits (in the queue, zone full, at
the goal), so the freeze attractor has no reward to feed on.

```bash
bash scripts/finetune/run_v1_action_rloo.sh
#   PROMPTS=64 GROUP_SIZE=4 TEMPERATURE=1.0   -> 256 completions/update (paper: 120 x 6)
#   NUM_HARVEST_EPISODES=120 HARVEST=1         -> (re)harvest with INIT_ADAPTER
#   SNAPSHOTS=experiment/train/data/snapshots  -> reuse an existing set
#   EVAL_EVERY=5 SAVE_EVERY=0
```

Two evals run during training (all in `training_log.jsonl`):
- `eval_acc` every `EVAL_EVERY=5` updates — greedy per-decision accuracy on
  held-out snapshots; verifiable, cannot be gamed by inaction, ~1 min.
- `ep_eval_*` every `EPISODE_EVAL_EVERY=10` updates — `EPISODE_EVAL_N=1` full
  episodes through the unchanged pipeline on fixed seeds with the current
  adapter (true env rewards): `ep_eval_return`, `ep_eval_violations`,
  `ep_eval_success`, `ep_eval_wait_frac`, per norm. Printed as `[eval uN]`.
  The best `ep_eval_success` (then return) is saved as `adapter_best` with
  `best_metric.json`. ~30 min per eval at 90 steps/episode (one episode is
  noisy — treat it as a sanity check, `eval_acc` as the curve).
Stop / select on those; then run the full `eval_checkpoint.py` on
`adapter_best`.

**On-policy re-harvest (default on, `REHARVEST_EVERY=20`,
`REHARVEST_EPISODES=16`).** The one structural difference from ViGaL: its
states are exogenous and cover the space by construction; ours come from a
policy, so a fixed harvest goes stale as soon as training moves the policy.
Run 108168 (bend overfit, SFT harvest only) showed it: held-out `eval_acc`
0.31 -> 0.53 by u24 while the real-game episode went from 2 to 15
violations and never waited — mid-queue "wait" states were only 9% of the
SFT harvest. So every N updates the trainer plays M episodes with
`adapter_latest` through the capture hook, labels every visited state with
the verifier (ground truth unchanged; only the state distribution follows
the policy) and adds them to the pool (`[reharvest uN]` line: states added,
the policy's accuracy on its own states, `wait_required` share, and the
game numbers of those episodes). Records + frames go to
`<RLOO_OUT>/reharvest/reharvest_uNNNN/`.

Caveats. (1) The initial snapshots are the *harvest* policy's states (memory
text included); the re-harvest above is what keeps the pool on-policy. (2) `legal` is judged against the other
agents' *current* positions, as the env does; the oracle plans against
*predicted* ones, so occasionally the oracle move itself is flagged — the
verifier then drops it from `good`. (3) Spacing / occupation-region norms are
checked pairwise by the env, not per move; the verifier's `check_norm_violation`
call is the planner's approximation there (fine for the 5 training norms,
which are line + movement norms).

Checkpoints go to `checkpoints/v1_action_rloo/<group>/<RUN_NAME>/` (`adapter_latest`, rewritten
every update); metrics to `training_log.jsonl` there.

## Watch-items (training_log.jsonl)

- `degenerate_groups` / `mean_group_spread` — the RLOO health metric (the
  analog of PPO's explained_variance). A degenerate group (all returns
  identical) contributes ZERO gradient; if all groups degenerate the update
  is a no-op. Persistent degeneracy means temperature too low, scenarios too
  easy/hard, or a collapsed policy.
- `parse_rate` collapsing — format breaking, same signal as PPO.
- **No KL anchor**: the paper drops it, but run 99152 (RUN_LOG 2026-08-20)
  showed an unanchored run here can degenerate in ~6 updates (word-salad
  reasoning, parse 1.00 -> 0.50). RLOO's zero-advantage-on-uniform-failure
  property is some protection the critic run didn't have, but if reasoning
  drifts, resubmit with `KL_BETA=0.05`.
