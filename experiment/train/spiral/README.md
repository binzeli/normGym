# SPIRAL-style trainers (v1/v2/v3)

Method family beside `vigal/`, adapting SPIRAL (arXiv:2506.24119, "Self-Play
on Zero-Sum Games Incentivizes Reasoning") to normGym's single-player
setting. **Written 2026-09-03, not yet run.**

## Why (the collapse diagnosis)

Every vigal run (v1 115799/179655, v2 119037/179656, v3 125557) ended the
same way: entropy drains under no-KL group-relative RLOO until the samples of
a group are identical, the few informative groups drive full-size
Adam-normalized steps, one bad step lands in degenerate output — and because
the RLOO baseline is the group mean, an all-equal group has advantage exactly
0: the collapsed state is *absorbing*. SPIRAL trains 400 stable steps with no
KL either; its stability comes from three mechanisms, adapted here:

| SPIRAL mechanism | here |
|---|---|
| RAE: per-(game, role) EMA baseline, A = R − b (Eq. 2) | `common.EmaBaselines`, per norm (v1/v2) / per norm×class (v3). Cross-batch, so a collapsed policy scores below its own history and keeps a restoring gradient — the trap loses its floor. **Deviation from the paper (added after run 295207): a fresh key initializes to its first reward, not 0.** b=0 is only unbiased when mean reward ≈ 0 (their self-play property); at our −0.85 it produced ~17 updates of uniformly negative advantages — "push down everything sampled" — and an entropy explosion (0.77→3.01) that collapsed parse. First-touch init centers advantages from episode 1. |
| Self-play auto-curriculum (win rate pinned ≈ 50%, reward variance never dies) | `common.DifficultyController`: a per-norm difficulty ladder (crowd capacity for v1; hard-example share for v3) stepped to hold success in a band. |
| Optimizer discipline: lr 1e-6 constant, grad clip 1.0, temp 1.0, sparse ±1 terminal reward, batch 128 trajectories | lr 3e-6 (LoRA) / 1e-6 (v3), grad clip 1.0, temp 1.0, terminal-only rewards; batches sized to our GPUs, so the collapse guard (`halt_on_collapse`) stays as a backstop. |

`--kl_beta` restores the SFT anchor in any trainer if a run drifts anyway —
off by default, faithful to the paper (KL coefficient 0.0 in their Table 6).

## The three versions

- **v1_action/** — episode-level: full interactive-stage episodes (the
  unchanged pipeline via `rollout.py`), ONE sparse terminal reward
  {−1, 0, +1} (violation-free success / partial / loss; an episode over the
  `max_unparsed_frac` format gate loses outright — SPIRAL's "invalid move
  loses the game"). Every trained turn carries the episode advantage
  (trajectory-level credit, SPIRAL Eq. 3), so norm evidence accrues in
  context — the single-snapshot identifiability gap of the vigal trainers
  does not exist here. Each norm trains at ITS benchmark capacity from
  `experiment/run/experiment_config.json` (`norm_capacity` override,
  else the global `capacity` — 10 for every queuing norm; user decision
  2026-09-03). `--capacity_ladder 4 6 8` enables the capacity curriculum
  instead; without it the 0-tier reward + EMA baseline carry the early
  signal — see the watch-table note below.
  `adapter_best` = held-out `eval_vf_success` (fixed scenarios at the same
  benchmark capacities).
- **v2_belief/** — same engine; the reward is a **verifiable identification**:
  after the episode the subject picks the crowd's norm out of the group's
  candidate descriptions (shuffled, unnamed; `identify.py`), 1/0 against the
  known ground truth. Replaces the vigal v2 LLM judge (cost, generosity to
  vague beliefs, the length-farming channel). Action reward deliberately 0
  (the vigal v2 decision). `adapter_best` = held-out `eval_id_acc`.
- **v3_judge/** — same task/data/reward as vigal v3 (imports `judge_task`,
  reuses `data/<group>/violation_judge/`); the recipe changes: 1 sample per
  record with per-norm-per-class EMA baselines instead of K=4 RLOO groups,
  lr 1e-6, hard-fraction ladder, halt on the "no-to-everything" signature.
  `adapter_best` = held-out `eval_balanced_acc`.

Shared plumbing (`config`, `modeling`, `policy_client`, `rollout*`) is the
version-neutral layer; nothing else is imported from `vigal/` except v3's
task definition, which must stay in sync with the harvested records.

## Running

Not maintained: the launchers were retired (2026-09-11). Each trainer's
docstring (`v1_action/train.py`, `v2_belief/train.py`, `v3_judge/train.py`)
describes its arguments.

The v1 launcher is the 2-nodes × 2-GPU heterogeneous job (the only v1
launcher — user decision 2026-09-05; the single-node 2/4-GPU variants were
removed after 294693 showed 4-GPU single-node slots don't materialize):
vLLM on GPU 0 of node 0 + 3 torchrun trainer ranks, 32 episodes/update —
close to SPIRAL's big-batch regime, since vLLM's continuous batching makes
32 episodes much cheaper than 2×16.

Outputs: `checkpoints/spiral_v<N>_<name>/<group>/<RUN_NAME>/` —
`training_log.jsonl` (per-update JSON incl. per-episode records),
`adapter_latest` / `adapter_best` / `best_metric.json`, and
`spiral_state.json` (EMA baselines + curriculum levels; restored on
`RESUME=1 START_UPDATE=<n>`).

## What to watch

| signal | reading |
|---|---|
| `mean_advantage` | should hover near 0 (baseline tracking); persistently large ⇒ baseline decay too slow for the drift |
| `levels` / `curriculum_changes` | inert on a single-rung ladder (the default). With a multi-rung ladder: rungs should climb as the band is beaten; stuck at 0 with success ≈ 0 ⇒ the lowest rung is still too hard |
| `win_frac` + `mid_frac` (v1) | both ≈ 0 for many updates ⇒ even "one goal cleanly" is out of reach at the benchmark capacity and no positive signal exists — the case for enabling the ladder (`CAPACITY_LADDER="4 6 8"`). At capacity 8 the historical vf rate was already ~0 (run1 re-harvests) and the benchmark's 10 is harder still, so early learning rides on the 0-tier (≥1 clean goal / success with violations), which the EMA baseline turns into a positive advantage against a −1 history |
| `turn_parse` (v1/v2) / `parse_rate`+`yes_frac` (v3) | the guard halts after `halt_patience` bad updates — if it fires, the run is over; keep `adapter_best` |
| `entropy` (v1/v2) | healthy SFT-anchored sampling runs ~0.8; a sustained climb is the 295207 runaway (uniformly negative advantages spreading probability into noise) and `ent_halt=2.0` now halts it. A slow monotonic *fall* toward 0 is the opposite (vigal-style) drain |
| `approx_kl` | the vigal cliffs followed 0.07–0.15/update; at lr 3e-6 expect ≪ that. Creeping up ⇒ raise `--kl_beta` before extending the run |
| `format_gated_frac` | episodes losing to the parse gate; sustained > 0.2 ⇒ format is degrading long before `turn_parse` shows it |

## Costs (estimates, unmeasured)

v1/v2: 16 episodes × ~90 steps ≈ 15–25 min generation on H100 + train half
over ≤ 24 turns/episode (+1 ID turn for v2) ⇒ ~30–45 min/update, 150 updates
≈ 3 days ⇒ one 72 h window with RESUME margin. v3: generation budget equals
the vigal 64×4 ⇒ ~4–6 min/update. v2 has **zero judge spend**.
