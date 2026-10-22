# Version 2 — belief-rewarded RLOO

> **Defaults changed 2026-08-28**: lr 1e-5, group size 4, hard_frac 0.25 + 40-update hard-pool age cap, unparsed −0.1, format bonus requires a stated belief (the belief reward composes with the actual bonus). See RUN_LOG.md.
> **2026-08-29:** truncated samples are dropped from their RLOO group (`drop_truncated=1`) instead of scoring 0, and `max_new_tokens` is 768 (was 1024) — run 119037 drifted into truncation and the 0-vs-0.1 gap became the dominant gradient ("write shorter"), drowning the belief reward. Not yet run.

> **Independence (2026-08-28):** v2 is self-contained. `step_rloo_belief_train.py`
> carries the full RLOO trainer (same recipe as v1/v3), `harvest_snapshots.py`
> and `snapshot_verifier.py` are v2's own copies; nothing is imported from
> `vigal/v1_action_rloo/` (v1). Shared, version-agnostic code stays at
> `experiment/train/` (`config`, `modeling`, `policy_client`, `rollout`,
> `rollout_pool`). `adapter_best` = the update with the best held-out
> `eval_belief` (v1's game-eval selector was dropped: run 119037 froze it at u9).

> **v2** of the RL lineage ([../README.md](../README.md)). Trainer:
> `step_rloo_belief_train.py`; judge: `belief_judge.py`; launcher:
> `scripts/finetune/run_v2_belief_rloo.sh`; outputs:
> `checkpoints/v2_belief_rloo/<group>/<RUN_NAME>/`.

v1 rewards only the `action` field of the policy's JSON. v2 keeps v1
unchanged (its trainer is imported, not copied) and replaces the reward with
one for the `belief` field — the norm hypothesis the agent states — so the
policy is trained to *say the rule* (user choice 2026-08-27: belief only;
the action is still measured by `acc` / `eval_acc` / the episode eval, but
never rewarded):

    r = action_coef * r_action + belief_coef * r_belief + r_format
        action_coef = 0 (default)   r_action = v1's verifier reward, off
        r_belief    = 1 iff the belief is rated strictly above the carried-over belief
                      and is tied-best in its group (genuine revision), else 0
        r_format    = 0.1 if the JSON parsed

## The judge — "genuine improvement" mode (default since 2026-08-27)

A live check on 8 snapshot groups showed the policy's 4 samples are
paraphrases of the belief carried over in the prompt, and a forced ranking
then breaks ties on wording (6/8 self-consistent, but the winners were
content-equal to the losers) — a binary top-tier reward on that is noise.
`GroupBeliefJudge.improvement_group(norm, beliefs, reference)` therefore
rates the **carried-over belief** (parsed from the prompt's
`Latest norm hypothesis (belief): …` line) together with the group's
distinct beliefs on the absolute 7-point rubric, `JUDGE_VOTES=2` calls
averaged, and rewards a sample with 1 **only if** it is rated strictly above
the reference and is tied-best in its group. Four restatements → all 0 → no
gradient (honest, not noise); a sample that revises the hypothesis toward the
true rule takes the whole leave-one-out advantage. The log line shows
`belief=<fraction rewarded>(lik <mean rating of revised samples> vs ref
<mean rating of the carried belief>, …)`.

**Length gate (2026-08-27, after run 117349).** Without it the policy found
the verbosity loophole: its beliefs grew 766 → 1,076 → 1,672 chars over the
u19/u39/u59 re-harvests, responses hit `max_new_tokens=1024` and a third of
its real-play turns became unparseable (u59 episode eval: tparse 0.66, 76
violations) while `eval_belief` still read 0.41. Now a belief longer than
`BELIEF_MAX_CHARS=600` (≈ the 99th percentile of the SFT beliefs: mean 349,
median 357, p90 506, max 650 chars; the true shape clause is ~120), or any truncated /
unparsed response, gets r_belief = 0 and is not sent to the judge — the
paper's format logic applied to the belief. The rubric also states that
length is not quality and that a multi-hypothesis belief must not outrank
a shorter one expressing the same rule. Log: `<gated frac> gated, <mean
belief chars>`.

## Rank mode (kept as `JUDGE_MODE=rank`)

`GroupBeliefJudge.rank_group(norm, beliefs)` sends the ground-truth norm
description and the K *distinct* beliefs of one RLOO group in **one** call
and asks for a ranking best→worst **with ties** ("candidates that express
the same rule, or are equally wrong, must share a tier"). The reward is
**binary like the paper's accuracy reward**: top tier 1, everyone else 0;
tied winners all get 1, and a group whose beliefs all land in one tier gets
all 1 → zero leave-one-out advantage, so being the least-bad wrong answer
earns nothing. Duplicated belief texts are sent once and share the reward.
`BELIEF_GRADED=1` gives the graded `(K − rank)/(K − 1)` instead.

Probe (`bend_left1_line`; correct / straight / mirror / unrelated / straight
again): ranks 1 / 3 / 2 / 5 / 3 → binary rewards 1 / 0 / 0 / 0 / 0
(graded: 1.0 / 0.5 / 0.75 / 0.0 / 0.5). Caveat: the judge does not
always tie two equally wrong beliefs (probe: two unrelated seat rules came
back 1.0 / 0.5) — a residual "least-bad" signal that the rubric asks it to
suppress but cannot guarantee.

- `score_group` (absolute 7-point rating, `JUDGE_MODE=score`) is kept; it is
  what the held-out `eval_belief` reports, since a single belief has no
  group to be ranked against.
- A failed call (network / unparseable after one retry) leaves the group
  with r_belief = 0 for every member → equal rewards → no gradient.
- With `ACTION_COEF=0` the pool is restricted to snapshots at step ≥
  `BELIEF_MIN_STEP` (earlier states have no evidence about the norm).
- Model: `openai/gpt-5.4-nano` via OpenRouter (`JUDGE_MODEL`), ~64 calls
  per update ≈ $0.02; needs `OPENROUTER_API_KEY` in the repo `.env` and
  internet on the compute node (verified 2026-08-27). Probe results and
  alternatives: docstring of `belief_judge.py`.

## Knobs (launcher env → trainer flags)

| Knob | Default | Meaning |
|---|---|---|
| `BELIEF_COEF` | 1.0 | weight of r_belief |
| `ACTION_COEF` | 0.0 | weight of v1's action reward (1.0 + `BELIEF_COEF=0` reproduces v1) |
| `JUDGE_MODE` | `improve` | `improve` (above carried belief & tied-best), `rank` (group ranking, ties), `score` (absolute 7-point) |
| `JUDGE_VOTES` | 2 | improve mode: judge calls averaged per group (~$0.04/update) |
| `BELIEF_MAX_CHARS` | 600 | beliefs longer than this (SFT p99), or truncated/unparsed responses, get 0 unjudged |
| `BELIEF_GRADED` | 0 | rank mode: 0 = binary top-tier 1/0 (paper), 1 = `(K−rank)/(K−1)` |
| `FORMAT_REWARD` | 0.1 | the paper's format term (both versions) |
| `BELIEF_MIN_STEP` | 10 | snapshots earlier in the episode carry no evidence and are not belief-graded (r_belief = 0 for the whole group) |
| `JUDGE_MODEL` | `openai/gpt-5.4-nano` | OpenRouter model id |
| `JUDGE_WORKERS` | 16 | concurrent judge calls |
| `BELIEF_EVAL_N` | 64 | held-out snapshots whose greedy belief is judged → `eval_belief` |

Everything else (snapshots, 64 × 4 batch, hard-half sampler, re-harvest,
episode eval, `adapter_best` selection) is v1's; see
[../vigal/README.md](../vigal/README.md).

## Log line

`[update N] reward=… acc=… … belief=0.412(0.78 graded, $0.43) …` — mean
r_belief over graded samples, share of samples graded, cumulative judge
cost. `training_log.jsonl` adds `mean_belief`, `mean_action_reward`,
`belief_groups_{eligible,scored,failed}`, `judge_{calls,failures,cost_usd}`,
`eval_belief`.

```bash
bash scripts/finetune/run_v2_belief_rloo.sh          # NORM_GROUP=queuing RUN_NAME=run1, from checkpoints/sft/queuing/current
INIT_ADAPTER=experiment/train/checkpoints/v1_action_rloo/queuing/line11/adapter_best RUN_NAME=line11_from_v1 \
    bash scripts/finetune/run_v2_belief_rloo.sh      # warm start from v1
```
