# Human-study metrics

`eval.py` is the human counterpart of [`experiment/evaluation/eval.py`](../../evaluation/eval.py):
same metric definitions and same aggregate labels, so a human summary can be put
side by side with an LLM summary.

Input is `experiment/human/data/<participant>/*.jsonl` — the raw event logs the
web app writes. Output is one summary per norm in `results/`:

```
results/metrics_summary_<norm>.json
```

## Run

```bash
python experiment/human/eval/eval.py                      # every participant in data/
python experiment/human/eval/eval.py --participants p1 p2
python experiment/human/eval/eval.py --judge              # + NormBeliefScore
python experiment/human/eval/eval.py --judge --judge-mid  # + belief trajectory
python experiment/human/eval/eval.py --out-dir some/dir
```

`--judge` sends each participant's final rule guess to the OpenRouter judge
(`experiment/utils/norm_judge.py`), so it needs `OPENROUTER_API_KEY`. Without it
the run still produces every behavioural metric and just warns.

## Metrics

Per run (one `*.jsonl`), identical in definition to the pipeline's:

| key | meaning |
|---|---|
| `violation_free_success` | `success * (1 - norm_violation)` — success that also kept the norm |
| `success` | mean over the participant's characters: 1.0 left through the door, 0.5 reached the second goal, 0.0 neither |
| `norm_violation` | 1 if the participant violated the norm at least once |
| `num_norm_violations` | count of participant violations |
| `collision` | 1 if the participant collided with anyone |
| `num_collisions` | count of participant collisions |
| `episode_length` | steps the participant spent on the map |
| `reward` | the participant's total reward, summed over their characters |

Per norm, the mean of each over that norm's runs — `ViolationFreeSuccessRate`,
`SuccessRate`, `ViolationRate`, `ViolationsPerEpisode`, `CollisionRate`,
`CollisionsPerEpisode`, `AverageLength`, `AverageReward` — reported separately
for the `interactive` stage, the same-world `test` run and the `new_map_test`
run, and again per participant.

With `--judge`, `NormBeliefLikert` (1–7 Likert; `NormBeliefScore` is the same rating rescaled to [0, 1]) grades the participant's final
inferred-rule guess against the ground-truth norm description; it is the human
counterpart of the LLM's interactive-stage norm belief. `--judge-mid` also
grades the mid-interactive guesses, giving a belief trajectory over steps.

## Differences from the pipeline

- **No `AverageAllReward` / `AverageRewardGap`.** Both need every agent's
  reward; the web app only logs the participant's own.
- **One run per test stage.** The pipeline does 3 same-world and 3 new-map runs
  per norm; a human session is too long for that, so each participant plays ONE
  run of each stage — one `test` and one `new_map_test` per norm, on the
  pipeline's own seeds and worlds. Which of the three they get is drawn per
  participant, so different participants contribute different settings; each run
  carries `stage_run` and `pipeline_run_id` (e.g. `new_map_test_002_library`) if
  you need to line a human run up with its llm/rl counterpart. Both stages are
  scored, over participants rather than over repeats.
- **`reward` is summed from `step_reward`, not read off `final_reward`.** In the
  interactive stage the participant is handed a fresh character after each one
  leaves, which resets the logged `cumulative_reward`; summing the step rewards
  gives the participant's total across all their characters, the analogue of the
  pipeline summing over subject instances. For a test run (a single character)
  the two agree exactly.
- **`success` is derived from goal progress**, since the human logs carry no
  `task_success` field. A character that left through the door has no target on
  its final step; one still on the map is scored by the goal it had reached.
