# Zero-shot baseline

Runs the experiment's test episodes directly, without the interactive learning
stage. Seeds, worlds, background agents and the stopping rule match
`run_experiment.py`, so zero-shot and learned results are directly comparable.

```bash
python experiment/run/zero_shot/run_zero_shot.py \
  --norm_type <norm> --ground_truth with \
  --agent_type memory_llm --provider openrouter --model openai/gpt-5.6-sol
python experiment/evaluation/eval.py \
  experiment/results/zero_shot/with_gt/memory_llm/openai_gpt-5.6-sol/cafe/<norm>/raw_runs
```

- `--ground_truth with|without`: whether an LLM subject sees the norm description.
- Every other flag is the same as `run_experiment.py`. `--config` defaults to the
  shared `experiment/run/experiment_config.json` (3 same-map + 3 new-map tests).
- `experiment_config_same_map1.json`: one same-map test per norm and no new-map
  tests, with the shared seeds and capacities. Pass it with `--config`.

