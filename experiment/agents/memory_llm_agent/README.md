# Memory LLM agent

`MemoryLLMAgent` is `LLMAgent` with a memory section in its prompt.

The plain agent is memoryless: it writes every step to the log file but is never
shown any of it again, so it decides from the current image and positions alone.
A penalty — the only ground-truth signal it ever gets — is gone from its prompt
the step after it appears. `LLMAgent._memory_prompt_parts` is the hook it leaves
empty and this agent fills in.

Everything else is inherited unchanged: action parsing, the auto-wait shortcut
when settled on a target, the settled-on-target override, exit requests, and the
memory file format. A `memory_llm` run is therefore directly comparable with an
`llm` run on the same seed.

## What goes into the prompt

1. **Recent steps** — the last `recent_steps` (default 10), each with the
   reasoning and the norm hypothesis the agent held at that step.
2. **Earlier penalised steps** — the `max_violation_steps` (default 5) most
   recent steps it was penalised on from *before* that window, rendered the same
   way. Penalties inside the window are excluded: they are already shown in
   full there, so the budget is never spent printing a step twice.
3. Its latest belief, once, at the end. In a test run the belief carried over
   from the interactive stage (`reference_memory_path`) is printed above the
   memory section; the plain agent ignores that file.

So the prompt holds at most 15 steps, and every one of them carries its
reasoning — versus nothing at all for the plain agent.

## Running it

```bash
python experiment/run/run_experiment.py --norm_type <norm> --agent_type memory_llm
```

`memory_llm` is a subject kind alongside `llm` / `rl` / the oracle baselines, so
it takes the same LLM `--provider` / `--model` flags and writes to its own
results tree (`results/memory_llm/<model>/<world>/<norm>/`).

## Configuration

Defaults live in `config.py` (`MemoryConfig`). Every field can be overridden
from the environment with the `NORMGYM_MEMORY_` prefix, so a sweep needs no code
change:

```bash
NORMGYM_MEMORY_RECENT_STEPS=20 \
NORMGYM_MEMORY_MAX_VIOLATION_STEPS=8 \
python experiment/run/run_experiment.py --norm_type <norm> --agent_type memory_llm
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `NORMGYM_MEMORY_RECENT_STEPS` | 10 | Most recent steps shown |
| `NORMGYM_MEMORY_MAX_VIOLATION_STEPS` | 5 | Earlier penalised steps shown |
| `NORMGYM_MEMORY_MAX_REASONING_CHARS` | 0 (no truncation) | Character cap per replayed reasoning line |

In code:

```python
from experiment.agents.memory_llm_agent import MemoryConfig, MemoryLLMAgent

agent = MemoryLLMAgent(..., memory_config=MemoryConfig(recent_steps=20))
```

## Memory file

The same `simulation_log/<run>.json` the plain agent writes, so
`append_final_hypothesis`, `eval.py`'s belief judge and
`mark_memory_norm_violation` all read it unchanged. Each step entry gains two
additive fields for analysis: `goal_index` (which goal the agent was pursuing)
and `target` (that goal's target cell).

## Files

| File | Contents |
| --- | --- |
| `memory_llm_agent.py` | `MemoryLLMAgent`: prompt memory section + memory writing |
| `episodic_memory.py` | Parsing the memory file, splitting it, rendering steps |
| `config.py` | `MemoryConfig` and its env-var overrides |
