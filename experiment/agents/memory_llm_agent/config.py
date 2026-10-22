"""Tunables for the full-memory LLM agent.

Every field can be overridden from the environment with the ``NORMGYM_MEMORY_``
prefix (e.g. ``NORMGYM_MEMORY_RECENT_STEPS=20``), so a sweep can change what the
agent remembers without touching the pipeline runners.
"""
import os
from dataclasses import dataclass, fields


@dataclass
class MemoryConfig:
    # The most recent steps, shown with the reasoning that went with each.
    recent_steps: int = 10
    # Penalised steps from before that window, most recent first. Penalties are
    # the only ground-truth signal the agent gets, so they are pinned into the
    # prompt instead of ageing out of the recent window.
    max_violation_steps: int = 5
    # Character cap on the per-step reasoning replayed in the prompt. 0 (the
    # default) keeps it in full; set a positive value to trim it if a long
    # episode stops fitting in the context window.
    max_reasoning_chars: int = 0

    def __post_init__(self):
        if self.recent_steps < 0 or self.max_violation_steps < 0:
            raise ValueError("recent_steps and max_violation_steps must be non-negative")

    @classmethod
    def from_env(cls, **overrides) -> "MemoryConfig":
        """Build a config from ``NORMGYM_MEMORY_*`` env vars, then apply overrides."""
        values = {}
        for field in fields(cls):
            raw = os.getenv(f"NORMGYM_MEMORY_{field.name.upper()}")
            if raw is None:
                continue
            if field.type is int or field.type == "int":
                values[field.name] = int(raw)
            elif field.type is float or field.type == "float":
                values[field.name] = float(raw)
            elif field.type is bool or field.type == "bool":
                values[field.name] = raw.strip().lower() in ("1", "true", "yes", "on")
            else:
                values[field.name] = raw
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)
