"""RL-TRAINING ONLY: an LLM agent prompted with its FULL episode history.

Never used by the pipeline's LLM experiments — ``MemoryLLMAgent`` (last 10
steps + pinned penalised steps + latest belief) remains the experiment agent,
and nothing in ``experiment/run`` selects this class unless
``subject_kind='full_memory_llm'`` is requested by name (the RLOO harvest
scripts do; see experiment/train/vigal/*/harvest_snapshots.py).

Differences from ``MemoryLLMAgent``, per the single-snapshot RLOO design
(2026-09-05): a snapshot is graded in isolation, so the prompt must carry
everything the model may need to infer the norm —

1. EVERY step lived so far goes into the memory section (no recent-steps
   window, no separate penalised-steps section — a penalty is already visible
   inline at its own step);
2. the "Latest norm hypothesis (belief)" line is OMITTED: the belief is what
   v1/v2 train the model to produce, so it must never appear as an input.

Everything else (response schema, action parsing, the memory file format,
``NORMGYM_MEMORY_MAX_REASONING_CHARS``) is inherited unchanged.
"""
from typing import Any, Dict, List, Optional

from experiment.agents.memory_llm_agent import MemoryLLMAgent
from experiment.agents.memory_llm_agent.episodic_memory import step_records


class FullMemoryLLMAgent(MemoryLLMAgent):
    """MemoryLLMAgent with the whole episode in the prompt and no belief line."""

    def _memory_prompt_parts(
        self, previous_memory: Optional[Dict[str, Any]]
    ) -> List[str]:
        return [
            self.MEMORY_SECTION_LABEL,
            self._format_full_memory(previous_memory),
        ]

    def _format_full_memory(self, memory: Optional[Dict[str, Any]]) -> str:
        if not isinstance(memory, dict):
            return "No previous memory."
        records = step_records(memory, self.agent_id)
        if not records:
            return "No previous steps."
        lines = [
            f"You have been in this environment for {len(records)} step(s) "
            f"(steps {records[0].step}-{records[-1].step}).",
            "All your steps so far (most recent last):",
        ]
        lines.extend(self._render_records(records))
        return "\n".join(lines)
