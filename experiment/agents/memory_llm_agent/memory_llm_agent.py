"""LLM agent that remembers the reasoning behind its past steps.

``LLMAgent`` is memoryless: it writes every step to the log file but is never
shown any of it again, so it decides from the current image alone — and a
penalty, the only ground-truth signal it ever gets, is gone from the prompt the
step after it appears.

``MemoryLLMAgent`` keeps the same environment behaviour and the same memory
file, and adds the memory section of the prompt that the base agent leaves
empty:

1. the last ``recent_steps`` steps (default 10), each with the reasoning that
   went with it;
2. the ``max_violation_steps`` (default 5) most recent steps it was penalised on
   from *before* that window, rendered the same way. The interactive stage gives
   every subject incarnation the same memory file, so these reach back across
   incarnations: a penalty earned under an earlier agent id stays in the prompt
   for the rest of the stage;
3. its latest norm hypothesis, once, at the end.

Everything else — action parsing, the auto-wait shortcut, the settled-on-target
override, the memory file format — is inherited unchanged, so a
memory run is directly comparable with a plain LLM run.
"""
from typing import Any, Dict, List, Optional

from experiment.agents.llm_agent import LLMAgent

from .config import MemoryConfig
from .episodic_memory import (
    StepRecord,
    render_belief,
    render_record,
    split_memory,
    step_records,
)


class MemoryLLMAgent(LLMAgent):
    """An LLMAgent prompted with the episode it has lived through."""

    MEMORY_SECTION_LABEL = "Your memory of this episode"

    def __init__(self, *args, memory_config: Optional[MemoryConfig] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_config = memory_config or MemoryConfig.from_env()

    # ------------------------------------------------------------------
    # Prompt memory section
    # ------------------------------------------------------------------

    def _render_records(self, records: List[StepRecord]) -> List[str]:
        return [
            render_record(
                record,
                self._resolve_name,
                max_reasoning_chars=self.memory_config.max_reasoning_chars,
            )
            for record in records
        ]

    def _memory_prompt_parts(
        self, previous_memory: Optional[Dict[str, Any]]
    ) -> List[str]:
        """The memory sections of the user prompt.

        ``LLMAgent`` is memoryless and returns nothing here; this agent replays
        the episode it has lived through, preceded by the belief carried over
        from the interactive stage when the pipeline supplied one.
        """
        parts: List[str] = []

        reference_memory = self._load_reference_memory()
        if reference_memory is not None:
            parts.extend([
                "Your final norm belief carried over from the interactive stage:",
                render_belief(reference_memory),
            ])

        parts.extend([
            self.MEMORY_SECTION_LABEL,
            self._format_memory(
                previous_memory,
                include_belief=self._tracks_belief(),
            ),
        ])
        return parts

    def _format_memory(
        self,
        memory: Optional[Dict[str, Any]],
        include_belief: bool = True,
    ) -> str:
        """Render the recent window and the earlier penalised steps."""
        if not isinstance(memory, dict):
            return "No previous memory."

        config = self.memory_config
        records = step_records(memory, self.agent_id)

        lines: List[str] = []
        if not records:
            lines.append("No previous steps.")
        else:
            recent, violations = split_memory(
                records, config.recent_steps, config.max_violation_steps
            )
            lines.append(
                f"You have been in this environment for {len(records)} step(s) "
                f"(steps {records[0].step}-{records[-1].step})."
            )
            if recent:
                lines.append("Your most recent steps (most recent last):")
                lines.extend(self._render_records(recent))
            if violations:
                lines.append(
                    "Earlier steps where you were penalised for violating the "
                    "social norm (most recent last):"
                )
                lines.extend(self._render_records(violations))

        if include_belief:
            lines.append(render_belief(memory))

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Memory writing
    # ------------------------------------------------------------------

    def _append_memory_step(
        self,
        output_dir: Optional[str],
        memory: Dict[str, Any],
        all_current_positions,
        reasoning: str,
        action_text: str,
        belief: str = "",
        confidence: Optional[float] = None,
    ) -> None:
        """Write the step exactly as ``LLMAgent`` does, then enrich it.

        The extra fields (the goal being pursued and its target cell) are
        additive: every reader of the memory file — the final-hypothesis pass,
        the belief judge, the violation marker — sees the format it already
        expects.
        """
        super()._append_memory_step(
            output_dir,
            memory,
            all_current_positions,
            reasoning,
            action_text,
            belief,
            confidence,
        )
        steps = memory.get("steps")
        if not isinstance(steps, list) or not steps:
            return
        entry = steps[-1]
        entry["goal_index"] = self.target_index
        entry["target"] = self._position_for_json(self.current_target)
        self._write_memory(output_dir, memory)
