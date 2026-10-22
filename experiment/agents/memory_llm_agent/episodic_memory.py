"""The agent's episodic memory: one record per step it has lived through.

The on-disk format is the same JSON the plain ``LLMAgent`` writes (see
``experiment/agents/llm_agent.py``) — a ``steps`` list plus the latest
``belief`` / ``confidence`` — so the pipeline's memory readers
(``append_final_hypothesis``, ``eval.py``'s belief judge,
``mark_memory_norm_violation``) keep working unchanged. This module only adds a
*view* over that file: parsing the raw dicts into ``StepRecord``s and rendering
the selected ones back into prompt text, including the reasoning the agent had
at that step. The plain ``LLMAgent`` writes the same file but never reads it
back into its prompt.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


Position = Optional[Tuple[int, int]]


def _as_position(value: Any) -> Position:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return (int(value[0]), int(value[1]))
        except (TypeError, ValueError):
            return None
    return None


@dataclass
class StepRecord:
    """One remembered step, parsed out of the memory JSON."""

    step: int
    position: Position = None
    other_positions: Dict[str, Position] = field(default_factory=dict)
    action: str = "unknown"
    reasoning: str = ""
    belief: str = ""
    confidence: Optional[float] = None
    violated: bool = False
    goal_index: Optional[int] = None
    target: Position = None

    @classmethod
    def from_entry(cls, entry: Dict[str, Any], agent_id: str) -> Optional["StepRecord"]:
        if not isinstance(entry, dict):
            return None
        step = entry.get("step")
        if not isinstance(step, int):
            return None

        others_raw = entry.get("other_agent_positions")
        others = {}
        if isinstance(others_raw, dict):
            others = {
                other_id: _as_position(pos)
                for other_id, pos in others_raw.items()
                if isinstance(other_id, str)
            }

        # ``norm_violation`` is true whenever *some* agent was marked that step;
        # only the subject's own penalties are a learning signal for it.
        # ``self_violation`` records that ownership as the agent that lived the
        # step saw it, which is what keeps a penalty visible after the subject is
        # replaced: the interactive stage gives every subject incarnation the same
        # memory file under a new agent id, so comparing ``agent_id`` against
        # ``norm_violation_agents`` would drop every earlier incarnation's penalty.
        # That comparison stays as the fallback for memory files written before
        # ``self_violation`` existed.
        self_violation = entry.get("self_violation")
        if isinstance(self_violation, bool):
            violated = self_violation
        else:
            violated = bool(entry.get("norm_violation"))
            violator_ids = entry.get("norm_violation_agents")
            if violated and isinstance(violator_ids, list):
                violated = agent_id in violator_ids

        confidence = entry.get("confidence")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            confidence = None

        goal_index = entry.get("goal_index")
        if not isinstance(goal_index, int):
            goal_index = None

        return cls(
            step=step,
            position=_as_position(entry.get("queried_agent_position")),
            other_positions=others,
            action=str(entry.get("action", "unknown")),
            reasoning=str(entry.get("reasoning", "") or ""),
            belief=str(entry.get("belief", "") or ""),
            confidence=float(confidence) if confidence is not None else None,
            violated=violated,
            goal_index=goal_index,
            target=_as_position(entry.get("target")),
        )


def step_records(memory: Optional[Dict[str, Any]], agent_id: str) -> List[StepRecord]:
    """Every step in the memory file, oldest first."""
    if not isinstance(memory, dict):
        return []
    entries = memory.get("steps")
    if not isinstance(entries, list):
        return []
    records = [StepRecord.from_entry(entry, agent_id) for entry in entries]
    return [r for r in records if r is not None]


def split_memory(
    records: Sequence[StepRecord], recent_steps: int, max_violation_steps: int
) -> Tuple[List[StepRecord], List[StepRecord]]:
    """Split the episode into (recent window, earlier penalised steps).

    The recent window is the last ``recent_steps`` steps. The penalised steps
    are the ``max_violation_steps`` most recent penalties from *before* that
    window: a penalty inside the window is already shown there in full, so
    including it again would spend the budget printing the same step twice.
    """
    recent = list(records[-recent_steps:]) if recent_steps else []
    recent_steps_seen = {record.step for record in recent}

    violations = [
        record for record in records
        if record.violated and record.step not in recent_steps_seen
    ]
    if max_violation_steps > 0:
        violations = violations[-max_violation_steps:]
    else:
        violations = []
    return recent, violations


# ---------------------------------------------------------------------------
# Rendering: the text that goes back into the prompt
# ---------------------------------------------------------------------------

def _format_position(position: Position) -> str:
    return "unknown" if position is None else f"({position[0]}, {position[1]})"


def _format_others(positions: Dict[str, Position], name_of: Callable[[str], str]) -> str:
    if not positions:
        return "none"
    return ", ".join(
        f"{name_of(other_id)} at {_format_position(pos)}"
        for other_id, pos in sorted(positions.items())
    )


def _truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def render_belief(memory: Optional[Dict[str, Any]]) -> str:
    """The latest norm hypothesis held in ``memory``, as one prompt line."""
    belief = memory.get("belief", "") if isinstance(memory, dict) else ""
    confidence = memory.get("confidence") if isinstance(memory, dict) else None
    if not isinstance(belief, str) or not belief.strip():
        return "Latest norm hypothesis (belief): none"
    conf_text = (
        f" (confidence: {confidence:.2f})"
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
        else ""
    )
    return f"Latest norm hypothesis (belief): {belief.strip()}{conf_text}"


def render_record(
    record: StepRecord,
    name_of: Callable[[str], str],
    *,
    max_reasoning_chars: int = 0,
) -> str:
    """One remembered step as prompt text, with the reasoning that went with it.

    The norm hypothesis the agent held at that step is deliberately left out:
    repeating it under every step is noise, and the latest hypothesis is shown
    once at the end of the memory section.
    """
    header = (
        f"- Step {record.step}: my position was {_format_position(record.position)}; "
        f"other agents' positions were {_format_others(record.other_positions, name_of)}; "
        f"my action was {record.action}."
    )
    if record.violated:
        header += " I had a violation mark this step."

    lines = [header]
    if record.reasoning:
        lines.append(
            f"    My reasoning then: {_truncate(record.reasoning, max_reasoning_chars)}"
        )
    return "\n".join(lines)
