"""Single source of truth for the natural-language goal descriptions shown to
LLM and human agents.

The text lives in experiment/utils/agent_goals.json (one entry per norm, mirroring
norm_descriptions.json). Each goal carries a 1-based ``step``, a ``free_choice``
flag, and a ``description`` that may contain a ``<target_goal_N>`` placeholder.

At runtime the placeholder for a *preset* goal is replaced with the agent's real
target position; a *free_choice* goal carries no placeholder and is shown
verbatim (the agent must pick a norm-compliant chair / door itself), so nothing
is substituted. Use this module instead of hard-coding goal sentences.
"""
import json
import os
from typing import List, Optional, Tuple

_GOALS_PATH = os.path.join(os.path.dirname(__file__), 'agent_goals.json')

# Cache: norm name -> entry dict. Loaded lazily on first use.
_GOALS_BY_NAME: Optional[dict] = None


def _load() -> dict:
    global _GOALS_BY_NAME
    if _GOALS_BY_NAME is None:
        with open(_GOALS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        _GOALS_BY_NAME = {entry['name']: entry for entry in data['agent_goals']}
    return _GOALS_BY_NAME


def _entry(norm: Optional[str]) -> dict:
    """The goal entry for `norm`. Norms without an entry (including no-norm agents)
    return an empty goal list, so callers degrade gracefully rather than raising."""
    return _load().get(norm) or {'name': norm, 'goals': []}


def _goal_for_step(norm: Optional[str], step: int) -> Optional[dict]:
    """The goal dict for a 1-based step, or None."""
    for g in _entry(norm)['goals']:
        if g['step'] == step:
            return g
    return None


def _render(goal: dict, position: Optional[Tuple[int, int]]) -> str:
    """Render a goal's description, substituting its position placeholder for a
    preset goal. Free-choice goals carry no placeholder and are returned as-is."""
    desc = goal['description']
    if not goal['free_choice'] and position is not None:
        desc = desc.replace(f"<target_goal_{goal['step']}>", str(tuple(position)))
    return desc


def is_free_choice_step(norm: Optional[str], target_index: int) -> bool:
    """Whether the goal at `target_index` (0-based) is a free choice (hidden cell,
    graded by norm compliance)."""
    g = _goal_for_step(norm, target_index + 1)
    return bool(g and g['free_choice'])


def current_goal_text(norm: Optional[str],
                      target_index: int,
                      position: Optional[Tuple[int, int]] = None) -> Optional[str]:
    """Natural-language description of the agent's current goal (0-based
    target_index). For a preset goal, pass `position` (the real target cell) to
    fill in the placeholder; for a free-choice goal pass None (nothing is
    substituted)."""
    g = _goal_for_step(norm, target_index + 1)
    if g is None:
        return None
    return _render(g, position)


def all_goals_text(norm: Optional[str],
                   target_positions: Optional[List[Tuple[int, int]]] = None) -> str:
    """The full ordered task — every goal for `norm`, one per line. Preset goals
    have their placeholder filled from `target_positions` (matched by step);
    free-choice goals are shown verbatim."""
    positions = list(target_positions) if target_positions else []
    lines = []
    for g in _entry(norm)['goals']:
        idx = g['step'] - 1
        pos = positions[idx] if idx < len(positions) else None
        lines.append(_render(g, pos))
    return "\n".join(lines)
