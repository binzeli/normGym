from typing import Tuple, Dict, Optional

from src.norms.lining import (
    ONE_BEND_LINE_VARIANTS,
    EDGE_LINE_VARIANTS,
    LSHAPE_LINE_VARIANTS,
    SQUARE_LINE_VARIANTS,
    STAIRCASE_LINE_VARIANTS,
    ZIGZAG_LINE_VARIANTS,
    check_line_norm,
    is_line_norm,
)
from src.norms.occupation import (
    OCCUPATION_NORMS, check_occupation_norm, is_occupation_norm,
)
from src.norms.navigation import (
    NAVIGATION_NORMS, check_navigation_norm, is_navigation_norm,
)


def known_norms() -> set:
    """Every norm string the checker can actually dispatch on.

    A norm outside this set is silently a no-op: check_norm_violation falls
    through every branch and returns False, the queue/occupation resolvers
    never run, and the whole simulation degenerates. is_known_norm() exists so
    callers can reject that up front instead of running a broken experiment.
    """
    return {
        "straight_line",
        *ZIGZAG_LINE_VARIANTS,
        *ONE_BEND_LINE_VARIANTS,
        *STAIRCASE_LINE_VARIANTS,
        *LSHAPE_LINE_VARIANTS,
        *SQUARE_LINE_VARIANTS,
        *EDGE_LINE_VARIANTS,
        *OCCUPATION_NORMS,
        *NAVIGATION_NORMS,
    }


def is_known_norm(norm: Optional[str]) -> bool:
    """True for None (a deliberate no-norm run) or a dispatchable norm."""
    return norm is None or norm in known_norms()


def check_norm_violation(norm: Optional[str],
                         from_pos: Tuple[int, int],
                         to_pos: Tuple[int, int],
                         target_position: Tuple[int, int],
                         all_current_positions: Dict[str, Tuple[int, int]],
                         world,
                         agent_id: str,
                         agent_classes: Optional[Dict[str, str]] = None) -> bool:
    """
    Check if moving from from_pos to to_pos violates the specified norm.

    agent_classes maps agent_id -> color class. It is optional: when omitted it
    is derived from world.agents, so callers (e.g. the planner) don't have to
    thread it through.

    Returns:
        True if the move violates the norm, False otherwise
    """
    if agent_classes is None:
        agent_classes = {a.agent_id: a.agent_class for a in getattr(world, 'agents', [])}

    # Check all line norm violations
    if norm and is_line_norm(norm):
        norm_violation_flag = check_line_norm(
            to_pos,
            all_current_positions,
            world,
            agent_id,
            target_position,
            agent_classes,
            norm=norm,
        )
        return norm_violation_flag

    # Check occupation norm violation (yellow-region capacity, door, seat rules)
    elif norm and is_occupation_norm(norm):
        return check_occupation_norm(
            norm, from_pos, to_pos, all_current_positions, world, agent_id,
            agent_classes, target_position=target_position,
        )

    # Check navigation norm violation (right-of-way variants, spacing, movement)
    elif norm and is_navigation_norm(norm):
        return check_navigation_norm(norm, from_pos, to_pos, all_current_positions, world, agent_id, agent_classes, target_position=target_position)

    return False
