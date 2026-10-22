"""
Action definitions for agents in the cafe simulation.
"""

from enum import Enum


class Action(Enum):
    """Available actions for agents"""
    MOVE_UP = "move_up"
    MOVE_DOWN = "move_down"
    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right"
    WAIT = "wait"
    PUSH = "push"  # env-controlled (collision resolution); agents cannot select
    ORDER = "order"  # env-controlled (auto-triggered at counter); agents cannot select


# The actions an agent is allowed to choose, in canonical index order.
# PUSH and ORDER are intentionally excluded — the environment sets those
# automatically. This is the single source of truth for the agent action set:
# the Gym action space, the index->Action mapping, and agents all reference it.
AGENT_ACTIONS = [
    Action.MOVE_UP,
    Action.MOVE_DOWN,
    Action.MOVE_LEFT,
    Action.MOVE_RIGHT,
    Action.WAIT,
]
