from typing import Tuple, Optional, List, Dict
from enum import Enum

from src.env.actions import Action


# Agent classes are colors. These three colors are the agent's class; there is
# no rank/ordering among them — they are just distinct categories.
AGENT_COLORS = ('orange', 'green', 'blue')


def agent_class_index(agent_class) -> int:
    """Encode an agent's color class as an integer for the observation vector.

    Returns the index into AGENT_COLORS, or -1 for an unknown class. This is a
    plain encoding for the numeric observation space, not a social rank.
    """
    try:
        return AGENT_COLORS.index(agent_class)
    except ValueError:
        return -1


class AgentType(Enum):
    """Types of agents in the cafe"""
    CUSTOMER = "customer"


class Agent:
    """
    Base agent class — all agents have position, type, speed, and status.
    Subclasses must implement get_action().

    Planning is delegated entirely to the functions in planner.py.
    """

    # Agents fall into two categories: oracle agents (scripted background agents
    # that know and follow the norm) and newcomer agents (human, LLM, RL, ...)
    # that must discover the norm. OracleAgent overrides this to True; every
    # other agent type is a newcomer by default.
    is_oracle: bool = False

    def __init__(
        self,
        agent_id: str,
        agent_class: str,
        position: Tuple[int, int],
        agent_type: AgentType = AgentType.CUSTOMER,
        goal: Optional[str] = "order",
        arrival_time: int = 0,
        entrance_id: int = 0,
        norm: Optional[str] = None,
        name: Optional[str] = None,
        target_positions: Optional[List[Tuple[int, int]]] = None,
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        # agent_class is the agent's color ('orange' | 'green' | 'blue').
        self.agent_class = agent_class
        # Human-readable display name. Defaults to the canonical display name
        # derived from agent_id so the renderer can read agent.name directly.
        if name is None:
            from src.env.agent_names import agent_display_name
            name = agent_display_name(agent_id)
        self.name = name

        # State attributes
        self.current_position = position
        self.previous_position: Optional[Tuple[int, int]] = position  # last integer position before step (rendering + collision-fault check)
        self.displaced_at_step: Optional[int] = None  # step at which collision resolution displaced this agent (its next move is forced, not chosen)
        self.next_position: Optional[Tuple[int, int]] = None
        self.goal = goal
        self.target_positions: List[Tuple[int, int]] = (
            list(target_positions) if target_positions is not None else []
        )
        self.target_index: int = 0
        self.previous_target_index: int = 0
        self.norm = norm
        self.arrival_time = arrival_time
        self.active = False
        self.counter_wait_time = 0
        self.target_wait_time = 0
        self.entrance_id = entrance_id
        self.done = False
        self.visited_x_cells: set = set()
        self.action: Optional[Action] = None
        self.previous_action: Optional[Action] = None  # rendering-only: action that caused last movement
        # Movement-norm bookkeeping, maintained by NormEnv after every step:
        # env steps on which the agent moved by its own choice (per-agent
        # rhythm norms).
        self.move_steps: set = set()

        # Reward tracking
        self.total_reward = 0.0

    # ------------------------------------------------------------------
    # Target management
    # ------------------------------------------------------------------

    @property
    def current_target(self) -> Optional[Tuple[int, int]]:
        """Return the active target position, or None if the list is empty."""
        if self.target_positions and self.target_index < len(self.target_positions):
            return self.target_positions[self.target_index]
        return None

    def display_target(self) -> Optional[Tuple[int, int]]:
        """The target cell safe to reveal to a viewer. Defaults to the current
        target; HumanAgent overrides this to hide free-choice occupation goals."""
        return self.current_target

    def advance_to_next_target(self) -> bool:
        """
        Advance to the next target in target_positions.

        Returns True if there is a next target,
        False if all targets are exhausted (agent.done set to True).
        """
        self.target_index += 1
        self.target_wait_time = 0
        self.counter_wait_time = 0
        if self.target_index < len(self.target_positions):
            return True
        self.done = True
        return False

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def get_action(
        self, world, all_current_positions, agent_classes, collision_actions=None,
        all_target_positions: Optional[Dict[str, Tuple[int, int]]] = None,
    ) -> Optional[Action]:
        """
        Choose and return an action for this step.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement get_action()")

    def __repr__(self):
        status_str = "done" if self.done else "active" if self.active else "waiting"
        return f"{self.agent_type.value}[{self.agent_id}] at {self.current_position} - {status_str}"
