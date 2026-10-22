from typing import Optional, Union

from src.env.actions import AGENT_ACTIONS, Action
from src.agents.base_agent import Agent


class RLAgent(Agent):
    """An agent whose action is supplied externally each step (e.g. by a DQN
    policy) instead of being computed by a planner.

    The RL wrapper sets the policy's chosen action through
    :meth:`set_policy_action`; ``get_action`` simply returns it. This is the
    single seam between an external RL policy and the simulation — all the
    other agent types decide their own actions internally.

    It inherits the base Agent so it participates in episode termination and
    rendering exactly like a scripted agent; only action selection differs.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The action the policy selected for the upcoming step. Defaults to WAIT
        # so the agent is well-defined before the first action is injected.
        self.pending_action: Optional[Action] = Action.WAIT

    def set_policy_action(self, action: Union[Action, int]) -> None:
        """Set the action selected by an external policy for the next step.

        The policy emits an integer action index, while evaluation code may already
        have an :class:`Action`. Keeping conversion here gives both paths one
        checked, canonical interface.
        """
        if isinstance(action, Action):
            if action not in AGENT_ACTIONS:
                raise ValueError(f"RL policy cannot select environment action {action!r}")
            self.pending_action = action
            return

        if isinstance(action, int) and not isinstance(action, bool):
            if 0 <= action < len(AGENT_ACTIONS):
                self.pending_action = AGENT_ACTIONS[action]
                return

        raise ValueError(
            f"RL policy action must be an Action or integer in "
            f"[0, {len(AGENT_ACTIONS) - 1}], got {action!r}"
        )

    def get_action(self, *args, **kwargs) -> Optional[Action]:
        """Return the externally-injected action (ignores planner arguments)."""
        if self.done:
            return None
        return self.pending_action if self.pending_action is not None else Action.WAIT
