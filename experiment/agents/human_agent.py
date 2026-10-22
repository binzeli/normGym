"""
Human-driven agent that receives actions from keyboard input instead of an LLM.
Supports an optional action_queue (threading.Queue) for web-based play:
  - action_queue=None  → terminal mode (blocks on stdin, original behaviour)
  - action_queue=Queue → web mode (blocks on queue.get() fed by the web server)
"""
import queue
import sys
from typing import Optional
from src.agents.base_agent import Agent, Action
from src.norms.occupation import (
    free_choice_goal_index,
    free_goal_settled,
)
from experiment.utils.agent_goals import current_goal_text

if sys.platform == "win32":
    import msvcrt
else:
    import termios
    import tty

_KEY_TO_ACTION = {
    '\x1b[A': Action.MOVE_UP,    # arrow up
    '\x1b[B': Action.MOVE_DOWN,  # arrow down
    '\x1b[D': Action.MOVE_LEFT,  # arrow left
    '\x1b[C': Action.MOVE_RIGHT, # arrow right
    ' ':       Action.WAIT,       # spacebar
}

_LEGEND = (
    "  ↑ = move up   ↓ = move down   ← = move left   → = move right   [space] = wait   q/Ctrl+C = quit"
)


def _read_key() -> str:
    """Read a single raw keypress from stdin. Arrow keys return 3-char escape sequences."""
    if sys.platform == "win32":
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            key = msvcrt.getwch()
            return {
                "H": "\x1b[A",
                "P": "\x1b[B",
                "K": "\x1b[D",
                "M": "\x1b[C",
            }.get(key, "")
        return ch

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == '\x1b':
            # Escape sequence: read up to 2 more bytes
            ch += sys.stdin.read(1)  # '['
            ch += sys.stdin.read(1)  # 'A'/'B'/'C'/'D'
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


class HumanAgent(Agent):
    """
    Agent controlled by keyboard input (terminal) or a threading.Queue (web).

    - action_queue=None  → terminal mode: blocks on stdin arrow-key presses.
    - action_queue=Queue → web mode: blocks on queue.get() so the web server
                           can feed actions from browser key presses.
    """

    # Renderers draw a "GOAL n" marker on this agent's current target cell (see
    # FlatRenderer.save_frame). Class flag so the renderer needs no agent import.
    show_goal_marker = True

    def __init__(self, *args, action_queue: Optional[queue.Queue] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_queue = action_queue
        self.pending_violation_feedback: Optional[int] = None  # step number of the violation
        # Optional callable(output_dir) set by the web server to push frames to
        # the browser before blocking on user input.
        self.emit_callback = None

    def _free_goal_active(self) -> bool:
        """True if the current goal is a hidden, free-choice occupation goal."""
        idx = free_choice_goal_index(self.norm, len(self.target_positions))
        return idx is not None and self.target_index == idx

    def display_target(self):
        """The target cell to reveal to the player, or None when the current goal
        is a hidden free-choice occupation goal (the player must choose a
        norm-compliant chair/door themselves)."""
        if self._free_goal_active():
            return None
        return self.current_target

    def goal_text(self) -> Optional[str]:
        """The player's CURRENT goal only, sourced from agent_goals.json. Goals are
        revealed one at a time: Goal 2 appears only after Goal 1 is finished. A
        preset goal shows its real target cell; a free-choice occupation goal is
        left unsubstituted (the player chooses a compliant chair/door)."""
        if self.done:
            return None
        return current_goal_text(self.norm, self.target_index, self.display_target())

    def _is_settled(self, world, all_current_positions) -> bool:
        """Whether the agent is resting on its current goal — the preset target
        cell, or (for a free-choice occupation goal) any chair/door for the seat
        and door norms (a wrong one still finishes the goal; the violation is
        flagged separately) / a permitted zone cell for the region norms."""
        if self._free_goal_active():
            return free_goal_settled(self, world, all_current_positions)
        return (self.current_target is not None
                and self.current_position == self.current_target)

    def get_action(self, world, all_current_positions, output_dir: Optional[str]) -> Action:
        """Return the next action.

        Terminal mode: blocks until the human presses a valid key.
        Web mode: blocks until an Action is placed in self.action_queue.
        """
        # Agent already finished — don't block or emit anything.
        if self.done:
            return Action.WAIT

        # Agent is sitting at a target position accumulating wait time — no need
        # to prompt; it will advance automatically once the wait is complete.
        if self.target_wait_time > 0 and self._is_settled(world, all_current_positions):
            print(f"  [auto-wait] settled at {self.current_position}, wait {self.target_wait_time}/3")
            return Action.WAIT

        other_positions = {
            agent_id: pos
            for agent_id, pos in all_current_positions.items()
            if agent_id != self.agent_id
        }
        goal_text = current_goal_text(self.norm, self.target_index, self.display_target())
        print(f"\n[HumanAgent {self.agent_id}] Step {self.current_step}")
        print(f"  Position: {self.current_position}")
        print(f"  {goal_text}" if goal_text else f"  Target: {self.display_target()}")
        print(f"  Other agents: {other_positions}")

        # --- Web mode ---
        if self.action_queue is not None:
            if self.emit_callback is not None:
                self.emit_callback(output_dir)
            print("  [web mode] waiting for browser action…")
            action = self.action_queue.get()  # blocks until web server puts an Action
            action_name = action.value.replace("_", " ")
            print(f"  Action received: {action_name}")
            return action

        # --- Terminal mode (original behaviour) ---
        print(_LEGEND)
        print("  Your move: ", end="", flush=True)

        while True:
            key = _read_key()
            if key in ('q', 'Q', '\x03'):  # q, Q, or Ctrl+C
                print("\nExiting simulation.")
                raise SystemExit(0)
            action = _KEY_TO_ACTION.get(key)
            if action is not None:
                action_name = action.value.replace("_", " ")
                print(action_name)
                return action
            # Invalid key — re-prompt on the same line
            print(f"\n  [Invalid key] Press an arrow key or spacebar.")
            print("  Your move: ", end="", flush=True)
