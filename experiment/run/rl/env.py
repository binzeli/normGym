"""DQN environment, observations, physical masks, and reward logging."""
from __future__ import annotations

from collections import deque
import contextlib
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from experiment.agents.rl_agent import RLAgent
from experiment.run.common import (
    build_initial_agents,
    build_scripted_pool,
    is_subject_agent,
    make_agent,
    pop_scripted_replacement,
    respawn_agent,
)
from src.agents.base_agent import agent_class_index
from src.env.actions import AGENT_ACTIONS, Action
from src.env.gym_env import NormEnv
from src.norms.occupation import (
    free_choice_goal_index,
    is_occupation_door_norm,
    is_occupation_region_norm,
    waiting_zone_cells,
)


# The training CLI sets dense shaping to 6.0 and disables it for sparse runs.
GOAL_PROGRESS_REWARD = 0.2
SUBJECT_FEATURES = 13
OTHER_FEATURES = 8


def _subject_reward_callback(log_dir: Path, log_every: int = 1):
    """Accumulate every step; emit scalars at log_every steps and episode ends."""
    if log_every < 1:
        raise ValueError("log_every must be positive")
    try:
        from stable_baselines3.common.callbacks import BaseCallback
    except ModuleNotFoundError as error:
        if error.name == "stable_baselines3":
            raise RuntimeError(
                "Stable-Baselines3 is required for RL training. Install the "
                "project dependencies (for example: pip install -r requirements.txt)."
            ) from error
        raise
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "TensorBoard is required for DQN reward tracking. "
            "Install the project dependencies (pip install -r requirements.txt)."
        ) from error

    class SubjectRewardCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.writer = None
            self.cumulative_reward = 0.0
            self.action_counts = np.zeros(len(AGENT_ACTIONS), dtype=np.int64)
            self.subject_action_steps = 0

        def _on_training_start(self) -> None:
            # The algorithm-specific <algorithm>_<n> directory is configured
            # by SB3 immediately before callbacks begin. Resolve it lazily in
            # _on_step as an additional safeguard across DQN runs.
            return None

        def _ensure_writer(self) -> None:
            if self.writer is not None:
                return
            # SB3 creates a distinct <algorithm>_<n> directory for every run.
            # Put custom rollout metrics beneath that same directory, rather
            # than merging all training attempts into one TensorBoard series.
            run_dirs = sorted(
                (path for path in log_dir.iterdir()
                 if path.is_dir() and path.name.rsplit("_", 1)[0] in {"DQN"}),
                key=lambda path: path.stat().st_mtime_ns,
            )
            run_dir = run_dirs[-1] if run_dirs else log_dir
            self.writer = SummaryWriter(log_dir=str(run_dir / "subject_step_rewards"))

        def _on_step(self) -> bool:
            rewards = self.locals.get("rewards")
            infos = self.locals.get("infos") or []
            info = infos[0] if infos else {}
            step_reward = float(rewards[0]) if rewards is not None else 0.0
            self.cumulative_reward += step_reward
            action_index = int(info.get("subject_policy_action", -1))
            if 0 <= action_index < len(AGENT_ACTIONS):
                self.action_counts[action_index] += 1
                self.subject_action_steps += 1
            dones = self.locals.get("dones")
            done = dones is not None and bool(dones[0])
            if self.num_timesteps % log_every == 0 or done:
                self._ensure_writer()
                values = {
                    "rollout/subject_cumulative_reward": self.cumulative_reward,
                    "rollout/subject_step_reward": step_reward,
                    "rollout/subject_active": float(info.get("subject_id") is not None),
                    "rollout/subject_goals_reached_cumulative": float(info.get("subject_goals_reached_total", 0)),
                    "rollout/subject_goals_reached_per_step": float(info.get("subject_goals_reached_this_step", 0)),
                }
                if 0 <= action_index < len(AGENT_ACTIONS):
                    values["rollout/subject_action_index"] = action_index
                    for index, action in enumerate(AGENT_ACTIONS):
                        values[f"rollout/subject_action_count/{action.value}"] = self.action_counts[index]
                        values[f"rollout/subject_action_fraction/{action.value}"] = self.action_counts[index] / self.subject_action_steps
                if "valid_action_count" in info:
                    values["rollout/valid_action_count"] = float(info["valid_action_count"])
                    values["rollout/subject_action_legal"] = float(bool(info.get("subject_action_legal")))
                for key, value in values.items():
                    self.writer.add_scalar(key, value, self.num_timesteps)
            if done:
                self.cumulative_reward = 0.0
            return True

        def _on_training_end(self) -> None:
            if self.writer is not None:
                self.writer.close()

    return SubjectRewardCallback()


class _DiscardStream:
    """A stdout stand-in that throws writes away instead of storing them."""

    def write(self, _text: str) -> int:
        return 0

    def flush(self) -> None:
        pass


_DISCARD_STREAM = _DiscardStream()


@contextlib.contextmanager
def _silenced_stdout(enabled: bool = True):
    """Drop everything the simulation prints inside this block.

    The scripted planner, the queue detector and NormEnv each print a block of
    debug text per agent per step.  That is worth reading when a run is
    rendered for a human, but a 1M-timestep training run emits roughly 20 GB of
    it, and on a shared filesystem the write throughput -- not the policy
    update -- becomes the wall-clock bottleneck: an 8-agent queue norm took 21
    hours, against under 5 once the stream was dropped.

    Learner rollouts are never read back, so the adapter discards that output
    rather than paying to store it.  Nothing in ``src/`` changes, so the
    rendered interactive/test stages and every non-RL subject keep printing
    exactly as before.  SB3's own progress tables are logged outside these
    blocks and are unaffected.
    """
    if not enabled:
        yield
        return
    saved = sys.stdout
    sys.stdout = _DISCARD_STREAM
    try:
        yield
    finally:
        sys.stdout = saved


class InteractiveDQNEnv(gym.Env):
    """A single-agent DQN view of an interactive multi-agent ``NormEnv``.

    The policy controls exactly one active :class:`RLAgent` at a time.  When
    that customer completes its goals, it is replaced from the same subject
    queue used by the experiment pipeline.  All other customers remain Oracle
    agents.  The episode ends only when the interactive horizon is reached.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        norm_type: str = "edge_left_line",
        capacity: int = 8,
        max_steps: int = 200,
        world_type: str = "cafe",
        ground_truth: str = "without",
        subject_position: int = 2,
        num_no_norm_agents: int = 1,
        map_type: str = "default",
        goal_progress_reward: bool = False,
        goal_progress_magnitude: float = GOAL_PROGRESS_REWARD,
        training_penalties: bool = True,
        norm_penalty: bool | None = None,
        collision_penalty: bool | None = None,
        task_goal_mode: str = "benchmark",
        quiet_simulation_output: bool = True,
    ):
        if capacity < 3:
            raise ValueError("capacity must be at least 3")
        if not 0 <= subject_position < capacity:
            raise ValueError("subject_position must be within the initial capacity")
        if not 0 <= num_no_norm_agents <= capacity - 1:
            raise ValueError("num_no_norm_agents must fit in the background slots")

        self.norm_type = norm_type
        self.capacity = capacity
        # Learner rollouts discard the simulation's debug printing; see
        # _silenced_stdout. Set False to watch a training episode.
        self.quiet_simulation_output = quiet_simulation_output
        self.max_steps = max_steps
        self.world_type = world_type
        self.ground_truth = ground_truth
        self.subject_position = subject_position
        self.num_no_norm_agents = num_no_norm_agents
        self.map_type = map_type
        self.goal_progress_reward = goal_progress_reward
        self.goal_progress_magnitude = float(goal_progress_magnitude)
        # False drops the collision and norm-violation terms from the reward
        # the policy TRAINS on, isolating "can it learn to reach the goal at
        # all". The environment still simulates, counts and logs both, and the
        # rendered test stages always score the full task reward, so eval.py
        # metrics stay comparable to every other subject kind either way.
        self.training_penalties = bool(training_penalties)
        self.norm_penalty = bool(training_penalties) if norm_penalty is None else bool(norm_penalty)
        self.collision_penalty = bool(training_penalties) if collision_penalty is None else bool(collision_penalty)
        if task_goal_mode not in {"benchmark", "category"}:
            raise ValueError("Unknown task goal mode")
        self.task_goal_mode = task_goal_mode
        self._distance_fields: Dict[tuple, Dict[tuple, int]] = {}

        # Rendering is intentionally disabled for learning: NormEnv otherwise
        # writes a PNG before querying every non-oracle agent.
        self.env = NormEnv(
            num_agents=capacity,
            max_steps=max_steps,
            norm_type=norm_type,
            render_mode=None,
            world_type=world_type,
            map_type=map_type,
        )
        height, width = self.env.world.height, self.env.world.width
        max_dimension = max(height, width)

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict({
            # The whole map as (layout, self, others) planes. An LLM subject
            # sees the full rendered world, is told its own position, and is
            # given every other agent's position, so all three stay here for
            # information parity. See _map_planes.
            "grid": spaces.Box(
                low=0.0, high=1.0, shape=(MAP_CHANNELS, height, width),
                dtype=np.float32,
            ),
            # Relative, normalized: goal delta, BFS steps-to-goal, unreachable
            # flag, the move just committed, active/done, which leg of the task,
            # the service wait, whether the last action drew a norm violation,
            # and how much of the stage is left. See policy_observation for why
            # the previous position is part of the state, and why nothing here
            # is absolute.
            "subject": spaces.Box(
                low=-1.0, high=1.0, shape=(SUBJECT_FEATURES,), dtype=np.float32,
            ),
            # Other agents relative to the subject, nearest first; unused slots
            # are all-zero (the last element is the occupancy flag). Positions
            # and violation markers only, matching what the LLM prompt lists.
            "other_agents": spaces.Box(
                low=-1.0, high=1.0, shape=(capacity - 1, OTHER_FEATURES), dtype=np.float32
            ),
            # Which physical actions are legal from the subject's current cell.
            # Part of the observation so DQN, which has no action-mask API, can
            # see legality directly instead of having its action indices
            # remapped underneath it.
            "action_mask": spaces.Box(
                low=0.0, high=1.0, shape=(len(AGENT_ACTIONS),), dtype=np.float32
            ),
        })

        self.waiting_subject: deque[RLAgent] = deque()
        self.waiting_norm_customer = deque()
        self.waiting_no_norm_customer = deque()
        self.next_replacement_step = 1
        self.last_spawn_entrance = None
        self.replaced_background_ids: set[str] = set()
        self.subject_goals_reached = 0

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None):
        with _silenced_stdout(self.quiet_simulation_output):
            return self._reset(seed=seed, options=options)

    def _reset(self, *, seed: Optional[int] = None,
               options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            # Agent creation and replacement utilities intentionally use the
            # repo's global RNG, so seed it alongside Gymnasium's generator.
            random.seed(seed)

        initial_agents = build_initial_agents(
            self.capacity,
            {self.subject_position},
            self.env.world,
            self.norm_type,
            client=None,
            ground_truth=self.ground_truth,
            subject_kind="rl",
            prompt_stage="interactive",
            num_no_norm_agents=self.num_no_norm_agents,
        )

        pool_size = self.capacity * 20
        subject_pool_size = pool_size // self.capacity
        customer_pool_size = pool_size - subject_pool_size
        agent_number = self.capacity + 1
        self.waiting_subject = deque()
        for _ in range(subject_pool_size):
            subject = make_agent(
                f"C{agent_number}",
                self.env.world,
                self.norm_type,
                client=None,
                subject_kind="rl",
                ground_truth=self.ground_truth,
                prompt_stage="interactive",
            )
            self.waiting_subject.append(subject)
            agent_number += 1

        (self.waiting_norm_customer,
         self.waiting_no_norm_customer,
         _) = build_scripted_pool(
            customer_pool_size,
            self.capacity - 1,
            self.num_no_norm_agents,
            self.env.world,
            self.norm_type,
            client=None,
            ground_truth=self.ground_truth,
            first_agent_number=agent_number,
        )
        self.next_replacement_step = 1
        self.last_spawn_entrance = None
        self.replaced_background_ids = set()
        self.subject_goals_reached = 0
        self.env.world.previous_queue = {}
        self.env.world.current_queue = {}
        self.env.num_agents = self.capacity
        _, info = self.env.reset(seed=seed, options={"agents": initial_agents})
        return self._observation(), info

    def step(self, action: int):
        with _silenced_stdout(self.quiet_simulation_output):
            return self._step(int(action))

    def _step(self, action: int):
        subject = self._active_subject()
        previous_target_index = subject.target_index if subject is not None else 0
        if subject is not None:
            subject.set_policy_action(int(action))

        _, _, terminated, truncated, info = self.env.step()
        subject_reward = 0.0
        shaping_reward = 0.0
        subject_id = subject.agent_id if subject is not None else None
        if subject_id is not None:
            subject_reward = self._training_reward(
                float(info.get("step_rewards", {}).get(subject_id, 0.0)),
                info.get("reward_components", {}).get(subject_id, {}),
            )
            # Learner-only shaping, added on top of the untouched task reward.
            # It never reaches env.step()'s own reward, agent.total_reward, or
            # the raw run logs, so eval.py still scores this subject on exactly
            # the reward the LLM and oracle subjects are scored on.
            shaping_reward = self._shaping_reward(subject)
            subject_reward += shaping_reward

        goals_reached_this_step = max(
            0, (subject.target_index - previous_target_index) if subject is not None else 0
        )
        self.subject_goals_reached += goals_reached_this_step
        subject_finished = bool(subject is not None and subject.done)
        if subject_finished:
            self._replace_subject(subject)
        self._replace_finished_background_agents()

        # A finite turnover pool must not make an RL episode end early.  The
        # interactive stage's actual horizon remains the source of truncation.
        if self.env.current_step < self.max_steps:
            terminated = False
            truncated = False

        info = {
            **info,
            "subject_id": subject_id,
            "subject_reward": subject_reward,
            "subject_policy_action": int(action) if subject is not None else -1,
            "subject_goals_reached_this_step": goals_reached_this_step,
            "subject_goals_reached_total": self.subject_goals_reached,
            "subject_finished": subject_finished,
        }
        return self._observation(), subject_reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Return physical-action validity for DQN.

        A movement action is valid only when it changes the subject's grid
        position.  This masks walls and world bounds, but deliberately does
        not mask occupied cells or norm-violating moves: those remain part of
        the social-navigation task and retain their existing environment
        consequences.  WAIT is always valid.
        """
        return policy_action_mask(self.env)

    def _training_reward(self, task_reward: float, components: Dict[str, float]) -> float:
        """The task reward the policy learns from, before shaping.

        With ``training_penalties`` on this is the task reward untouched. With
        it off, the collision and norm-violation terms are subtracted back out
        using the amounts the env itself charged -- taken from its breakdown
        rather than recomputed here, so the collision-blame rules live in one
        place.  Only the learning signal changes: the env's ``step_rewards``,
        ``agent.total_reward`` and the raw run logs eval.py reads keep the full
        penalty, and the rendered test stages always score the full reward.
        """
        return (task_reward
                - (0.0 if self.collision_penalty else float(components.get("collision", 0.0)))
                - (0.0 if self.norm_penalty else float(components.get("violation", 0.0))))

    def _goal_distance_field(self, goals) -> Dict[tuple, int]:
        """Steps to the nearest cell of ``goals`` (see module-level distance_field)."""
        return distance_field(self.env.world, goals)

    def _goal_cells(self, subject, target_index: int) -> frozenset:
        """The cells that would complete this subject's goal (see goal_cells)."""
        return goal_cells(self.env, subject, target_index)


    def _shaping_reward(self, subject) -> float:
        """Dense progress signal for the RL subject only (0.0 when disabled)."""
        if not self.goal_progress_reward:
            return 0.0
        # previous_target_index refers to the goal that was active for the move
        # just committed, matching how the env scores that move.
        target_index = getattr(subject, "previous_target_index", subject.target_index)
        targets = getattr(subject, "target_positions", [])
        if not (0 <= target_index < len(targets)):
            return 0.0
        previous_position = getattr(subject, "previous_position", None)
        current_position = getattr(subject, "current_position", None)
        if previous_position is None or current_position is None:
            return 0.0
        goal_cells = self._goal_cells(subject, target_index)
        if not goal_cells:
            return 0.0
        field = self._goal_distance_field(goal_cells)
        # A cell can be missing from the field only if it is walled off from the
        # goal; treat that as "no information" rather than infinite progress.
        before = field.get(tuple(previous_position))
        after = field.get(tuple(current_position))
        if before is None or after is None or before == after:
            return 0.0
        magnitude = self.goal_progress_magnitude
        return magnitude if after < before else -magnitude

    def _active_subject(self) -> Optional[RLAgent]:
        for agent in self.env.agents:
            if is_subject_agent(agent) and agent.active and not agent.done:
                if not isinstance(agent, RLAgent):
                    raise TypeError(f"Expected RLAgent subject, found {type(agent).__name__}")
                return agent
        return None

    def _replace_subject(self, subject: RLAgent) -> None:
        if not self.waiting_subject:
            return
        replacement = self.waiting_subject.popleft()
        self.next_replacement_step, self.last_spawn_entrance = respawn_agent(
            self.env, replacement, self.next_replacement_step, self.last_spawn_entrance
        )

    def _replace_finished_background_agents(self) -> None:
        for agent in list(self.env.agents):
            if (
                not agent.done
                or is_subject_agent(agent)
                or agent.agent_id in self.replaced_background_ids
            ):
                continue
            replacement = pop_scripted_replacement(
                agent, self.waiting_norm_customer, self.waiting_no_norm_customer
            )
            if replacement is not None:
                self.replaced_background_ids.add(agent.agent_id)
                self.next_replacement_step, self.last_spawn_entrance = respawn_agent(
                    self.env, replacement, self.next_replacement_step, self.last_spawn_entrance
                )

    def _observation(self) -> Dict[str, np.ndarray]:
        for agent in self.env.agents:
            if is_subject_agent(agent):
                agent.task_goal_mode = self.task_goal_mode
        return policy_observation(self.env, self.capacity)


def _cell_scale(world) -> float:
    """Normaliser for raw CellType values, so the map cannot swamp by magnitude."""
    scale = getattr(world, "_policy_cell_scale", None)
    if scale is None:
        scale = float(max(1, int(world.grid.max())))
        world._policy_cell_scale = scale
    return scale


# Map planes: static layout, where I am, where everybody else is.
MAP_CHANNELS = 3


def _map_planes(world, position, others) -> np.ndarray:
    """The whole map as (layout, self, others) planes.

    Three things make this the right shape rather than a bare cell-type grid:

    * An LLM subject is shown the entire rendered world AND the layout again as
      exact per-zone coordinates, so the full map is what parity requires.
    * It is told "Your current position is (r, c)" and sees its own gold star on
      the frame. Every other feature here is RELATIVE (a goal delta, other agents
      offset from the subject), so without a self plane nothing tied the policy
      to the map at all and the layout was a constant it could not use.
    * It is given every other agent's absolute position in the prompt text. The
      others plane carries the same fact in the place a convolution can read it.

    Layout is normalized rather than raw so a cell id of 12 does not dominate a
    delta of 0.13 by magnitude; the others plane carries (class + 1) / 3 so an
    empty cell (0.0) stays distinct from class 0.
    """
    planes = np.zeros((MAP_CHANNELS, world.height, world.width), dtype=np.float32)
    planes[0] = world.grid.astype(np.float32) / _cell_scale(world)
    if position is not None and 0 <= position[0] < world.height and 0 <= position[1] < world.width:
        planes[1, position[0], position[1]] = 1.0
    for agent in others:
        cell = policy_state_position(agent)
        if 0 <= cell[0] < world.height and 0 <= cell[1] < world.width:
            planes[2, cell[0], cell[1]] = (agent_class_index(agent.agent_class) + 1) / 3.0
    return planes


def violation_agent_ids(env: NormEnv) -> frozenset:
    """Agent ids carrying a norm-violation marker in the current step.

    ``NormEnv`` detects a violation during step k and injects it into step k+1's
    log entry, so the newest entry names the agents whose PREVIOUS action
    violated -- exactly the set the renderer draws red cross marks on, and the
    set the LLM prompt describes in words. Reading it here keeps the two
    subjects on the same signal without touching the shared environment.
    """
    log = getattr(env, "episode_log", None)
    if not log:
        return frozenset()
    return frozenset(
        violation[0] for violation in log[-1].get("norm_violations", []) if violation
    )


def _relative_agent_features(agent, position, scale: float,
                             violated: bool) -> np.ndarray:
    """One other agent, described relative to the subject.

    Deliberately carries NO target: the LLM prompt lists other agents' positions
    and nothing else, so handing the policy everyone's destination would give it
    information the baseline cannot have -- and precisely the information the
    queueing, seat and door norms are about.
    """
    other = policy_state_position(agent)
    if position is None:
        position = other
    return np.array([
        (other[0] - position[0]) / scale,
        (other[1] - position[1]) / scale,
        (other[0] - agent.current_position[0]) / scale,
        (other[1] - agent.current_position[1]) / scale,
        agent_class_index(agent.agent_class) / 3.0,
        1.0 if agent.active else 0.0,
        1.0 if violated else 0.0,  # red cross mark on the rendered frame
        1.0,  # slot occupied; empty slots stay all-zero
    ], dtype=np.float32)


def distance_field(world, goals) -> Dict[tuple, int]:
    """Steps to the NEAREST cell of ``goals``, for every reachable cell.

    Manhattan distance is the wrong potential on this map.  The counter is an
    unwalkable cell directly beside the service spot, so cells on its far side
    score well on Manhattan while being several steps further away in reality --
    for the cafe's edge_left_line goal (1,5), cell (1,7) is Manhattan 2 but 4
    steps away.  Shaping on Manhattan rewarded moving into that pocket and
    penalised the detour around the counter that is the only real progress.

    ``goals`` may hold several cells: a compliance-graded goal is satisfied by
    any of them, so seeding the queue with all of them at distance 0 yields
    "distance to the closest acceptable cell" in one pass.  The field is cached
    on the world, which the layout being static per episode makes safe, and is
    read by both the shaping reward and the observation.
    """
    # A bare (row, col) is a single goal, not an iterable of two goals:
    # frozenset((1, 5)) would silently become {1, 5} and field a pair of ints.
    if (isinstance(goals, tuple) and len(goals) == 2
            and all(isinstance(value, int) for value in goals)):
        goals = frozenset({goals})
    else:
        goals = frozenset(goals)

    cache = getattr(world, "_policy_distance_fields", None)
    if cache is None:
        cache = {}
        world._policy_distance_fields = cache
    cached = cache.get(goals)
    if cached is not None:
        return cached

    field = {goal: 0 for goal in goals}
    frontier = deque(goals)
    while frontier:
        row, col = frontier.popleft()
        for neighbour in ((row - 1, col), (row + 1, col),
                          (row, col - 1), (row, col + 1)):
            if neighbour in field:
                continue
            if not (0 <= neighbour[0] < world.height
                    and 0 <= neighbour[1] < world.width):
                continue
            # Goal cells themselves can be non-walkable furniture (a chair, the
            # service spot); interior cells must be traversable.
            if not (world.is_walkable(neighbour) or world.is_sittable(neighbour)):
                continue
            field[neighbour] = field[(row, col)] + 1
            frontier.append(neighbour)
    cache[goals] = field
    return field


def goal_cells(env: NormEnv, subject, target_index: int) -> frozenset:
    """The cells the subject may legitimately aim at for goal ``target_index``.

    Matched to what an LLM subject is told, so the RL subject trains on the same
    information and its numbers stay comparable:

    - Arrival goals: the concrete preset cell, which is exactly what the LLM
      prompt substitutes (``_target_prompt_text``).
    - Free-choice goals (one per occupation norm): the LLM is NOT given a cell.
      It is told "the chair/door you choose for your current goal above" and has
      to pick a compliant one itself; under ground_truth="without" it is not even
      told the norm. So this returns the whole goal CATEGORY -- every chair, every
      waiting-zone cell, every door -- and deliberately does NOT filter by
      ``free_goal_satisfied``. Category membership is visible on the map the LLM
      sees; which member complies is the task, and handing that over would make
      the occupation norms measure navigation instead of norm inference.

    This still fixes the original defect: the preset cell for a free-choice goal
    is drawn by ``random.choice(world.entrances)`` with no norm awareness, so
    aiming at it sent a subject to a door that could never complete the goal.
    Aiming at the category cannot be unwinnable, without revealing the answer.
    """
    targets = getattr(subject, "target_positions", [])
    if not (0 <= target_index < len(targets)):
        return frozenset()
    preset = frozenset({tuple(targets[target_index])})
    if getattr(subject, "is_oracle", False):
        return preset
    norm = getattr(subject, "norm", None)
    free_index = free_choice_goal_index(norm, len(targets))
    if free_index is None or free_index != target_index:
        return preset

    world = env.world
    if is_occupation_door_norm(norm):
        category = set(getattr(world, "entrances", []))
    elif is_occupation_region_norm(norm):
        category = set(waiting_zone_cells(world))
    else:  # seat norms
        category = {
            (row, col)
            for row in range(world.height) for col in range(world.width)
            if world.is_sittable((row, col))
        }
    return frozenset(category) or preset


def policy_state_position(agent) -> tuple:
    """The cell the agent's *next* action will be applied from.

    ``NormEnv.step`` commits a move at the START of the following step: the
    action chosen during step k sets ``next_position``, which only becomes
    ``current_position`` at the top of step k+1.  A policy queried after
    step k therefore has to reason about ``next_position`` — reading
    ``current_position`` describes where the agent already acted from, one
    move stale, and produced a mask for the wrong cell.
    """
    if agent.next_position is not None:
        return agent.next_position
    return agent.current_position


def policy_observation(env: NormEnv, capacity: int) -> Dict[str, np.ndarray]:
    """Build the fixed policy observation for a live rendered ``NormEnv`` too.

    Every agent is reported at the position its next action applies from
    (``policy_state_position``) plus the position it is moving out of.  Both
    halves matter:

    * the first keeps the observation and the action mask describing the cell
      the policy is actually about to move from;
    * the second makes the reward Markov.  ``NormEnv._compute_reward`` scores
      the move it just committed, so the reward handed back for step k is the
      goal progress of the move from ``current_position`` to
      ``next_position``.  With both endpoints in the observation that reward is
      a function of the observed state, which is a well-formed MDP; with only
      one of them the learner sees ±GOAL_PROGRESS_REWARD arriving seemingly at
      random (a WAIT could be "paid" for an earlier move).

    What is and is not included is pinned to what an LLM subject receives, so
    the two are compared on the same information:

    * other agents carry position, class and violation marker, but NO target.
      The prompt lists other agents' positions only; nothing tells it where
      anyone is headed, and that is exactly what the queueing, seat and door
      norms turn on.
    * violation markers are included for the subject and for every other agent.
      The LLM is told in words when its own previous action violated, and sees
      red cross marks on other violators, so withholding these left the RL
      subject inferring a hidden norm from an unattributed scalar penalty.
    * the stage clock is included; the LLM is told the stage's step budget.
    """
    if capacity < 2:
        raise ValueError("capacity must reserve a slot for at least one background agent")
    subject = None
    for agent in env.agents:
        if is_subject_agent(agent) and agent.active and not agent.done:
            subject = agent
            break
    world = env.world
    scale = float(max(world.height, world.width))
    violated_ids = violation_agent_ids(env)
    subject_features = np.zeros(SUBJECT_FEATURES, dtype=np.float32)
    position = None
    if subject is not None:
        # Report the goal the env will actually grade. For a free-choice goal
        # the preset target is an arbitrary member of the accepted set and may
        # be unsatisfiable (see goal_cells), so aim at the accepted set instead
        # -- otherwise the observed goal and the shaped/graded goal disagree.
        cells = goal_cells(env, subject, subject.target_index)
        position = policy_state_position(subject)
        field = distance_field(world, cells) if cells else {}
        target = None
        if cells:
            # The representative goal is the one the BFS field says is nearest,
            # matching the cell the shaping reward is actually pulling toward.
            # Each seed in a multi-source field has distance zero. Compare
            # single-goal distances FROM the subject, with stable coordinate ties.
            distances = {cell: distance_field(world, {cell}).get(tuple(position), float("inf"))
                         for cell in cells}
            target = min(cells, key=lambda cell: (distances[cell], cell))
        steps_to_goal = field.get(tuple(position))
        # An LLM subject on a free-choice goal is told "the chair/door you
        # choose", never a cell, and never which one complies. The flag mirrors
        # that: the agent knows it is on a free-choice goal, and the delta below
        # points at the nearest member of the goal CATEGORY (visible on the map
        # it observes) rather than at a compliant cell.
        free_index = free_choice_goal_index(
            getattr(subject, "norm", None),
            len(getattr(subject, "target_positions", []) or [1]),
        )
        is_free_choice = 1.0 if free_index == subject.target_index else 0.0

        # RELATIVE, normalized geometry. Absolute coordinates forced the policy
        # to memorise "at (1,2) with target (5,0) go left" per cell, which is
        # why a policy that navigated leg 1 competently froze on leg 2 and why
        # nothing transferred to a new world (SuccessRate 0.000 on every
        # new_map_test of both 1M sweeps). A delta plus a BFS distance makes
        # the direction of progress the same signal everywhere on the map.
        delta_row = (target[0] - position[0]) / scale if target else 0.0
        delta_col = (target[1] - position[1]) / scale if target else 0.0
        moved_row = (position[0] - subject.current_position[0]) / scale
        moved_col = (position[1] - subject.current_position[1]) / scale
        num_targets = max(1, len(getattr(subject, "target_positions", []) or [1]))
        # The LLM is told in words when its own previous action violated the
        # norm, and is told how many steps the stage runs for. Without these the
        # RL subject had to infer a hidden norm from an unattributed -20 folded
        # into a scalar reward, and had no clock at all.
        own_violation = 1.0 if subject.agent_id in violated_ids else 0.0
        max_steps = max(1, int(getattr(env, "max_steps", 1) or 1))
        steps_left = max(0, max_steps - int(getattr(env, "current_step", 0)))
        subject_features = np.array([
            delta_row, delta_col,
            (steps_to_goal / scale) if steps_to_goal is not None else -1.0,
            1.0 if steps_to_goal is None else 0.0,   # goal unreachable from here
            moved_row, moved_col,
            1.0 if subject.active else 0.0,
            1.0 if subject.done else 0.0,
            subject.target_index / num_targets,      # which leg of the task
            min(subject.target_wait_time, 3) / 3.0,
            is_free_choice,                          # goal cell is not disclosed
            own_violation,                           # my last action was penalised
            steps_left / max_steps,                  # stage clock
        ], dtype=np.float32)

    others = np.zeros((capacity - 1, OTHER_FEATURES), dtype=np.float32)
    # Sorted by distance to the subject, not by agent id. Sorting by the id
    # string made a slot's meaning permute arbitrarily as agents turned over,
    # so "slot 0" carried no consistent signal across steps.
    active_others = [
        agent for agent in env.agents
        if not is_subject_agent(agent) and agent.active and not agent.done
    ]
    if position is not None:
        active_others.sort(key=lambda agent: (
            abs(agent.current_position[0] - position[0])
            + abs(agent.current_position[1] - position[1])))
    else:
        active_others.sort(key=lambda agent: agent.agent_id)
    for index, agent in enumerate(active_others[:capacity - 1]):
        others[index] = _relative_agent_features(
            agent, position, scale, agent.agent_id in violated_ids
        )

    return {
        "grid": _map_planes(world, position, active_others[:capacity - 1]),
        "subject": subject_features,
        "other_agents": others,
        "action_mask": policy_action_mask(env).astype(np.float32),
    }


def policy_action_mask(env: NormEnv) -> np.ndarray:
    """Physical action mask for a training or rendered-policy environment.

    Masked against ``policy_state_position``: the mask has to describe the cell
    the action will actually be applied from, not the one the agent has already
    left (see policy_state_position).
    """
    subject = next(
        (agent for agent in env.agents
         if is_subject_agent(agent) and agent.active and not agent.done),
        None,
    )
    mask = np.zeros(len(AGENT_ACTIONS), dtype=bool)
    wait_index = AGENT_ACTIONS.index(Action.WAIT)
    mask[wait_index] = True
    if subject is None:
        return mask

    position = policy_state_position(subject)
    for index, action in enumerate(AGENT_ACTIONS):
        if action == Action.WAIT:
            continue
        mask[index] = env.world.transition(position, action) != position
    return mask
