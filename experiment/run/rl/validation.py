"""Validation layouts, headless rollouts, and passive lifecycle recording."""
from __future__ import annotations

import random
from collections import Counter

import numpy as np
import torch

from experiment.run.common import (
    RunRecorder, _random_subject_positions, build_initial_agents, build_scripted_pool,
    is_subject_agent, pop_scripted_replacement, respawn_agent, release_region_holders,
)
from experiment.evaluation.eval import run_metrics
from experiment.run.rl.env import (
    InteractiveDQNEnv, _silenced_stdout, distance_field, policy_observation, policy_state_position,
)
from experiment.run.rl.provenance import digest
from src.env.actions import AGENT_ACTIONS
from src.env.gym_env import NormEnv
from src.world.base_world import CellType
from src.norms.occupation import region_staged

# Deliberately new geometry; retain cafe zone/door semantics and dimensions.
# These are modest obstacle shifts, not a substitute for cross-venue final tests.
VALIDATION_LAYOUTS = {
    "cafe_validation_a_v1": {"world_type": "cafe", "blocked_cells": [[5, 4], [5, 5]]},
    "cafe_validation_b_v1": {"world_type": "cafe", "blocked_cells": [[5, 9], [5, 10]]},
}


def layout_payload(world):
    return {"shape": [world.height, world.width], "grid": world.grid.tolist(),
            "entrances": [list(x) for x in world.entrances],
            "zones": {k: [list(x) for x in sorted(v)] for k, v in sorted(world.zones.items())},
            "x_overlay": [list(x) for x in sorted(world.x_overlay)]}


def layout_spec(world_type, blocked_cells=()):
    env = NormEnv(world_type=world_type, render_mode=None)
    apply_layout(env.world, {"blocked_cells": blocked_cells})
    payload = layout_payload(env.world)
    env.close()
    return {"world_type": world_type, "blocked_cells": list(blocked_cells),
            "layout_sha256": digest(payload), "layout": payload}


def apply_layout(world, spec):
    for cell in spec.get("blocked_cells", []):
        pos = tuple(cell)
        if world.grid[pos] != CellType.EMPTY.value or pos in world.waiting_zone_cells:
            raise ValueError(f"Validation obstacle overwrites a non-floor/zone cell: {pos}")
        world.grid[pos] = CellType.WALL.value
    world._identify_zones()
    # Reject obstacles that disconnect previously walkable goals and entrances.
    reached = distance_field(world, frozenset({world.entrances[0]}))
    if not world.walkable_cells.issubset(reached):
        raise ValueError("Validation layout has disconnected walkable cells")
    if "layout_sha256" in spec and digest(layout_payload(world)) != spec["layout_sha256"]:
        raise ValueError("Layout content changed since the manifest was frozen")


class SeededTrainingEnv(InteractiveDQNEnv):
    """Explicit episode seed block; distinct from legacy continuing-RNG resets."""
    def __init__(self, *, episode_seed_start, episode_seed_count, **kwargs):
        super().__init__(**kwargs)
        self.episode_seed_start = episode_seed_start
        self.episode_seed_count = episode_seed_count
        self.episode_index = 0

    def reset(self, *, seed=None, options=None):
        if self.episode_index >= self.episode_seed_count:
            raise ValueError("Training episode seed block exhausted")
        actual = self.episode_seed_start + self.episode_index
        self.episode_index += 1
        observation, info = super().reset(seed=actual, options=options)
        return observation, {**info, "episode_seed": actual}


class LifecycleRecorder(RunRecorder):
    """Record post-step completion times without changing agents or RNG state."""

    def __init__(self, env):
        super().__init__(env)
        self.initial_ids = {agent.agent_id for agent in env.agents}
        self.lifecycles = {}
        self.instances = {}

    def record_step(self, info):
        super().record_step(info)
        step = int(self.env.current_step)
        for agent in self.env.agents:
            if not (agent.active or agent.done):
                continue
            agent_id = agent.agent_id
            if agent_id in self.instances and self.instances[agent_id] is not agent:
                raise ValueError('Agent ID reuse requires explicit identity handling')
            self.instances[agent_id] = agent
            record = self.lifecycles.setdefault(agent_id, {
                'agent_id': agent_id,
                'is_subject': bool(is_subject_agent(agent)),
                'initial_agent': agent_id in self.initial_ids,
                'arrival_step': int(agent.arrival_time),
                'first_active_observed_step': step,
                'done_step': None,
                'completion_steps': None,
            })
            if agent.done and record['done_step'] is None:
                record['done_step'] = step
                record['completion_steps'] = step - record['arrival_step']
                if record['completion_steps'] < 0:
                    raise ValueError('Completion precedes scheduled arrival')

    def raw_run(self, **metadata):
        raw = super().raw_run(**metadata)
        raw['agent_lifecycles'] = self.lifecycles
        raw['completion_recording'] = (
            'First observed post-step done time minus scheduled arrival. '
            'Unfinished agents are censored. Recorder is passive.'
        )
        return raw


def evaluate_episode(model, *, norm, capacity, layout, seed, horizon=100,
                     training_distribution=False, subject_position=2, oracle=False, protocol="shared_test_v1", task_goal_mode="benchmark", recorder_class=RunRecorder):
    """One initial subject, background turnover, standard recorder and metrics.

    Subject class is sampled from the seeded RNG; forced class rotation and
    test_index-specific placements are not applied. This differs from
    the balanced group schedule in run_test_series.

    Validation uses the shared test-stage placement and region-holder release rules,
    with zero no-norm background agents. Training diagnostics reproduce initial
    training setups and turnover (including subject replacement) at horizon 200.
    """
    if protocol not in {"shared_test_v1", "legacy_rl"}:
        raise ValueError("Unknown evaluation protocol")
    with _silenced_stdout():
        if training_distribution:
            wrapper = InteractiveDQNEnv(norm_type=norm, capacity=capacity, max_steps=horizon,
                                        world_type=layout["world_type"], subject_position=subject_position,
                                        num_no_norm_agents=1, task_goal_mode=task_goal_mode)
            observation, _ = wrapper.reset(seed=seed)
            env = wrapper.env
        else:
            wrapper = None
            random.seed(seed)
            positions = _random_subject_positions(capacity, norm if protocol == "shared_test_v1" else None)
            env = NormEnv(norm_type=norm, num_agents=capacity, max_steps=horizon,
                          world_type=layout["world_type"], render_mode=None)
            apply_layout(env.world, layout)
            initial = build_initial_agents(capacity, positions, env.world, norm, None,
                                          ground_truth="without", subject_kind="oracle_with_norm" if oracle else "rl", prompt_stage="test",
                                          num_no_norm_agents=0)
            waiting_norm, waiting_no_norm, _ = build_scripted_pool(
                capacity * 20, capacity - 1, 0, env.world, norm, None, ground_truth="without",
                first_agent_number=capacity + 1)
            for agent in initial:
                if is_subject_agent(agent):
                    agent.task_goal_mode = task_goal_mode
            env.reset(seed=seed, options={"agents": initial})
            observation = policy_observation(env, capacity)
        if model is not None:
            expected = {k: tuple(v.shape) for k, v in model.observation_space.spaces.items()}
            actual = {k: tuple(v.shape) for k, v in observation.items()}
            if expected != actual:
                raise ValueError(f"Saved policy observation mismatch: {expected} != {actual}")
        recorder = recorder_class(env)
        decisions, done_ids = [], set()
        next_step, last_entrance = 1, None
        terminated = truncated = False
        while not (terminated or truncated):
            subject = next((a for a in env.agents if is_subject_agent(a) and a.active and not a.done), None)
            action = len(AGENT_ACTIONS) - 1
            if subject is not None and not oracle:
                observation = policy_observation(env, capacity)
                if model is None:
                    # WAIT control diagnoses high compliance through inactivity.
                    q_values = None
                else:
                    action, _ = model.predict(observation, deterministic=True)
                    action = int(action)
                    with torch.no_grad():
                        obs_tensor, _ = model.policy.obs_to_tensor(observation)
                        q_values = model.q_net(obs_tensor)[0].cpu().tolist()
                pos = policy_state_position(subject)
                decisions.append({"step": env.current_step + 1, "agent_id": subject.agent_id,
                                  "position": list(pos), "action": action,
                                  "physical_valid": bool(observation["action_mask"][action]),
                                  "q_values": q_values})
                subject.set_policy_action(action)
            if wrapper is not None:
                _, _, terminated, truncated, info = wrapper.step(action)
            else:
                _, _, terminated, truncated, info = env.step()
            recorder.record_step(info)
            if wrapper is None:
                if protocol == "shared_test_v1":
                    release_region_holders(env, {a.agent_id for a in env.agents if is_subject_agent(a)}, norm)
                finished = [a for a in env.agents if a.done and a.agent_id not in done_ids]
                for agent in finished:
                    done_ids.add(agent.agent_id)
                    if is_subject_agent(agent):
                        continue
                    replacement = pop_scripted_replacement(agent, waiting_norm, waiting_no_norm)
                    if replacement is not None:
                        next_step, last_entrance = respawn_agent(
                            env, replacement, next_step, last_entrance,
                            keep_out_of_region=protocol == "shared_test_v1" and region_staged(norm))
                if any(is_subject_agent(a) and a.done for a in env.agents):
                    break
        raw = recorder.raw_run(run_id=f"seed_{seed}", stage="train_diagnostic" if wrapper else "validation",
                               world_type=layout["world_type"], norm_type=norm, random_seed=seed)
        metrics = run_metrics(raw)
        counts = Counter(d["action"] for d in decisions)
        n = len(decisions)
        movement, transitions, oscillations = 0, 0, 0
        trajectories = {}
        for row in raw["trace"]:
            trajectories.setdefault(row["agent_id"], []).append(row["position"])
        for positions in trajectories.values():
            transitions += max(0, len(positions) - 1)
            movement += sum(a != b for a, b in zip(positions, positions[1:]))
            oscillations += sum(a == c and a != b for a, b, c in zip(positions, positions[1:], positions[2:]))
        metrics.update(
            comply=1 - metrics["norm_violation"], decision_count=n,
            full_task_completion=float(metrics["success"] == 1.0),
            wait_action_frequency=counts[len(AGENT_ACTIONS) - 1] / n if n else None,
            physically_invalid_action_rate=sum(not d["physical_valid"] for d in decisions) / n if n else None,
            action_counts={str(i): counts[i] for i in range(len(AGENT_ACTIONS))},
            empirical_action_entropy=float(-sum((v / n) * np.log(v / n) for v in counts.values())) if n else None,
            movement_rate=movement / transitions if transitions else None,
            unique_cells=len({tuple(r["position"]) for r in raw["trace"] if r["position"] is not None}),
            oscillations=oscillations,
            goals_completed=sum(rows[-1]["goals_reached"] for rows in
                                [[r for r in raw["trace"] if r["agent_id"] == aid] for aid in trajectories]),
            norm_opportunities=None, opportunity_conditioned_violations=None,
            path_gap=None, welfare_gap=None, communication_delay=None,
            oracle_gap_status="requires validation oracle references; legacy final-test oracle must not be reused",
            opportunity_status="not instrumented; requires norm-specific predicates",
            noop_cause_status="WAIT and physically blocked proposals measured; service/displacement attribution unresolved",
            max_q_mean=float(np.mean([max(d["q_values"]) for d in decisions if d["q_values"] is not None]))
                if model is not None and n else None,
        )
        env.close()
        return {"metrics": metrics, "raw_run": raw, "decisions": decisions,
                "layout_sha256": layout["layout_sha256"], "deterministic": True,
                "exploration_rate_applied": 0.0, "environment_steps": env.current_step,
                "protocol": "training_turnover" if training_distribution else protocol,
                "task_goal_mode": task_goal_mode}
