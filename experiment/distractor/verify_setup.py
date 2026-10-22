"""Check that a distractor run differs from the baseline only where it is meant to.

For every norm of the chosen mode and every stage setting (interactive /
same-world test in the main world, test in the norm's new-map world), the
initial batch plus one respawn is built twice from the same seed — once with
the pipeline as is, once with the distractor installed — and compared:
  exit_right / exit_bottom : ids, classes, entrances, arrival times, norms, the
      subject's goals and the RNG state afterwards are identical; each background
      agent differs only in its exit door, which is the distractor door.
  move1_wait1 : everything above is identical, goals included; a background
      oracle that moved last turn now waits.
  free exit   : for that norm only, the last goal is the free one, any door
      completes it, and the prompt text no longer names a door; another norm is
      answered exactly as before.
No LLM calls, no rendering.

    python experiment/distractor/verify_setup.py                 # auto (the 8 GPT norms)
    DISTRACTOR=exit_bottom python experiment/distractor/verify_setup.py
"""
import os
import random
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from types import SimpleNamespace

from experiment.agents import llm_agent as llm_agent_module
from experiment.distractor import run_distractor as rd
from experiment.run import common, run_experiment
from experiment.utils import agent_goals
from src.agents.oracle_agent import OracleAgent
from src.env import gym_env as gym_env_module
from src.env.actions import Action
from src.env.gym_env import WORLD_CLASSES
from src.norms import occupation

EXIT_BOTTOM_NORMS = [
    'bend_left1_line', 'bend_left2_line', 'edge_left_line',
    'movement_move1_wait1', 'movement_move3_wait1', 'movement_no_even_turn',
    'movement_no_turn_mod3_0', 'movement_no_turn_mod3_1',
    'occupation_door_A_forbidden', 'occupation_door_C_forbidden',
    'occupation_seat_bottom_left_only', 'occupation_seat_bottom_right_only',
    'xcell_forbidden',
]
UNRELATED_NORM = 'occupation_door_C_forbidden'

PATCHED = [(common, 'make_agent'), (common, 'respawn_agent'),
           (run_experiment, 'make_agent'), (run_experiment, 'respawn_agent'),
           (OracleAgent, 'get_action'),
           (occupation, 'free_choice_goal_index'), (occupation, 'free_goal_satisfied'),
           (gym_env_module, 'free_choice_goal_index'),
           (llm_agent_module, 'free_choice_goal_index'),
           (agent_goals, '_entry')]
ORIGINALS = [(owner, name, getattr(owner, name)) for owner, name in PATCHED]


def uninstall():
    for owner, name, fn in ORIGINALS:
        setattr(owner, name, fn)


def snapshot(agent):
    return {
        'id': agent.agent_id,
        'subject': common.is_subject_agent(agent),
        'class': agent.agent_class,
        'position': tuple(agent.current_position),
        'arrival': agent.arrival_time,
        'norm': agent.norm,
        'targets': [tuple(t) for t in agent.target_positions],
    }


def build(norm, world_type, stage, seed, capacity, num_no_norm):
    random.seed(seed)
    world = WORLD_CLASSES[world_type](map_type='default')
    world.norm_type = norm
    agents = common.build_initial_agents(
        capacity, {capacity // 2}, world, norm, None, 'without',
        subject_kind='oracle_with_norm', prompt_stage=stage,
        num_no_norm_agents=num_no_norm, subject_class='green',
        test_index=1 if stage == 'test' else None,
    )
    # One respawn of a background agent, as the stages do when one leaves.
    env = SimpleNamespace(world=world, norm_type=norm, agents=list(agents),
                          current_step=5, num_agents=len(agents),
                          add_agent=lambda a: None)
    background = next(a for a in agents if not common.is_subject_agent(a))
    common.respawn_agent(env, background, 0, None)
    return world, agents, [snapshot(a) for a in agents], random.getstate()


def free_exit_answers(norm, world):
    """What the three free-exit decision points currently say about `norm`."""
    doors = [tuple(d) for d in world.entrances]
    floor = next((r, c) for r in range(world.height) for c in range(world.width)
                 if (r, c) not in doors)
    return {
        'index': [m.free_choice_goal_index(norm, 2)
                  for m in (occupation, gym_env_module, llm_agent_module)],
        'doors_complete': [occupation.free_goal_satisfied(norm, d, 'C1', 'green', {}, world)
                           for d in doors],
        'floor_completes': occupation.free_goal_satisfied(norm, floor, 'C1', 'green', {}, world),
        'free_steps': [agent_goals.is_free_choice_step(norm, i) for i in (0, 1)],
        'text': agent_goals.all_goals_text(norm, [(1, 1), (2, 2)]),
    }


def check_free_exit(norm, world, free_exit, before, unrelated_before):
    problems = []
    now = free_exit_answers(norm, world)
    if not free_exit:
        return [] if now == before else ['free-exit answers changed without free exit']
    if now['index'] != [1, 1, 1]:
        problems.append(f"free goal index {now['index']}")
    if not all(now['doors_complete']) or now['floor_completes']:
        problems.append('any-door completion wrong')
    if now['free_steps'] != [False, True]:
        problems.append(f"free steps {now['free_steps']}")
    if rd.FREE_EXIT_GOAL_TEXT not in now['text'] or '(2, 2)' in now['text'] \
            or '(1, 1)' not in now['text']:
        problems.append(f"goal text {now['text']!r}")
    if free_exit_answers(UNRELATED_NORM, world) != unrelated_before:
        problems.append(f'{UNRELATED_NORM} answers changed')
    return problems


def check_rhythm(world, agents):
    world.agents = agents
    background = next(a for a in agents if not common.is_subject_agent(a))
    background.done, background.current_step, background.move_steps = False, 7, {6}
    if background.get_action(world, {}, {}) != Action.WAIT:
        return ['background oracle that just moved did not wait']
    return []


def main():
    mode = os.environ.get('DISTRACTOR', 'auto')
    norms = list(rd.NORM_DISTRACTORS) if mode == 'auto' else EXIT_BOTTOM_NORMS
    config = run_experiment.load_experiment_config(run_experiment.DEFAULT_CONFIG_PATH)
    failures = 0
    for norm in norms:
        seed = config['norm_seeds'][norm]
        capacity = config.get('norm_capacity', {}).get(norm, config['capacity'])
        new_world = config['norm_new_map_worlds'][norm][0]
        for world_type, stage in ((config['world_type'], 'interactive'),
                                  (config['world_type'], 'test'),
                                  (new_world, 'test')):
            uninstall()
            base_world, _, base, base_rng = build(norm, world_type, stage, seed, capacity,
                                                  config['num_no_norm_agents'])
            answers_before = free_exit_answers(norm, base_world)
            unrelated_before = free_exit_answers(UNRELATED_NORM, base_world)

            distractor, free_exit = rd.install(norm, mode, False)
            world, agents, dist, dist_rng = build(norm, world_type, stage, seed, capacity,
                                                  config['num_no_norm_agents'])
            side = rd.EXIT_SIDES.get(distractor)
            door = rd.distractor_door(world, side) if side else None

            problems = []
            if base_rng != dist_rng:
                problems.append('RNG state differs')
            for b, d in zip(base, dist):
                if b['subject'] or door is None:
                    if b != d:
                        problems.append(f"{b['id']} changed")
                    continue
                if {k: v for k, v in b.items() if k != 'targets'} != \
                        {k: v for k, v in d.items() if k != 'targets'}:
                    problems.append(f"{b['id']} changed beyond its goals")
                if b['targets'][:-1] != d['targets'][:-1]:
                    problems.append(f"{b['id']} first goal changed")
                if d['targets'][-1] != door:
                    problems.append(f"{b['id']} exits {d['targets'][-1]}, not {door}")
            problems += check_free_exit(norm, world, free_exit, answers_before, unrelated_before)
            if distractor == 'move1_wait1':
                problems += check_rhythm(world, agents)
            uninstall()
            if free_exit_answers(norm, world) != answers_before:
                problems.append('uninstall did not restore the pipeline')

            what = distractor + (f' -> {door}' if door else '') + (' +free_exit' if free_exit else '')
            status = 'FAIL ' + '; '.join(problems) if problems else 'ok'
            failures += bool(problems)
            print(f"{norm:36s} {world_type:15s} {stage:11s} {what:32s} {status}")
    print('ALL OK' if not failures else f'{failures} FAILURE(S)')
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
