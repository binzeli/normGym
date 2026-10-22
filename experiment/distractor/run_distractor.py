"""Distractor experiment: every background agent shows one extra regularity that
is NOT the run's norm, and the question is whether the subject mistakes it for one.

The run is the unmodified main pipeline (experiment/run/run_experiment.py:
same CLI, same experiment_config.json, same seeds, stages, agents and eval); the
distractor is applied in memory and nothing in src/ or experiment/run/ is
edited. The checker only knows the run's norm_type, so nobody is ever penalised
for the distractor regularity.

DISTRACTOR (environment variable) picks the regularity:
  auto         (default) per norm, from NORM_DISTRACTORS below; a norm that is not
               listed there is refused rather than run without a distractor.
  exit_right / exit_bottom
               after a background agent's goals are drawn, its exit door (the
               last goal) is replaced by the world's right / bottom door. The
               agents still enter through any door and follow the real norm.
               Nothing is drawn from the RNG by the override, so with the same
               seed the run is identical to the baseline (layout, spawns,
               classes, the subject's goals) except where the background exits.
  move1_wait1  every background agent also keeps the movement_move1_wait1 rhythm
               (never two moves in a row) on top of the real norm. Waiting is
               allowed by every norm, so no background agent violates the real
               norm. Goals and spawns are untouched; only the pace differs.
  none         no distractor; only meaningful with FREE_EXIT=1 (the control for
               the free-exit prompt change below).

Free exit: under the norms in FREE_EXIT_NORMS (auto mode), or under any
non-occupation norm when FREE_EXIT=1, the subject's exit door becomes its own
choice, exactly as the door norms already do it: the prompt says "leave through
a door" instead of naming one, and reaching any door completes the goal. This
lets the subject act on an exit-door distractor. The hidden door is still drawn,
so the RNG stream is unchanged.

Results go to experiment/results_distractor/ (experiment/results_distractor_control/
for DISTRACTOR=none), mirroring experiment/results/; each norm directory gets a
distractor.json saying what was applied. run_experiment.main() wipes the norm's
output directory before a run, so this launcher refuses the baseline results tree.

Usage (same flags as run_experiment.py):
    python experiment/distractor/run_distractor.py --norm_type <norm> \
        --agent_type memory_llm --provider openrouter --model openai/gpt-5.6-sol
"""
import copy
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from experiment.agents import llm_agent as llm_agent_module
from experiment.run import common, run_experiment
from experiment.utils import agent_goals
from src.agents.oracle_agent import OracleAgent
from src.env import gym_env as gym_env_module
from src.env.actions import Action
from src.norms import occupation
from src.norms.navigation import _rhythm_requires_rest
from src.norms.occupation import door_side, is_occupation_norm

# auto mode: the norms GPT-5.6 solved 100% violation-free in both test stages
# (movement_move1_wait1, the distractor rhythm itself, excluded).
NORM_DISTRACTORS = {
    'movement_no_turn_mod3_0': 'exit_right',
    'movement_move2_wait1': 'exit_right',
    'movement_move3_wait1': 'exit_right',
    'occupation_door_A_forbidden': 'move1_wait1',
    'occupation_door_B_forbidden': 'move1_wait1',
    'occupation_door_class_specific_BCA': 'move1_wait1',
    'occupation_door_class_specific_CAB': 'move1_wait1',
    'occupation_seat_no_bottom_left': 'move1_wait1',
}
FREE_EXIT_NORMS = {n for n, d in NORM_DISTRACTORS.items() if d == 'exit_right'}
FREE_EXIT_GOAL_TEXT = "Goal 2: Once you are finished, leave through a door."

EXIT_SIDES = {'exit_right': 'right', 'exit_bottom': 'bottom'}
DISTRACTOR_MODES = ('auto', 'exit_right', 'exit_bottom', 'move1_wait1', 'none')
DISTRACTOR_RHYTHM = 'movement_move1_wait1'
DISTRACTOR_OUTPUT_DIR = 'results_distractor'
CONTROL_OUTPUT_DIR = 'results_distractor_control'
BASELINE_OUTPUT_DIR = os.path.join(common.EXPERIMENT_OUTPUT_ROOT, 'results')


def resolve(norm, mode=None, free_exit_flag=None):
    """(distractor, free_exit) applied to a run of `norm`."""
    mode = os.environ.get('DISTRACTOR', 'auto') if mode is None else mode
    if free_exit_flag is None:
        free_exit_flag = os.environ.get('FREE_EXIT', '') == '1'
    if mode not in DISTRACTOR_MODES:
        sys.exit(f"[FATAL] DISTRACTOR must be one of {DISTRACTOR_MODES}, got {mode!r}")
    if mode == 'auto':
        if norm not in NORM_DISTRACTORS:
            sys.exit(f"[FATAL] no distractor is assigned to {norm!r} (see NORM_DISTRACTORS); "
                     "set DISTRACTOR explicitly")
        return NORM_DISTRACTORS[norm], norm in FREE_EXIT_NORMS
    if free_exit_flag and is_occupation_norm(norm):
        sys.exit(f"[FATAL] FREE_EXIT=1 is for non-occupation norms; {norm!r} already "
                 "has its own free-choice goal")
    if mode == 'none' and not free_exit_flag:
        sys.exit("[FATAL] DISTRACTOR=none without FREE_EXIT=1 is just the baseline pipeline")
    return mode, free_exit_flag


# ----------------------------------------------------------------- exit door

def distractor_door(world, side):
    doors = [tuple(d) for d in getattr(world, 'entrances', [])
             if door_side(world, tuple(d)) == side]
    if len(doors) != 1:
        raise RuntimeError(
            f"{type(world).__name__} has {len(doors)} {side} door(s); "
            "the distractor needs exactly one"
        )
    return doors[0]


def install_exit_door(side):
    make_agent_, respawn_agent_ = common.make_agent, common.respawn_agent

    def send_out(agent, world):
        if common.is_subject_agent(agent) or not agent.target_positions:
            return
        agent.target_positions[-1] = distractor_door(world, side)
        # Under the door norms an oracle re-picks its exit door once (see
        # OracleAgent.get_action); latch it so the distractor door stands.
        agent._occupation_door_fixed = True

    def make_agent(agent_id, world, *args, **kwargs):
        agent = make_agent_(agent_id, world, *args, **kwargs)
        send_out(agent, world)
        return agent

    def respawn_agent(env, agent, *args, **kwargs):
        result = respawn_agent_(env, agent, *args, **kwargs)
        send_out(agent, env.world)
        return result

    # run_experiment imported these names from common, so both modules hold
    # their own reference; every call site in either must see the wrapper.
    _patch((common, run_experiment), 'make_agent', make_agent)
    _patch((common, run_experiment), 'respawn_agent', respawn_agent)


# -------------------------------------------------------------------- rhythm

def install_rhythm():
    get_action_ = OracleAgent.get_action

    def get_action(self, world, *args, **kwargs):
        """A background oracle rests whenever the distractor rhythm says so, and
        otherwise acts exactly as it would have. This is how an oracle follows
        the real movement_move1_wait1 too: its planner drops the moves the
        rhythm forbids, which leaves WAIT (agent.move_steps is kept by the env
        under every norm)."""
        if (not self.done and not common.is_subject_agent(self)
                and _rhythm_requires_rest(DISTRACTOR_RHYTHM, self.agent_id, world)):
            return Action.WAIT
        return get_action_(self, world, *args, **kwargs)

    OracleAgent.get_action = get_action


# ----------------------------------------------------------------- free exit

def install_free_exit(norm):
    """Make the exit door of `norm` the interactive subject's free choice, the
    way the door norms have it. Three places decide that per norm: which goal
    is free, when it counts as reached, and the goal text in the prompt."""
    goal_index_, satisfied_, entry_ = (occupation.free_choice_goal_index,
                                       occupation.free_goal_satisfied,
                                       agent_goals._entry)

    def free_choice_goal_index(agent_norm, num_targets):
        if agent_norm == norm and num_targets >= 1:
            return num_targets - 1
        return goal_index_(agent_norm, num_targets)

    def free_goal_satisfied(agent_norm, pos, *args, **kwargs):
        if agent_norm == norm:
            world = kwargs.get('world', args[-1] if args else None)
            return tuple(pos) in {tuple(d) for d in getattr(world, 'entrances', [])}
        return satisfied_(agent_norm, pos, *args, **kwargs)

    def _entry(agent_norm):
        entry = entry_(agent_norm)
        if agent_norm != norm:
            return entry
        entry = copy.deepcopy(entry)
        last = max(entry['goals'], key=lambda g: g['step'])
        last['free_choice'] = True
        last['description'] = FREE_EXIT_GOAL_TEXT
        return entry

    # gym_env and llm_agent imported free_choice_goal_index by name;
    # free_goal_settled looks free_goal_satisfied up in occupation itself.
    _patch((occupation, gym_env_module, llm_agent_module),
           'free_choice_goal_index', free_choice_goal_index)
    _patch((occupation,), 'free_goal_satisfied', free_goal_satisfied)
    _patch((agent_goals,), '_entry', _entry)


def _patch(modules, name, replacement):
    for module in modules:
        if not callable(getattr(module, name, None)):
            raise RuntimeError(
                f"{module.__name__}.{name} not found; the pipeline changed "
                "and the distractor would silently not apply"
            )
        setattr(module, name, replacement)


def install(norm, mode=None, free_exit_flag=None):
    distractor, free_exit = resolve(norm, mode, free_exit_flag)
    if distractor in EXIT_SIDES:
        install_exit_door(EXIT_SIDES[distractor])
    elif distractor == 'move1_wait1':
        install_rhythm()
    if free_exit:
        install_free_exit(norm)
    return distractor, free_exit


def _record_setup(setup):
    """Write distractor.json into the norm directory once main() has created it."""
    run_experiment_ = run_experiment.run_experiment

    def run_with_record(*args, **kwargs):
        norm_dir = os.path.dirname(os.path.dirname(kwargs['results_path']))
        with open(os.path.join(norm_dir, 'distractor.json'), 'w', encoding='utf-8') as f:
            json.dump(setup, f, indent=1)
        return run_experiment_(*args, **kwargs)

    _patch((run_experiment,), 'run_experiment', run_with_record)


def _arg(argv, flag):
    value = None
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            value = argv[i + 1]
        elif arg.startswith(flag + '='):
            value = arg.split('=', 1)[1]
    return value


def _force_output_dir(argv, default_dir):
    """Default --output_dir to the distractor tree and never allow the baseline's."""
    output_dir = _arg(argv, '--output_dir')
    if output_dir is None:
        output_dir = default_dir
        argv = argv + ['--output_dir', output_dir]
    resolved = (output_dir if os.path.isabs(output_dir)
                else os.path.join(common.EXPERIMENT_OUTPUT_ROOT, output_dir))
    if os.path.realpath(resolved) == os.path.realpath(BASELINE_OUTPUT_DIR):
        sys.exit(f"[FATAL] --output_dir resolves to the baseline results tree "
                 f"({BASELINE_OUTPUT_DIR}); a run would wipe the baseline norm directory.")
    return argv


if __name__ == "__main__":
    argv = sys.argv[1:]
    norm = _arg(argv, '--norm_type')
    if norm is None:
        sys.exit("[FATAL] --norm_type is required")
    distractor, free_exit = install(norm)
    sys.argv = [sys.argv[0]] + _force_output_dir(
        argv, CONTROL_OUTPUT_DIR if distractor == 'none' else DISTRACTOR_OUTPUT_DIR)
    setup = {
        'norm_type': norm,
        'distractor': distractor,
        'exit_side': EXIT_SIDES.get(distractor),
        'rhythm': DISTRACTOR_RHYTHM if distractor == 'move1_wait1' else None,
        'subject_free_exit': free_exit,
    }
    _record_setup(setup)
    print(f"[OK] Distractor for {norm}: {distractor}"
          + (" + subject exit door is a free choice" if free_exit else ""))
    try:
        run_experiment.main()
    except run_experiment.InsufficientCreditsError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        sys.exit(run_experiment.EXIT_CODE_NO_CREDITS)
