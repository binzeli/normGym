"""Oracle-planner baselines for the experiment pipeline.

The test stages of ``run_experiment.py`` and nothing else: the ``--num_tests``
same-world runs followed by one run per world in ``--new_map_worlds``, with no
interactive stage — an oracle planner has nothing to learn there. The new-world
block mirrors run_experiment.py's stage 3 (same seeds, same worlds from the
shared config) so an oracle run pairs with the LLM / RL run of the same index;
pass ``--new_map_worlds ''`` to skip it. The only other change is the subject
agent kind. Everything the subject meets is unchanged: the same per-norm seed
from the shared config, so the subject enters in the same slot (if the LLM
subject was the 4th agent to enter in a run, so is the oracle), against the same
scripted customers, in the same world.

Every step of every run keeps its rendered PNG under ``simulation_figures/``.

Two subject kinds, each written to its own output tree:

    oracle_with_norm     the planner knows the norm and plans around it — the
                         ceiling a norm-aware agent can reach on these runs.
    oracle_without_norm  the planner ignores the norm and takes the shortest path
                         to the same goals — the norm-blind floor.

Results follow run_experiment.py's layout, results/<agent_type>/<world>/<norm>/,
so the baselines sit beside the LLM / RL trees (results/llm/<model>/<world>/<norm>/).
Each run dumps the same raw run log as run_experiment.py, so eval.py reads them
with no changes.

Usage:
    python experiment/run/run_oracle.py --norm_type <norm> [--agent_type oracle_with_norm|oracle_without_norm|both]
    python experiment/evaluation/eval.py <printed raw_runs dir>
"""
import os
import sys
import json
import argparse
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from experiment.run.common import (
    EXPERIMENT_OUTPUT_ROOT,
    ORACLE_SUBJECT_KINDS,
    WORLD_TYPE_CHOICES,
    force_load_repo_env,
    norm_choices,
    run_test_series,
    run_with_raw_stdout_log,
)
from experiment.run.run_experiment import (
    DEFAULT_CONFIG_PATH,
    _resolve_capacity,
    _resolve_new_map_worlds,
    _resolve_seed,
    _resolve_setting,
    load_experiment_config,
)


def run_oracle_tests(capacity, test_max_steps, results_path, figures_root,
                     raw_runs_dir, norm_type, subject_kind, random_seed,
                     world_type, env_kwargs, num_tests, new_map_worlds=None):
    """Run the same-world and new-world test stages with an oracle subject.

    Every run re-seeds the global RNG with the same stage-derived seed
    run_experiment.py uses (base seed + run index, the interactive stage's base
    seed being skipped here), and draws its setup in the same order, so an oracle
    run and an LLM / RL run of the same norm and index face the identical setting:
    same entry slot for the subject, same scripted customers, same arrival times,
    same world. The subject's color also rotates through the same fixed
    orange/green/blue schedule as run_experiment.py's test stage.

    The new-world block runs one test per world in new_map_worlds, continuing the
    seed sequence past the same-world block exactly as run_experiment.py's stage 3
    does.
    """
    new_map_worlds = list(new_map_worlds or [])

    stage_results = run_test_series(
        count=num_tests,
        capacity=capacity,
        test_max_steps=test_max_steps,
        figures_root=figures_root,
        norm_type=norm_type,
        client=None,
        subject_kind=subject_kind,
        world_type=world_type,
        env_kwargs=env_kwargs,
        reference_memory_path=None,
        num_no_norm_agents=0,
        raw_runs_dir=raw_runs_dir,
        random_seed=random_seed,
        seed_offset=0,
        stage_label='test',
        keep_all_frames=True,
    )

    stage_results += run_test_series(
        count=len(new_map_worlds),
        capacity=capacity,
        test_max_steps=test_max_steps,
        figures_root=figures_root,
        norm_type=norm_type,
        client=None,
        subject_kind=subject_kind,
        world_type=new_map_worlds,
        env_kwargs=env_kwargs,
        reference_memory_path=None,
        num_no_norm_agents=0,
        raw_runs_dir=raw_runs_dir,
        random_seed=random_seed,
        seed_offset=num_tests,
        stage_label='new_map_test',
        keep_all_frames=True,
    )

    result = {
        'mode': 'oracle',
        'agent_type': subject_kind,
        'random_seed': random_seed,
        'norm_type': norm_type,
        'world_type': world_type,
        'new_map_worlds': new_map_worlds,
        'capacity': capacity,
        'test_max_steps': test_max_steps,
        'figures_root': figures_root,
        'raw_runs_dir': raw_runs_dir,
        'stages': stage_results,
    }

    with open(results_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n[OK] Oracle results ({num_tests} same-world tests + "
          f"{len(new_map_worlds)} new-world tests) saved to JSON: {results_path}")
    print(f"[OK] Step figures saved under: {figures_root}")
    print(f"[OK] Raw run logs saved to: {raw_runs_dir}")
    print(f"[OK] Compute metrics with: python experiment/evaluation/eval.py {raw_runs_dir}")
    return result


def run_one_baseline(subject_kind, args, capacity, test_max_steps, num_tests,
                     world_type, random_seed, env_kwargs, new_map_worlds):
    """Set up the output tree for one oracle subject kind and run its test stage."""
    base_output_dir = (
        args.output_dir
        if os.path.isabs(args.output_dir)
        else os.path.join(EXPERIMENT_OUTPUT_ROOT, args.output_dir)
    )
    # Same layout run_experiment.py writes: results/<agent_type>/<world>/<norm>/,
    # so oracle baselines sit beside the LLM / RL results
    # (results/llm/<model>/<world>/<norm>/) and eval.py reads either unchanged.
    base_dir = os.path.join(base_output_dir, subject_kind, world_type, args.norm_type)
    figures_root = os.path.join(base_dir, 'simulation_figures' + args.figures_suffix)
    logs_dir = os.path.join(base_dir, 'simulation_log')
    raw_runs_dir = os.path.join(base_dir, 'raw_runs')

    if os.path.exists(figures_root):
        shutil.rmtree(figures_root)
        print("[OK] Cleared existing simulation_figures directory")

    # eval.py reads every run log in raw_runs_dir, so stale runs from a previous
    # experiment must not survive into this one's metrics.
    if os.path.exists(raw_runs_dir):
        shutil.rmtree(raw_runs_dir)
        print("[OK] Cleared existing raw_runs directory")

    os.makedirs(figures_root, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(raw_runs_dir, exist_ok=True)

    logs_path = os.path.join(logs_dir, f"oracle_results_{args.norm_type}.json")

    print("\n" + "=" * 70)
    print(f"ORACLE BASELINE — subject = {subject_kind}")
    print("=" * 70)
    print(f"[OK] Norm: {args.norm_type} (seed {random_seed}, same setting as every other subject kind)")
    print(f"[OK] Same-world tests: {num_tests} in {world_type}; new-world tests: {new_map_worlds}")
    print(f"[OK] Output: {base_dir}\n")

    txt_log_path = os.path.splitext(logs_path)[0] + '.txt'
    run_with_raw_stdout_log(
        txt_log_path,
        run_oracle_tests,
        capacity=capacity,
        test_max_steps=test_max_steps,
        results_path=logs_path,
        figures_root=figures_root,
        raw_runs_dir=raw_runs_dir,
        norm_type=args.norm_type,
        subject_kind=subject_kind,
        random_seed=random_seed,
        world_type=world_type,
        env_kwargs=env_kwargs,
        num_tests=num_tests,
        new_map_worlds=new_map_worlds,
    )
    print(f"[OK] Simulation log saved to TXT: {txt_log_path}")
    return raw_runs_dir


def main():
    parser = argparse.ArgumentParser(
        description='Run the oracle-planner baselines: the same-world and new-world '
                    'test runs of run_experiment.py (no interactive stage) with an '
                    'oracle planner as the subject agent.'
    )
    parser.add_argument('--norm_type', type=str, required=True,
                        choices=norm_choices(),
                        help='Norm type to use')
    parser.add_argument('--agent_type', type=str, default='both',
                        choices=list(ORACLE_SUBJECT_KINDS) + ['both'],
                        help='Oracle subject kind: oracle_with_norm plans around the '
                             'norm, oracle_without_norm ignores it, both runs the two '
                             'baselines back-to-back (default: both)')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH,
                        help='Path to the shared experiment config JSON holding the '
                             'per-norm seeds and fixed settings (default: '
                             'experiment/run/experiment_config.json)')
    parser.add_argument('--capacity', type=int, default=None,
                        help='Max agents on the map at once (default: from config, else 8)')
    parser.add_argument('--test_max_steps', type=int, default=None,
                        help='Maximum steps per test run (default: from config, else 150)')
    parser.add_argument('--num_tests', type=int, default=None,
                        help='Number of same-world test runs (default: from config, else 3)')
    parser.add_argument('--random_seed', type=int, default=None,
                        help='Base seed for the stage setups; the same seed gives every '
                             'subject kind (llm, rl, oracle_*) the identical setting '
                             '(default: per-norm seed from config)')
    parser.add_argument('--world_type', type=str, default=None,
                        choices=WORLD_TYPE_CHOICES,
                        help='World type for the same-world tests (default: from config, else cafe)')
    parser.add_argument('--new_map_worlds', type=str, nargs='*', default=None,
                        help="Worlds for the new-world test stage, one run each "
                             "(default: this norm's worlds from the config). Pass "
                             "with no values to skip the new-world stage.")
    parser.add_argument('--figures_suffix', type=str, default='',
                        help='Optional suffix for figures directory')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results (default: results)')
    parser.add_argument('--no_sprites', action='store_true',
                        help='Disable PNG sprite rendering and use matplotlib fallback')
    parser.add_argument('--isometric', action='store_true',
                        help='Use 2.5D isometric rendering instead of flat top-down view')
    parser.add_argument('--entrance_side', type=str, default='bottom',
                        choices=['bottom', 'left', 'right'],
                        help='Which side the entrance is on (default: bottom)')
    parser.add_argument('--map_type', type=str, default='default',
                        choices=['default', 'navigation', 'affordance'],
                        help='Map layout type (default: default)')
    parser.add_argument('--interpolation_steps', type=int, default=0,
                        help='Number of intermediate frames between simulation steps')
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    capacity = _resolve_capacity(args.capacity, config, args.norm_type)
    test_max_steps = _resolve_setting(args.test_max_steps, config, 'test_max_steps', 150)
    num_tests = _resolve_setting(args.num_tests, config, 'num_tests', 3)
    world_type = _resolve_setting(args.world_type, config, 'world_type', 'cafe')
    random_seed = _resolve_seed(args.random_seed, config, args.norm_type)

    # An explicit empty --new_map_worlds skips the stage; omitting the flag falls
    # back to this norm's worlds in the config.
    new_map_worlds = (
        [] if args.new_map_worlds == []
        else _resolve_new_map_worlds(args.new_map_worlds, config, args.norm_type,
                                     world_type, random_seed)
    )
    unknown = [w for w in new_map_worlds if w not in WORLD_TYPE_CHOICES]
    if unknown:
        raise ValueError(f"Unknown --new_map_worlds: {unknown}")
    if world_type in new_map_worlds:
        raise ValueError(
            f"--new_map_worlds must not include the main world_type ({world_type})"
        )

    if num_tests < 0:
        raise ValueError("num_tests must be non-negative")
    if capacity < 3:
        raise ValueError("capacity must be at least 3 so the subject can enter in 3rd place or later")

    force_load_repo_env()

    env_kwargs = {
        'use_sprites': not args.no_sprites,
        'use_isometric': args.isometric,
        'entrance_side': args.entrance_side,
        'map_type': args.map_type,
        'interpolation_steps': args.interpolation_steps,
    }

    subject_kinds = (
        list(ORACLE_SUBJECT_KINDS) if args.agent_type == 'both' else [args.agent_type]
    )
    raw_runs_dirs = {
        subject_kind: run_one_baseline(
            subject_kind, args, capacity, test_max_steps, num_tests,
            world_type, random_seed, env_kwargs, new_map_worlds,
        )
        for subject_kind in subject_kinds
    }

    print("\n" + "=" * 70)
    print("DONE — compute the metrics with:")
    for subject_kind, raw_runs_dir in raw_runs_dirs.items():
        print(f"  {subject_kind}:")
        print(f"    python experiment/evaluation/eval.py {raw_runs_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
