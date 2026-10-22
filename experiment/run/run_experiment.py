"""Main experiment pipeline.

One experiment = three stages, run back-to-back for a single subject agent
kind (--agent_type llm | memory_llm | rl):

1. Interactive stage — the subject agent lives among scripted (oracle)
   customers that come and go and are replaced. When the subject finishes its
   goals it re-enters at a new entrance, until max_steps is reached.
2. Same-world test runs (--num_tests) — scripted customers still come and go,
   but each run ends as soon as the subject agent finishes all its goals.
   An LLM subject sees only the interactive-stage memory as reference.
3. New-world test runs — same protocol as stage 2, but each run happens on a
   different world (--new_map_worlds). The LLM subject still carries only the
   interactive-stage memory.

Usage:
    python experiment/run/run_experiment.py --norm_type <norm> [--agent_type llm|memory_llm|rl] ...

Zero-shot runs (stages 2-3 only, no interactive learning) use
experiment/run/zero_shot/run_zero_shot.py, which shares this module's code.
"""
import os
import sys
import random
import json
import argparse
import shutil
from collections import deque

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from experiment.run.common import (
    EXPERIMENT_OUTPUT_ROOT,
    SUBJECT_KINDS,
    WORLD_TYPE_CHOICES,
    RunRecorder,
    _random_subject_positions,
    build_initial_agents,
    build_scripted_pool,
    filter_subject_episode_frames,
    force_load_repo_env,
    is_llm_subject_kind,
    is_subject_agent,
    make_agent,
    make_env,
    norm_choices,
    pop_scripted_replacement,
    respawn_agent,
    run_test_series,
    run_with_raw_stdout_log,
    subject_class_schedule,
    summarize_episode,
    summarize_norm_violations,
    write_raw_run,
)
from experiment.utils.client import create_llm_client, InsufficientCreditsError


def run_interactive_simulation(env, initial_agents, waiting_subject,
                               waiting_norm_customer, waiting_no_norm_customer,
                               figures_root, norm_type):
    """Interactive stage: everyone (subject included) is replaced when done."""
    print("=" * 70)
    print(f"Interactive Simulation - WITH {norm_type.upper()} Norm")
    print("=" * 70)
    print(f"Figures will be saved to: {figures_root}/\n")

    obs, info = env.reset(options={'agents': initial_agents})
    env.output_dir = figures_root
    env.render()

    recorder = RunRecorder(env)
    done_agent_ids: set = set()
    terminated = False
    truncated = False
    step_count = 0
    next_replacement_step = 1
    last_spawn_entrance = None

    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step()
        step_count = info['step']
        recorder.record_step(info)
        env.render()

        newly_done = [
            agent for agent in env.agents
            if agent.done and agent.agent_id not in done_agent_ids
        ]

        for agent in newly_done:
            done_agent_ids.add(agent.agent_id)
            if is_subject_agent(agent):
                new_agent = waiting_subject.popleft() if waiting_subject else None
            else:
                new_agent = pop_scripted_replacement(
                    agent, waiting_norm_customer, waiting_no_norm_customer
                )
            if new_agent is None:
                continue

            next_replacement_step, last_spawn_entrance = respawn_agent(
                env, new_agent, next_replacement_step, last_spawn_entrance
            )

    episode = summarize_episode(env, done_agent_ids, step_count, figures_root)
    episode['recorder'] = recorder
    return episode


def run_interactive_stage(capacity, max_steps, figures_root, norm_type, client,
                          subject_kind, subject_positions, ground_truth,
                          world_type, env_kwargs,
                          num_no_norm_agents, raw_runs_dir, random_seed=42):
    """Stage 1: build env + replacement pools and run the interactive simulation.

    The subject and each of its replacements walk down
    subject_class_schedule(random_seed, ...), so consecutive subject
    incarnations always have different colors and the first three between them
    cover orange / green / blue.
    """
    num_subjects = len(subject_positions)
    background_slots = capacity - num_subjects
    if num_no_norm_agents < 0:
        raise ValueError("num_no_norm_agents must be non-negative")
    if num_no_norm_agents > background_slots:
        raise ValueError(
            "num_no_norm_agents cannot exceed the number of non-subject "
            f"background slots ({background_slots})"
        )
    pool_size = capacity * 20
    subject_pool_size = pool_size * num_subjects // capacity
    customer_pool_size = pool_size - subject_pool_size

    figures_dir = os.path.join(figures_root, 'interactive')
    env = make_env(norm_type, figures_dir, capacity, world_type, max_steps, **env_kwargs)

    # One color per subject incarnation: index 0 is the initial subject, the
    # rest are handed out to the replacement pool in pop order below.
    subject_classes = subject_class_schedule(random_seed, 1 + subject_pool_size)

    initial_agents = build_initial_agents(
        capacity, subject_positions, env.world, norm_type, client, ground_truth,
        subject_kind=subject_kind,
        prompt_stage='interactive',
        max_steps=max_steps,
        memory_filename="interactive_stage.json",
        num_no_norm_agents=num_no_norm_agents,
        subject_class=subject_classes[0],
    )
    subject_ids = {a.agent_id for a in initial_agents if is_subject_agent(a)}
    no_norm_ids = {
        a.agent_id for a in initial_agents
        if not is_subject_agent(a) and a.norm is None
    }
    print(f"[INFO] Initial agents: {[a.agent_id for a in initial_agents]}")
    print(f"[INFO] Subject-controlled ({subject_kind}): {sorted(subject_ids)}")
    print(f"[INFO] No-norm background agents: {sorted(no_norm_ids)}")
    print(f"[INFO] Subject entry position: {sorted(subject_positions)[0] + 1}")
    print(f"[INFO] Subject color order (initial + replacements): "
          f"{', '.join(subject_classes[:6])}, ...")
    print(f"[INFO] Ground truth: {ground_truth}")

    agent_id_counter = capacity + 1
    waiting_subject: deque = deque()
    for pool_index in range(subject_pool_size):
        agent = make_agent(
            f"C{agent_id_counter}",
            env.world,
            norm_type,
            client,
            subject_kind=subject_kind,
            ground_truth=ground_truth,
            prompt_stage='interactive',
            max_steps=max_steps,
            memory_filename="interactive_stage.json",
            subject_class=subject_classes[pool_index + 1],
        )
        agent_id_counter += 1
        waiting_subject.append(agent)

    waiting_norm_customer, waiting_no_norm_customer, agent_id_counter = build_scripted_pool(
        customer_pool_size, background_slots, num_no_norm_agents,
        env.world, norm_type, client, ground_truth,
        first_agent_number=agent_id_counter,
    )

    print(
        f"[INFO] Mode=interactive: max_steps={max_steps}, "
        f"pool={len(waiting_subject)} subject agents and "
        f"{len(waiting_norm_customer)} norm Customer agents and "
        f"{len(waiting_no_norm_customer)} no-norm Customer agents\n"
    )

    if os.path.exists(figures_dir):
        shutil.rmtree(figures_dir)
    os.makedirs(figures_dir)

    episode = run_interactive_simulation(
        env=env,
        initial_agents=initial_agents,
        waiting_subject=waiting_subject,
        waiting_norm_customer=waiting_norm_customer,
        waiting_no_norm_customer=waiting_no_norm_customer,
        figures_root=figures_dir,
        norm_type=norm_type,
    )
    filter_subject_episode_frames(figures_dir, episode)

    raw_run_path = write_raw_run(
        episode, raw_runs_dir, 'interactive',
        stage='interactive',
        agent_type=subject_kind,
        norm_type=norm_type,
        world_type=world_type,
        ground_truth=ground_truth,
        capacity=capacity,
        max_steps=max_steps,
        subject_position=sorted(subject_positions)[0] + 1,
        num_no_norm_agents=num_no_norm_agents,
    )

    return {
        'stage': 'interactive',
        'ground_truth': ground_truth,
        'norm_type': norm_type,
        'world_type': world_type,
        'capacity': capacity,
        'subject_position': sorted(subject_positions)[0] + 1,
        'figures_dir': figures_dir,
        'raw_run_path': raw_run_path,
        'steps': episode['steps'],
        'num_collisions': episode['num_collisions'],
        'norm_violations': summarize_norm_violations(episode['episode_log']),
        'agent_rewards': episode['agent_rewards'],
        'agent_classes': episode['agent_classes'],
        'final_states': episode['final_states'],
    }


def run_experiment(capacity, max_steps, results_path, figures_root, raw_runs_dir,
                   norm_type, client, subject_kind,
                   ground_truth, random_seed=42, world_type='cafe',
                   env_kwargs=None,
                   num_tests=3, test_max_steps=150,
                   new_map_worlds=None, num_no_norm_agents=0, mode='experiment'):
    """Run the test protocol, optionally preceded by interactive learning.

    Every run dumps its raw per-step log to raw_runs_dir; metrics are derived
    from those files by experiment/evaluation/eval.py, which this pipeline never
    calls itself.

    Every stage's setup (subject entry slot, agent colors, entrances, arrival
    times, replacement-pool composition, test worlds) is drawn from the global
    RNG immediately after re-seeding it with a stage-derived seed. Nothing
    before a stage's setup depends on the subject kind, so with the same
    --random_seed and --norm_type, LLM / RL / any other subject experiences
    the identical setting in every stage; only in-run randomness (replacement
    timing and entrances) follows from the subject's own behavior.

    num_no_norm_agents applies to the interactive stage only; test runs use
    norm-following scripted agents exclusively.

    zero_shot skips interactive learning and reference memory, but retains the
    test seeds, setup, background turnover, stopping rule and raw run IDs.
    """
    if mode not in ('experiment', 'zero_shot'):
        raise ValueError(f"Unknown experiment mode: {mode}")
    env_kwargs = env_kwargs or {}
    new_map_worlds = new_map_worlds or []
    stage_results = []
    subject_positions = None
    interactive_memory_path = None
    reference_memory_path = None

    # ---------------- Stage 1: interactive ----------------
    if mode == 'experiment':
        random.seed(random_seed)
        subject_positions = _random_subject_positions(capacity, norm_type)
        ordinals = [
            f"{p + 1}{'th' if 4 <= (p + 1) % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get((p + 1) % 10, 'th')}"
            for p in sorted(subject_positions)
        ]
        print(f"[OK] Subject position (seed {random_seed}): the {', '.join(ordinals)} agent to enter the env\n")

        interactive_result = run_interactive_stage(
            capacity=capacity,
            max_steps=max_steps,
            figures_root=figures_root,
            norm_type=norm_type,
            client=client,
            subject_kind=subject_kind,
            subject_positions=subject_positions,
            ground_truth=ground_truth,
            world_type=world_type,
            env_kwargs=env_kwargs,
            num_no_norm_agents=num_no_norm_agents,
            raw_runs_dir=raw_runs_dir,
            random_seed=random_seed,
        )
        interactive_result['random_seed'] = random_seed

        interactive_memory_path = os.path.join(
            os.path.dirname(figures_root), 'simulation_log', 'interactive_stage.json'
        )

        print("\n" + "=" * 70)
        print("INTERACTIVE SUMMARY")
        print("=" * 70)
        print(f"\nWith {norm_type.upper()} Norm:")
        print(f"  Steps: {interactive_result['steps']}")
        print(f"  Collisions: {interactive_result['num_collisions']['total']}")
        print(f"  Norm violations: {interactive_result['norm_violations']['total']}")
        print(f"  Shared memory: {interactive_memory_path}")
        print("  Rewards per Agent:")
        for agent_id in sorted(interactive_result['agent_rewards'].keys()):
            print(f"    {agent_id}: {interactive_result['agent_rewards'][agent_id]:.2f}")
        print("\n" + "=" * 70)

        reference_memory_path = (
            interactive_memory_path if is_llm_subject_kind(subject_kind) else None
        )
        stage_results.append(interactive_result)

    if num_tests + len(new_map_worlds) >= 100:
        print("[WARN] More than 99 runs: derived seeds may overlap the next "
              "norm's seed block in the config")

    # The normal experiment still withholds GT in tests, regardless of the
    # interactive condition. Zero-shot varies GT only in the subject's prompt.
    test_options = (
        dict(ground_truth=ground_truth, prompt_stage='zero_shot', mode=mode)
        if mode == 'zero_shot' else {}
    )

    # ---------------- Stage 2: same-world tests ----------------
    # Stage-derived seeds are identical for every subject kind, but each test
    # run gets its own setup (subject entry slot, scripted arrivals, ...).
    # Per-norm base seeds in the config are spaced 100 apart, so run seeds
    # stay inside the norm's own block as long as there are < 100 runs.
    stage_results += run_test_series(
        count=num_tests,
        capacity=capacity,
        test_max_steps=test_max_steps,
        figures_root=figures_root,
        norm_type=norm_type,
        client=client,
        subject_kind=subject_kind,
        world_type=world_type,
        env_kwargs=env_kwargs,
        reference_memory_path=reference_memory_path,
        num_no_norm_agents=0,
        raw_runs_dir=raw_runs_dir,
        random_seed=random_seed,
        seed_offset=0,
        stage_label='test',
        **test_options,
    )

    # ---------------- Stage 3: new-world tests ----------------
    stage_results += run_test_series(
        count=len(new_map_worlds),
        capacity=capacity,
        test_max_steps=test_max_steps,
        figures_root=figures_root,
        norm_type=norm_type,
        client=client,
        subject_kind=subject_kind,
        world_type=new_map_worlds,
        env_kwargs=env_kwargs,
        reference_memory_path=reference_memory_path,
        num_no_norm_agents=0,
        raw_runs_dir=raw_runs_dir,
        random_seed=random_seed,
        seed_offset=num_tests,
        stage_label='new_map_test',
        **test_options,
    )

    result = {
        'mode': mode,
        'agent_type': subject_kind,
        'random_seed': random_seed,
        'norm_type': norm_type,
        'world_type': world_type,
        'new_map_worlds': new_map_worlds,
        'capacity': capacity,
        'max_steps': test_max_steps if mode == 'zero_shot' else max_steps,
        'test_max_steps': test_max_steps,
        'subject_position': (sorted(subject_positions)[0] + 1
                             if subject_positions is not None else None),
        'interactive_memory_path': interactive_memory_path,
        'raw_runs_dir': raw_runs_dir,
        'stages': stage_results,
    }
    if mode == 'zero_shot':
        result['ground_truth'] = ground_truth

    with open(results_path, 'w') as f:
        json.dump(result, f, indent=2)

    learning_label = 'interactive + ' if mode == 'experiment' else 'zero-shot; '
    print(
        f"\n[OK] Experiment results ({learning_label}{num_tests} same-world tests + "
        f"{len(new_map_worlds)} new-world tests) saved to JSON: {results_path}"
    )
    print(f"[OK] Raw run logs saved to: {raw_runs_dir}")
    print(f"[OK] Compute metrics with: python experiment/evaluation/eval.py {raw_runs_dir}")
    return result


def _default_new_map_worlds(world_type, seed, count=3):
    """`count` worlds sampled from the canonical list (excluding the main world)
    with a seed-derived RNG, so each norm gets its own fixed world trio even
    without a config entry."""
    candidates = [w for w in WORLD_TYPE_CHOICES if w != world_type]
    return random.Random(seed).sample(candidates, count)


def _resolve_new_map_worlds(cli_worlds, config, norm_type, world_type, seed):
    """CLI flag > per-norm worlds from config > config-wide worlds > seeded default."""
    if cli_worlds:
        return cli_worlds
    norm_worlds = config.get('norm_new_map_worlds', {})
    if isinstance(norm_worlds, dict) and norm_type in norm_worlds:
        return norm_worlds[norm_type]
    return config.get('new_map_worlds') or _default_new_map_worlds(world_type, seed)


DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'experiment_config.json')


def load_experiment_config(config_path):
    """Load the shared experiment config (fixed per-norm seeds and settings)."""
    if not config_path or not os.path.exists(config_path):
        print(f"[WARN] Experiment config not found: {config_path}; using built-in defaults")
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Experiment config must be a JSON object: {config_path}")
    return config


def _resolve_setting(cli_value, config, key, fallback):
    """CLI flag (if given) > config file > built-in fallback."""
    if cli_value is not None:
        return cli_value
    return config.get(key, fallback)


def _resolve_capacity(cli_capacity, config, norm_type):
    """CLI flag > per-norm capacity from config > config-wide capacity > 8.

    A norm gets its own batch size when the size itself is part of the setting.
    occupation_region_capacity_1 runs a 6-agent batch — 5 scripted customers plus
    the subject, which enters last — because the waiting zone holds one agent at a
    time: at the full 8 the extra traffic knocks the subject off the zone's edge
    while it queues, and it never settles there long enough to take the slot the
    staged holder frees.
    """
    if cli_capacity is not None:
        return cli_capacity
    norm_capacity = config.get('norm_capacity', {})
    if isinstance(norm_capacity, dict) and norm_type in norm_capacity:
        return norm_capacity[norm_type]
    return config.get('capacity', 8)


def _resolve_seed(cli_seed, config, norm_type):
    """CLI flag > per-norm seed from config > config-wide seed > 42."""
    if cli_seed is not None:
        return cli_seed
    norm_seeds = config.get('norm_seeds', {})
    if isinstance(norm_seeds, dict) and norm_type in norm_seeds:
        return norm_seeds[norm_type]
    return config.get('random_seed', 42)


def main(mode='experiment'):
    """Main function - handles argument parsing and directory setup.

    mode='zero_shot' (used by zero_shot/run_zero_shot.py) skips the interactive
    stage and runs the same test episodes directly.
    """
    if mode not in ('experiment', 'zero_shot'):
        raise ValueError(f"Unknown experiment mode: {mode}")
    parser = argparse.ArgumentParser(
        description=(
            'Run the full experiment pipeline: interactive stage, same-world '
            'test runs, and new-world test runs.'
            if mode == 'experiment' else
            'Run the same-world and new-world test runs zero-shot, without '
            'the interactive learning stage.'
        )
    )
    parser.add_argument('--agent_type', type=str, default='llm',
                        choices=list(SUBJECT_KINDS),
                        help='Kind of subject agent under study (default: llm; '
                             'memory_llm is the LLM subject prompted with the '
                             'reasoning behind its past steps)')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH,
                        help='Path to the shared experiment config JSON holding the '
                             'per-norm seeds and fixed settings (default: '
                             'experiment/run/experiment_config.json)')
    parser.add_argument('--capacity', type=int, default=None,
                        help='Max agents on the map at once (default: from config, else 8)')
    parser.add_argument('--max_steps', type=int, default=None,
                        help='Maximum steps for the interactive stage (default: from config, else 100)')
    parser.add_argument('--test_max_steps', type=int, default=None,
                        help='Maximum steps per test run (default: from config, else 150)')
    parser.add_argument('--random_seed', type=int, default=None,
                        help='Seed for all stage setups; the same seed gives every '
                             'subject kind (llm, rl, ...) the identical setting '
                             '(default: per-norm seed from config)')
    parser.add_argument('--norm_type', type=str, required=True,
                        choices=norm_choices(),
                        help='Norm type to use')
    parser.add_argument('--figures_suffix', type=str, default='',
                        help='Optional suffix for figures directory')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results (default: results)')
    parser.add_argument('--ground_truth', type=str, default='without',
                        choices=['with', 'without'],
                        help='Whether an LLM subject can access the norm description')
    parser.add_argument('--num_no_norm_agents', type=int, default=None,
                        help='Number of scripted no-norm background agents in the '
                             'interactive stage (default: from config, else 1)')
    parser.add_argument('--provider', type=str, default='openrouter',
                        choices=['openrouter', 'hf'],
                        help='LLM provider to use (default: openrouter; ignored for '
                             'non-LLM --agent_type). "hf" runs any open-weight '
                             'VLM locally via Hugging Face')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name to use (overrides default for the provider)')
    parser.add_argument('--world_type', type=str, default=None,
                        choices=WORLD_TYPE_CHOICES,
                        help='World type for the interactive stage and same-world tests '
                             '(default: from config, else cafe)')
    parser.add_argument('--new_map_worlds', type=str, nargs='+', default=None,
                        choices=WORLD_TYPE_CHOICES,
                        help='Worlds for the new-map test runs, one run per world '
                             '(default: from config, else first 3 worlds different from --world_type)')
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
    parser.add_argument('--num_tests', type=int, default=None,
                        help='Number of same-world test runs in either mode '
                             '(default: from config, else 3)')
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    capacity = _resolve_capacity(args.capacity, config, args.norm_type)
    max_steps = _resolve_setting(args.max_steps, config, 'max_steps', 100)
    test_max_steps = _resolve_setting(args.test_max_steps, config, 'test_max_steps', 150)
    num_tests = _resolve_setting(args.num_tests, config, 'num_tests', 3)
    world_type = _resolve_setting(args.world_type, config, 'world_type', 'cafe')
    num_no_norm_agents = _resolve_setting(
        args.num_no_norm_agents, config, 'num_no_norm_agents', 1
    )
    random_seed = _resolve_seed(args.random_seed, config, args.norm_type)
    print(f"[OK] Experiment config: {args.config}")
    print(f"[OK] Seed for norm '{args.norm_type}': {random_seed} "
          f"(identical setting for every --agent_type)")

    if num_tests < 0:
        raise ValueError("num_tests must be non-negative")
    if capacity < 3:
        raise ValueError("capacity must be at least 3 so the subject can enter in 3rd place or later")

    force_load_repo_env()

    model = args.model
    client = None
    if is_llm_subject_kind(args.agent_type):
        client = create_llm_client(provider=args.provider, model=model)
        client_model = getattr(client, 'model', None)
        if not isinstance(client_model, str):
            client_model = model  # HFClient.model is the torch module
        print(f"[OK] Using {args.provider} provider with model: {client_model}")
    else:
        print(f"[OK] Subject agent type: {args.agent_type} (no LLM client needed)")

    new_map_worlds = _resolve_new_map_worlds(
        args.new_map_worlds, config, args.norm_type, world_type, random_seed
    )
    if world_type in new_map_worlds:
        raise ValueError(
            f"--new_map_worlds must not include the main world_type ({world_type})"
        )

    base_output_dir = (
        args.output_dir
        if os.path.isabs(args.output_dir)
        else os.path.join(EXPERIMENT_OUTPUT_ROOT, args.output_dir)
    )
    if mode == 'zero_shot':
        base_output_dir = os.path.join(
            base_output_dir, 'zero_shot', f'{args.ground_truth}_gt'
        )

    # Results are grouped by subject agent kind so different subjects never
    # overwrite each other (llm and memory_llm are separate kinds, so their runs
    # never mix). LLM runs additionally get a per-model directory so
    # different models never overwrite each other: this pipeline writes
    # results/llm/<model>/<world>/<norm>/, e.g. results/llm/openai_gpt-5.5/cafe/<norm>/,
    # while non-LLM subjects keep results/<agent_type>/<world>/<norm>/, e.g.
    # results/rl/cafe/<norm>/. eval.py / generate_report.py read whichever
    # directory they are pointed at, so both layouts work unchanged.
    path_parts = [base_output_dir, args.agent_type]
    if is_llm_subject_kind(args.agent_type):
        # HFClient.model is the loaded torch module, not a name string; fall
        # back to the CLI value so the per-model directory stays a plain path.
        model_name = getattr(client, 'model', None)
        if not isinstance(model_name, str):
            model_name = model or 'default'
        # Model ids like "openai/gpt-5.5" contain path separators; flatten them
        # so the model stays a single directory level.
        path_parts.append(model_name.replace('/', '_').replace(':', '_'))
    path_parts.extend([world_type, args.norm_type])
    norm_base_dir = os.path.join(*path_parts)
    figures_root = os.path.join(norm_base_dir, 'simulation_figures' + args.figures_suffix)
    logs_dir = os.path.join(norm_base_dir, 'simulation_log')
    raw_runs_dir = os.path.join(norm_base_dir, 'raw_runs')
    logs_path = os.path.join(logs_dir, f"experiment_results_{args.norm_type}.json")

    # Always wipe this norm's simulation_figures and simulation_log before a run
    # so stale frames, memory, and logs never mix with the new run, whether or
    # not a previous result is present.
    for stale_dir in (figures_root, logs_dir):
        if os.path.exists(stale_dir):
            shutil.rmtree(stale_dir)
            print(f"[OK] Cleared existing {os.path.basename(stale_dir)} directory")

    # eval.py reads every run log in raw_runs_dir, so stale runs from a previous
    # experiment must not survive into this one's metrics.
    if os.path.exists(raw_runs_dir):
        shutil.rmtree(raw_runs_dir)
        print("[OK] Cleared existing raw_runs directory")

    os.makedirs(figures_root, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(raw_runs_dir, exist_ok=True)
    print(f"[OK] Created fresh simulation_figures directory: {figures_root}\n")

    # Flush the running cost report after every billed call so spend is visible
    # while the run is still in progress, not only in the end-of-run write below.
    if client is not None and hasattr(client, "enable_incremental_cost_report"):
        client.enable_incremental_cost_report(norm_base_dir)

    print(f"[OK] Mode: {mode} ({args.agent_type} subject)")
    print(f"[OK] Ground truth: {args.ground_truth}")
    print(f"[OK] Same-world tests: {num_tests}; new-world tests: {new_map_worlds}\n")

    env_kwargs = {
        'use_sprites': not args.no_sprites,
        'use_isometric': args.isometric,
        'entrance_side': args.entrance_side,
        'map_type': args.map_type,
        'interpolation_steps': args.interpolation_steps,
    }

    txt_log_path = os.path.splitext(logs_path)[0] + '.txt'
    run_with_raw_stdout_log(
        txt_log_path,
        run_experiment,
        capacity=capacity,
        max_steps=max_steps,
        results_path=logs_path,
        figures_root=figures_root,
        raw_runs_dir=raw_runs_dir,
        norm_type=args.norm_type,
        client=client,
        subject_kind=args.agent_type,
        ground_truth=args.ground_truth,
        random_seed=random_seed,
        world_type=world_type,
        env_kwargs=env_kwargs,
        num_tests=num_tests,
        test_max_steps=test_max_steps,
        new_map_worlds=new_map_worlds,
        num_no_norm_agents=num_no_norm_agents,
        mode=mode,
    )
    print(f"[OK] Simulation log saved to TXT: {txt_log_path}")

    # Report OpenRouter spend for this run (only OpenRouter clients track cost).
    if client is not None and hasattr(client, "cost_summary_line"):
        print(f"[OK] {client.cost_summary_line()}")
        cost_path = client.write_cost_report(norm_base_dir)
        if cost_path:
            print(f"[OK] Cost report saved to: {cost_path}")


# Distinct exit code for "provider account is out of credits" so batch scripts
# can stop the whole job instead of retrying the remaining norms/models.
EXIT_CODE_NO_CREDITS = 3



def cli(mode='experiment'):
    try:
        main(mode)
    except InsufficientCreditsError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        print(
            "[FATAL] Aborting: no LLM calls can succeed without credits, so "
            "continuing would only produce invalid all-'wait' results.",
            file=sys.stderr,
        )
        sys.exit(EXIT_CODE_NO_CREDITS)


if __name__ == "__main__":
    cli()
