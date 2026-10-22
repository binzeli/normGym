import os
import sys
import random
import json
import argparse
import shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.env.gym_env import NormEnv
from src.agents.oracle_agent import OracleAgent
from src.agents.agent_init import get_agent_entry_position, get_agent_targets
from src.norms.occupation import is_occupation_door_norm, is_occupation_seat_norm

def _build_agents(num_agents, arrival_times, env, norm_type=None):
    """Build a list of OracleAgent agents.
    
    norm_type — norm assigned to each agent (controls their behaviour).
    Target positions are computed using env.norm_type so that no-norm agents
    navigate to the same kind of targets as norm agents.
    """
    world = env.world
    target_norm_type = env.norm_type
    # Assign entrances in arrival order so last_entrance avoids consecutive
    # agents entering through the same door at runtime.
    sorted_indices = sorted(range(num_agents), key=lambda i: arrival_times[i])
    agent_classes = [random.choice(['orange', 'green', 'blue']) for _ in range(num_agents)]
    positions = [None] * num_agents
    claimed_targets: set = set()
    last_entrance = None
    for i in sorted_indices:
        position = get_agent_entry_position(world, norm_type, last_entrance)
        last_entrance = position
        positions[i] = position

    agents = []
    for i in range(num_agents):
        position = positions[i]
        target_positions = get_agent_targets(position, world, target_norm_type, extra_claimed=claimed_targets)
        if target_positions:
            claimed_targets.add(target_positions[0])
        agent = OracleAgent(
            agent_id=f"C{i+1}",
            agent_class=agent_classes[i],
            position=position,
            arrival_time=arrival_times[i],
            norm=norm_type,
            target_positions=target_positions
        )
        # Which goal the agent chooses follows the run's norm family: the exit
        # door under a door norm, the seat under a seat norm, neither otherwise.
        # The no-norm agents of this run carry norm=None and cannot see the norm
        # themselves, hence the flags (see OracleAgent.get_action).
        if is_occupation_door_norm(target_norm_type):
            agent.free_choice_door = True
        elif is_occupation_seat_norm(target_norm_type):
            agent.free_choice_seat = True
        agents.append(agent)
    return agents


def _copy_agents_with_norm(agents, norm_type):
    """Create copies of agents with a different norm, sharing all other init data."""
    copies = []
    for a in agents:
        copy = OracleAgent(
            agent_id=a.agent_id,
            agent_class=a.agent_class,
            position=a.current_position,
            arrival_time=a.arrival_time,
            norm=norm_type,
            target_positions=list(a.target_positions),
        )
        # Carried over so the no-norm copy still chooses the same goal its
        # norm-aware twin does — it is dropping the norm, not the task.
        for flag in ('free_choice_door', 'free_choice_seat'):
            if getattr(a, flag, False):
                setattr(copy, flag, True)
        copies.append(copy)
    return copies


def _make_env(norm_type, figures_dir, num_agents, world_type, use_isometric, map_type, interpolation_steps=0):
    """Construct a NormEnv with the given parameters."""
    return NormEnv(
        num_agents=num_agents,
        max_steps=35,
        norm_type=norm_type,
        output_dir=figures_dir,
        render_mode='rgb_array',
        world_type=world_type,
        use_isometric=use_isometric,
        map_type=map_type,
        interpolation_steps=interpolation_steps
    )


def run_one_simulation(env, agents, norm_type=None, run_number=1, arrival_times=None,
                   all_results=None, all_episodes_no_norms=None,
                   all_episodes_with_norms=None, figures_root=None):
    """Run a single simulation with pre-built env and agents."""
    print("=" * 70)
    print(f"Cafe Simulation - {'WITH' if norm_type else 'WITHOUT'} Norms")
    print("=" * 70)

    # Create figures subfolder for this simulation under outputs/simulation_figures
    run_folder = f"{'with_' + norm_type if norm_type else 'no_norm'}_run{run_number}"
    figures_dir = os.path.join(figures_root, run_folder)

    if os.path.exists(figures_dir):
        for file in os.listdir(figures_dir):
            if file.endswith('.png'):
                os.remove(os.path.join(figures_dir, file))
        print(f"\nCleared existing figures from: {figures_dir}/")
    else:
        os.makedirs(figures_dir)
    print(f"Figures will be saved to: {figures_dir}/")

    env.output_dir = figures_dir
    if env.output_dir and not os.path.exists(env.output_dir):
        os.makedirs(env.output_dir)

    print(f"\nStarting simulation...")
    print("-" * 70)

    # Reset environment with pre-built agents
    obs, info = env.reset(options={'agents': agents})
    
    # Run simulation step by step
    step_count = 0
    env.render()
    
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step()
        step_count = info['step']
        
        env.render()
        
    # Print every agent's target positions at the end of the simulation
    print("\n[Target positions]")
    for a in env.agents:
        print(f"  {a.agent_id}: current={a.current_position}, targets={list(a.target_positions)}")

    # Compile episode data
    total_norm_violations = sum(len(s.get('norm_violations', [])) for s in env.episode_log)
    episode = {
        'run': run_number,
        'run_folder': run_folder,
        'steps': env.current_step,
        'customers_count': len(env.agents),
        'total_collisions': env.total_collisions,
        'total_norm_violations': total_norm_violations,
        'agent_rewards': {a.agent_id: a.total_reward for a in env.agents},
        'agent_classes': {a.agent_id: a.agent_class for a in env.agents},
        'episode_log': env.episode_log,
        'final_states': {a.agent_id: ('done' if a.done else 'active') for a in env.agents},
        'agents': [{'id': a.agent_id, 'type': a.agent_type.value} for a in env.agents]
    }
    
    print(f"\n[OK] All figures saved to '{figures_dir}/' directory")
    print(f"  Total figures: {step_count}")
    
    # Store results
    if all_results is not None:
        # Group rewards by class
        class_rewards = {}
        agent_classes_dict = episode.get('agent_classes', {})
        for agent_id, reward in episode['agent_rewards'].items():
            agent_class = agent_classes_dict.get(agent_id, 'Unknown')
            if agent_class not in class_rewards:
                class_rewards[agent_class] = []
            class_rewards[agent_class].append(reward)
        
        # Calculate average reward by class
        avg_reward_by_class = {}
        for agent_class, rewards in class_rewards.items():
            avg_reward_by_class[agent_class] = sum(rewards) / len(rewards) if rewards else 0
        
        agent_ids = list(episode['agent_rewards'].keys())
        
        all_results.append({
            'run': run_number,
            'norm_type': norm_type if norm_type else 'no_norms',
            'run_folder': run_folder,
            'arrival_times': {
                agent_id: arrival_times[i] if i < len(arrival_times) else None
                for i, agent_id in enumerate(agent_ids)
            },
            'agent_classes': episode['agent_classes'],
            'steps': episode['steps'],
            'total_collisions': episode['total_collisions'],
            'total_norm_violations': episode['total_norm_violations'],
            'agent_rewards': episode['agent_rewards'],
            'avg_reward_by_class': avg_reward_by_class
        })
    
    # Store episodes for summary calculations
    if norm_type:
        all_episodes_with_norms.append(episode)
    else:
        all_episodes_no_norms.append(episode)

    return episode


def run_all_simulations(num_agents, num_simulations, results_path, figures_root, norm_type=None, world_type='cafe',
                        use_isometric=False, cycle_all_agents=False, rounds_per_agent=1, sweep_compare=False, manifest_path=None,
                        map_type='default', interpolation_steps=0):
    """Run all simulations and handle results storage"""
    all_results = []
    all_episodes_no_norms = []
    all_episodes_with_norms = []
    run_manifest = []

    if cycle_all_agents:
        if norm_type is None:
            raise ValueError("cycle_all_agents requires --norm_type")
        if rounds_per_agent < 1:
            raise ValueError("rounds_per_agent must be >= 1")
        if num_simulations != 1:
            print("[INFO] --num_simulations is ignored when --cycle_all_agents is enabled.")
        if not sweep_compare:
            print("[INFO] Enforcing comparison mode for sweep runs (no_norm + with_norm).")
            sweep_compare = True

        total_runs = num_agents * rounds_per_agent
        print(f"[OK] Sweep mode enabled: {num_agents} agents x {rounds_per_agent} rounds/agent = {total_runs} queried-agent runs")
        print(f"[OK] Sweep comparison is enabled: total episodes = {total_runs * 2}\n")

        global_run = 0
        for scenario_round in range(1, rounds_per_agent + 1):
            arrival_times = random.sample(range(1, 15), num_agents)
            print("-" * 70)
            print(f"SCENARIO ROUND {scenario_round}/{rounds_per_agent} (shared across all queried agents)")
            print("-" * 70)
            print(f"Arrival times: {arrival_times} for agent {', '.join([f'C{i+1}' for i in range(num_agents)])}\n")

            for queried_agent_id in range(num_agents):
                global_run += 1
                print("~" * 70)
                print(f"GLOBAL RUN {global_run}/{total_runs}: Scenario {scenario_round}, Queried Agent C{queried_agent_id + 1}")
                print("~" * 70)

                # Create env for this run
                run_folder_no_norm = f"no_norm_run{global_run}"
                run_folder_with_norm = f"with_{norm_type}_run{global_run}"
                env_with_norm = _make_env(norm_type, os.path.join(figures_root, run_folder_with_norm),
                                          num_agents,
                                          world_type, use_isometric, map_type,
                                          interpolation_steps)

                # Build agents once; derive no-norm copy with identical initialization
                agents_with_norm = _build_agents(num_agents, arrival_times,
                                                 env_with_norm, norm_type=norm_type)
                agents_no_norm = _copy_agents_with_norm(agents_with_norm, norm_type=None)
                episode_no_norms = run_one_simulation(
                    env_with_norm,
                    agents_no_norm,
                    norm_type=None,
                    run_number=global_run,
                    arrival_times=arrival_times,
                    all_results=all_results,
                    all_episodes_no_norms=all_episodes_no_norms,
                    all_episodes_with_norms=all_episodes_with_norms,
                    figures_root=figures_root,
                )

                episode_with_norms = run_one_simulation(
                    env_with_norm,
                    agents_with_norm,
                    norm_type=norm_type,
                    run_number=global_run,
                    arrival_times=arrival_times,
                    all_results=all_results,
                    all_episodes_no_norms=all_episodes_no_norms,
                    all_episodes_with_norms=all_episodes_with_norms,
                    figures_root=figures_root,
                )

                run_manifest.append({
                    'global_run': global_run,
                    'scenario_round': scenario_round,
                    'queried_agent_id': queried_agent_id,
                    'queried_agent_label': f"C{queried_agent_id + 1}",
                    'norm_type': 'no_norms',
                    'run_folder': f"no_norm_run{global_run}",
                    'arrival_times': arrival_times,
                    'steps': episode_no_norms['steps'],
                    'total_collisions': episode_no_norms['total_collisions'],
                    'total_norm_violations': episode_no_norms['total_norm_violations']
                })
                run_manifest.append({
                    'global_run': global_run,
                    'scenario_round': scenario_round,
                    'queried_agent_id': queried_agent_id,
                    'queried_agent_label': f"C{queried_agent_id + 1}",
                    'norm_type': norm_type,
                    'run_folder': f"with_{norm_type}_run{global_run}",
                    'arrival_times': arrival_times,
                    'steps': episode_with_norms['steps'],
                    'total_collisions': episode_with_norms['total_collisions'],
                    'total_norm_violations': episode_with_norms['total_norm_violations']
                })

        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[OK] Results saved to JSON: {results_path}")

        if manifest_path:
            with open(manifest_path, 'w') as f:
                json.dump(run_manifest, f, indent=2)
            print(f"[OK] Run manifest saved to JSON: {manifest_path}")
        return

    
    for run in range(1, num_simulations + 1):
        # Generate random arrival times for this run (ensure all are unique)
        arrival_times = random.sample(range(1, 15), num_agents)

        print(f"Arrival times: {arrival_times} for agent {', '.join([f'C{i+1}' for i in range(num_agents)])}\n")

        # Create env for this run
        run_folder_no_norm = f"no_norm_run{run}"
        run_folder_with_norm = f"with_{norm_type}_run{run}"
        env_with_norm = _make_env(norm_type, os.path.join(figures_root, run_folder_with_norm),
                                  num_agents,
                                  world_type, use_isometric, map_type,
                                  interpolation_steps)

        # Build agents once; derive no-norm copy with identical initialization
        agents_with_norm = _build_agents(num_agents, arrival_times,
                                         env_with_norm, norm_type=norm_type)
        agents_no_norm = _copy_agents_with_norm(agents_with_norm, norm_type=None)

        # Run simulation without norms
        episode_no_norms = run_one_simulation(
            env_with_norm,
            agents_no_norm,
            norm_type=None,
            run_number=run,
            arrival_times=arrival_times,
            all_results=all_results,
            all_episodes_no_norms=all_episodes_no_norms,
            all_episodes_with_norms=all_episodes_with_norms,
            figures_root=figures_root,
        )

        print("\n\n")

        # Run simulation with specified norm (same arrival times and entrances)
        norm_display_name = norm_type if norm_type else 'NO NORM'
        print("-" * 70)
        print(f"SIMULATION {run}B: WITH {norm_display_name.upper()} NORM")
        print("-" * 70 + "\n")
        episode_with_norms = run_one_simulation(
            env_with_norm,
            agents_with_norm,
            norm_type=norm_type,
            run_number=run,
            arrival_times=arrival_times,
            all_results=all_results,
            all_episodes_no_norms=all_episodes_no_norms,
            all_episodes_with_norms=all_episodes_with_norms,
            figures_root=figures_root,
        )


        # Print comparison for this run
        print("\n" + "~" * 70)
        print(f"RUN {run} COMPARISON")
        print("~" * 70)
        print(f"\nNo Norms:")
        print(f"  Steps: {episode_no_norms['steps']}")
        print(f"  Collisions: {episode_no_norms['total_collisions']}")
        print(f"  Norm Violations: {episode_no_norms['total_norm_violations']}")
        print(f"  Rewards: {episode_no_norms['agent_rewards']}")
        print(f"\nWith {norm_display_name.upper()} Norm:")
        print(f"  Steps: {episode_with_norms['steps']}")
        print(f"  Collisions: {episode_with_norms['total_collisions']}")
        print(f"  Norm Violations: {episode_with_norms['total_norm_violations']}")
        print(f"  Rewards: {episode_with_norms['agent_rewards']}")
        print("~" * 70)
        print("\n")
    
    # Save results to JSON file
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n[OK] Results saved to JSON: {results_path}")

    # Print overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    
    # Calculate averages for no norms from JSON results
    avg_collisions_no_norms = sum(r['total_collisions'] for r in all_results if r['norm_type'] == 'no_norms') / num_simulations
    avg_violations_no_norms = sum(r['total_norm_violations'] for r in all_results if r['norm_type'] == 'no_norms') / num_simulations
    class_rewards_no_norms = {}
    
    for result in all_results:
        if result['norm_type'] == 'no_norms':
            for agent_class, avg_reward in result.get('avg_reward_by_class', {}).items():
                if agent_class not in class_rewards_no_norms:
                    class_rewards_no_norms[agent_class] = []
                class_rewards_no_norms[agent_class].append(avg_reward)
    
    # Calculate averages for with norms from JSON results
    avg_collisions_with_norms = sum(r['total_collisions'] for r in all_results if r['norm_type'] != 'no_norms') / num_simulations
    avg_violations_with_norms = sum(r['total_norm_violations'] for r in all_results if r['norm_type'] != 'no_norms') / num_simulations
    class_rewards_with_norms = {}
    
    for result in all_results:
        if result['norm_type'] != 'no_norms':
            for agent_class, avg_reward in result.get('avg_reward_by_class', {}).items():
                if agent_class not in class_rewards_with_norms:
                    class_rewards_with_norms[agent_class] = []
                class_rewards_with_norms[agent_class].append(avg_reward)
    
    norm_display_name = norm_type if norm_type else 'NO NORM'
    
    print(f"\nNo Norms (Average across {num_simulations} runs):")
    print(f"  Average Collisions: {avg_collisions_no_norms:.2f}")
    print(f"  Average Norm Violations: {avg_violations_no_norms:.2f}")
    print(f"  Average Rewards by Class:")
    for agent_class in sorted(class_rewards_no_norms.keys(), key=lambda x: (x == 'Unknown', x)):
        avg_reward = sum(class_rewards_no_norms[agent_class]) / len(class_rewards_no_norms[agent_class])
        print(f"    Class {agent_class}: {avg_reward:.2f}")
    
    # Calculate and print average rewards per agent for no norms
    print(f"  Average Rewards per Agent:")
    agent_rewards_no_norms = {}
    for result in all_results:
        if result['norm_type'] == 'no_norms':
            for agent_id, reward in result.get('agent_rewards', {}).items():
                if agent_id not in agent_rewards_no_norms:
                    agent_rewards_no_norms[agent_id] = []
                agent_rewards_no_norms[agent_id].append(reward)
    for agent_id in sorted(agent_rewards_no_norms.keys()):
        avg_reward = sum(agent_rewards_no_norms[agent_id]) / len(agent_rewards_no_norms[agent_id])
        print(f"    {agent_id}: {avg_reward:.2f}")
    
    print(f"\nWith {norm_display_name.upper()} Norm (Average across {num_simulations} runs):")
    print(f"  Average Collisions: {avg_collisions_with_norms:.2f}")
    print(f"  Average Norm Violations: {avg_violations_with_norms:.2f}")
    print(f"  Average Rewards by Class:")
    for agent_class in sorted(class_rewards_with_norms.keys(), key=lambda x: (x == 'Unknown', x)):
        avg_reward = sum(class_rewards_with_norms[agent_class]) / len(class_rewards_with_norms[agent_class])
        print(f"    Class {agent_class}: {avg_reward:.2f}")
    
    # Calculate and print average rewards per agent for with norms
    print(f"  Average Rewards per Agent:")
    agent_rewards_with_norms = {}
    for result in all_results:
        if result['norm_type'] != 'no_norms':
            for agent_id, reward in result.get('agent_rewards', {}).items():
                if agent_id not in agent_rewards_with_norms:
                    agent_rewards_with_norms[agent_id] = []
                agent_rewards_with_norms[agent_id].append(reward)
    for agent_id in sorted(agent_rewards_with_norms.keys()):
        avg_reward = sum(agent_rewards_with_norms[agent_id]) / len(agent_rewards_with_norms[agent_id])
        print(f"    {agent_id}: {avg_reward:.2f}")
    
    print("\n" + "=" * 70)




def print_step_details(env, step_count, info):
    """Print details of a simulation step"""
    print(f"\nStep {step_count}:")
    
    # Get collision info from last episode log entry
    if env.episode_log:
        last_step = env.episode_log[-1]
        collisions_resolved = last_step.get('collisions_resolved', 0)
        if collisions_resolved > 0:
            print(f"  [Collision resolution: {collisions_resolved} position(s) had conflicts]")
    
    # Create agent lookup for class info
    agent_lookup = {a.agent_id: a for a in env.agents if a.active}
    
    # Print agent actions from last step
    if env.episode_log:
        last_step = env.episode_log[-1]
        for agent_id, action_data in last_step['actions'].items():
            action_str = action_data['action'] if action_data['action'] else 'None'
            agent = agent_lookup.get(agent_id)
            agent_class_str = f", agent_class={agent.agent_class}" if agent else ""
            
            # Check if action was WAIT due to blocking
            blocked_str = ""
            if action_str == 'wait' and agent and hasattr(agent, '_blocked_by_stronger'):
                blocked_str = " [BLOCKED by stronger agent]"
                delattr(agent, '_blocked_by_stronger')
            
            print(f"  {agent_id}: action={action_str}, "
                  f"pos={action_data['position']}{agent_class_str}{blocked_str}")
    
    # Check for collisions at END of step
    positions = {}
    for agent in env.agents:
        if agent.active and not agent.done:
            pos = agent.current_position
            if pos not in positions:
                positions[pos] = []
            positions[pos].append(agent.agent_id)
    
    # Report any positions with multiple agents
    for pos, agent_ids in positions.items():
        if len(agent_ids) > 1:
            print(f"  [WARN] COLLISION at {pos}: {agent_ids}")


def main():
    """Main function - handles argument parsing and directory setup only"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run cafe simulation multiple times.')
    parser.add_argument('--num_agents', type=int, default=3, help='Number of agents in the simulation')
    parser.add_argument('--num_simulations', type=int, default=1, help='Number of simulation runs')
    parser.add_argument('--random_seed', type=int, default=None, help='Random seed for reproducibility')
    _norm_choices = [n['name'] for n in json.load(open(os.path.join(os.path.dirname(__file__), '../src/norms/norm_descriptions.json'), encoding='utf-8'))['norms']]
    parser.add_argument('--norm_type', type=str, default=None,
                       choices=_norm_choices,
                       help='Norm type to use (omit for no norm)')
    parser.add_argument('--figures_suffix', type=str, default='',
                       help='Optional suffix for figures directory (e.g., "_1agent", "_2agents")')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for results (default: outputs)')
    parser.add_argument('--world_type', type=str, default='cafe',
                       choices=['cafe', 'movie_theater', 'bank', 'pharmacy', 'grocery_store', 'clothing_store', 'library', 'post_office', 'hospital', 'barbershop'],
                       help='World type to use (default: cafe)')
    parser.add_argument('--isometric', action='store_true',
                       help='Use 2.5D isometric rendering instead of flat top-down view')
    parser.add_argument('--cycle_all_agents', action='store_true',
                       help='Sweep mode: cycle through all agents as queried agent')
    parser.add_argument('--rounds_per_agent', type=int, default=1,
                       help='Number of scenario rounds per agent in sweep mode (default: 1)')
    parser.add_argument('--sweep_compare', action='store_true',
                       help='Enable comparison mode for sweep runs (no_norm + with_norm)')
    parser.add_argument('--map_type', type=str, default='default',
                       choices=['default', 'navigation', 'affordance'],
                       help='Map layout type (default: default)')
    parser.add_argument('--interpolation_steps', type=int, default=0,
                       help='Number of intermediate frames between simulation steps for smooth rendering (0 = disabled)')
    args = parser.parse_args()
    
    # Convert empty string to None if provided
    if args.norm_type == '':
        args.norm_type = None

    # Set random seed if provided
    if args.random_seed is not None:
        random.seed(args.random_seed)
    
    # Setup directories
    # Use provided output_dir or default to project_root/outputs
    if os.path.isabs(args.output_dir):
        base_output_dir = args.output_dir
    else:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base_output_dir = os.path.join(project_root, args.output_dir)
    
    # Create base directory for world_type/norm_type (or 'no_norm' if no norm type)
    norm_folder = args.norm_type if args.norm_type else 'no_norm'
    
    # Create subdirectories
    figures_root = os.path.join(base_output_dir, "simulation_figures", "simulation_figures" + args.figures_suffix)
    logs_dir = os.path.join(base_output_dir, "simulation_log")
    results_dir = os.path.join(base_output_dir, "results")
    
    # Clear figures directory if it exists
    if os.path.exists(figures_root):
        shutil.rmtree(figures_root)
        print(f"[OK] Cleared existing simulation_figures directory")
    
    # Create all necessary directories
    os.makedirs(figures_root, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    print(f"[OK] Created fresh simulation_figures directory: {figures_root}\n")
    
    logs_path = os.path.join(logs_dir, f"simulation_results_{args.norm_type}_{args.num_agents}agents.json")
    manifest_path = os.path.join(results_dir, f"run_manifest_{args.norm_type}_{args.num_agents}agents.json")
    # Run all simulations
    run_all_simulations(
        args.num_agents,
        args.num_simulations,
        logs_path,
        figures_root,
        norm_type=args.norm_type,
        world_type=args.world_type,
        use_isometric=args.isometric,
        cycle_all_agents=args.cycle_all_agents,
        rounds_per_agent=args.rounds_per_agent,
        sweep_compare=args.sweep_compare,
        manifest_path=manifest_path,
        map_type=args.map_type,
        interpolation_steps=args.interpolation_steps
    )


if __name__ == "__main__":
    main()
