import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import os
import sys
import matplotlib
matplotlib.use('Agg')
from collections import deque
import tempfile
import matplotlib.image as mpimg
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.world.cafe import CafeWorld
from src.world.movie_theater import MovieTheaterWorld
from src.world.bank import BankWorld
from src.world.pharmacy import PharmacyWorld
from src.world.grocery_store import GroceryStoreWorld
from src.world.clothing_store import ClothingStoreWorld
from src.world.library import LibraryWorld
from src.world.post_office import PostOfficeWorld
from src.world.hospital import HospitalWorld
from src.world.barbershop import BarbershopWorld
from src.world.base_world import CellType

WORLD_CLASSES = {
    'cafe': CafeWorld,
    'movie_theater': MovieTheaterWorld,
    'bank': BankWorld,
    'pharmacy': PharmacyWorld,
    'grocery_store': GroceryStoreWorld,
    'clothing_store': ClothingStoreWorld,
    'library': LibraryWorld,
    'post_office': PostOfficeWorld,
    'hospital': HospitalWorld,
    'barbershop': BarbershopWorld,
}
from src.agents.base_agent import Agent, agent_class_index
from experiment.agents.llm_agent import LLMAgent
from src.norms.lining import generate_queue_line, get_queue, get_queue_start, is_line_norm
from src.norms.navigation import is_spacing_norm, is_navigation_norm, is_xcell_norm, xcell_requires_visit
from src.norms.occupation import (
    region_capacity,
    is_occupation_region_norm,
    free_choice_goal_index,
    free_goal_settled,
    advance_seat_in_order_frontier,
    is_seat_in_order_norm,
)
from src.env import collision_handler
from src.env.actions import Action, AGENT_ACTIONS
from src.env.rewards import (
    COLLISION_PENALTY,
    GOAL_REWARD,
    NORM_VIOLATION_PENALTY,
    STEP_COST,
)
from src.norms.norm_checker import check_norm_violation, is_known_norm, known_norms
from src.env.sprite_loader import SpriteCache
from src.env.norm_conflict_resolver import (
    resolve_spacing_norm_conflict,
    check_spacing_violations,
    resolve_occupation_region_norm_conflict,
    check_occupation_region_violations,
    resolve_queue_norm_conflict,
)

class NormEnv(gym.Env):
    """
    Gym environment for social norm simulation.
    
    The environment simulates a world where multiple agents (customers) need to
    navigate to a goal, complete tasks while potentially following social norms 
    to reduce collisions.
    
    This class combines simulation logic with Gym environment interface.
    
    Observation Space:
        - Grid representation of the world
        - Agent positions, classes, and states

    Action Space:
        - Multi-agent discrete actions (one action per agent), MultiDiscrete([5] * num_agents)
        - Each agent can: move up, move down, move left, move right, or wait
          (indices 0-4, matching _action_to_enum)
        
    Reward:
        - -0.1 per step (action cost)
        - -5 for collisions
        - -20 for norm violations
        - +10 for task completion
    """


    def __init__(
        self,
        num_agents: int = 3,
        max_steps: int = 25,
        norm_type: Optional[str] = None,
        render_mode: Optional[str] = None,
        output_dir: Optional[str] = None,
        queried_agent_id: Optional[int] = None,
        show_queried_agent_star: bool = False,
        world_type: str = 'cafe',
        use_isometric: bool = False,
        map_type: str = 'default',
        interpolation_steps: int = 0
    ):
        """
        Initialize the environment.

        Args:
            num_agents: Number of customer agents (default: 3)
            max_steps: Maximum steps per episode (default: 25)
            norm_type: Social norm to apply ('straight_line', line-variants, spacing/movement/xcell/occupation norms, or None)
            render_mode: Rendering mode. 'rgb_array' renders each step and saves
                the frame(s) to output_dir; None disables rendering.
            output_dir: Directory to save visualization images
            queried_agent_id: Index of queried agent for star rendering and arrow suppression.
                If omitted, no queried agent is set.
            show_queried_agent_star: Whether to render a star on the queried agent.
                Useful for ActionQA data generation where queried agent is not an LLMAgent.
            use_isometric: Whether to use 2.5D isometric rendering (default: False)
        """
        super().__init__()

        self.num_agents = num_agents
        self.max_steps = max_steps
        if not is_known_norm(norm_type):
            raise ValueError(f"Unknown norm_type {norm_type!r}. Known norms: {', '.join(sorted(known_norms()))}")
        self.norm_type = norm_type
        assert render_mode in (None, 'rgb_array'), (
            f"Unsupported render_mode {render_mode!r}; expected None or 'rgb_array'"
        )
        self.render_mode = render_mode
        self.output_dir = output_dir
        self.show_queried_agent_star = show_queried_agent_star
        self.world_type = world_type
        self.map_type = map_type

        # Queried agent tracking (set via set_queried_agent)
        self.queried_agent_id = None
        self.queried_agent_label = None
        default_queried_agent_id = queried_agent_id if queried_agent_id is not None else (
            None
        )
        self.set_queried_agent(default_queried_agent_id)

        # Create world
        world_cls = WORLD_CLASSES.get(world_type, CafeWorld)
        self.world = world_cls(map_type=map_type)
        # is_at_counter resolves the service cell from the norm (edge lines are
        # served from the counter's side cell, not the cell in front of it).
        self.world.norm_type = norm_type

        # Initialize sprite cache (required; sprite rendering is always enabled).
        self.use_isometric = use_isometric
        self.iso_renderer = None
        assets_dir = os.path.join(os.path.dirname(__file__), '../../assets/sprites')
        self.sprite_cache = SpriteCache(assets_dir, px_per_cell=80)
        self.sprite_cache.load_world_sprites(world_type)
        from src.env.flat_renderer import FlatRenderer
        self.flat_renderer = FlatRenderer(self.sprite_cache, self.world)
        if use_isometric:
            self.sprite_cache.load_iso_sprites(world_type)
            from src.env.iso_renderer import IsometricRenderer
            self.iso_renderer = IsometricRenderer(self.sprite_cache, self.world)
        
        # Create output directory if specified
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Define observation space
        max_cell_value = max(c.value for c in CellType)
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(
                low=0,
                high=max_cell_value,  # Highest CellType value (derived, not hardcoded)
                shape=(self.world.height, self.world.width),
                dtype=np.int32
            ),
            'agents': spaces.Box(
                low=-1,
                high=max(self.world.height, self.world.width, 10),
                shape=(num_agents, 6),  # [pos_r, pos_c, class, active, done, counter_wait]
                dtype=np.float32
            )
        })
        
        # Define action space - one action per agent.
        # Each agent picks one of the discrete actions in AGENT_ACTIONS
        # (index order: MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, WAIT).
        self.action_space = spaces.MultiDiscrete([len(AGENT_ACTIONS)] * num_agents)
        
        # Simulation state
        self.agents: List[Agent] = []
        self.current_step = 0
        self.episode_log: List[Dict] = []
        self.total_collisions = 0
        self.episode_rewards = []
        self._pending_norm_violations: List[Tuple] = []  # violations detected this step, injected into next step's data
        self._last_reward_components: Dict[str, Dict[str, float]] = {}

        # Interpolation / smooth rendering
        self.interpolation_steps = interpolation_steps  # 0 = disabled
        self._last_terminated = False
        self._pending_push_pairs: List[Dict] = []  # push pairs deferred to next render's sub-frames
        # Optional callback invoked with each frame path right after it is saved
        # during render(). The web viewer uses it to stream frames to the browser
        # one-by-one instead of waiting for all sub-frames to finish rendering.
        self.frame_callback = None
        # Set True when step() renders this step's frames in-step (before prompting
        # an interactive agent), so the game loop's render() can skip re-drawing.
        self._rendered_during_step = False
    
    # ========================= Gym Environment Methods =========================
    
    def _action_to_enum(self, action_idx: int) -> Action:
        """Convert action index into the corresponding Action enum.

        Indices map to AGENT_ACTIONS; out-of-range indices fall back to WAIT.
        """
        if 0 <= action_idx < len(AGENT_ACTIONS):
            return AGENT_ACTIONS[action_idx]
        return Action.WAIT
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation of the environment."""
        grid_obs = self.world.grid.astype(np.int32)
        
        agent_obs = np.zeros((self.num_agents, 6), dtype=np.float32)
        for i, agent in enumerate(self.agents):
            agent_obs[i] = [
                agent.current_position[0],
                agent.current_position[1],
                agent_class_index(agent.agent_class),
                1.0 if agent.active else 0.0,
                1.0 if agent.done else 0.0,
                agent.counter_wait_time
            ]
        
        return {
            'grid': grid_obs,
            'agents': agent_obs
        }
    
    def _get_info(self, step_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get additional info about the current state.
        
        Args:
            step_data: Optional step data to include in info
            
        Returns:
            Dictionary containing step information
        """
        info = {
            'step': self.current_step,
            'total_collisions': self.total_collisions,
            'agents_done': sum(1 for a in self.agents if a.done),
            'agent_rewards': {a.agent_id: a.total_reward for a in self.agents},
            'queried_agent_id': self.queried_agent_id,
            'queried_agent_label': self.queried_agent_label
        }
        
        # Add step_data if provided
        if step_data:
            info['actions'] = step_data.get('actions', {})
            info['collisions'] = step_data.get('collisions', {})
            info['agent_states'] = step_data.get('agent_states', {})
            info['norm_violations'] = step_data.get('norm_violations', [])
        
        return info
    
    def add_agent(self, agent: Agent):
        """Add an agent to the simulation."""
        self.agents.append(agent)

    def set_queried_agent(self, queried_agent_id: Optional[int]) -> None:
        """Set the queried agent used for centered star and arrow suppression."""
        if queried_agent_id is None:
            self.queried_agent_id = None
            self.queried_agent_label = None
            return

        if queried_agent_id < 0 or queried_agent_id >= self.num_agents:
            raise ValueError(f"queried_agent_id must be between 0 and {self.num_agents - 1}")

        self.queried_agent_id = queried_agent_id
        self.queried_agent_label = f"C{queried_agent_id + 1}"

    def _is_queried_agent(self, agent_id: str) -> bool:
        return self.queried_agent_label is not None and agent_id == self.queried_agent_label
    
    def reset(self, seed: Optional[int] = None,
    options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Must contain 'agents': a list of pre-built Agent objects to register.

        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)

        options = options or {}
        agents = options.get('agents')
        if not agents:
            raise ValueError("reset() requires options['agents']: a non-empty list of pre-built Agent objects.")

        # Reset simulation state
        self.agents = []
        self.current_step = 0
        self.episode_log = []
        self.total_collisions = 0
        self.episode_rewards = []
        self._pending_norm_violations = []
        self._last_terminated = False
        self._pending_push_pairs = []
        self.world.zigzag_variant = None  # Reset zigzag variant for new episode
        
        for agent in agents:
            self.add_agent(agent)
        
        # Sync previous_position with starting position for all agents
        for agent in self.agents:
            agent.previous_position = agent.current_position

        observation = self._get_observation()
        info = self._get_info()

        return observation, info
    
    def step(self, actions: Optional[np.ndarray] = None) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            actions: Array of actions for each agent (if None, agents use their default behavior)
        
        Returns:
            observation: Current observation after step
            reward: Total reward for this step (sum of all agent rewards)
            terminated: Whether episode is complete (all agents done)
            truncated: Whether episode was truncated (max steps reached)
            info: Additional information
        """

        self.current_step += 1

        # set agent.active to True for agents that have arrived
        self._update_agent_active_status()
        # update agent current position 
        for agent in self.agents:
            if agent.next_position is not None:
                agent.previous_position = agent.current_position  # save for sub-frame rendering
                agent.current_position = agent.next_position
            agent.next_position = None
            # Save visual state before it's updated so sub-frames show pre-step appearance
            agent.previous_target_wait_time = agent.target_wait_time
            agent.previous_target_index = getattr(agent, 'target_index', 0)
            agent.previous_done = agent.done

        # If an agent was pushed away from their target, clear the wait timer
        for agent in self.agents:
            if agent.target_wait_time != 0 and not self._agent_at_goal(agent):
                agent.target_wait_time = 0

        # Track X-cell visits for xcell_must_visit
        norm = getattr(self, 'norm_type', None)
        if norm and xcell_requires_visit(norm):
            x_overlay = getattr(self.world, 'x_overlay', set())
            for agent in self.agents:
                if not agent.active or agent.done:
                    continue
                r, c = agent.current_position
                cell = self.world.grid[r, c]
                is_x_cell = (cell == CellType.AFFORDANCE_X.value) or (agent.current_position in x_overlay)
                if is_x_cell:
                    agent.visited_x_cells.add(agent.current_position)

        # Advance agents through their target sequence.
        # - At the first target (index 0): require a 3-step wait before advancing.
        # - Past the first target (index > 0): advance immediately on arrival.
        # - hold_at_first_target pins an agent to its first goal indefinitely,
        #   overriding the 3-step wait. The test stage sets it on the scripted
        #   agents staged to hold the waiting zone, and clears it once the subject
        #   reaches the zone boundary (see release_region_holders).
        # For interactive agents on a free-choice goal, "at goal" means the agent
        # settled on a norm-compliant chair / door rather than a preset cell.
        for agent in self.agents:
            if not agent.active or agent.done or agent.current_target is None:
                continue
            if self._agent_at_goal(agent):
                if agent.target_index == 0 and len(agent.target_positions) > 1:
                    if getattr(agent, 'hold_at_first_target', False):
                        continue
                    if agent.target_wait_time < 3:
                        agent.target_wait_time += 1
                    else:
                        agent.advance_to_next_target()
                elif agent.target_index > 0:
                    agent.advance_to_next_target()

        # Keep the seat-in-order fill frontier in step with current occupancy.
        self._sync_seat_in_order_frontier()

        # Save the action that caused this step's movement (for sub-frame arrow display)
        for agent in self.agents:
            agent.previous_action = agent.action

        step_data = self._simulation_step()
        
        # Assign rewards and calculate total
        total_reward, step_rewards = self._compute_reward(step_data)
        
        # Get observation
        observation = self._get_observation()
        
        # Check termination conditions
        terminated = self._is_done()
        truncated = self.current_step >= self.max_steps and not terminated
        
        # Get info with step_data
        info = self._get_info(step_data)
        info['step_rewards'] = step_rewards
        info['reward_components'] = self._last_reward_components
        
        # Print step data for debugging/monitoring
        self._print_step_data(step_data)
        
        self._last_terminated = terminated or truncated

        return observation, total_reward, terminated, truncated, info
    
    # ========================= Simulation Logic Methods =========================
    
    def _simulation_step(self) -> Dict[str, Any]:
        """
        Execute one simulation step.
        
        Flow:
        1. Get all agent actions (includes ordering transitions, collision detection/resolution, and planning)
        2. Execute all agent actions (set next positions)
        
        Returns:
            Step data including actions taken
        """
        
        step_data = {
            'step': self.current_step,
            'actions': {},  # Dict[agent_id, {'action': str, 'current_position': tuple, 'next_position': tuple}]
            'collisions': {},  # Dict[position, List[Agent]] - only present if collisions detected
            'agent_states': {},  # Dict[agent_id, {'active': bool, 'done': bool, 'class': str}]
            'norm_violations': [],  # List of (agent_id, from_pos, to_pos) tuples — non-spacing violations are injected from the previous step
            'possible_norm_violations': {},  # Dict[agent_id, List[str]] of actions blocked by norm check
        }

        # Inject non-spacing violations detected in the previous step.
        # This means the violation appears in the step_data of the step AFTER it was committed,
        # so renderers and LLM agents see it at the right moment.
        if self._pending_norm_violations:
            step_data['norm_violations'].extend(self._pending_norm_violations)
            violated_agent_ids = {v[0] for v in self._pending_norm_violations}
            has_no_norm_system_agent = any(
                a.is_oracle and a.norm is None
                for a in self.agents
            )
            for agent in self.agents:
                if (
                    not agent.is_oracle
                    and agent.active
                    and not agent.done
                    and (agent.agent_id in violated_agent_ids or has_no_norm_system_agent)
                ):
                    agent.pending_violation_feedback = self.current_step
                    if isinstance(agent, LLMAgent):
                        agent.pending_violation_agent_ids = sorted(violated_agent_ids)
            self._pending_norm_violations = []
        
        for agent in self.agents:
            step_data['agent_states'][agent.agent_id] = {
                'active': agent.active,
                'done': agent.done,
                'agent_class': agent.agent_class,
                'arrival_time': agent.arrival_time
            }
        
        print(f"\n{'='*70}")
        print(f"STEP {step_data['step']}")
        print(f"{'='*70}")
        print(f"Agent States:")
        for agent_id, state in step_data['agent_states'].items():
            print(f"  {agent_id}: active={state['active']}, done={state['done']}, "
                  f"agent_class={state['agent_class']}, arrival={state['arrival_time']}")

        # Update current step for all agents before action selection so LLM prompts see the correct step
        for agent in self.agents:
            agent.current_step = self.current_step

        # Get all agent actions (includes ordering, collision resolution, and planning)
        agent_actions = self._get_all_agent_actions(step_data)
        
        # Execute all agent actions (set next positions)
        step_data = self._get_all_agent_next_positions(step_data, agent_actions)
        
        # Check for norm violations by no-norm agents
        self._check_norm_violations(step_data)

        # Record each agent's chosen movement for the per-agent rhythm norms,
        # which read its history. Done
        # after the violation check so this step is judged on the history up to
        # the previous step, the same state the agents planned against.
        self._record_agent_moves(step_data)
    
        # Store for next step
        self.episode_log.append(step_data)
        
        return step_data

    def _compute_reward(self, step_data: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Assign rewards to agents based on step outcomes and return total reward.
        
        Args:
            step_data: Dictionary containing step information including actions and collisions
            
        Returns:
            Tuple of (total_reward, step_rewards_dict)
            - total_reward: Total reward for this step (sum of all agent step rewards)
            - step_rewards_dict: Dictionary mapping agent_id to their reward for this step
        """
        step_rewards = {}
        # Read-only breakdown for reward ablations; the shared reward is unchanged.
        reward_components = {}
        
        # Initialize all agent rewards to 0 for this step
        for agent in self.agents:
            step_rewards[agent.agent_id] = 0.0
            reward_components[agent.agent_id] = dict(goal=0.0, step=0.0, collision=0.0, violation=0.0)
        
        # Apply collision penalties — only to agents that moved into the
        # collision cell by their own choice. An agent that stood still and was
        # collided into is not at fault, and neither is an agent whose move was
        # itself a displacement forced by last step's collision resolution.
        if 'collisions' in step_data:
            for pos, conflicting_agents in step_data['collisions'].items():
                for agent in conflicting_agents:
                    if not agent.active or agent.done:
                        continue
                    if agent.previous_position == agent.current_position:
                        continue  # stayed put; another agent ran into them
                    if agent.displaced_at_step == self.current_step - 1:
                        continue  # moved only because the env displaced them
                    step_rewards[agent.agent_id] -= COLLISION_PENALTY
                    reward_components[agent.agent_id]['collision'] -= COLLISION_PENALTY
        
        # Apply norm violation penalties
        for violation in step_data.get('norm_violations', []):
            agent_id = violation[0]
            if agent_id in step_rewards:
                step_rewards[agent_id] -= NORM_VIOLATION_PENALTY
                reward_components[agent_id]['violation'] -= NORM_VIOLATION_PENALTY

        # Apply other rewards
        for agent in self.agents:
            if agent.active:
                # Reward for each newly completed goal this step
                completions = agent.target_index - getattr(agent, 'previous_target_index', 0)
                if completions > 0:
                    step_rewards[agent.agent_id] += GOAL_REWARD
                    reward_components[agent.agent_id]['goal'] += GOAL_REWARD
                if not agent.done:
                    # Step cost for agents that took actions
                    if agent.agent_id in step_data['actions']:
                        step_rewards[agent.agent_id] -= STEP_COST
                        reward_components[agent.agent_id]['step'] -= STEP_COST
        
        self._last_reward_components = reward_components

        # Update agent total rewards
        for agent in self.agents:
            agent.total_reward += step_rewards[agent.agent_id]
        
        # Calculate total reward
        total_reward = sum(step_rewards.values())
        self.episode_rewards.append(total_reward)
        
        return total_reward, step_rewards


    def _update_agent_active_status(self) -> bool:
        for agent in self.agents:
            if not agent.active:
                if self.current_step >= agent.arrival_time:
                    agent.active = True
                    # Name agents in on-map arrival order (not by global id) so
                    # no two visible agents share a name. Assigned once, the first
                    # time the agent enters the map.
                    if not getattr(agent, '_arrival_named', False):
                        self._assign_arrival_name(agent)
                        agent._arrival_named = True

            if agent.done:
                agent.active = False

    def _assign_arrival_name(self, agent: Agent) -> None:
        """Give a newly arrived agent a display name not currently shown on the
        map. Names free up automatically when an agent leaves (done agents are
        not counted as in-use), so the 50-name pool never runs out for the
        handful of agents visible at once. The participant's "You" label and any
        other interactive agent's name are left untouched."""
        if not agent.is_oracle:
            return
        from src.env.agent_names import AGENT_NAMES
        used = {
            a.name for a in self.agents
            if a is not agent and a.active and not a.done
        }
        for name in AGENT_NAMES:
            if name not in used:
                agent.name = name
                return
        # More visible agents than available names: fall back to the raw id.
        agent.name = agent.agent_id
    
    

    def _can_move_to_position(self, agent: Agent, target_position: Tuple[int, int]) -> bool:
        """Check if agent can move to target position."""
        occupant = None
        for other_agent in self.agents:
            if other_agent.active and not other_agent.done and other_agent.current_position == target_position:
                occupant = other_agent
                break
        
        if not occupant:
            return True
        
        if occupant.action == Action.MOVE_UP or occupant.action == Action.MOVE_DOWN or occupant.action == Action.MOVE_LEFT or occupant.action == Action.MOVE_RIGHT:
            # Occupant is moving away, so allow move
            return True

        # Any class can move into any other class's position (collision resolved separately)
        return True


    def _get_all_agent_actions(self, step_data: Dict[str, Any]) -> Dict[str, Action]:
        """
        set agent.action for all agents including:
        1. Ordering transitions
        2. Collision detection and resolution
        3. Planning actions for remaining agents
        
        Args:
            step_data: Step data dictionary to populate with collisions info and current positions
            
        Returns:
            Dictionary mapping agent_id to Action
        """
        agent_actions = {}
        collision_actions = {}
        
        # STEP 1: Detect and resolve collisions
        collisions = collision_handler.detect_collisions(self.agents)
        
        if collisions:
            self.total_collisions += len(collisions)
            step_data['collisions'] = collisions
            
            collision_actions = collision_handler.resolve_collisions(
                self.agents, self.world, collisions)
            
            if collision_actions:
                # Add collision resolution actions to agent_actions
                agent_actions.update(collision_actions)
                # Agents displaced by the resolver (everyone except the PUSH
                # winner) did not choose their move: exempt them from this
                # step's norm-violation checks, and from a collision penalty
                # if the forced move lands them in a new collision next step.
                displaced_ids = {
                    aid for aid, act in collision_actions.items()
                    if act != Action.PUSH
                }
                step_data['collision_displaced'] = displaced_ids
                for agent in self.agents:
                    if agent.agent_id in displaced_ids:
                        agent.displaced_at_step = self.current_step

        # STEP 2: Get planning actions for remaining agents
        candidates = []
        for agent in self.agents:
            if not agent.active:
                continue
            if agent.agent_id in agent_actions:
                continue
            candidates.append(agent)
        
        # get current positions for all agents
        all_current_positions = {}
        agent_classes = {}
        all_target_positions = {}
        all_target_indices = {}
        for agent in self.agents:
                if agent.active and not agent.done:
                    all_current_positions[agent.agent_id] = agent.current_position
                    agent_classes[agent.agent_id] = agent.agent_class
                    all_target_positions[agent.agent_id] = agent.current_target
                    all_target_indices[agent.agent_id] = agent.target_index

        print("all agent current positions:", all_current_positions)

        # Make agents accessible via world so norm checks can read state
        self.world.agents = self.agents

        # Render the current frame before LLM/human agents are called so the image on
        # disk matches the positions text in the user prompt (both reflect step N).
        has_interactive_agent = (
            any(not a.is_oracle for a in candidates)
            or any(
                not a.is_oracle and a.active and not a.done and a not in candidates and a.agent_id in agent_actions
                for a in self.agents
            )
        )
        if has_interactive_agent and self.render_mode == 'rgb_array' and self.output_dir:
            prev_step_data = self.episode_log[-1] if self.episode_log else None
            prompt_step_data = step_data if step_data.get('norm_violations') else prev_step_data
            # Also write the previous step's sub-frames so they are on disk before
            # the human/LLM agent is prompted (render() won't run until after step()).
            if self.interpolation_steps > 1 and self.current_step > 0:
                prev_step = self.current_step - 1
                for k in range(1, self.interpolation_steps):
                    frac = k / self.interpolation_steps
                    interp_positions = {
                        agent.agent_id: (
                            agent.previous_position[0] + (agent.current_position[0] - agent.previous_position[0]) * frac,
                            agent.previous_position[1] + (agent.current_position[1] - agent.previous_position[1]) * frac,
                        )
                        for agent in self.agents if agent.active
                    }
                    sub_path = os.path.join(self.output_dir, f"step_{prev_step:03d}_sub{k}.png")
                    self._visualize_step(
                        save_path=sub_path,
                        step_data=prev_step_data,
                        interp_positions=interp_positions,
                        interp_frac=frac,
                        display_step=prev_step,
                        push_pairs_override=self._pending_push_pairs,
                    )
                    # Stream this frame now. This in-step render runs AFTER the
                    # previous step's action has been committed (positions updated
                    # at the top of step()), so it shows the result of the human's
                    # last move and animates it right before they're prompted. The
                    # game loop's post-step render() produces the same filenames and
                    # is deduped by the streamer, so streaming here (not there) is
                    # what keeps the displayed motion in sync with the keypress
                    # instead of lagging a step behind.
                    # Sub-frames never draw the violation X, so pass no violations.
                    if self.frame_callback is not None:
                        self.frame_callback(sub_path, [])
            if self.interpolation_steps > 0:
                fig_path = os.path.join(self.output_dir, f"step_{self.current_step:03d}_sub0.png")
            else:
                fig_path = os.path.join(self.output_dir, f"step_{self.current_step:03d}.png")
            # The keyframe shows agents at their freshly committed positions, so the
            # norm-violation X must reflect THIS step's violations (already injected
            # above), not the previous step's — otherwise the X (drawn at the agent's
            # current cell) only reaches the browser one step late. Keep prev_step_data
            # for action/push overlays, but swap in the current violations.
            # Web viewer: swap in unconditionally (even when empty), so a violation
            # already drawn on the previous step's keyframe never resurfaces here.
            # Each violation instance then appears on exactly one streamed keyframe,
            # and a repeat violation (same agent, same cells — e.g. retrying a move
            # after a collision pushed the agent back) still gets its own X + modal.
            keyframe_step_data = prev_step_data
            if prev_step_data is not None and (
                self.frame_callback is not None or step_data.get('norm_violations')
            ):
                keyframe_step_data = {**prev_step_data,
                                      'norm_violations': step_data.get('norm_violations', [])}
            keyframe_violations = (keyframe_step_data or {}).get('norm_violations', [])
            self._visualize_step(save_path=fig_path, step_data=keyframe_step_data)
            if self.frame_callback is not None:
                # Pass the violations actually drawn on this keyframe so the viewer
                # can fire the modal with the X frame. step_data isn't in episode_log
                # yet at this point, so the callback must not rely on episode_log[-1].
                self.frame_callback(fig_path, keyframe_violations)
                # Tell render() this step's frames are already drawn + streamed.
                self._rendered_during_step = True

        for agent in self.agents:
            if not agent.active or agent.done:
                continue
            if not isinstance(agent, LLMAgent):
                continue
            if agent not in candidates and agent.agent_id in agent_actions:
                agent.get_action(
                    self.world,
                    all_current_positions,
                    self.output_dir,
                    actual_action=agent_actions[agent.agent_id],
                )

        for agent in candidates:
            if not agent.is_oracle:
                action = agent.get_action(self.world, all_current_positions, self.output_dir)
            else:
                action = agent.get_action(self.world, all_current_positions, agent_classes, collision_actions,
                                          violation_log=step_data['possible_norm_violations'],
                                          all_target_positions=all_target_positions,
                                          all_target_indices=all_target_indices)
            agent_actions[agent.agent_id] = action
            # Set next_position immediately so later agents see this reservation
            # during yellow-capacity checks (agents_in_yellow uses next_position if set)
            if action is not None:
                next_pos = self.world.transition(agent.current_position, action)
                agent.next_position = next_pos if next_pos else agent.current_position
        
        # Set all actions on agents at once
        for agent_id, action in agent_actions.items():
            agent = next((a for a in self.agents if a.agent_id == agent_id), None)
            if agent:
                agent.action = action
        
        # store the queue line
        if self.norm_type and is_line_norm(self.norm_type):
            target_position = get_queue_start(self.world, self.norm_type)
            queue_cells = generate_queue_line(target_position, self.world, self.norm_type)
            eligible_agents = {a.agent_id for a in self.agents if a.target_index == 0}
            self.world.previous_queue = get_queue(
                queue_cells,
                all_current_positions,
                previous_queue=self.world.previous_queue,
                eligible_agents=eligible_agents,
            )
            if len(self.world.previous_queue) == 1 and self.world.previous_queue[0] == None:
                self.world.previous_queue = {} 
            print("the world queue:", self.world.previous_queue)
            
        print("all agent's actions:", agent_actions)
        return agent_actions
    
    def _collect_push_pairs(self, step_data: Dict[str, Any]) -> None:
        """Build pusher->pushee mappings for visualization."""
        collisions = step_data.get('collisions', {})
        actions = step_data.get('actions', {})
        push_pairs = []

        if not collisions:
            step_data['push_pairs'] = push_pairs
            return

        for pos, agents in collisions.items():
            if not agents:
                continue
            # Prefer the agent whose action is PUSH; fall back to first in list
            pusher = next((a for a in agents if getattr(a, 'action', None) == Action.PUSH), None)
            if pusher is None:
                pusher = agents[0]

            pusher_pos = actions.get(pusher.agent_id, {}).get('current_position', pusher.current_position)

            for agent in agents:
                if agent == pusher:
                    continue
                target = actions.get(agent.agent_id, {}).get('next_position', agent.next_position)
                if target is None:
                    continue
                push_pairs.append({
                    'pusher': pusher.agent_id,
                    'pushed': agent.agent_id,
                    'start_pos': pusher_pos,
                    'target_pos': target,
                })

        step_data['push_pairs'] = push_pairs

    
    def _get_all_agent_next_positions(self, step_data: Dict[str, Any], agent_actions: Dict[str, Action]) -> None:
        """
        Execute all agent actions by setting agent.next_position.
        Agents already in step_data['actions'] (from ordering/collisions) are skipped.
        
        Args:
            step_data: Step data dictionary to populate with action info
            agent_actions: Dictionary mapping agent_id to Action
        """
        # Sort by arrival_time (earlier arrivals execute first) for determinism
        sorted_agent_ids = sorted(
            agent_actions.keys(),
            key=lambda agent_id: next(
                (a.arrival_time, a.agent_id)
                for a in self.agents if a.agent_id == agent_id
            ),
        )
        
        agent_classes = {a.agent_id: a.agent_class for a in self.agents}

        for agent_id in sorted_agent_ids:
            agent = next((a for a in self.agents if a.agent_id == agent_id), None)
            if not agent:
                continue
            
            # Execute movement actions
            intended_pos = self.world.transition(agent.current_position, agent.action)
                
            if self._can_move_to_position(agent, intended_pos):
                agent.next_position = intended_pos
            else:
                agent.next_position = agent.current_position
                agent._blocked_by_stronger = True

            # Only resolve norm conflicts for oracle agents that have a personal
            # norm (newcomer agents choose their own actions freely). Skip agents
            # whose move was forced by collision displacement: vetoing it would
            # put them back on top of the agent they collided with — and an
            # agent escaping an overlap necessarily passes through adjacency, so
            # the spacing re-check would trap the pair on one cell forever.
            # (Displaced agents are already exempt from violation flags/penalties.)
            if (agent.norm is not None and agent.is_oracle
                    and agent.agent_id not in step_data.get('collision_displaced', set())):
                # For spacing-norm agents: resolve same-cell or spacing conflicts.
                if is_spacing_norm(agent.norm):
                    resolve_spacing_norm_conflict(
                        agent, step_data, agent_classes,
                        self.agents, self.world, self._can_move_to_position,
                    )
                # For occupation region agents (capacity / single-class): yield if
                # entering the zone would break the norm given others' next positions.
                elif is_occupation_region_norm(agent.norm):
                    resolve_occupation_region_norm_conflict(
                        agent, step_data, agent_classes,
                        self.agents, self.world, self._can_move_to_position,
                    )

            step_data['actions'][agent_id] = {
                'action': agent.action,
                'current_position': agent.current_position,
                'next_position': agent.next_position,
                'intended_position': intended_pos  # Track intended position for norm violation checks
            }

        # Line (queue) norms only: fill the open end-of-queue cell if a norm-following
        # oracle agent is standing next to it but nobody committed to stepping in. Runs
        # once, after every next_position is set, since it reasons about all of them at
        # once — unlike the per-agent spacing/occupation resolvers above.
        if is_line_norm(self.norm_type):
            resolve_queue_norm_conflict(
                self.agents, step_data, self.world, self.norm_type,
                self._can_move_to_position,
            )

        print("all agent's next positions:", {a.agent_id: a.next_position for a in self.agents})
        collision_handler.resolve_swap_pairs(self.agents, self.world, step_data, self._is_queried_agent)
        self._collect_push_pairs(step_data)
        return step_data

    def _current_goal_is_free_choice(self, agent) -> bool:
        """True if the agent's CURRENT goal is a free, hidden, compliance-graded
        choice. Only newcomer (non-oracle) agents have free-choice goals; oracle
        agents always plan toward a preset compliant cell."""
        if agent.is_oracle:
            return False
        idx = free_choice_goal_index(agent.norm, len(agent.target_positions))
        return idx is not None and agent.target_index == idx

    def _sync_seat_in_order_frontier(self) -> None:
        """Advance the seat-in-order fill frontier to match current occupancy,
        so free-choice agents are graded against the right next seat even when
        no oracle agent advances it during planning.

        Call once per step, after the goal-advance loop (which latches a settled
        agent to its seat) — advancing the frontier then never strands an agent
        mid-sit. Idempotent for fixed positions, so it is safe alongside oracle
        advancement. No-op under any norm outside the in-order family."""
        if not is_seat_in_order_norm(self.norm_type):
            return
        positions = {
            a.agent_id: a.current_position
            for a in self.agents if a.active and not a.done
        }
        advance_seat_in_order_frontier(positions, self.world, self.norm_type)

    def _agent_at_goal(self, agent) -> bool:
        """Whether the agent has reached its current goal. For free-choice goals
        this is settling on any chair / door (seat and door norms are norm-blind
        here, so task success is independent of violations) or a permitted
        waiting-zone cell (region norms), latched once settled; otherwise it is
        arrival at the preset target cell."""
        if self._current_goal_is_free_choice(agent):
            # RL tasks score reaching the public goal category separately
            # from compliance; all other subjects retain the benchmark behavior.
            if (getattr(agent, "task_goal_mode", "benchmark") == "category"
                    and is_occupation_region_norm(agent.norm)):
                from src.norms.occupation import waiting_zone_cells
                return agent.current_position in waiting_zone_cells(self.world)
            positions = {
                a.agent_id: a.current_position
                for a in self.agents if a.active and not a.done
            }
            return free_goal_settled(agent, self.world, positions)
        return (agent.current_target is not None
                and agent.current_position == agent.current_target)

    def _record_agent_moves(self, step_data: Dict[str, Any]) -> None:
        """Update agent.move_steps from this step's moves.

        A move counts only when the agent actually changes cell and the move was
        its own choice: a move forced by collision displacement is not a move
        in its rhythm.
        """
        displaced_ids = step_data.get('collision_displaced', set())
        for agent in self.agents:
            action_data = step_data['actions'].get(agent.agent_id)
            if action_data is None or agent.agent_id in displaced_ids:
                continue
            from_pos = action_data.get('current_position')
            to_pos = action_data.get('next_position')
            if from_pos is None or to_pos is None or tuple(from_pos) == tuple(to_pos):
                continue
            agent.move_steps.add(self.current_step)

    def _check_norm_violations(self, step_data: Dict[str, Any]) -> None:
        """Check every agent that acted this step against the global norm.

        No agent is exempt for being a norm-following oracle: an oracle plans to
        comply, but whether it actually complied is measured here like everyone
        else. (Displaced and colliding agents stay exempt — their move was forced,
        not chosen.)
        """

        if not self.norm_type:
            return

        # Build current positions dict
        all_current_positions = {a.agent_id: a.current_position for a in self.agents if a.active}
        agent_classes = {a.agent_id: a.agent_class for a in self.agents}

        # Refresh world.current_queue so it reflects actual positions at this step,
        # not a stale value left by oracle agents that were resolved in the ordering phase.
        if is_line_norm(self.norm_type):
            target_position = get_queue_start(self.world, self.norm_type)
            queue_cells = generate_queue_line(target_position, self.world, self.norm_type)
            eligible_agents = {a.agent_id for a in self.agents if a.target_index == 0}
            self.world.current_queue = get_queue(
                queue_cells,
                all_current_positions,
                previous_queue=self.world.previous_queue,
                eligible_agents=eligible_agents,
            )

        colliding_agents = {
            a.agent_id
            for agents_list in step_data.get('collisions', {}).values()
            for a in agents_list
        }
        # Agents whose move this step was forced by collision displacement are
        # never at fault for a violation that move causes.
        displaced_ids = step_data.get('collision_displaced', set())

        # For spacing norms: pairwise check over all agents, independent of per-agent filters.
        if is_spacing_norm(self.norm_type):
            spacing_violations = check_spacing_violations(
                step_data, agent_classes, self.norm_type, self.agents,
                exempt_agent_ids=displaced_ids,
            )
            self._pending_norm_violations.extend(spacing_violations)
            return

        # For occupation region norms: flag agents that entered the zone in
        # violation of the capacity / single-class rule.
        if is_occupation_region_norm(self.norm_type):
            region_violations = check_occupation_region_violations(
                step_data, self.norm_type, agent_classes, self.agents, self.world,
                exempt_agent_ids=displaced_ids,
            )
            self._pending_norm_violations.extend(region_violations)
            # Whether a violating entry also costs the agent its waiting-zone
            # goal is not decided here: the zone cell is a free-choice goal, so
            # _agent_at_goal grades a newcomer's position against the capacity /
            # single-class rule directly (see occupation.free_goal_satisfied),
            # while an oracle completes on arrival at its preset cell.
            return

        for agent in self.agents:
            if agent.agent_id not in step_data['actions']:
                continue

            # Skip agents currently in a collision
            if agent.agent_id in colliding_agents:
                continue

            # Skip agents whose move was forced by collision displacement — including
            # bystanders pushed out of the way by a displacement chain. They did not
            # choose this move, so it is not their violation.
            if agent.agent_id in displaced_ids:
                continue

            action_data = step_data['actions'][agent.agent_id]
            from_pos = action_data.get('current_position')
            to_pos = action_data.get('next_position')  # Use next_position (actual post-collision movement) not intended_position

            if from_pos is None or to_pos is None:
                continue
            if from_pos == to_pos:
                # Staying put is only judged under line norms, where waiting at a
                # non-queue cell is itself a violation (see lining.check_queue_line),
                # and only when the agent chose to stay: a move that collision /
                # swap resolution reverted is a forced stay, not a wait.
                intended = action_data.get('intended_position')
                chose_to_stay = intended is None or tuple(intended) == tuple(from_pos)
                if not (is_line_norm(self.norm_type) and chose_to_stay):
                    continue

            # Get target position for this agent. For interactive agents on a
            # free-choice occupation goal there is no preset target cell — grade
            # any chair/door they step onto (target_position=None lets the
            # occupation check flag every non-compliant chair/door, not just one).
            if self._current_goal_is_free_choice(agent):
                target_pos = None
            else:
                target_pos = agent.current_target
                if target_pos is None:
                    continue

            # Skip agents heading to exit (already served at counter, not joining any queue)
            # Only applies to line norms — other norms check all moves regardless of target index
            if agent.target_index >= 1 and is_line_norm(self.norm_type):
                continue

            # An agent that was obliged to take the end-of-queue cell but lost it to a
            # peer this step is judged with that cell counted as occupied — the queue
            # norm cannot require a move into a cell another agent is taking. Its other
            # queue rules still apply.
            positions_for_check = all_current_positions
            claim = step_data.get('queue_slot_claim')
            if claim and agent.agent_id in claim['yielded']:
                positions_for_check = {**all_current_positions,
                                       claim['claimer']: claim['slot']}

            # Check if this move violates the global norm
            violates = check_norm_violation(
                self.norm_type,
                from_pos,
                to_pos,
                target_pos,
                positions_for_check,
                self.world,
                agent.agent_id,
                agent_classes
            )

            if violates:
                # Store in pending buffer — will be injected into next step's data so that
                # renderers and LLM agents see it at the step when the agent is at to_pos.
                self._pending_norm_violations.append((agent.agent_id, from_pos, to_pos))
                print(f"  [NORM VIOLATION] {agent.agent_id} violated {self.norm_type} norm: {from_pos} -> {to_pos} (will appear in next step)")

    def _is_done(self) -> bool:
        """Check if episode should end."""
        customers = [a for a in self.agents if isinstance(a, Agent)]

        if not customers:
            return True

        all_done = all(c.done for c in customers)
        return all_done or self.current_step >= self.max_steps
    
    def _print_step_data(self, step_data: Dict[str, Any]) -> None:
        """Print detailed information about the current step."""
        
        
        # Create agent lookup for class info
        agent_lookup = {a.agent_id: a for a in self.agents if a.active}
        
        # Print agent actions from step data
        for agent_id, action_data in step_data['actions'].items():
            action_str = action_data['action'] if action_data['action'] else 'None'
            agent = agent_lookup.get(agent_id)
            agent_class_str = f", agent_class={agent.agent_class}" if agent else ""
            
            # Check if action was WAIT due to blocking
            blocked_str = ""
            if action_str == 'wait' and agent and hasattr(agent, '_blocked_by_stronger'):
                blocked_str = " [BLOCKED by stronger agent]"
                delattr(agent, '_blocked_by_stronger')
            
            # Use 'position' key if it exists, otherwise use 'current_position'
            pos = action_data.get('position', action_data.get('current_position', 'N/A'))
            print(f"  {agent_id}: action={action_str}, "
                  f"pos={pos}{agent_class_str}{blocked_str}")
        
        # Check for collisions at END of step
        positions = {}
        for agent in self.agents:
            if agent.active and not agent.done:
                pos = agent.current_position
                if pos not in positions:
                    positions[pos] = []
                positions[pos].append(agent.agent_id)
        
        # Report any positions with multiple agents
        for pos, agent_ids in positions.items():
            if len(agent_ids) > 1:
                print(f"  [WARN] COLLISION at {pos}: {agent_ids}")

        # Print norm violations (actions that actually violated the norm)
        norm_violations = step_data.get('norm_violations', [])
        if norm_violations:
            print(f"  [NORM VIOLATIONS] {norm_violations}")
        else:
            print(f"  [NORM VIOLATIONS] none")

        # Print possible norm violations recorded during planning
        possible_norm_violations = step_data.get('possible_norm_violations', {})
        if possible_norm_violations:
            print(f"  [POSSIBLE NORM VIOLATIONS] {possible_norm_violations}")
        else:
            print(f"  [POSSIBLE NORM VIOLATIONS] none")
    
    # ========================= Visualization Methods =========================
    
    def render(self):
        """Render the current state of the environment."""
        if self.render_mode is None:
            return

        if self.render_mode == 'rgb_array':
            step_data = self.episode_log[-1] if self.episode_log else None

            if self.interpolation_steps > 0:
                fig_path = os.path.join(self.output_dir, f"step_{self.current_step:03d}_sub0.png")
            else:
                fig_path = os.path.join(self.output_dir, f"step_{self.current_step:03d}.png")

            # When a frame_callback (web viewer) is attached, step() already
            # rendered AND streamed this step's frames in-step — right after the
            # action was committed, so the displayed motion stays in sync with the
            # keypress. Re-rendering identical frames here would just waste ~0.5s
            # (4 matplotlib renders) and delay the next in-step render, so skip the
            # drawing and only run the side-effects below. (Eval/LLM paths have no
            # frame_callback and render normally.)
            skip_draw = self.frame_callback is not None and self._rendered_during_step
            self._rendered_during_step = False

            if not skip_draw:
                # --- Sub-frames between previous step and current step ---
                # Use push pairs from the PREVIOUS step so the push arrow only appears
                # while agents are departing (not while they are still approaching).
                if self.interpolation_steps > 1 and self.current_step > 0:
                    prev_step = self.current_step - 1
                    sub_frame_push_pairs = self._pending_push_pairs

                    for k in range(1, self.interpolation_steps):
                        frac = k / self.interpolation_steps
                        interp_positions = {
                            agent.agent_id: (
                                agent.previous_position[0] + (agent.current_position[0] - agent.previous_position[0]) * frac,
                                agent.previous_position[1] + (agent.current_position[1] - agent.previous_position[1]) * frac,
                            )
                            for agent in self.agents if agent.active
                        }
                        sub_path = os.path.join(self.output_dir, f"step_{prev_step:03d}_sub{k}.png")
                        self._visualize_step(
                            save_path=sub_path,
                            step_data=step_data,
                            interp_positions=interp_positions,
                            interp_frac=frac,
                            display_step=prev_step,
                            push_pairs_override=sub_frame_push_pairs
                        )
                        if self.frame_callback is not None:
                            self.frame_callback(sub_path, [])

                # --- Keyframe for the current (integer) step ---
                # When interpolation is active, name as _sub0 so all frames share a
                # consistent step_NNN_subK pattern and sort unambiguously.
                kf_violations = (step_data or {}).get('norm_violations', [])
                self._visualize_step(
                    save_path=fig_path,
                    step_data=step_data
                )
                if self.frame_callback is not None:
                    self.frame_callback(fig_path, kf_violations)

            # Save this step's push pairs so the next render's sub-frames can show
            # the push arrow while agents are animating away from the collision.
            self._pending_push_pairs = step_data.get('push_pairs', []) if step_data else []

            # --- Auto-assemble MP4 when episode ends ---
            if self._last_terminated:
                self._save_video()

            img = mpimg.imread(fig_path)
            return img
    
    def _save_video(self, fps: int = 2):
        """Assemble all step PNG frames (including sub-frames) into simulation.mp4."""
        import re
        import subprocess
        import shutil
        import sys

        def _natural_sort_key(fname):
            parts = re.split(r'(\d+)', fname)
            return [int(p) if p.isdigit() else p for p in parts]

        frames_dir = self.output_dir
        all_files = sorted(
            [f for f in os.listdir(frames_dir) if f.endswith('.png') and f.startswith('step_')],
            key=_natural_sort_key
        )

        if not all_files:
            print("[render] No step frames found — skipping MP4 assembly.")
            return

        video_path = os.path.join(frames_dir, "simulation.mp4")

        # Resolve ffmpeg binary: system ffmpeg → imageio_ffmpeg in current env →
        # imageio_ffmpeg discovered via subprocess (handles venv/miniconda differences)
        ffmpeg_bin = shutil.which('ffmpeg')
        if not ffmpeg_bin:
            try:
                import imageio_ffmpeg
                ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                pass
        if not ffmpeg_bin:
            try:
                result = subprocess.run(
                    [sys.executable, '-c',
                     'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())'],
                    capture_output=True, text=True, check=True
                )
                ffmpeg_bin = result.stdout.strip() or None
            except Exception:
                pass
        if not ffmpeg_bin:
            # Last resort: try other python3 interpreters on PATH
            for py in ['python3', 'python']:
                py_path = shutil.which(py)
                if py_path and py_path != sys.executable:
                    try:
                        result = subprocess.run(
                            [py_path, '-c',
                             'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())'],
                            capture_output=True, text=True, check=True
                        )
                        ffmpeg_bin = result.stdout.strip() or None
                        if ffmpeg_bin:
                            break
                    except Exception:
                        pass

        if not ffmpeg_bin:
            print("[render] No ffmpeg available — skipping MP4 assembly.")
            print("         Run: pip install imageio[ffmpeg]")
            return

        list_file = os.path.join(frames_dir, '_frame_list.txt')
        try:
            with open(list_file, 'w', encoding='utf-8') as f:
                for fname in all_files:
                    f.write(f"file '{os.path.join(frames_dir, fname)}'\n")
                    f.write(f"duration {1.0 / fps}\n")
            cmd = [
                ffmpeg_bin, '-y', '-f', 'concat', '-safe', '0',
                '-i', list_file,
                '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                '-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
                video_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"  [Video saved to {video_path}]")
        except Exception as e:
            print(f"[render] ffmpeg failed: {e}")
        finally:
            if os.path.exists(list_file):
                os.remove(list_file)

    def _visualize_step(self, save_path: str = None, step_data: Optional[Dict[str, Any]] = None,
                         interp_positions: Optional[Dict[str, Tuple[float, float]]] = None,
                         interp_frac: float = 1.0,
                         display_step: Optional[int] = None,
                         push_pairs_override: Optional[List[Dict]] = None):
        """Visualize the current state of the simulation."""
        # Isometric rendering path
        if self.use_isometric and self.iso_renderer is not None:
            if step_data is None and self.episode_log:
                step_data = self.episode_log[-1]
            self.iso_renderer.save_frame(
                save_path, self.agents, self.current_step,
                step_data=step_data,
                queried_agent_label=self.queried_agent_label,
                interp_positions=interp_positions,
                display_step=display_step,
                push_pairs_override=push_pairs_override
            )
            if save_path:
                print(f"  [Figure saved to {save_path}]")
            return

        # Flat (top-down) rendering path
        if step_data is None and self.episode_log:
            step_data = self.episode_log[-1]
        self.flat_renderer.save_frame(
            save_path, self.agents, self.current_step,
            step_data=step_data,
            queried_agent_label=self.queried_agent_label,
            show_queried_agent_star=self.show_queried_agent_star,
            interp_positions=interp_positions,
            interp_frac=interp_frac,
            display_step=display_step,
            push_pairs_override=push_pairs_override,
        )
