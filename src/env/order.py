"""
Order queue management for cafe simulation.
Handles ordering transitions and counter wait time.
"""

from typing import List, Optional, Dict
from src.env.actions import Action

order_wait_time = 2


def get_order_action(world, all_current_positions, agent):
    """Determine if the agent should perform an ORDER action at the counter."""

    if getattr(world, 'map_type', 'default') != 'default':
        return None

    # Only apply ordering logic while the agent is still on its first target
    if getattr(agent, 'target_index', 0) > 0:
        return None

    at_counter = check_if_at_counter(agent, world)
    if at_counter:
        num_agents_at_counter = count_agents_at_counter(all_current_positions, world)
        ordering = check_if_already_ordering(agent)
        if ordering:
            if num_agents_at_counter == 1:
                if still_ordering(agent):
                    return Action.ORDER
            else:
                agent.counter_wait_time = 0 # Pause ordering for now if more than one agent at counter
        
        else: #if not already ordering, set action to order
            if num_agents_at_counter == 1:
                agent.counter_wait_time = 1
                return Action.ORDER            
    return None

    


def check_if_already_ordering(agent):
    if agent.active and not agent.done and agent.action == Action.ORDER:
        return True
    return False
    

def check_if_at_counter(agent, world):
    if (agent.active and not agent.done and world.is_at_counter(agent.current_position)):
        return True
    return False


def count_agents_at_counter(all_current_positions, world):
    
    count = 0
    for aid, pos in all_current_positions.items():
        if world.is_at_counter(pos):
            count += 1
    return count
    
    
def still_ordering(agent):
    """Handle an agent that is currently ordering."""
    # Still ordering, increment wait time
    agent.counter_wait_time += 1
    if agent.counter_wait_time >= order_wait_time:
        agent.advance_to_next_target()
    return True




