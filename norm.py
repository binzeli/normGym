from typing import List, Dict, Optional
from enum import Enum
from agent import Agent, Action, AgentStatus


class Norm:
    """
    Represents the queuing norm in the cafe.
    Customers must queue before doing anything else.
    """
    
    def __init__(self):
        """Initialize the queuing norm"""
        self.violation_log: List[Dict] = []
    
    def is_valid_action(self, agent: Agent, action: Action, queue: List[Agent]) -> bool:
        """
        Check if an action is valid according to queuing norm.
        Customers must queue and cannot cut in line.
        
        Args:
            agent: The agent taking the action
            action: The action being taken
            queue: Current queue state
        
        Returns:
            True if action complies with norm, False if it violates
        """
        
        # Cutting in line is ALWAYS a violation
        if action == Action.CUT_IN_LINE:
            return False
        
        return True
    
    def check_violation(self, agent: Agent, action: Action, queue: List[Agent]) -> bool:
        """
        Check if an action violates the norm.
        
        Returns:
            True if violation occurs, False if compliant
        """
        is_valid = self.is_valid_action(agent, action, queue)
        
        if not is_valid:
            # Log violation
            self.violation_log.append({
                'agent_id': agent.agent_id,
                'action': action.value if action else None,
                'status': agent.status.value,
                'position': agent.position,
                'timestep': len(self.violation_log)
            })
        
        return not is_valid
    
    def get_violation_count(self) -> int:
        """Get total number of violations recorded"""
        return len(self.violation_log)
    
    def __repr__(self):
        return "Norm(QUEUING)"