from typing import Tuple, Optional, List, Dict
from enum import Enum


class AgentType(Enum):
    """Types of agents in the cafe"""
    # Customers
    CUSTOMER = "customer"
    

class AgentStatus(Enum):
    """Current status/state of an agent"""
    ENTERING = "entering"
    MOVING = "moving"
    QUEUING = "queuing"


class Action(Enum):
    """Available actions for agents"""
    MOVE = "move"
    QUEUE = "queue"
    YIELD = "yield"
    CUT_IN_LINE = "cut_in_line"
    ORDER = "order"
    PICK_UP = "pick_up"
    WAIT = "wait"


class Agent:
    """
    Base agent class - all agents have position, type, speed, and status
    Subclasses must implement update() method for behavior
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_type: AgentType,
        position: Tuple[int, int],
        speed: float = 1.0
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        
        # State attributes
        self.speed = speed
        self.position = position
        self.status = AgentStatus.ENTERING
        self.target_position: Optional[Tuple[int, int]] = None
        self.queue_position: Optional[int] = None # Position in queue if queuing (0 = front)
        self.order_ready = False
        self.patience = 100
        
    
    def move_towards(self, target: Tuple[int, int], world) -> bool:
        """
        Move one step towards target position.
        Returns True if moved, False if blocked or arrived.
        """
        if self.position == target:
            return False
        
        
        current_r, current_c = self.position
        target_r, target_c = target
        
        # Simple greedy movement
        next_pos = self.position
        
        if current_r < target_r and world.is_walkable((current_r + 1, current_c)):
            next_pos = (current_r + 1, current_c)
        elif current_r > target_r and world.is_walkable((current_r - 1, current_c)):
            next_pos = (current_r - 1, current_c)
        elif current_c < target_c and world.is_walkable((current_r, current_c + 1)):
            next_pos = (current_r, current_c + 1)
        elif current_c > target_c and world.is_walkable((current_r, current_c - 1)):
            next_pos = (current_r, current_c - 1)
        
        if next_pos != self.position:
            self.position = next_pos
            return True
        
        return False
    
    def update(self, world, queue: List['Agent'], norm) -> Optional[Action]:
        """
        Choose and return an action for this step.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement update()")
    
    def __repr__(self):
        return f"{self.agent_type.value}[{self.agent_id}] at {self.position} - {self.status.value}"



class Customer(Agent):
    """Customer who queues for ordering"""
    
    def __init__(self, agent_id: str, position: Tuple[int, int]):
        super().__init__(agent_id, AgentType.CUSTOMER, position, speed=1.0)
    
    
    def update(self, world, queue: List['Agent'], norm) -> Optional[Action]:
        """
        Customer state machine: ENTERING -> MOVING_TO_QUEUE -> QUEUING
        Can also attempt to CUT_IN_LINE
        """
        
        if self.status == AgentStatus.ENTERING:
            # Set target to counter area (open space in front of counter)
            counter_zone = world.zones['counter']
            if counter_zone and not self.target_position:
                # Find position in front of counter
                open_space = world.zones['open_space']
                queue_spots = [pos for pos in open_space if pos[0] > counter_zone[0][0] and 4 <= pos[1] <= 8]
                
                if queue_spots:
                    # Find first unoccupied spot
                    occupied = {a.position for a in queue}
                    for pos in queue_spots:
                        if pos not in occupied:
                            self.target_position = pos
                            break
                    
                    # If all spots taken, queue at the last position
                    if not self.target_position:
                        self.target_position = queue_spots[-1]
            
            if self.target_position:
                self.status = AgentStatus.MOVING
                return Action.MOVE
        
        elif self.status == AgentStatus.MOVING:
            # Move towards counter area
            if self.target_position and self.position != self.target_position:
                return Action.MOVE
            else:
                # Arrived at queue position
                self.status = AgentStatus.QUEUING
                return Action.QUEUE
        
        elif self.status == AgentStatus.QUEUING:
            # Wait in queue or try to cut in line
            if len(queue) > 1:
                return Action.CUT_IN_LINE
            return Action.WAIT
        
        return Action.WAIT