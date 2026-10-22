"""
You are shown an image from a cafe grid-world simulation.

ROLE
- You are the agent marked by a centered gold star.
- You enter the cafe through the green door and your goal is to reach the orange counter to order food.
- If you do not see yourself on the grid, then that means you just entered the green door.
- The target cell is (2,6), which is the front of the counter.

ENVIRONMENT & DYNAMICS
- Agents are represented as colored figures (orange, green, and blue).
- All non-starred agents have arrows indicating their next movement direction.
- The starred agent's arrows are intentionally hidden in the image.
- For non-starred agents, no arrow means that agent is waiting.
- An agent's color is its social class. Orange is the highest class (class 1), green is the middle class (class 2), and blue is the lowest class (class 3).
- If a collision occurs: if one agent is waiting, the moving agent stays and pushes the waiting agent to another position; if both are moving to the same spot, one is randomly chosen to stay and the other is pushed.
- Red grid cells represent tables and chairs. They are static obstacles, NOT agents. Do not treat them as agents and do not expect them to move.

SOCIAL NORMS
{NORMS_DESCRIPTION}

ACTIONS
You may take exactly one of the following actions each step:
- move up
- move down
- move left
- move right
- wait

REWARDS & PENALTIES
- Reaching (2,6): +10
- Each collision: -5
- Each timestep: -0.1
- Any violation of social norms listed above: -20
- If you collide with another agent: -5

TASK
Looking at this image, analyze the current state and determine the best next action for {AGENT_NAME}.

Consider:
1. The current position of {AGENT_NAME} and the path to the orange counter.
2. Any obstacles or other agents nearby.
3. Any social norms that should be followed.
4. The class of nearby agents and expected next movements of non-starred agents from their arrows.

What should be the next action for {AGENT_NAME}? Provide your recommendation with a clear action name (move up, move down, move left, move right, wait, or order) and give the reasoning behind your choice.
"""
