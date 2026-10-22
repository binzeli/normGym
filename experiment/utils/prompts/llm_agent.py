
"""
You are shown an image from a grid-world simulation.

SETTING
{WORLD_DESCRIPTION}

ROLE
- You are the agent with a gold star.
- Your task has the following goals, in order:
{GOAL_DESCRIPTION}
- You are currently working on the goal whose target is {TARGET_POSITION}.

ENVIRONMENT & DYNAMICS
- Agents are represented as figures with different clothing colors (orange, green, and blue).
- An agent's color is its social group and in this world we have 3 groups: orange, green and blue.
- If a collision occurs: if one agent is waiting, the moving agent stays and pushes the waiting agent to another position; if both are moving to the same spot, one is randomly chosen to stay and the other is pushed.
- An agent can display states on their upper right: a timer indicates the agent is completing its first target goal, a red exclamation mark indicates the agent is in a push/collision state, and a green checkmark indicates the first target is finished. 

OBJECT POSITIONS
{OBJECT_POSITIONS}

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
- Reaching your target cell {TARGET_POSITION}: +10
- Each collision: -5
- Each timestep: -0.1
- Any violation of social norms listed above: -20

TASK
Given the current step image, which shows the environment and all agents' current positions, decide the next action for yourself.

ReAct PROCEDURE (output your reasoning step by step)
1) Observe:
   - Identify your current grid position.
   - Identify your current target cell {TARGET_POSITION} and any blocked cells/obstacles.
   - Identify nearby agents, their intended next moves, and their class.
2) Reason:
   - Propose some candidate actions that move you toward {TARGET_POSITION} efficiently.
   - For each candidate, assess:
     - Collision risk next step (including push outcomes based on class), taking into account other agents' arrows.
     - Whether it violates any social norms.
3) Act:
   - State your chosen action.

"""
