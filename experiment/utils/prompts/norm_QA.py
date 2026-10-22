"""
You are shown a sequence of images from a cafe grid-world simulation.
Each agent enters the cafe through the green door and aims to reach the orange counter to order food, so the goal of each agent is to reach the position (2,6) which is the front of the counter.
Agents are represented as shapes, and color intensity indicates social class (darker = higher class number).
If a collision occurs: if one agent is waiting, the moving agent stays and pushes the waiting agent; if both are moving to the same spot, one is randomly chosen to stay.
The actions that an agent can take are: move up, move down, move right, move left, wait (no arrows), order (agent showing orange)

Describe what is happening in the images.
Note any collisions or notable interactions between agents.
Identify any social norms that appear to be followed, and briefly explain your reasoning. Note the social norms here may be arbitrary rules specific to this simulation. So look for patterns of behavior (such as waiting) that suggest norms are being followed.
Keep in mind that if there are no social norms, because 
If no norms are clearly visible, say so.
"""