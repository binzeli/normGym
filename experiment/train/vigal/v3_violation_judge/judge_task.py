"""v3 task definition: prompt, response schema, parser and 1/0 grader.

The policy is an OBSERVER. It sees the frame before a move, is told which
agent moves where, and answers whether that move violates the (unstated)
social norm the group follows. Ground truth is the env's own verdict, so the
reward is the paper's: 1 if the yes/no is right, 0 otherwise, + 0.1 format.
Since 2026-09-05 the output is {"reasoning", "violation"} only — the belief
field was dropped (the parser still tolerates one from old checkpoints).
Nothing here imports v1 (experiment/train/vigal) or v2 (v2_belief_rloo).
"""
import json
import re
from typing import Any, Dict, Optional

SYSTEM_TEMPLATE = """You are shown an image from a grid-world simulation.

SETTING
{WORLD_DESCRIPTION}

ROLE
- You are an observer, not one of the agents. You watch the group and judge their moves.
- The group follows a social norm that is not described to you. Your job is to infer that norm from how the agents behave and use it to judge moves.

ENVIRONMENT & DYNAMICS
- Agents are represented as figures with different clothing colors (orange, green, and blue); each agent's name is written next to it.
- An agent's color is its social group and in this world we have 3 groups: orange, green and blue.
- If a collision occurs: if one agent is waiting, the moving agent stays and pushes the waiting agent to another position; if both are moving to the same spot, one is randomly chosen to stay and the other is pushed.
- An agent can display states on their upper right: a timer indicates the agent is completing its first target goal, a red exclamation mark indicates the agent is in a push/collision state, and a green checkmark indicates the first target is finished.
- A red cross mark on an agent means that agent's PREVIOUS move violated the social norm and it was penalised at that step. The move you are asked about has not happened yet in the image, so its own outcome is never shown.
{SETTING_NOTE}

OBJECT POSITIONS
{OBJECT_POSITIONS}

TASK
You are told which agent is about to move and where. Decide whether that move violates the group's social norm.

PROCEDURE (reason step by step before answering)
1) Observe: where the named agent is, where it is going, and {OBSERVE_HINT}.
2) Infer: {INFER_HINT}, using the current picture, the agent's recent moves and any red-cross feedback seen so far.
3) Judge: does the described move break that rule?

OUTPUT
Respond with ONLY a JSON object, no other text:
{{"reasoning": "<your step-by-step reasoning>", "violation": "yes" or "no"}}
"""

ACTION_TEXT = {
    'MOVE_UP': 'move up', 'MOVE_DOWN': 'move down', 'MOVE_LEFT': 'move left',
    'MOVE_RIGHT': 'move right', 'WAIT': 'wait',
}


def action_text(action: Any) -> str:
    name = getattr(action, 'name', None) or str(action)
    name = name.split('.')[-1].upper()
    return ACTION_TEXT.get(name, name.lower().replace('_', ' '))


# Norm-category presets for the category-specific parts of the system prompt,
# one per norm family of the paper (config.py:NORM_GROUPS tags each group
# with its category; harvest_judge.py defaults to the active group's tag).
# 'queue' is the wording every run up to and including run2 used verbatim.
# The other families must NOT assume the agents' goal is the counter, so
# each gets its own preset; 'generic' is a family-agnostic fallback.
CATEGORY_HINTS = {
    'queue': dict(
        setting_note='- Agents come to the counter at {COUNTER} to order, then leave '
                     'through a door. Most of them follow the norm; a few do not know it.',
        observe_hint='how the other agents are arranged — especially those waiting '
                     'for the counter',
        infer_hint='what rule the agents who wait for the counter seem to follow '
                   '(for example, a fixed shape or path of the waiting line, or how '
                   'one joins and advances in it)',
    ),
    'region': dict(
        setting_note='- Each agent pursues its own task in this world and then leaves '
                     'through a door. Most of them follow the norm; a few do not know it.',
        observe_hint='which cells agents step on or avoid, which seats they take, '
                     'and which doors they leave by',
        infer_hint='what rule restricts where agents may end up (for example, a '
                   'forbidden zone, a door everyone must use or must avoid, or a '
                   'region of seats that may or may not be taken)',
    ),
    'trajectory': dict(
        setting_note='- Each agent pursues its own task in this world and then leaves '
                     'through a door. Most of them follow the norm; a few do not know it.',
        observe_hint="the pattern of each agent's own movement over time — when it "
                     'moves, when it waits, and along which path',
        infer_hint="what rule shapes each agent's own motion (for example, "
                   'circulating one way around the room, moving only on certain '
                   'turns, alternating moves and waits, or visiting a special zone '
                   'before sitting)',
    ),
    'allocation': dict(
        setting_note='- Each agent pursues its own task in this world and then leaves '
                     'through a door. Most of them follow the norm; a few do not know it.',
        observe_hint='how agents of different colors share zones, seats, and doors, '
                     'and how many use each at once',
        infer_hint='what rule divides the contested space among agents (for example, '
                   'a cap on how many may occupy a zone, doors or seating regions '
                   'tied to agent color, or seats filling in a fixed order)',
    ),
    'generic': dict(
        setting_note='- Each agent pursues its own task in this world and then leaves '
                     'through a door. Most of them follow the norm; a few do not know it.',
        observe_hint='how the other agents are arranged and behaving',
        infer_hint='what rule the group seems to follow (for example, a pattern in how '
                   'agents move, where they may stand or sit, or the order in which '
                   'they act)',
    ),
}


def build_system_prompt(world_description: str, object_positions: str, counter: str,
                        category: str = 'queue') -> str:
    hints = CATEGORY_HINTS[category]
    return SYSTEM_TEMPLATE.format(WORLD_DESCRIPTION=world_description,
                                  OBJECT_POSITIONS=object_positions,
                                  SETTING_NOTE=hints['setting_note'].format(COUNTER=counter),
                                  OBSERVE_HINT=hints['observe_hint'],
                                  INFER_HINT=hints['infer_hint'])


def build_user_prompt(rec: Dict[str, Any], show_feedback: bool = True) -> str:
    """rec fields: step, agent_name, from_pos, to_pos, action, positions_text,
    history_text, feedback_text."""
    fr, to = tuple(rec['from_pos']), tuple(rec['to_pos'])
    act = rec['action_text']
    move = (f"{rec['agent_name']} is at {fr} and is about to {act}"
            + (f", which would take it to {to}." if act != 'wait' else " (stay at its cell)."))
    parts = [
        f"Current step: {rec['step']}.",
        "The image shows the environment and all agents' positions right before this move.",
        f"Agents' current positions are {rec['positions_text']}. Positions are (row, column). Row 0 is the top, column 0 is the left.",
    ]
    if rec.get('memory_text'):
        parts.append("Your memory of this episode (every agent's position at each step so far):\n"
                     + rec['memory_text'])
    parts.append(move)
    if rec.get('history_text'):
        parts.append(f"{rec['agent_name']}'s recent moves: {rec['history_text']}.")
    if show_feedback and rec.get('feedback_text'):
        parts.append(f"Red-cross feedback seen so far in this episode: {rec['feedback_text']}.")
    parts.append(f"Question: does this move by {rec['agent_name']} violate the group's social norm? "
                 "Answer with the JSON object described in the instructions.")
    return "\n".join(parts)


_YES = {'yes', 'y', 'true', 'violation', 'violates'}
_NO = {'no', 'n', 'false', 'none', 'compliant'}


def parse_response(text: str) -> Optional[Dict[str, Any]]:
    """-> {'violation': bool, 'reasoning': str, 'belief': str} or None."""
    if not text:
        return None
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        # tolerate a trailing-comma / unescaped-newline JSON: fall back to a regex on the verdict
        v = re.search(r'"violation"\s*:\s*"?\s*([A-Za-z]+)', m.group(0))
        if not v:
            return None
        obj = {'violation': v.group(1)}
    v = obj.get('violation')
    if isinstance(v, bool):
        verdict = v
    elif isinstance(v, str) and v.strip().lower() in _YES:
        verdict = True
    elif isinstance(v, str) and v.strip().lower() in _NO:
        verdict = False
    else:
        return None
    return {'violation': verdict,
            'reasoning': str(obj.get('reasoning', '') or ''),
            'belief': str(obj.get('belief', '') or '')}


def is_degenerate(text: str, ngram: int = 8, repeats: int = 4) -> bool:
    """Repetition loop: some 8-word sequence occurs >= 4 times (run 120409
    looped '...would be a collision, but the norm might be...' to the token
    limit on hard states). Normal answers repeat no 8-gram more than twice."""
    words = (text or '').split()
    if len(words) < ngram * repeats:
        return False
    seen: Dict[str, int] = {}
    for k in range(len(words) - ngram + 1):
        g = ' '.join(words[k:k + ngram])
        seen[g] = seen.get(g, 0) + 1
        if seen[g] >= repeats:
            return True
    return False


def grade(label: int, response: Optional[str], correct_reward: float = 1.0,
          format_reward: float = 0.1, truncated: bool = False) -> Dict[str, Any]:
    """1/0 on the verdict + 0.1 if the JSON parsed (the paper's reward).
    Truncated or degenerate (looping) responses count as unparsed."""
    if is_degenerate(response or ''):
        truncated = True
    parsed = parse_response(response or '') if not truncated else None
    if parsed is None:
        return {'reward': 0.0, 'parsed': False, 'truncated': bool(truncated),
                'correct': False, 'pred': None, 'category': 'unparsed'}
    pred = int(parsed['violation'])
    correct = pred == int(label)
    cat = ('tp' if pred and correct else 'tn' if (not pred and correct)
           else 'fp' if pred else 'fn')
    return {'reward': (correct_reward if correct else 0.0) + format_reward,
            'parsed': True, 'truncated': False, 'correct': correct, 'pred': pred,
            'category': cat, 'belief': parsed['belief']}
