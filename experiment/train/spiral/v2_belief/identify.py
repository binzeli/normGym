"""v2 identification task: a VERIFIABLE end-of-episode belief test.

The vigal v2 rewarded free-text beliefs through an LLM judge — which cost
money, rated shapeless beliefs generously, and let the policy farm length
until the run collapsed. Here the belief reward is a game outcome instead
(the SPIRAL principle): after the episode, the subject is shown the true
norm's description among the other candidates' descriptions (shuffled,
unnamed) and must pick the one the crowd was following. The ground-truth
norm is known, so the reward is an exact 1/0 — no judge, no length channel.

The identification prompt carries the real disambiguating evidence:
- the sequence of everyone's positions over the episode (extracted from the
  subject's own turn prompts, subsampled every ``stride`` steps) — the queue
  shape over time, which a single frame cannot show;
- the final frame image.

Candidate descriptions come from ``src/norms/norm_descriptions.json``.
"""
import json
import os
import re
import string
from typing import Dict, List, Optional, Tuple

from experiment.train.config import PROJECT_ROOT
from experiment.train.policy_client import Turn

NORM_DESCRIPTIONS_PATH = os.path.join(PROJECT_ROOT, 'src', 'norms',
                                      'norm_descriptions.json')

_POSITIONS_RE = re.compile(r"Other agents' current positions are ([^\n]+)")
_SELF_RE = re.compile(r'Your current position is (\(\d+, \d+\))')

ID_SYSTEM_PROMPT = """You just finished an interactive episode in a grid-world simulation: you were one of several agents in a cafe, most of which follow a shared social norm about how to line up and wait for the counter. Your task now is to identify that norm.

You are given:
- a log of everyone's positions over the episode (positions are (row, column); row 0 is the top, column 0 is the left);
- the final image of the episode;
- a list of candidate norm descriptions, labeled with letters.

Exactly one candidate is the norm the group was following. Compare the shape and evolution of the waiting line in the log against each candidate's description and pick the letter of the matching one.

Respond with ONLY a JSON object, no other text:
{"reasoning": "<step-by-step comparison of the evidence against the candidates>", "choice": "<letter>"}"""


def load_descriptions() -> Dict[str, str]:
    with open(NORM_DESCRIPTIONS_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {n['name']: n['description'] for n in data['norms']}


def episode_history(ep, stride: int) -> str:
    """Everyone's positions every ``stride`` steps, from the subject's own
    turn prompts (the recorded observations, not privileged state)."""
    lines = []
    for i, turn in enumerate(ep.turns):
        if stride > 1 and i % stride:
            continue
        others = _POSITIONS_RE.search(turn.user_prompt)
        me = _SELF_RE.search(turn.user_prompt)
        if not others:
            continue
        step = f'step {turn.step}' if turn.step is not None else f'turn {i}'
        mine = f'me at {me.group(1)}, ' if me else ''
        lines.append(f'- {step}: {mine}{others.group(1)}')
    return '\n'.join(lines)


def build_id_turn(ep, candidates: List[str], descriptions: Dict[str, str],
                  rng, stride: int = 3,
                  desc_chars: int = 0) -> Optional[Tuple[Turn, str]]:
    """-> (Turn with empty response, correct letter), or None when the episode
    can't be turned into a valid question (norm not in the candidate list,
    or no usable history)."""
    if ep.norm_type not in candidates:
        return None
    history = episode_history(ep, stride)
    if not history:
        return None
    order = list(candidates)
    rng.shuffle(order)
    letters = string.ascii_uppercase[:len(order)]
    answer = letters[order.index(ep.norm_type)]
    blocks = []
    for letter, name in zip(letters, order):
        desc = descriptions[name]
        if desc_chars and len(desc) > desc_chars:
            desc = desc[:desc_chars].rstrip() + ' …'
        blocks.append(f'{letter}. {desc}')
    user_prompt = (
        'POSITION LOG (subsampled)\n' + history +
        '\n\nCANDIDATE NORMS\n' + '\n\n'.join(blocks) +
        '\n\nWhich candidate is the social norm this group was following? '
        'Answer with the JSON object described in the instructions.'
    )
    image = ep.turns[-1].image_paths[-1:] if ep.turns[-1].image_paths else []
    return Turn(system_prompt=ID_SYSTEM_PROMPT, user_prompt=user_prompt,
                image_paths=image, response=''), answer


_CHOICE_RE = re.compile(r'"choice"\s*:\s*"?\s*([A-Za-z])')


def parse_choice(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            c = str(obj.get('choice', '')).strip().upper()
            if len(c) == 1 and c.isalpha():
                return c
        except json.JSONDecodeError:
            pass
    m = _CHOICE_RE.search(text)
    return m.group(1).upper() if m else None
