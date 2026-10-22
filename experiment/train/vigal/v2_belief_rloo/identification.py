"""v2 identification reward: a VERIFIABLE belief test on single snapshots.

Replaces the LLM belief judge (2026-09-05, user decision): the judge cost
money, rated shapeless beliefs generously, and opened a length-farming
channel. Borrowing the SPIRAL v2 principle (see
experiment/train/spiral/v2_belief/identify.py), the belief reward becomes a
game outcome instead: the snapshot keeps its full-memory history, the action
question at the tail is replaced by a lettered list of ALL training norms'
descriptions (shuffled per snapshot, unnamed), and the model must pick the
one the crowd is following. The ground-truth norm is known, so the reward is
an exact 1/0 — no judge, no length channel.

Unlike SPIRAL v2 (end-of-episode, position log + final frame), the question
is asked mid-episode on a harvested snapshot: the evidence is the snapshot's
own full memory + current frame. Snapshots below ``--min_step`` carry too
little evidence and are filtered out of the pool by the trainer.
"""
import json
import os
import random
import re
import string
from typing import Any, Dict, List, Optional

from experiment.train.config import PROJECT_ROOT

NORM_DESCRIPTIONS_PATH = os.path.join(PROJECT_ROOT, 'src', 'norms',
                                      'norm_descriptions.json')

# The harvested (interactive-stage) user prompt ends with this line followed
# by the action/belief JSON instructions; everything from it onward is
# replaced by the identification question.
ACTION_TAIL_MARKER = 'Decide the next action for yourself.'

ID_INSTRUCTIONS = """You are NOT asked to choose a move this step. Instead, identify the social norm this group is following.
Below are candidate norm descriptions, labeled with letters. Exactly one of them is the norm the group follows.
{candidates}
Compare your memory of the episode so far ({evidence_hint}) against each candidate and pick the letter of the matching one.
Respond with ONLY a JSON object, no other text:
{{"reasoning": "<step-by-step comparison of the evidence against the candidates>", "choice": "<letter>"}}"""

# Category-specific evidence hint inside ID_INSTRUCTIONS, one per norm
# family of the paper (config.py:NORM_GROUPS tags each group with its
# category; trainers default to the active group's tag). 'queue' is the
# wording every run up to and including run2 used verbatim; the agents' goal
# is NOT always the counter, so the other families get their own hint
# instead of inheriting the waiting-line one. 'generic' is a family-agnostic
# fallback (e.g. for mixed-norm pools).
EVIDENCE_HINTS = {
    'queue': 'the shape of the waiting line, how agents join and advance in it, '
             'and any penalties you saw or received',
    'region': 'which cells agents step on or avoid, which seats they take, '
              'which doors they leave by, and any penalties you saw or received',
    'trajectory': "the pattern of each agent's own motion — the direction it "
                  'circulates, on which turns it moves or waits, where it goes '
                  'before sitting, and any penalties you saw or received',
    'allocation': 'how agents share contested spots — how many occupy a zone at '
                  'once, which door or seat agents of each color use, the order '
                  'in which seats fill, and any penalties you saw or received',
    'generic': 'how agents move, where they stand or sit, the order in which '
               'they act, and any penalties you saw or received',
}


def load_descriptions() -> Dict[str, str]:
    with open(NORM_DESCRIPTIONS_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {n['name']: n['description'] for n in data['norms']}


def prepare_identification(records: List[Dict[str, Any]],
                           candidate_norms: List[str], seed: int,
                           category: str = 'queue') -> None:
    """Rewrite each snapshot's user_prompt tail in place and set
    ``rec['id_answer']`` (the correct letter).

    The candidate order is a deterministic per-record shuffle keyed on
    (seed, record id), so reruns/resumes see identical prompts while the
    answer letter is not learnable across records. Already-prepared records
    (e.g. on a second call) are left untouched.
    """
    desc = load_descriptions()
    norms = sorted(set(candidate_norms))
    missing = [n for n in norms if n not in desc]
    if missing:
        raise ValueError(f'norms without a description: {missing}')
    if len(norms) > len(string.ascii_uppercase):
        raise ValueError(f'{len(norms)} candidates > 26 letters')
    for rec in records:
        if rec.get('id_answer'):
            continue
        if rec['norm'] not in norms:
            raise ValueError(f"snapshot {rec['id']}: norm {rec['norm']!r} "
                             f'not among the candidates {norms}')
        if ACTION_TAIL_MARKER not in rec['user_prompt']:
            raise ValueError(f"snapshot {rec['id']}: prompt has no "
                             f'{ACTION_TAIL_MARKER!r} tail to replace')
        rng = random.Random(f"{seed}:{rec['id']}")
        order = norms[:]
        rng.shuffle(order)
        letters = string.ascii_uppercase[:len(order)]
        candidates = '\n'.join(f'{letter}. {desc[n]}'
                               for letter, n in zip(letters, order))
        head = rec['user_prompt'].split(ACTION_TAIL_MARKER)[0].rstrip()
        rec['user_prompt'] = head + '\n' + ID_INSTRUCTIONS.format(
            candidates=candidates, evidence_hint=EVIDENCE_HINTS[category])
        rec['id_answer'] = letters[order.index(rec['norm'])]
        rec['id_num_candidates'] = len(order)


def parse_choice(text: str) -> Optional[str]:
    """-> the chosen letter (upper-case) or None."""
    if not text:
        return None
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if not m:
        return None
    choice = None
    try:
        obj = json.loads(m.group(0))
        choice = obj.get('choice')
    except json.JSONDecodeError:
        # tolerate broken JSON: fall back to a regex on the choice field
        v = re.search(r'"choice"\s*:\s*"?\s*([A-Za-z])', m.group(0))
        if v:
            choice = v.group(1)
    if not isinstance(choice, str):
        return None
    choice = choice.strip().rstrip('.').upper()
    if len(choice) == 1 and choice in string.ascii_uppercase:
        return choice
    return None


def is_degenerate(text: str, ngram: int = 8, repeats: int = 4) -> bool:
    """Repetition loop (same guard as v3's judge_task): some 8-word sequence
    occurs >= 4 times. Normal answers repeat no 8-gram more than twice."""
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


def grade_choice(rec: Dict[str, Any], response: Optional[str],
                 correct_reward: float = 1.0,
                 format_reward: float = 0.1) -> Dict[str, Any]:
    """1/0 on the letter + format bonus if the JSON parsed. Truncated or
    degenerate (looping) responses count as unparsed, reward 0."""
    response = response or ''
    truncated = not response.rstrip().endswith('}') or is_degenerate(response)
    choice = None if truncated else parse_choice(response)
    if choice is None:
        return {'reward': 0.0, 'category': 'unparsed', 'parsed': False,
                'truncated': truncated, 'correct': False, 'choice': None,
                'format_bonus': 0.0}
    correct = choice == rec['id_answer']
    return {'reward': (correct_reward if correct else 0.0) + format_reward,
            'category': 'good' if correct else 'wrong', 'parsed': True,
            'truncated': False, 'correct': correct, 'choice': choice,
            'format_bonus': format_reward}
