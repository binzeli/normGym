"""LLM judge for norm-belief grading.

Compares an agent's stated norm belief (e.g. the final belief from the
interactive stage) against the ground-truth norm description and returns a
rating on a 7-point Likert scale (integer 1-7; 7 means the belief expresses
the same rule as the ground truth, 1 means unrelated). The rating is returned
both raw (``likert``) and rescaled to [0, 1] (``score = (likert - 1) / 6``) so
existing 0-1 consumers keep working. The judge is an LLM call through
OpenRouterClient; the model defaults to GPT-5.6-sol and can be overridden via
the NORM_JUDGE_MODEL env var.

Usage (library):
    from experiment.utils.norm_judge import judge_norm_belief, ground_truth_norm_description
    result = judge_norm_belief(belief, ground_truth_norm_description('occupation_seat_no_bottom_left'))
    result['likert']  # int in 1..7
    result['score']   # float in [0, 1], (likert - 1) / 6

Usage (CLI, for a quick check):
    python experiment/utils/norm_judge.py --norm occupation_seat_no_bottom_left \
        --belief "Agents must not sit on chairs in the bottom-left region."
"""
import json
import os
import re
import sys
from typing import Any, Dict, Optional

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiment.utils.client import OpenRouterClient

DEFAULT_JUDGE_MODEL = os.getenv('NORM_JUDGE_MODEL', 'openai/gpt-5.6-sol')

_NORM_DESCRIPTIONS_PATH = os.path.join(
    _PROJECT_ROOT, 'src', 'norms', 'norm_descriptions.json'
)

_SYSTEM_PROMPT = (
    "You grade how well an agent inferred a social norm. You are given the "
    "ground-truth norm and the agent's stated belief, and you output a single "
    "rating on a 7-point Likert scale (a whole number from 1 to 7)."
)

_PROMPT_TEMPLATE = """Ground-truth norm:
{ground_truth}

Agent's stated norm belief:
{belief}

Rate how well the agent's belief matches the ground-truth norm on a 7-point
Likert scale, using whole numbers only:
- 7: the belief expresses the same rule as the ground truth (wording may differ).
- 6: the same rule with a minor error or omission, or with extra incorrect
  constraints bundled in alongside the correct rule.
- 5: the belief identifies the right kind of rule — the correct behavior it
  constrains (e.g. which seat to take, which door to use, how to move) and the
  correct feature it depends on (e.g. region, color, order, direction) — but gets
  the specific rule wrong (e.g. "only sit in the bottom-left region" when the
  true rule is "never sit in the bottom-left region").
- 4: the belief constrains the right behavior but hinges on the wrong feature,
  or captures only about half of the rule.
- 3: the belief constrains the right behavior but captures only a small
  fragment of the rule.
- 2: only vaguely related — touches the same general behavior in passing but
  the stated rule is otherwise off.
- 1: unrelated to the ground-truth rule.

Judge only the content of the rule — ignore differences in phrasing, level of
detail, or hedging about confidence.

A belief counts as expressing the same rule when it captures the rule's
defining content — the specific constraint that distinguishes this norm from
its neighbours (for a queueing norm the shape of the line; for a movement norm
the direction or turn rule; for a seating or region norm which places are
allowed or forbidden for whom; for a door norm which door and for whom). It
does not also need to restate incidental details such as where a line starts,
how agents advance, capacities, exact lengths, or what happens when the rule
cannot be followed.

Respond with only a JSON object:
{{ "score": <integer from 1 to 7>, "reasoning": "<one or two sentences>" }}"""

LIKERT_MIN, LIKERT_MAX = 1, 7


def likert_to_score(likert: int) -> float:
    """Rescale a 1-7 Likert rating to [0, 1] (4 -> 0.5)."""
    return round((likert - LIKERT_MIN) / (LIKERT_MAX - LIKERT_MIN), 3)


def ground_truth_norm_description(norm_name: str) -> str:
    """The ground-truth description for ``norm_name`` from norm_descriptions.json."""
    with open(_NORM_DESCRIPTIONS_PATH, encoding='utf-8') as f:
        norms = json.load(f)['norms']
    for norm in norms:
        if norm.get('name') == norm_name:
            return norm['description']
    raise KeyError(f"Norm '{norm_name}' not found in {_NORM_DESCRIPTIONS_PATH}")


def _parse_judge_response(text: str) -> Dict[str, Any]:
    """Extract {'likert', 'score', 'reasoning'} from the judge's reply.

    The rating must be a whole number from 1 to 7; out-of-range values are
    clamped and non-integers rounded, with the raw value kept in ``raw``.
    """
    rating: Optional[float] = None
    reasoning = ''

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed.get('score'), (int, float)):
                rating = float(parsed['score'])
            reasoning = str(parsed.get('reasoning', ''))
        except json.JSONDecodeError:
            pass

    if rating is None:
        # Fallback: first standalone digit 1-7 in the reply.
        number = re.search(r'(?<![\d.])([1-7])(?![\d.])', text)
        if number:
            rating = float(number.group(1))

    if rating is None:
        raise ValueError(f"Could not parse a rating from judge response: {text!r}")

    likert = int(min(LIKERT_MAX, max(LIKERT_MIN, round(rating))))
    return {'likert': likert, 'score': likert_to_score(likert),
            'raw': rating, 'reasoning': reasoning}


def judge_norm_belief(
    belief: Optional[str],
    ground_truth: str,
    model: Optional[str] = None,
    client: Optional[OpenRouterClient] = None,
) -> Dict[str, Any]:
    """Grade ``belief`` against ``ground_truth`` with an LLM judge.

    Returns {'likert': int in 1..7, 'score': float in [0, 1], 'reasoning': str,
    'judge_model': str, 'belief': str, 'ground_truth': str}. An empty/missing
    belief is rated 1 (score 0.0) without an API call.
    """
    belief = (belief or '').strip()
    if not belief:
        return {
            'likert': LIKERT_MIN,
            'score': 0.0,
            'reasoning': 'No norm belief was stated.',
            'judge_model': None,
            'belief': belief,
            'ground_truth': ground_truth,
        }

    client = client or OpenRouterClient(model=model or DEFAULT_JUDGE_MODEL)
    reply = client.response(
        image_paths=[],
        prompt=_PROMPT_TEMPLATE.format(ground_truth=ground_truth, belief=belief),
        system_prompt=_SYSTEM_PROMPT,
        max_completion_tokens=2048,
    )
    result = _parse_judge_response(reply)
    result['judge_model'] = client.model
    result['belief'] = belief
    result['ground_truth'] = ground_truth
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Grade a norm belief with the LLM judge.')
    parser.add_argument('--norm', required=True, help='Ground-truth norm name (norm_descriptions.json)')
    parser.add_argument('--belief', required=True, help="The agent's stated norm belief")
    parser.add_argument('--model', default=None, help=f'Judge model (default: {DEFAULT_JUDGE_MODEL})')
    args = parser.parse_args()

    result = judge_norm_belief(
        args.belief, ground_truth_norm_description(args.norm), model=args.model
    )
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
