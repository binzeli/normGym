"""Group-relative belief judge for v2 (belief-rewarded RLOO).

One LLM call per RLOO group: the judge sees the ground-truth norm description
and the K distinct beliefs the policy stated for the SAME snapshot and either
RANKS them (``rank_group``, the training reward — user choice 2026-08-27) or
rates each on the 7-point rubric of experiment/utils/norm_judge.py
(``score_group``, kept for the absolute held-out ``eval_belief`` metric).

Rank mode: the judge orders the candidates best-to-worst with ties allowed
(candidates expressing the same rule share a tier). Reward is BINARY like
the RLOO paper's accuracy reward (user choice 2026-08-27): the top tier
gets 1, every other candidate 0; tied winners all get 1, and a group where
every belief is equally good/bad is one tier -> all 1 -> zero LOO advantage
(no reward for being the least-bad wrong answer). ``binary=False`` gives
the graded (K - rank) / (K - 1) instead. Rankings are relative to
the group, so they are not cached; identical texts are sent once. Rating the
group jointly makes the ratings mutually calibrated (the same belief cannot
get a 6 in one sample and a 3 in another of the same group), which is what
the leave-one-out advantage needs; the scale stays absolute so a group of
four equally wrong beliefs is scored equally (no reward for being the least
bad wrong answer — the wait-hack lesson of run 108906).

r_belief = (likert - 1) / 6 in [0, 1]. Empty belief -> 0 without a call.
Ratings are cached per (model, norm, belief) so the models' habit of
repeating a belief verbatim for many steps costs one call.

Default judge: openai/gpt-5.4-nano (user choice 2026-08-27). Probe on
bend_left1_line with {correct, straight-line, mirror-image, unrelated}
beliefs: 5.4-nano rated 7/3/5/1 (in $0.20/M, ~$0.00035 per group call);
gpt-4.1-mini 7/5/5/1; gpt-4.1-nano 4/2/4/2 (unusable). Reasoning models
need the 1024-token completion budget (gpt-5-mini returned nothing at 64).
"""
import json
import os
import re
import sys
import threading
from typing import Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

from experiment.utils.client import OpenRouterClient
from experiment.utils.norm_judge import (
    LIKERT_MAX,
    LIKERT_MIN,
    ground_truth_norm_description,
    likert_to_score,
)

DEFAULT_JUDGE_MODEL = os.getenv('BELIEF_JUDGE_MODEL', 'openai/gpt-5.4-nano')


def likert_to_sharp_score(likert: int) -> float:
    """Likert 1-5 -> 0, 6 -> 0.5, 7 -> 1. Calibration (RUN_LOG 2026-08-29):
    the judge rates a generic shapeless belief 3.5-5, so the linear mapping
    pays ~0.4-0.67 for staying vague and a specific-but-wrong guess less --
    a gradient AWAY from exploring specific hypotheses. Zeroing <= 5 keeps
    only near-correct beliefs rewarded."""
    return max(0.0, (likert - 5) / (LIKERT_MAX - 5))

_SYSTEM_PROMPT = (
    "You grade how well an agent inferred a social norm. You are given the "
    "ground-truth norm and several candidate beliefs the agent stated for the "
    "same situation, and you rate EACH candidate independently on a 7-point "
    "Likert scale (whole numbers 1 to 7). Candidates that express the same "
    "rule must get the same rating."
)

_RUBRIC = """Rate how well each belief matches the ground-truth norm on a 7-point
Likert scale, using whole numbers only:
- 7: the belief expresses the same rule as the ground truth (wording may differ).
- 6: the same rule with a minor error or omission, or with extra incorrect
  constraints bundled in alongside the correct rule.
- 5: the belief identifies the right kind of rule — the correct behavior it
  constrains and the correct feature it depends on — but gets the specific
  rule wrong (e.g. the mirror image of the true line shape, the opposite
  turning direction, or "only sit in the bottom-left region" when the true
  rule is "never sit in the bottom-left region").
- 4: the belief constrains the right behavior but hinges on the wrong feature,
  or captures only about half of the rule.
- 3: the belief constrains the right behavior but captures only a small
  fragment of the rule.
- 2: only vaguely related — touches the same general behavior in passing but
  the stated rule is otherwise off.
- 1: unrelated to the ground-truth rule, or too vague to act on.

Judge only the content of the rule — ignore differences in phrasing, level of
detail, or hedging about confidence. Length is not quality: a longer belief,
or one that lists several competing hypotheses, must NOT be rated above a
shorter belief that expresses the same rule. A belief that lists several
hypotheses is rated by the one it commits to; if it commits to none, rate
what all of them share, and never credit it for merely mentioning the right
rule among alternatives.

A belief counts as expressing the same rule (7) when it captures the rule's
defining content — the specific constraint that distinguishes this norm from
its neighbours (for a queueing norm the shape of the line; for a movement norm
the direction or turn rule; for a seating or region norm which places are
allowed or forbidden for whom; for a door norm which door and for whom). It
does not also need to restate incidental details such as where a line starts,
how agents advance, capacities, exact lengths, or what happens when the rule
cannot be followed.

Respond with only a JSON object mapping each candidate number to its rating:
{"ratings": {"1": <1-7>, "2": <1-7>, ...}, "reasoning": "<one or two sentences>"}"""

_RANK_RUBRIC = """Rank the candidate beliefs from the one that best matches the ground-truth
norm to the one that matches it worst. Judge only the content of the rule —
ignore phrasing, level of detail and hedging about confidence. Two candidates
that express the same rule (or are equally wrong) MUST share a tier: do not
break ties arbitrarily. A belief that lists several hypotheses is judged by
the one it commits to. A belief is a full match when it captures the rule's
defining content — the constraint that distinguishes this norm from its
neighbours (a line's shape, a turn or direction rule, which places or doors
are allowed for whom) — even if it omits incidental details (where a line
starts, how agents advance, capacities, lengths).

Respond with only a JSON object whose "ranking" is a list of tiers, best
first; each tier is a flat list of candidate numbers (one number when the
tier has a single candidate). Every candidate must appear exactly once.
Example for four candidates where 2 is best, 1 and 3 are tied, 4 is worst:
{"ranking": [[2], [1, 3], [4]], "reasoning": "<one or two sentences>"}"""

_PROMPT_TEMPLATE = """Ground-truth norm:
{ground_truth}

Candidate beliefs (each stated by the agent for the same situation):
{candidates}

{rubric}"""


def _parse_ratings(text: str, n: int) -> Optional[List[int]]:
    match = re.search(r'\{.*\}', text or '', re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    ratings = parsed.get('ratings') if isinstance(parsed, dict) else None
    if not isinstance(ratings, dict):
        return None
    out = []
    for i in range(1, n + 1):
        v = ratings.get(str(i), ratings.get(i))
        if not isinstance(v, (int, float)):
            return None
        out.append(int(min(LIKERT_MAX, max(LIKERT_MIN, round(v)))))
    return out


def _parse_ranking(text: str, n: int) -> Optional[List[int]]:
    """-> competition rank (1 = best) per candidate, or None if malformed.

    Tolerant of what gpt-5.4-nano actually emits (job 117139): a tier given
    as a bare number, tiers nested one level too deep (``[[[1],[2],[3]]]`` =
    three ordered tiers), and candidates left out of the ranking (placed in a
    final tier below every listed one). A candidate listed twice is malformed.
    """
    match = re.search(r'\{.*\}', text or '', re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    tiers = parsed.get('ranking') if isinstance(parsed, dict) else None
    if not isinstance(tiers, list):
        return None
    # normalise: every tier becomes a flat list of ints, nested lists split
    flat_tiers: List[List[int]] = []

    def push(tier):
        if isinstance(tier, (int, float)):
            flat_tiers.append([int(tier)])
        elif isinstance(tier, list):
            if tier and all(isinstance(c, list) for c in tier):
                for sub in tier:            # nested one level too deep
                    push(sub)
            elif tier and all(isinstance(c, (int, float)) for c in tier):
                flat_tiers.append([int(c) for c in tier])
            elif tier:
                for sub in tier:            # mixed: split into singletons
                    push(sub)

    for tier in tiers:
        push(tier)
    ranks: Dict[int, int] = {}
    pos = 1
    for tier in flat_tiers:
        members = []
        for c in tier:
            if c < 1 or c > n or c in ranks or c in members:
                return None
            members.append(c)
        for c in members:
            ranks[c] = pos
        pos += len(members)
    if not ranks:
        return None
    for c in range(1, n + 1):               # omitted candidates: last tier
        if c not in ranks:
            ranks[c] = pos
    return [ranks[i] for i in range(1, n + 1)]


def ranks_to_rewards(ranks: List[int], binary: bool = True) -> List[float]:
    """binary (default, paper-style 1/0): the top tier gets 1, everyone else
    0. graded: (K - rank) / (K - 1)."""
    k = len(ranks)
    if binary:
        return [1.0 if r == 1 else 0.0 for r in ranks]
    if k < 2:
        return [1.0] * k
    return [(k - r) / (k - 1) for r in ranks]


def reference_belief(user_prompt: str) -> str:
    """The belief carried over in the snapshot's prompt (episodic_memory.
    render_belief line); '' when the prompt says 'none'."""
    m = re.search(r'Latest norm hypothesis \(belief\): (.*)', user_prompt or '')
    if not m:
        return ''
    text = m.group(1).strip()
    if text.lower() == 'none':
        return ''
    return re.sub(r'\s*\(confidence: [0-9.]+\)\s*$', '', text).strip()


class GroupBeliefJudge:
    """score_group(norm, beliefs) -> [score in [0,1] or None per belief].

    None means the judge call failed (network, unparseable reply); the caller
    treats that as "no belief signal for this group" (all members equal), so
    a flaky judge degrades to v1 behaviour instead of injecting noise.
    """

    def __init__(self, model: str = DEFAULT_JUDGE_MODEL, cache_path: Optional[str] = None,
                 max_completion_tokens: int = 1024):
        self.model = model
        self.client = OpenRouterClient(model=model)
        self.max_completion_tokens = max_completion_tokens
        self.cache_path = cache_path
        self.cache: Dict[str, int] = {}
        self._lock = threading.Lock()
        self.calls = 0
        self.failures = 0
        self._descriptions: Dict[str, str] = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, encoding='utf-8') as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self.cache[rec['key']] = int(rec['likert'])
                    except (ValueError, KeyError):
                        continue

    @property
    def cost_usd(self) -> float:
        return float(self.client.total_cost_usd)

    def _description(self, norm: str) -> str:
        if norm not in self._descriptions:
            self._descriptions[norm] = ground_truth_norm_description(norm)
        return self._descriptions[norm]

    def _key(self, norm: str, belief: str) -> str:
        return json.dumps([self.model, norm, belief.strip()])

    def _remember(self, key: str, likert: int) -> None:
        with self._lock:
            self.cache[key] = likert
            if self.cache_path:
                with open(self.cache_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({'key': key, 'likert': likert}) + '\n')

    def score_group(self, norm: str, beliefs: List[Optional[str]],
                    eligible: Optional[List[bool]] = None, max_chars: int = 0,
                    sharp: bool = False) -> List[Optional[float]]:
        """Absolute per-belief scores. Over-long beliefs (> max_chars) and
        members flagged ineligible score 0 without a judge call, like empty
        ones. sharp=True maps Likert 5/6/7 -> 0/0.5/1 (<= 5 pays nothing) so
        the judge's generosity toward vague beliefs (a shapeless norm rated 5)
        leaves no reward plateau to farm; the cache stores raw Likerts, so
        both mappings share it."""
        conv = likert_to_sharp_score if sharp else likert_to_score
        cleaned = [(b or '').strip() for b in beliefs]
        ok = [bool(b) and (max_chars <= 0 or len(b) <= max_chars)
              and (eligible is None or eligible[i]) for i, b in enumerate(cleaned)]
        scores: List[Optional[float]] = [None] * len(cleaned)
        todo: List[str] = []
        for i, b in enumerate(cleaned):
            if not ok[i]:
                scores[i] = 0.0
            elif self._key(norm, b) in self.cache:
                scores[i] = conv(self.cache[self._key(norm, b)])
            elif b not in todo:
                todo.append(b)
        if todo:
            ratings = self._call(norm, todo)
            if ratings is None:
                self.failures += 1
                return [None] * len(cleaned)
            for b, r in zip(todo, ratings):
                self._remember(self._key(norm, b), r)
            for i, b in enumerate(cleaned):
                if scores[i] is None:
                    scores[i] = conv(self.cache[self._key(norm, b)])
        return scores

    def rank_group(self, norm: str, beliefs: List[Optional[str]],
                   binary: bool = True) -> List[Optional[float]]:
        """Rank-based rewards per belief (binary: 1 for the judge's top tier,
        0 otherwise — the RLOO paper's 1/0; graded: (K-rank)/(K-1));
        [None]*K if the call failed. Empty beliefs are ranked last without
        being sent; a group with < 2 distinct non-empty beliefs needs no call."""
        cleaned = [(b or '').strip() for b in beliefs]
        distinct = []
        for b in cleaned:
            if b and b not in distinct:
                distinct.append(b)
        if len(distinct) >= 2:
            ranks = self._call(norm, distinct, mode='rank')
            if ranks is None:
                self.failures += 1
                return [None] * len(cleaned)
            rank_of = dict(zip(distinct, ranks))
        elif len(distinct) == 1:
            rank_of = {distinct[0]: 1}
        else:
            rank_of = {}
        # empty beliefs share the tier after every ranked one
        empty_rank = len(distinct) + 1
        per_text = {b: rank_of.get(b, empty_rank) for b in set(cleaned)}
        # competition ranks over the group's members (duplicates share a rank)
        order = sorted(set(per_text.values()))
        rewards = {}
        pos = 1
        for r in order:
            members = [b for b in cleaned if per_text[b] == r]
            for b in set(members):
                rewards[b] = pos
            pos += len(members)
        return ranks_to_rewards([rewards[b] for b in cleaned], binary=binary)

    def improvement_group(self, norm: str, beliefs: List[Optional[str]], reference: str,
                          votes: int = 2, max_chars: int = 0,
                          eligible: Optional[List[bool]] = None) -> Dict[str, object]:
        """Reward genuine belief revision (user choice 2026-08-27).

        One call (x ``votes``, averaged) rates the carried-over ``reference``
        belief and the group's distinct beliefs on the absolute 7-point scale.
        A sample earns 1 iff its rating is strictly above the reference's AND
        it is tied-best within the group; everything else 0. Four restatements
        of the reference -> all 0 -> no gradient; a sample that revises the
        hypothesis toward the true rule takes the whole LOO advantage.
        Returns {'rewards': [..] (None per member if the judge failed),
        'ref_likert': float|None, 'likerts': [float|None]}.
        """
        cleaned = [(b or '').strip() for b in beliefs]
        reference = (reference or '').strip()
        # Length / format gate (user choice 2026-08-27, after run 117349 grew
        # beliefs 766 -> 1672 chars and truncated a third of its real-play
        # turns): over-long beliefs and members flagged ineligible (truncated
        # or unparsed responses) are never judged and never rewarded.
        ok = [bool(b) and (max_chars <= 0 or len(b) <= max_chars)
              and (eligible is None or eligible[i]) for i, b in enumerate(cleaned)]
        distinct = []
        for b, good in zip(cleaned, ok):
            if good and b != reference and b not in distinct:
                distinct.append(b)
        out = {'rewards': [None] * len(cleaned), 'ref_likert': None,
               'likerts': [None] * len(cleaned), 'gated': [not g for g in ok]}
        if not distinct:                       # nothing but restatements / empties
            out['rewards'] = [0.0] * len(cleaned)
            return out
        candidates = ([reference] if reference else []) + distinct
        sums = [0.0] * len(candidates)
        for _ in range(max(1, votes)):
            ratings = self._call(norm, candidates, mode='score')
            if ratings is None:
                self.failures += 1
                return out
            for i, r in enumerate(ratings):
                sums[i] += r
        means = [x / max(1, votes) for x in sums]
        ref_likert = means[0] if reference else float(LIKERT_MIN)
        by_text = dict(zip(distinct, means[len(candidates) - len(distinct):]))
        likerts = [by_text.get(b, ref_likert if (b == reference and g) else float(LIKERT_MIN))
                   for b, g in zip(cleaned, ok)]
        best = max(likerts)
        out['rewards'] = [1.0 if (g and l > ref_likert and l == best) else 0.0
                          for g, l in zip(ok, likerts)]
        out['ref_likert'] = ref_likert
        out['likerts'] = likerts
        return out

    def _call(self, norm: str, beliefs: List[str], mode: str = 'score') -> Optional[List[int]]:
        candidates = '\n\n'.join(f'[{i + 1}] {b}' for i, b in enumerate(beliefs))
        prompt = _PROMPT_TEMPLATE.format(ground_truth=self._description(norm),
                                         candidates=candidates,
                                         rubric=_RANK_RUBRIC if mode == 'rank' else _RUBRIC)
        # One retry: the judge occasionally omits a candidate from its JSON
        # (seen on job 117111) and a wasted group is worse than a second call.
        for attempt in range(2):
            self.calls += 1
            try:
                reply = self.client.response(image_paths=[], prompt=prompt,
                                             system_prompt=_SYSTEM_PROMPT,
                                             max_completion_tokens=self.max_completion_tokens,
                                             temperature=0.0 if attempt == 0 else 0.3)
            except Exception as exc:  # network / provider errors
                print(f'[judge] call failed: {exc}', flush=True)
                continue
            ratings = (_parse_ranking(reply, len(beliefs)) if mode == 'rank'
                       else _parse_ratings(reply, len(beliefs)))
            if ratings is not None:
                return ratings
            print(f'[judge] unparseable reply (attempt {attempt + 1}): {str(reply)[:200]!r}', flush=True)
        return None


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Probe the group judge on a norm.')
    p.add_argument('--norm', default='bend_left1_line')
    p.add_argument('--model', default=DEFAULT_JUDGE_MODEL)
    p.add_argument('--beliefs', nargs='+', required=True)
    p.add_argument('--mode', choices=['improve', 'rank', 'score'], default='improve')
    p.add_argument('--reference', default='', help='carried-over belief (improve mode)')
    p.add_argument('--max_chars', type=int, default=600)
    p.add_argument('--graded', action='store_true')
    a = p.parse_args()
    j = GroupBeliefJudge(a.model)
    if a.mode == 'improve':
        res = j.improvement_group(a.norm, a.beliefs, a.reference, max_chars=a.max_chars)
    elif a.mode == 'rank':
        res = j.rank_group(a.norm, a.beliefs, binary=not a.graded)
    else:
        res = j.score_group(a.norm, a.beliefs)
    print(json.dumps({a.mode: res, 'calls': j.calls, 'cost_usd': j.cost_usd}, indent=1))
