"""Build (and verify) the fixed assignment design for the LLM-judge study.

The design answers two questions:
  1. WHICH of the 500 judged beliefs are shown to humans at all, and
  2. WHO sees which ones, in what order.

Defaults: 30 participants x 10 items each = 300 ratings, every item rated by
ONE participant (so 300 distinct items). Items are sampled stratified over the
28 cells (4 norm families x 7 judge Likert levels) so the human ratings cover
the judge's whole scale, and each participant's 10 items are dealt so that
their judge levels and families are as spread out as possible (the judge
rating is hidden from them until they have given their own rating, but a set
of 10 that was all 1/7 or all 7/7 would still be a poor use of a rater).
Beliefs that are literally 'none' (the model offered no hypothesis) are
excluded by default: they are trivially a 1/7 and would only inflate
agreement. --include_none keeps them in the pool. Long texts are excluded
too so an item stays readable at a glance: beliefs over --max_belief_words
(default 70) and norms whose ground-truth text is over --max_gt_words
(default 170). Every queuing norm's text is 237-265 words (they share a long
description of lining up along a path), so the default drops the queuing
family; --max_gt_words 0 keeps it.

Usage:
    python experiment/human_judge/make_study_design.py                   # writes study_design.json
    python experiment/human_judge/make_study_design.py --check           # verify the existing file
    python experiment/human_judge/make_study_design.py --raters_per_item 2   # 100 items, 2 raters each
"""
import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_beliefs import DEFAULT_TXT, parse_file  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(_HERE, 'study_design.json')
FAMILIES = ['queuing', 'region', 'trajectory', 'allocation']
LIKERT = list(range(1, 8))


def rated_entries(data_dir, ignore_raters=()):
    """Entry ids that already have a rating file under data_dir (any
    participant whose id does not end with one of ignore_raters)."""
    import glob
    out = set()
    for path in glob.glob(os.path.join(data_dir, '*', '*_item*_*.json')):
        pid = os.path.basename(os.path.dirname(path))
        if any(pid.endswith(suf) for suf in ignore_raters):
            continue
        try:
            with open(path, encoding='utf-8') as f:
                out.add(int(json.load(f)['entry']))
        except Exception:
            continue
    return out


def sample_items(items, n_items, rng, levels=LIKERT, families=FAMILIES, use_last=()):
    """`use_last`: entry ids to draw only once a cell has nothing else left
    (already-rated items when a follow-up wave may re-rate)."""
    """Stratified sample: an equal quota per judge Likert level (the axis the
    study is about), and within each level round-robin over the 4 norm
    families. A family that runs dry inside a level (e.g. queuing has a single
    7/7 belief once 'none' is excluded) just hands its turn to the others, so
    the level's quota is still met."""
    cells = defaultdict(list)
    for it in items:
        cells[(it['judge_likert'], it['family'])].append(it)
    use_last = set(use_last)
    for key in cells:
        rng.shuffle(cells[key])
        # pop() takes from the end, so fresh items go last (= drawn first)
        cells[key].sort(key=lambda it: it['entry'] not in use_last)
    # Equal quota per level, water-filled: a level with fewer items than its
    # share contributes everything it has and the shortfall is spread evenly
    # over the levels that still have room.
    avail = {k: sum(len(cells[(k, f)]) for f in families) for k in levels}
    if sum(avail.values()) < n_items:
        raise ValueError(f'only {sum(avail.values())} eligible items for a design of {n_items}')
    quotas, remaining, open_levels = {}, n_items, list(levels)
    for k in sorted(levels, key=lambda k: avail[k]):
        share = remaining // len(open_levels)
        quotas[k] = min(avail[k], share)
        remaining -= quotas[k]
        open_levels.remove(k)
    for k in sorted(levels, key=lambda k: -(avail[k] - quotas[k])):  # leftover from rounding
        while remaining and quotas[k] < avail[k]:
            quotas[k] += 1
            remaining -= 1
    chosen = []
    for k in levels:
        fams = [f for f in families if cells.get((k, f))]
        got, i = 0, 0
        while got < quotas[k]:
            live = [f for f in fams if cells[(k, f)]]
            # fresh items anywhere in this level come before any fallback item
            fresh = [f for f in live if cells[(k, f)][-1]['entry'] not in use_last]
            live = fresh or live
            if not live:
                raise ValueError(f'not enough items at judge level {k} for a quota of {quotas[k]}')
            f = live[i % len(live)]
            chosen.append(cells[(k, f)].pop())
            got += 1
            i += 1
    return chosen


def deal(chosen, n_participants, per_participant, raters_per_item, rng):
    """Assign each item to `raters_per_item` distinct participants so that every
    participant gets exactly `per_participant` items with judge levels and
    families spread as evenly as possible. Greedy: items are handed out in a
    random order, each copy to the participant with room whose current mix
    would benefit most (fewest of that Likert level, then that family)."""
    slots = [[] for _ in range(n_participants)]
    lik = [Counter() for _ in range(n_participants)]
    fam = [Counter() for _ in range(n_participants)]
    order = list(chosen)
    rng.shuffle(order)
    # hardest first: process items level by level so that each level is spread
    # before the next one crowds the slots
    order.sort(key=lambda it: (it['judge_likert'], rng.random()))
    for it in order:
        taken = set()
        for _ in range(raters_per_item):
            cands = [p for p in range(n_participants)
                     if p not in taken and len(slots[p]) < per_participant]
            if not cands:
                raise ValueError('no participant has room left: check n * per == items * raters')
            rng.shuffle(cands)
            p = min(cands, key=lambda p: (lik[p][it['judge_likert']], fam[p][it['family']], len(slots[p])))
            slots[p].append(it['entry'])
            lik[p][it['judge_likert']] += 1
            fam[p][it['family']] += 1
            taken.add(p)
    for s in slots:
        rng.shuffle(s)  # presentation order is random
    return slots


def _words(s: str) -> int:
    return len(s.split())


def build(items, n_participants, per_participant, raters_per_item, seed, include_none,
          max_belief_words=None, max_gt_words=None, families=FAMILIES, levels=LIKERT,
          exclude_entries=(), allow_rerate=False):
    """`exclude_entries` are items already rated in an earlier wave: left out
    entirely, or (allow_rerate) used only where a level runs out of fresh
    items, so they get a second rater."""
    exclude_entries = set(exclude_entries)
    pool = [it for it in items if include_none or not it['belief_is_none']]
    pool = [it for it in pool if it['family'] in families and it['judge_likert'] in levels
            and (allow_rerate or it['entry'] not in exclude_entries)]
    if max_belief_words:
        pool = [it for it in pool if _words(it['belief']) <= max_belief_words]
    if max_gt_words:
        pool = [it for it in pool if _words(it['ground_truth_short']) <= max_gt_words]
    total = n_participants * per_participant
    if total % raters_per_item:
        raise ValueError('n_participants * items_per_participant must be divisible by raters_per_item')
    n_items = total // raters_per_item
    rng = random.Random(seed)
    chosen = sample_items(pool, n_items, rng, levels=list(levels), families=list(families),
                          use_last=exclude_entries if allow_rerate else ())
    schedule = deal(chosen, n_participants, per_participant, raters_per_item, rng)
    by_entry = {it['entry']: it for it in chosen}
    return {
        'source': os.path.basename(DEFAULT_TXT),
        'seed': seed,
        'n_participants': n_participants,
        'items_per_participant': per_participant,
        'raters_per_item': raters_per_item,
        'include_none': include_none,
        'max_belief_words': max_belief_words,
        'max_gt_words': max_gt_words,
        'families': list(families),
        'judge_levels': list(levels),
        'excluded_entries': sorted(exclude_entries),
        'allow_rerate': allow_rerate,
        'rerated_entries': sorted(it['entry'] for it in chosen if it['entry'] in exclude_entries),
        'pool_size': len(pool),
        'n_items': n_items,
        'items': {str(e): by_entry[e] for e in sorted(by_entry)},
        'schedule': schedule,
    }


def check(design, items_all=None):
    """Raise ValueError on anything that violates the design's own contract."""
    sched = design['schedule']
    items = design['items']
    n, per, r = design['n_participants'], design['items_per_participant'], design['raters_per_item']
    if len(sched) != n:
        raise ValueError(f'schedule has {len(sched)} slots, expected {n}')
    uses = Counter()
    for i, row in enumerate(sched):
        if len(row) != per:
            raise ValueError(f'slot {i} has {len(row)} items, expected {per}')
        if len(set(row)) != len(row):
            raise ValueError(f'slot {i} repeats an item: {row}')
        for e in row:
            if str(e) not in items:
                raise ValueError(f'slot {i} references unknown item {e}')
            uses[e] += 1
    if len(uses) != design['n_items']:
        raise ValueError(f'{len(uses)} distinct items used, expected {design["n_items"]}')
    bad = {e: c for e, c in uses.items() if c != r}
    if bad:
        raise ValueError(f'items not rated exactly {r} times: {bad}')
    for e, it in items.items():
        if it['family'] not in design.get('families', FAMILIES):
            raise ValueError(f"item {e} is a {it['family']} norm, not in the design's families")
        if it['judge_likert'] not in design.get('judge_levels', LIKERT):
            raise ValueError(f"item {e} has judge level {it['judge_likert']}, not in the design's levels")
        if int(e) in set(design.get('excluded_entries', [])) and not design.get('allow_rerate'):
            raise ValueError(f'item {e} is in the excluded (already rated) set')
        if design.get('max_belief_words') and _words(it['belief']) > design['max_belief_words']:
            raise ValueError(f"item {e} belief exceeds {design['max_belief_words']} words")
        if design.get('max_gt_words') and _words(it['ground_truth_short']) > design['max_gt_words']:
            raise ValueError(f"item {e} true norm (short) exceeds {design['max_gt_words']} words")
    if items_all is not None:
        src = {it['entry']: it for it in items_all}
        for e, it in items.items():
            s = src.get(int(e))
            if s is None or s['belief'] != it['belief'] or s['judge_likert'] != it['judge_likert']:
                raise ValueError(f'item {e} does not match the source txt any more')
            if s['ground_truth_short'] != it.get('ground_truth_short'):
                raise ValueError(f'item {e}: norm_short.json changed for {it["norm"]} — regenerate the design')
    return True


def report(design):
    items = design['items']
    lines = [f"{design['n_participants']} participants x {design['items_per_participant']} items, "
             f"{design['raters_per_item']} rater(s)/item -> {design['n_items']} items "
             f"(seed {design['seed']}, none beliefs {'kept' if design['include_none'] else 'excluded'}, "
             f"belief <= {design.get('max_belief_words') or 'any'} words, "
             f"true norm <= {design.get('max_gt_words') or 'any'} words, pool {design.get('pool_size', '?')}; "
             f"families {design.get('families', FAMILIES)}, judge levels {design.get('judge_levels', LIKERT)}, "
             f"{len(design.get('excluded_entries', []))} already-rated entries "
             f"{'used only as fallback: ' + str(len(design.get('rerated_entries', []))) + ' re-rated' if design.get('allow_rerate') else 'excluded'})"]
    bw = [_words(it['belief']) for it in items.values()]
    gw = [_words(it['ground_truth_short']) for it in items.values()]
    lines.append(f'belief words: median {sorted(bw)[len(bw) // 2]}, max {max(bw)};  '
                 f'true-norm (short, as shown) words: median {sorted(gw)[len(gw) // 2]}, max {max(gw)}')
    lines.append(f"norms covered: {len({it['norm'] for it in items.values()})}")
    rer = set(design.get('rerated_entries', []))
    lines.append('items per judge Likert: ' + ', '.join(
        f'{k}:{v}' + (f' ({sum(1 for it in items.values() if it["judge_likert"] == k and it["entry"] in rer)} re-rated)' if rer else '')
        for k, v in sorted(Counter(it['judge_likert'] for it in items.values()).items())))
    lines.append('items per family: ' + ', '.join(
        f'{k}:{v}' for k, v in sorted(Counter(it['family'] for it in items.values()).items())))
    lines.append('items per model: ' + ', '.join(
        f'{k}:{v}' for k, v in sorted(Counter(it['model'] for it in items.values()).items())))
    spreads = []
    for i, row in enumerate(design['schedule']):
        lv = sorted(items[str(e)]['judge_likert'] for e in row)
        fm = Counter(items[str(e)]['family'] for e in row)
        spreads.append(f'  slot {i:2d}: likert {lv}  families {dict(sorted(fm.items()))}')
    lines.append('per participant:')
    lines.extend(spreads)
    return '\n'.join(lines)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--txt', default=DEFAULT_TXT)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--n_participants', type=int, default=30)
    ap.add_argument('--items_per_participant', type=int, default=10)
    ap.add_argument('--raters_per_item', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--include_none', action='store_true',
                    help="keep beliefs that are literally 'none' in the pool")
    ap.add_argument('--max_belief_words', type=int, default=70,
                    help='only beliefs up to this many words (0 = no cap)')
    ap.add_argument('--max_gt_words', type=int, default=0,
                    help='only norms whose SHORT ground-truth text (norm_short.json, what the participant '
                         'sees) is up to this many words (0 = no cap; the short texts are all under 90 words)')
    ap.add_argument('--families', nargs='+', default=FAMILIES, choices=FAMILIES,
                    help='norm families to draw from (default: all four)')
    ap.add_argument('--judge_levels', nargs='+', type=int, default=LIKERT, choices=LIKERT,
                    help='judge Likert levels to draw from (default: 1-7)')
    ap.add_argument('--exclude_rated_in', default=None, metavar='DATA_DIR',
                    help='skip entries that already have a rating file under this data dir (a follow-up wave)')
    ap.add_argument('--allow_rerate', action='store_true',
                    help='with --exclude_rated_in: when a level runs out of fresh items, fall back to '
                         'already-rated ones (they get a second rater) instead of shrinking that level')
    ap.add_argument('--ignore_rater', nargs='*', default=[], metavar='PID_SUFFIX',
                    help='with --exclude_rated_in: ratings by participant ids ending in these do not count')
    ap.add_argument('--check', action='store_true',
                    help='only verify the existing design file (exit 1 if invalid)')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    items_all = parse_file(args.txt)
    if args.check:
        with open(args.out, encoding='utf-8') as f:
            design = json.load(f)
        try:
            check(design, items_all)
        except ValueError as e:
            print(f'INVALID design {args.out}: {e}')
            sys.exit(1)
        if not args.quiet:
            print(f'OK {args.out}\n' + report(design))
        sys.exit(0)

    excluded = rated_entries(args.exclude_rated_in, args.ignore_rater) if args.exclude_rated_in else set()
    design = build(items_all, args.n_participants, args.items_per_participant,
                   args.raters_per_item, args.seed, args.include_none,
                   args.max_belief_words or None, args.max_gt_words or None,
                   families=args.families, levels=sorted(args.judge_levels), exclude_entries=excluded,
                   allow_rerate=args.allow_rerate)
    check(design, items_all)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(design, f, indent=1, ensure_ascii=False)
    print(report(design))
    print(f'wrote {args.out}')
