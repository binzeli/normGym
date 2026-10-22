"""Build the human study's fixed assignment design: experiment/human/study_design.json.

Protocol (2026-09): every one of the 50 experiment norms is played by exactly 2
participants — once as scenario 1 and once as scenario 2 — and each participant's
two norms come from DIFFERENT norm categories (queue / region / trajectory /
allocation, the paper's four families as tagged in experiment/train/config.py:
NORM_GROUPS). 50 participants fill the design exactly.

Construction: one cycle over the norms in which no two neighbours share a
category; participant i plays (cycle[i], cycle[i+1]). Every norm then has one
predecessor and one successor, i.e. it is first for one participant and second
for another. A seeded random search picks, among many such cycles, one whose
directed category pairings are as even as the family sizes allow (queue has 16
norms, so 32 of the 50 pairs necessarily contain a queue norm).

The output is committed so the design is fixed; the app (app.py --assignment
design) only reads it. Re-run this script ONLY to change the design on purpose:
a different file archives the running assignment ledger and restarts the
design from slot 1.

    python experiment/human/make_study_design.py            # writes study_design.json (all 50 norms)
    python experiment/human/make_study_design.py --check    # verify the committed file

A design over a SUBSET of the norms (e.g. the validated half) uses the same
construction on that subset — K norms -> K participants, every norm still played
twice, once per order position, cross-category pairs:

    python experiment/human/make_study_design.py --norms_file experiment/human/norms_25.txt \
        --out experiment/human/study_design_25.json
    python experiment/human/make_study_design.py --check --out experiment/human/study_design_25.json
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _ROOT)

DESIGN_PATH = os.path.join(_HERE, 'study_design.json')
NORMS_JSON = os.path.join(_ROOT, 'src', 'norms', 'norm_descriptions.json')
TUTORIAL_NORM = 'straight_line'
SEED = 2026
RESTARTS = 4000


def norm_categories() -> dict:
    """norm -> category, from the training config's NORM_GROUPS (the paper's
    Table 1 families). Imported lazily: the training config is only needed to
    (re)generate the design, never to run the study."""
    from experiment.train.config import NORM_GROUPS
    cats = {}
    for group in NORM_GROUPS.values():
        for n in group['train'] + group['test']:
            cats[n] = group['category']
    return cats


def experiment_norms() -> list:
    with open(NORMS_JSON, encoding='utf-8') as f:
        return [n['name'] for n in json.load(f)['norms']]


def _random_cycle(norms, cat_of, rng, front=()):
    """A random cyclic order of `norms` with no two neighbours of one category.
    Greedy with a bias toward the category that has most norms left (which is
    what keeps the walk from painting itself into a corner); restarts on a
    dead end. `front`: norms that must occupy the FIRST positions of the cycle
    (in some order), so they land in the earliest slots."""
    front = list(front)
    while True:
        pool = [n for n in norms if n not in front]
        rng.shuffle(pool)
        if front:
            # Keep the given order (it is a priority order) when it already
            # alternates categories; otherwise fall back to a random order.
            head = front[:]
            if any(cat_of[a] == cat_of[b] for a, b in zip(head, head[1:])):
                rng.shuffle(head)
                if any(cat_of[a] == cat_of[b] for a, b in zip(head, head[1:])):
                    continue
            seq = head
        else:
            seq = [pool.pop()]
        ok = True
        while pool:
            allowed = [n for n in pool if cat_of[n] != cat_of[seq[-1]]]
            if not allowed:
                ok = False
                break
            remaining = Counter(cat_of[n] for n in pool)
            top = max(remaining[cat_of[n]] for n in allowed)
            cands = ([n for n in allowed if remaining[cat_of[n]] == top]
                     if rng.random() < 0.7 else allowed)
            pick = rng.choice(cands)
            pool.remove(pick)
            seq.append(pick)
        if ok and cat_of[seq[0]] != cat_of[seq[-1]]:
            return seq


def _score(seq, cat_of):
    """Lower is better: even directed pair counts, (a,b) ~ (b,a), and the two
    participants of one norm meeting partners from different categories."""
    k = len(seq)
    pairs = Counter((cat_of[seq[i]], cat_of[seq[(i + 1) % k]]) for i in range(k))
    spread = max(pairs.values()) - min(pairs.values())
    asym = sum(abs(pairs[(a, b)] - pairs[(b, a)]) for (a, b) in pairs if a < b)
    same_partner = sum(1 for i in range(k)
                       if cat_of[seq[(i - 1) % k]] == cat_of[seq[(i + 1) % k]])
    return spread * 3 + asym * 2 + same_partner


def build_design(norms, cat_of, seed=SEED, restarts=RESTARTS, front=()) -> dict:
    rng = random.Random(seed)
    best, best_score = None, float('inf')
    for _ in range(restarts):
        seq = _random_cycle(norms, cat_of, rng, front)
        s = _score(seq, cat_of)
        if s < best_score:
            best, best_score = seq, s
    k = len(best)
    schedule = [[best[i], best[(i + 1) % k]] for i in range(k)]
    return {
        '_comment': (
            'Fixed assignment design for the human study, generated by '
            'make_study_design.py — do not edit by hand. schedule[i] is the '
            'ordered norm list of participant slot i (0-based; scenario 1, '
            'scenario 2). Every norm appears once as scenario 1 and once as '
            'scenario 2, and the two norms of a slot are from different '
            'categories. app.py --assignment design hands slots out in '
            'arrival order (lowest free slot first).'),
        'tutorial_norm': TUTORIAL_NORM,
        'blocks': 2,
        'seed': seed,
        'categories': {n: cat_of[n] for n in norms},
        'schedule': schedule,
    }


def verify(design: dict, valid_norms=None) -> list:
    """Hard-constraint check. Returns a list of problems (empty = OK).
    `valid_norms`: the experiment's norm catalog; every norm of the design must
    be in it (the design itself may cover only a subset of the catalog)."""
    problems = []
    cat_of = design['categories']
    sched = design['schedule']
    pool = sorted(cat_of)
    blocks = design.get('blocks', 2)
    if valid_norms is not None:
        unknown = sorted(set(pool) - set(valid_norms))
        if unknown:
            problems.append(f'design names norms not in the catalog: {unknown}')
    in_sched = {n for row in sched for n in row}
    if in_sched != set(pool):
        problems.append(f'schedule and category map disagree on {sorted(in_sched ^ set(pool))}')
    if design.get('tutorial_norm') in cat_of:
        problems.append('tutorial norm is also a test norm')
    for i, row in enumerate(sched):
        if len(row) != blocks:
            problems.append(f'slot {i} has {len(row)} norms, expected {blocks}')
        if len({cat_of.get(n) for n in row}) != len(row):
            problems.append(f'slot {i} repeats a category: {row}')
    # A hand-made follow-up design may deliberately give some norms more plays
    # than others ("balanced_positions": false); then only the per-slot rules
    # above apply.
    if design.get('balanced_positions', True):
        for pos in range(blocks):
            seen = Counter(row[pos] for row in sched if len(row) > pos)
            bad = {n: c for n in pool for c in [seen.get(n, 0)] if c != 1}
            if bad:
                problems.append(f'position {pos + 1}: norms not played exactly once: {bad}')
    return problems


def summarize(design: dict) -> str:
    cat_of = design['categories']
    sched = design['schedule']
    sizes = Counter(cat_of.values())
    pairs = Counter((cat_of[a], cat_of[b]) for a, b in sched)
    lines = [f'{len(sched)} slots x {design["blocks"]} norms; categories: {dict(sizes)}',
             'directed category pairs (scenario 1 -> scenario 2):']
    for (a, b), c in sorted(pairs.items()):
        lines.append(f'  {a:>10} -> {b:<10} {c}')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--check', action='store_true',
                    help='verify the committed design instead of regenerating it')
    ap.add_argument('--out', default=DESIGN_PATH)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--norms', nargs='+', metavar='NORM',
                    help='build the design over just these norms (default: all experiment norms)')
    ap.add_argument('--norms_file',
                    help='file with one norm name per line (# comments allowed); '
                         'same as --norms')
    ap.add_argument('--front', nargs='+', metavar='NORM', default=[],
                    help='norms to put in the EARLIEST slots (e.g. norms with no '
                         'data yet), so the first arrivals play them')
    args = ap.parse_args()

    catalog = [n for n in experiment_norms() if n != TUTORIAL_NORM]
    if args.check:
        with open(args.out, encoding='utf-8') as f:
            design = json.load(f)
    else:
        norms = catalog
        subset = list(args.norms or [])
        if args.norms_file:
            with open(args.norms_file, encoding='utf-8') as f:
                subset += [ln.split('#')[0].strip() for ln in f
                           if ln.split('#')[0].strip()]
        if subset:
            unknown = sorted(set(subset) - set(catalog))
            if unknown:
                sys.exit(f'unknown norms in the subset: {unknown}')
            norms = [n for n in catalog if n in set(subset)]   # catalog order, de-duplicated
        cat_of = norm_categories()
        missing = sorted(set(norms) - set(cat_of))
        if missing:
            sys.exit(f'norms without a category in experiment/train/config.py: {missing}')
        bad_front = sorted(set(args.front) - set(norms))
        if bad_front:
            sys.exit(f'--front norms not in the design: {bad_front}')
        design = build_design(norms, cat_of, seed=args.seed, front=args.front)
        if args.front:
            design['front'] = list(args.front)
            design['_comment'] += (' The "front" norms were pinned to the earliest '
                                   'slots so the first arrivals play them.')
        if subset:
            design['_comment'] += (f' This design covers a SUBSET of {len(norms)} of the '
                                   f'{len(catalog)} experiment norms.')
    problems = verify(design, catalog)
    print(summarize(design))
    if problems:
        print('PROBLEMS:')
        for p in problems:
            print('  -', p)
        sys.exit(1)
    if not args.check:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(design, f, indent=1)
            f.write('\n')
        print(f'wrote {os.path.relpath(args.out, _ROOT)}')
    for i, row in enumerate(design['schedule'], 1):
        print(f'  slot {i:2d}: ' + '  ->  '.join(f'{n} ({design["categories"][n]})' for n in row))


if __name__ == '__main__':
    main()
