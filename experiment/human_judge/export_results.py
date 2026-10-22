"""Flatten the collected ratings to a CSV and compute the agreement statistics.

Usage:
    python experiment/human_judge/export_results.py                 # data/ -> results/ratings.csv + stats
    python experiment/human_judge/export_results.py --data path/to/unzipped/data

Statistics reported (over every submitted rating, overall and per family):
  * distribution and mean of the 1-5 agreement rating, and its mean per judge level;
  * for records that also carry the participant's own 1-7 rating (wave 1):
    Spearman/Pearson between human and judge, exact and within-1 agreement,
    mean absolute difference, mean human rating per judge level, and
    inter-rater agreement treating human and judge as two raters: Cohen's
    kappa (unweighted, linear- and quadratic-weighted; also unweighted on the
    scale collapsed to low 1-2 / mid 3-5 / high 6-7) and Krippendorff's alpha
    (interval and nominal).
Pure Python (no numpy/scipy needed), so it runs anywhere the app does.
"""
import argparse
import csv
import glob
import io
import json
import math
import os
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
COLUMNS = ['participant_id', 'index', 'entry', 'norm', 'family', 'model',
           'judge_likert', 'human_likert', 'agree_with_judge', 'comment',
           'ms_to_own_rating', 'ms_to_reveal', 'ms_to_answer', 'ms_to_submit',
           'timestamp', 'belief', 'judge_reason']


def collect_ratings(data_dir: str) -> list:
    rows = []
    for path in sorted(glob.glob(os.path.join(data_dir, '*', '*_item*_*.json'))):
        try:
            with open(path, encoding='utf-8') as f:
                rows.append(json.load(f))
        except Exception as e:
            print(f'skip {path}: {e}')
    return rows


def to_csv(rows: list) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, extrasaction='ignore')
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue()


def _rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def pearson(x, y):
    n = len(x)
    if n < 2:
        return float('nan')
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy) if sxx and syy else float('nan')


def spearman(x, y):
    return pearson(_rank(x), _rank(y))


def cohen_kappa(x, y, weights=None, categories=None):
    """Cohen's kappa between two raters over the same items. weights: None
    (unweighted), 'linear' or 'quadratic' (for ordinal scales)."""
    cats = sorted(categories or set(x) | set(y))
    idx = {c: i for i, c in enumerate(cats)}
    k, n = len(cats), len(x)
    if n == 0 or k < 2:
        return float('nan')
    obs = [[0.0] * k for _ in range(k)]
    for a, b in zip(x, y):
        obs[idx[a]][idx[b]] += 1
    px = [sum(row) / n for row in obs]
    py = [sum(obs[i][j] for i in range(k)) / n for j in range(k)]
    def w(i, j):
        if weights is None:
            return 0.0 if i == j else 1.0
        d = abs(i - j) / (k - 1)
        return d if weights == 'linear' else d * d
    dis_obs = sum(w(i, j) * obs[i][j] / n for i in range(k) for j in range(k))
    dis_exp = sum(w(i, j) * px[i] * py[j] for i in range(k) for j in range(k))
    return 1 - dis_obs / dis_exp if dis_exp else float('nan')


def krippendorff_alpha(x, y, metric='interval'):
    """Krippendorff's alpha for two raters, every item rated by both.
    metric: 'interval' (squared difference) or 'nominal'."""
    n = len(x)
    if n < 2:
        return float('nan')
    vals = list(x) + list(y)
    def delta(a, b):
        return (a - b) ** 2 if metric == 'interval' else (0.0 if a == b else 1.0)
    d_obs = sum(delta(a, b) for a, b in zip(x, y)) / n              # mean within-item disagreement
    m = len(vals)
    d_exp = sum(delta(vals[i], vals[j]) for i in range(m) for j in range(i + 1, m)) / (m * (m - 1) / 2)
    return 1 - d_obs / d_exp if d_exp else float('nan')


def _bin3(v):
    """Collapse 1-7 to low (1-2) / mid (3-5) / high (6-7)."""
    return 'low' if v <= 2 else ('high' if v >= 6 else 'mid')


def stats(rows: list) -> dict:
    def block(rs):
        a = [r['agree_with_judge'] for r in rs]
        agree_per_level = defaultdict(list)
        for r in rs:
            agree_per_level[r['judge_likert']].append(r['agree_with_judge'])
        out = {
            'n': len(rs),
            'n_participants': len({r['participant_id'] for r in rs}),
            'agree_dist': dict(sorted(Counter(a).items())),
            'mean_agree': sum(a) / len(a) if a else None,
            'share_agree_or_strongly': (sum(v >= 4 for v in a) / len(a)) if a else None,
            'mean_agree_per_judge_level': {k: sum(v) / len(v) for k, v in sorted(agree_per_level.items())},
        }
        # Wave-1 records also carry the participant's own 1-7 rating.
        hj = [(r['human_likert'], r['judge_likert']) for r in rs if r.get('human_likert') is not None]
        if hj:
            h = [x for x, _ in hj]
            j = [y for _, y in hj]
            per_level = defaultdict(list)
            for x, y in hj:
                per_level[y].append(x)
            out.update({
                'n_with_human_likert': len(hj),
                'human_likert_dist': dict(sorted(Counter(h).items())),
                'spearman_human_vs_judge': spearman(h, j) if len(hj) > 1 else None,
                'pearson_human_vs_judge': pearson(h, j) if len(hj) > 1 else None,
                'exact_agreement': sum(x == y for x, y in hj) / len(hj),
                'within_1_agreement': sum(abs(x - y) <= 1 for x, y in hj) / len(hj),
                'mean_abs_diff': sum(abs(x - y) for x, y in hj) / len(hj),
                'mean_human_minus_judge': sum(x - y for x, y in hj) / len(hj),
                'mean_human_per_judge_level': {k: sum(v) / len(v) for k, v in sorted(per_level.items())},
                # inter-rater agreement, human vs judge as two raters on the same items
                'kappa_unweighted': cohen_kappa(h, j, None, range(1, 8)),
                'kappa_linear': cohen_kappa(h, j, 'linear', range(1, 8)),
                'kappa_quadratic': cohen_kappa(h, j, 'quadratic', range(1, 8)),
                'kappa_3bin_unweighted': cohen_kappa([_bin3(v) for v in h], [_bin3(v) for v in j], None, ['low', 'mid', 'high']),
                'krippendorff_alpha_interval': krippendorff_alpha(h, j, 'interval'),
                'krippendorff_alpha_nominal': krippendorff_alpha(h, j, 'nominal'),
            })
        return out
    out = {'overall': block(rows)}
    fams = defaultdict(list)
    for r in rows:
        fams[r['family']].append(r)
    out['per_family'] = {f: block(rs) for f, rs in sorted(fams.items())}
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default=os.path.join(_HERE, 'data'))
    ap.add_argument('--out_dir', default=os.path.join(_HERE, 'results'))
    args = ap.parse_args()
    rows = collect_ratings(args.data)
    if not rows:
        print(f'no ratings found under {args.data}')
        raise SystemExit(1)
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, 'ratings.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        f.write(to_csv(rows))
    st = stats(rows)
    json_path = os.path.join(args.out_dir, 'stats.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(st, f, indent=2)
    print(json.dumps(st['overall'], indent=2))
    print(f'wrote {csv_path} ({len(rows)} ratings) and {json_path}')
