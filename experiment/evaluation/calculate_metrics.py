"""The benchmark's test metrics for one model, by norm family.

Prints one table per test split (``test`` = same layout, ``new_map_test`` =
new layout), with a row per norm family (region / trajectory / queue /
allocation, see ``FAMILIES``) plus a pooled ALL row. Each row pools every run
of the family's norms, each run weighted equally, and shows mean ± std across
runs. A trailing ``(k)`` marks a cell resting on fewer runs than the row's
``n``, and ``—`` a cell with no defined run.

Every value comes from the metrics summaries written by eval.py
(``<results-root>/<agent>/<model>/<world>/<norm>/metrics_summary_<norm>.json``).

Each metric is computed per test episode and averaged over all test episodes.

Task outcome:

* **Joint** (↑) — 1 only when the run reached ALL its goals and never
  violated the norm, else 0.
* **Task** (↑) — goal success: 1 all goals, 0.5 one of two, 0 none.
* **Comply** (↑) — 1 − norm_violation: 1 when the norm was never violated.

Individual and community costs, measured over the full episode: a test run
ends when the subject finishes, so eval.py replays it exactly and keeps the
episode running to its end (100 steps), observing the community over the same
horizon under every subject. These are the definitions of panel (A) of the
paper's main figure:

* **Path length** (↓) — steps from the subject's arrival on the map until it
  completed its task; a run that never finished counts up to the episode's
  end (a lower bound on the true time).
* **Collision** (↓) — 1 if the subject collided with another agent at least
  once in the episode, else 0.
* **Comm. path length** (↓) — mean steps the other agents took to reach their
  goals (finish step − arrival step), over those that finished in the episode.
* **Comm. welfare** (↑) — mean end-of-episode reward of the other agents:
  every agent that entered the map during the episode except the subject.

Runs whose replay diverged from the recorded run (``replay_verified`` false,
e.g. runs recorded with older simulator code) are left out of the four cost
columns and counted.

Usage:
    python experiment/evaluation/calculate_metrics.py memory_llm/anthropic_claude-opus-5
    python experiment/evaluation/calculate_metrics.py memory_llm/openai_gpt-5.6-sol --norms edge_left_line xcell_forbidden
    python experiment/evaluation/calculate_metrics.py --list        # show available models
    python experiment/evaluation/calculate_metrics.py <agent/model> --results-root <other tree>
"""
import argparse
import json
import math
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_RESULTS_ROOT = os.path.join(PROJECT_ROOT, 'experiment', 'results')

SPLITS = [('test', 'same layout'), ('new_map_test', 'new layout')]

# Labels are inlined (not imported from plot/common.py) so this file runs with
# a bare Python install: common.py pulls in matplotlib/numpy.
MODEL_LABELS = {
    'anthropic_claude-opus-5': 'Claude Opus 5',
    'google_gemini-3.1-pro-preview': 'Gemini 3.1 Pro',
    'openai_gpt-5.6-sol': 'GPT-5.6-sol',
    'openai_gpt-5.5': 'GPT-5.5',
    'openai_gpt-4o-mini': 'GPT-4o-mini',
    'google_gemini-3.6-flash': 'Gemini 3.6 Flash',
    'Qwen_Qwen3-VL-4B-Instruct': 'Qwen3-VL-4B',
    'Qwen_Qwen3-VL-8B-Instruct': 'Qwen3-VL-8B',
    'meta-llama_Llama-3.2-11B-Vision-Instruct': 'Llama 3.2 11B Vision',
    'zai-org_GLM-4.6V-Flash': 'GLM-4.6V-Flash',
    'mistralai_Ministral-3-8B-Instruct': 'Ministral 3 8B',
    'mistralai_Ministral-3-8B-Instruct-2512-BF16': 'Ministral 3 8B',
}

# The four norm families of the paper's norm-library table, covering all 50
# latent norms: region norms constrain which cells the agent may end up in
# (decidable from the entered cell alone), trajectory norms the form of its
# own motion (its own transition, the clock, or its own history), queue norms
# the shape of the line at the counter, and allocation norms how a contested
# resource is divided among co-present agents.
FAMILIES = [
    ('Region', [
        'xcell_forbidden',
        'occupation_door_all_exit_A',
        'occupation_door_all_exit_B',
        'occupation_door_all_exit_C',
        'occupation_door_A_forbidden',
        'occupation_door_B_forbidden',
        'occupation_door_C_forbidden',
        'occupation_seat_bottom_left_only',
        'occupation_seat_top_left_only',
        'occupation_seat_bottom_right_only',
        'occupation_seat_no_bottom_left',
        'occupation_seat_no_bottom_right',
    ]),
    ('Trajectory', [
        'movement_clockwise_move',
        'movement_counterclockwise_move',
        'movement_no_odd_turn',
        'movement_no_even_turn',
        'movement_no_turn_mod3_0',
        'movement_no_turn_mod3_1',
        'movement_no_turn_mod3_2',
        'movement_move1_wait1',
        'movement_move2_wait1',
        'movement_move3_wait1',
        'xcell_must_visit',
    ]),
    ('Queue', [
        'bend_right1_line',
        'bend_right2_line',
        'bend_left1_line',
        'bend_left2_line',
        'zigzag_right1_line',
        'zigzag_right2_line',
        'zigzag_left1_line',
        'zigzag_left2_line',
        'staircase_right_line',
        'staircase_left_line',
        'lshape_right_line',
        'lshape_left_line',
        'square_right_line',
        'square_left_line',
        'edge_left_line',
        'edge_right_line',
    ]),
    ('Allocation', [
        'occupation_region_capacity_1',
        'occupation_region_capacity_2',
        'occupation_region_single_class_only',
        'occupation_door_class_specific',
        'occupation_door_class_specific_BCA',
        'occupation_door_class_specific_CAB',
        'occupation_seat_class_specific',
        'occupation_seat_class_specific_rot1',
        'occupation_seat_class_specific_rot2',
        'occupation_seat_in_order',
        'occupation_seat_in_order_right',
    ]),
]
NORM_FAMILY = {n: fam for fam, norms in FAMILIES for n in norms}

# (column header, key, decimals). Arrows mark the direction that is better.
COLUMNS = [
    ('Joint ↑', 'joint', 3),
    ('Task ↑', 'task', 3),
    ('Comply ↑', 'comply', 3),
    ('Path length ↓', 'path_length', 1),
    ('Collision ↓', 'collision', 3),
    ('Comm. path length ↓', 'community_path_length', 1),
    ('Comm. welfare ↑', 'community_welfare', 2),
]
COST_KEYS = ('path_length', 'collision', 'community_path_length',
             'community_welfare')


def norm_family(norm):
    return NORM_FAMILY.get(norm, 'Other')


def model_label(model):
    return MODEL_LABELS.get(model, model)


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float('nan')


def _std(xs):
    """Sample std (ddof=1); NaN with fewer than two values."""
    xs = list(xs)
    if len(xs) < 2:
        return float('nan')
    mu = _mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def _load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


# ------------------------------------------------------------ metric scoring

def run_values(run):
    """The seven metrics of one run of a metrics summary (NaN where undefined).

    The cost values are left out (NaN) when the run has no verified replay."""
    success = float(run['success'])
    violation = float(run['norm_violation'])
    values = {
        'joint': 1.0 if (success >= 1.0 and not violation) else 0.0,
        'task': success,
        'comply': 1.0 - violation,
    }
    verified = run.get('replay_verified')
    for key in COST_KEYS:
        value = run.get(key) if verified else None
        values[key] = float('nan') if value is None else float(value)
    return values


def aggregate(values):
    """{'num_runs', key: mean, key_std: std, key_n: defined count} over a list
    of per-run metric dicts. Undefined entries are dropped; ``key_n`` is
    recorded when a cell rests on fewer runs than the row's total."""
    out = {'num_runs': len(values)}
    for _, key, _ in COLUMNS:
        col = [v[key] for v in values if not math.isnan(v[key])]
        out[key] = _mean(col)
        out[key + '_std'] = _std(col)
        if len(col) != len(values):
            out[key + '_n'] = len(col)
    return out


# -------------------------------------------------------------- data loading

def load_runs(base, norms=None):
    """({norm: {split: [per-run metric dicts]}}, counts of runs left out of
    the cost columns: {'no_replay': n, 'diverged': n})."""
    out, skipped = {}, {'no_replay': 0, 'diverged': 0}
    for norm in sorted(os.listdir(base)):
        if norms and norm not in norms:
            continue
        path = os.path.join(base, norm, f'metrics_summary_{norm}.json')
        if not os.path.isfile(path):
            continue
        summary = _load(path)
        out[norm] = {}
        for split, _ in SPLITS:
            runs = summary.get(split, {}).get('runs', [])
            for run in runs:
                if 'replay_verified' not in run:
                    skipped['no_replay'] += 1
                elif not run['replay_verified']:
                    skipped['diverged'] += 1
            out[norm][split] = [run_values(r) for r in runs]
    return out, skipped


def build_split_rows(norm_runs, split):
    """By-family rows for one split, then the pooled ALL row."""
    per_norm = {n: s[split] for n, s in norm_runs.items() if s.get(split)}
    families = [(fam, [n for n in norms if n in per_norm])
                for fam, norms in FAMILIES]
    other = sorted(n for n in per_norm if n not in NORM_FAMILY)
    if other:
        families.append(('Other', other))

    rows, pooled, pooled_norms = [], [], []
    for fam, norms in families:
        if not norms:
            continue
        fam_vals = [v for norm in norms for v in per_norm[norm]]
        plural = 's' if len(norms) != 1 else ''
        rows.append((f'{fam} ({len(norms)} norm{plural})', aggregate(fam_vals)))
        pooled.extend(fam_vals)
        pooled_norms.extend(norms)
    if pooled:
        rows.append((f'ALL ({len(pooled_norms)} norms)', aggregate(pooled)))
    return rows


# ---------------------------------------------------------------- printing

def _fmt(m, key, prec):
    """mean ± std; '—' when undefined everywhere, bare mean for a single
    value, a trailing (k) when the cell rests on fewer runs than the row's n."""
    mean, std = m[key], m[key + '_std']
    if math.isnan(mean):
        return '—'
    if abs(mean) < 0.5 * 10 ** -prec:
        mean = 0.0
    n = m.get(key + '_n')
    tail = f' ({n})' if n is not None else ''
    if math.isnan(std):
        return f'{mean:.{prec}f}{tail}'
    return f'{mean:.{prec}f} ± {std:.{prec}f}{tail}'


def format_table(rows, title, label_head='Family'):
    label_w = max([len(label_head)] + [len(lbl) for lbl, _ in rows])
    col_w = 20
    header = (f"{label_head:<{label_w}}  {'n':>3}  "
              + '  '.join(f'{h:>{col_w}}' for h, _, _ in COLUMNS))
    lines = [title, '=' * len(header), header, '-' * len(header)]
    for label, m in rows:
        if label.startswith('ALL'):
            lines.append('-' * len(header))
        vals = '  '.join(f'{_fmt(m, key, prec):>{col_w}}'
                         for _, key, prec in COLUMNS)
        lines.append(f"{label:<{label_w}}  {m['num_runs']:>3}  {vals}")
    return '\n'.join(lines)


def list_models(results_root, world):
    found = []
    if os.path.isdir(results_root):
        for agent in sorted(os.listdir(results_root)):
            adir = os.path.join(results_root, agent)
            if agent.startswith('oracle_') or not os.path.isdir(adir):
                continue
            for model in sorted(os.listdir(adir)):
                if os.path.isdir(os.path.join(adir, model, world)):
                    found.append(f'{agent}/{model}')
    return found


def main():
    parser = argparse.ArgumentParser(
        description='Print the benchmark test metrics for one model, by norm '
                    'family and overall, on the same-layout and new-layout '
                    'test splits.')
    parser.add_argument('model', nargs='?', metavar='AGENT/MODEL',
                        help='e.g. memory_llm/anthropic_claude-opus-5')
    parser.add_argument('--results-root', default=DEFAULT_RESULTS_ROOT,
                        help='results tree holding <agent>/<model>/<world>/<norm>/')
    parser.add_argument('--world', default='cafe')
    parser.add_argument('--norms', nargs='+', default=None,
                        help='restrict to these norms (default: every norm with a metrics file)')
    parser.add_argument('--list', action='store_true',
                        help='list the AGENT/MODEL trees under --results-root and exit')
    args = parser.parse_args()

    if args.list or not args.model:
        print('Available models under', args.results_root)
        for spec in list_models(args.results_root, args.world):
            print(' ', spec)
        if not args.list:
            raise SystemExit('\nPass one as: python calculate_metrics.py AGENT/MODEL')
        return

    if '/' not in args.model:
        raise SystemExit(f'Expected AGENT/MODEL, got: {args.model}')
    agent, model = args.model.split('/', 1)
    base = os.path.join(args.results_root, agent, model, args.world)
    if not os.path.isdir(base):
        raise SystemExit(f'No results directory: {base}')
    norm_runs, skipped = load_runs(base, args.norms)
    if not norm_runs:
        raise SystemExit(f'No metrics_summary files under {base}')
    if skipped['no_replay']:
        print(f"[NOTE] {skipped['no_replay']} runs have no full-episode metrics "
              f"yet; their cost columns are left out. Re-run "
              f"experiment/evaluation/eval.py on their raw_runs directories")
    if skipped['diverged']:
        print(f"[NOTE] {skipped['diverged']} runs whose replay diverged from "
              f"the original are left out of the cost columns")

    print(f'\n{model_label(model)} ({agent}) — {len(norm_runs)} norms')
    for split, split_name in SPLITS:
        rows = build_split_rows(norm_runs, split)
        if rows:
            print()
            print(format_table(rows, f'{split_name} test — by family'))


if __name__ == '__main__':
    main()
