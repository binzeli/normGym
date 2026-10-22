"""Community metrics over the full episode, from the replayed test runs.

Reads the runs written by experiment/evaluation/replay/replay_tests.py
(``experiment/results_replay/<spec>/cafe/<norm>/raw_runs/test_<k>.json`` and,
for the matched baseline, ``experiment/results_replay/oracle_like/<spec>/...``: the norm oracle playing the subject in that tree's own
setting) and computes, per run pair, the two community metrics of
experiment/evaluation/calculate_metrics.py at a common horizon -- the episode's end
(``max_steps``, 100) -- instead of at the step the subject happened to
finish:

* **welfare gap** -- the oracle run's bystander mean reward at the end
  minus the model run's, bystanders being every agent that entered the map
  other than the subject. Positive = the community earned less with the
  model in the slot.
* **community delay** -- over the bystanders that finished in both runs
  (matched by agent id: the pairing shares the seed and arrival schedule),
  the mean of (finish step under the model - finish step under the oracle).
  Positive = the community was slower with the model in the slot.

Runs whose replay diverged from the original before the subject finished
(``verify.mismatches > 0``) are left out and counted; the frontier models'
Sep-5 runs on the first 21 queue norms are the known case (older simulator
code). A run whose matched oracle run is missing is left out as well.

``aggregate(spec)`` returns the pooled dict plot_cost_externality.py reads
(``key``, ``key_std``, ``key_n``, ``num_runs``), so the figure can swap the
episode-end community panels for these without touching its other code.
"""
import json
import math
import os

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                         'results_replay')
WORLD = 'cafe'


def _load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _bystanders(run):
    subject = set(run['subject_agent_ids'])
    return {a: r for a, r in run['agent_rewards'].items() if a not in subject}


def run_pair_values(model_run, oracle_run):
    """{'welfare_gap', 'community_delay'} for one matched pair; NaN where
    undefined."""
    mine, theirs = _bystanders(model_run), _bystanders(oracle_run)
    welfare_gap = float('nan')
    if mine and theirs:
        welfare_gap = (sum(theirs.values()) / len(theirs)
                       - sum(mine.values()) / len(mine))
    exclude = set(model_run['subject_agent_ids']) | set(oracle_run['subject_agent_ids'])
    mine_done, theirs_done = model_run['completions'], oracle_run['completions']
    diffs = [mine_done[a] - theirs_done[a]
             for a in mine_done.keys() & theirs_done.keys() if a not in exclude]
    delay = sum(diffs) / len(diffs) if diffs else float('nan')
    return {'welfare_gap': welfare_gap, 'community_delay': delay}


def run_pairs(spec, data_root=DATA_ROOT, norms=None):
    """[(norm, run_id, values)] over the spec's verified, matched runs, plus
    {'diverged': n, 'unmatched': n} for the ones left out."""
    base = os.path.join(data_root, spec, WORLD)
    like = os.path.join(data_root, 'oracle_like', spec, WORLD)
    out, skipped = [], {'diverged': 0, 'unmatched': 0}
    for norm in sorted(os.listdir(base)):
        if norms and norm not in norms:
            continue
        runs_dir = os.path.join(base, norm, 'raw_runs')
        for name in sorted(os.listdir(runs_dir)):
            if not name.startswith('test_'):      # same-layout runs only
                continue
            run = _load(os.path.join(runs_dir, name))
            verify = run.get('verify') or {}
            if verify.get('mismatches'):
                skipped['diverged'] += 1
                continue
            oracle_path = os.path.join(like, norm, 'raw_runs', name)
            if not os.path.isfile(oracle_path):
                skipped['unmatched'] += 1
                continue
            out.append((norm, run['run_id'], run_pair_values(run, _load(oracle_path))))
    return out, skipped


def aggregate(spec, data_root=DATA_ROOT, norms=None):
    """The pooled row for the figure: mean, std and defined count per metric
    over every matched run, plus the skip counts."""
    pairs, skipped = run_pairs(spec, data_root, norms)
    out = {'num_runs': len(pairs), 'skipped': skipped,
           'norms': len({n for n, _, _ in pairs})}
    for key in ('welfare_gap', 'community_delay'):
        col = [v[key] for _, _, v in pairs if not math.isnan(v[key])]
        n = len(col)
        mean = sum(col) / n if n else float('nan')
        std = (math.sqrt(sum((x - mean) ** 2 for x in col) / (n - 1))
               if n > 1 else float('nan'))
        out[key], out[key + '_std'] = mean, std
        if n != len(pairs):
            out[key + '_n'] = n
    return out
