"""Export Joint and Task summaries as CSV, JSON, and LaTeX tables."""
from __future__ import annotations

import csv
import json

LABELS = {'no_penalty': 'No penalty', 'penalty': 'Penalty'}


def write_table(path, caption, label, headers, rows):
    lines = [r'\begin{table}[t]', r'\centering\small', '\\caption{'+caption+'}',
             '\\label{'+label+'}', '\\begin{tabular}{'+'l'*(len(headers)-2)+'rr}',
             r'\toprule', ' & '.join(headers)+r' \\', r'\midrule']
    for row in rows:
        lines.append(' & '.join(str(v) for v in row)+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    path.write_text('\n'.join(lines)+'\n')


def build_tables(data, output):
    tables = data['tables']
    with (output/'final_test_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tables[0]))
        writer.writeheader()
        writer.writerows(tables)
    counts = sorted({r['episodes'] for r in tables if r['family']=='All'})
    common = ('Joint success requires both goals and no norm violation. Task scores episodes '
              'as 1, 0.5, or 0 for two, one, or no goals. Both are percentages. '
              f'Each architecture and condition has {counts[0] if len(counts)==1 else counts} episodes per evaluation layout condition.')
    layout_label = {'test':'Seen layout', 'new_map_test':'Unseen layouts'}
    def values(row):
        return [f'{100*row["joint"]:.1f}', f'{100*row["task"]:.1f}']
    for policy in ['mlp', 'cnn']:
        rows = [[layout_label[r['split']], r['family'], *values(r)] for r in tables
                if r['condition']=='penalty' and r['variant']==policy+'_sparse']
        write_table(output/f'{policy}_by_family.tex', f'Sparse-penalty DQN-{policy.upper()} by norm family. '+common,
                    f'tab:rl-{policy}-category', ['Evaluation layout','Family','J','T'], rows)
    rows = [[r['variant'].split('_')[0].upper(), LABELS[r['condition']], layout_label[r['split']], *values(r)]
            for r in tables if r['family']=='All' and r['variant'].endswith('sparse')]
    write_table(output/'penalty_ablation.tex', 'Penalty ablation: with versus without norm and collision penalties, using sparse rewards. '+common,
                'tab:rl-penalty-ablation', ['Model','Condition','Evaluation layout','Joint success','Task success'], rows)
    rows = [[r['variant'].split('_')[0].upper(), r['variant'].split('_')[1].title(), layout_label[r['split']], *values(r)]
            for r in tables if r['family']=='All' and r['condition']=='penalty']
    write_table(output/'reward_shaping_ablation.tex', 'Reward-shaping ablation: sparse versus dense rewards with both penalties active. '+common,
                'tab:rl-reward-ablation', ['Model','Reward','Evaluation layout','Joint success','Task success'], rows)
    # Export sparse-penalty summaries for downstream analysis.
    primary = [r for r in tables if r['condition']=='penalty' and r['variant'].endswith('sparse')]
    (output/'main_table_sparse_penalty.json').write_text(json.dumps(primary, indent=2)+'\n')
