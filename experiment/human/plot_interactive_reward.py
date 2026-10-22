"""Plot the running-total reward over steps for the three tested norms, averaged
over the study participants' interactive-stage runs.

Reward line: the participant's own ``step_reward`` accumulated continuously
(un-reset) across their successive interactive characters, so each line is a
smooth cumulative trajectory. Per norm we average this running total across the
study participants at each step index (mean over whoever is still on the map).

    python experiment/human/plot_interactive_reward.py
"""
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'old_results')
STUDY_PARTICIPANTS = ['Austen', 'Chris', 'Gillian', 'Maria', 'MatthewRenze', 'Navya', 'rebekah']

NORMS = ['occupation_seat_no_bottom_left', 'movement_clockwise_move', 'spacing_no_adjacent']
NORM_LABELS = {
    'occupation_seat_no_bottom_left': 'Occupation (no bottom-left seat)',
    'movement_clockwise_move': 'Movement (clockwise)',
    'spacing_no_adjacent': 'Spacing (no adjacent)',
}
# Validated categorical slots 1-3 (dataviz palette), light mode.
NORM_COLORS = {
    'occupation_seat_no_bottom_left': '#2a78d6',   # blue
    'movement_clockwise_move': '#008300',     # green
    'spacing_no_adjacent': '#e87ba4',         # magenta
}


def running_total(jsonl_path: str) -> np.ndarray:
    """Cumulative sum of the participant's step_reward over the whole session."""
    rewards = []
    for line in open(jsonl_path, encoding='utf-8'):
        e = json.loads(line)
        if e.get('event') == 'step':
            rewards.append(float(e.get('step_reward', 0.0)))
    return np.cumsum(rewards) if rewards else np.array([])


def norm_series() -> Dict[str, List[np.ndarray]]:
    """{norm: [running_total per participant]} from the study interactive runs.
    If a participant has several interactive files for a norm, keep the longest."""
    series: Dict[str, List[np.ndarray]] = defaultdict(list)
    for p in STUDY_PARTICIPANTS:
        best: Dict[str, np.ndarray] = {}
        for f in sorted(glob.glob(os.path.join(RESULTS_DIR, p, '*_interactive_ep*.jsonl'))):
            norm = next((n for n in NORMS if n in os.path.basename(f)), None)
            if norm is None:
                continue
            rt = running_total(f)
            if norm not in best or len(rt) > len(best[norm]):
                best[norm] = rt
        for norm, rt in best.items():
            if len(rt):
                series[norm].append(rt)
    return series


def mean_forward_filled(runs: List[np.ndarray]) -> np.ndarray:
    """Mean running-total at each step over all runs, holding a run's final value
    after it ends. Once a participant's interactive session finishes they accrue
    no more reward, so their total is flat at its last value — forward-filling
    keeps n constant and avoids the upward jump that dropping finished (often very
    negative) participants would create."""
    if not runs:
        return np.array([])
    max_len = max(len(r) for r in runs)
    filled = np.empty((len(runs), max_len))
    for i, r in enumerate(runs):
        filled[i, :len(r)] = r
        filled[i, len(r):] = r[-1]  # hold final value
    return filled.mean(axis=0)


def main():
    series = norm_series()

    plt.rcParams.update({
        'font.size': 11, 'axes.edgecolor': '#c9c8c3', 'axes.linewidth': 0.8,
        'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb',
    })
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    for norm in NORMS:
        runs = series.get(norm, [])
        if not runs:
            continue
        mean = mean_forward_filled(runs)
        x = np.arange(1, len(mean) + 1)
        color = NORM_COLORS[norm]
        ax.plot(x, mean, color=color, linewidth=2.0, solid_capstyle='round',
                label=f"{NORM_LABELS[norm]} (n={len(runs)})")
        # Direct end-label (relief for the low-contrast series; good practice <=4 lines).
        ax.annotate(NORM_LABELS[norm], xy=(x[-1], mean[-1]),
                    xytext=(6, 0), textcoords='offset points',
                    color=color, fontsize=9.5, fontweight='bold',
                    va='center', ha='left')

    ax.axhline(0, color='#c9c8c3', linewidth=0.8, zorder=0)
    ax.set_xlabel('Interactive step')
    ax.set_ylabel('Running-total reward (mean over participants)')
    ax.set_title('Interactive-stage reward trajectory by norm', fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color='#ecebe6', linewidth=0.8, zorder=0)
    ax.legend(frameon=False, loc='lower left', fontsize=9.5)
    ax.margins(x=0.14)  # room for end-labels
    fig.tight_layout()

    out = os.path.join(RESULTS_DIR, 'interactive_reward_by_norm.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved plot to: {out}")
    for norm in NORMS:
        runs = series.get(norm, [])
        lens = [len(r) for r in runs]
        print(f"  {norm}: n={len(runs)} participants, step counts={sorted(lens)}")


if __name__ == '__main__':
    main()
