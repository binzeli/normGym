"""Baseline vs distractor, per norm, from the files the pipeline already writes.

For every norm under the distractor results tree, reads distractor.json (what
was applied), metrics_summary_<norm>.json (eval.py), simulation_log/interactive_stage.json
and raw_runs/ under both trees and reports:
  - violation-free success in the same-world and new-map test stages,
  - the norm-belief judge score of the final interactive belief (the only thing
    the subject carries into the tests),
  - whether that final belief states the distractor rule (an exit-door rule for
    that side, or a move/wait rhythm), the first interactive step whose belief
    did, and the share of interactive steps whose belief did,
  - which door the subject actually left through, per finished life, as
    left/bottom/right counts (interactive | tests).
The rule match is a keyword screen, run on the baseline too as the control (a
baseline belief can already talk about exiting or pacing); read what it flags.
Under a free-exit norm the baseline subject was told which door to use, so its
door counts are the random assignment, not a choice.

    python experiment/distractor/analyze.py [--model openai_gpt-5.6-sol] [--show_beliefs] [--csv out.csv]
"""
import argparse
import csv
import glob
import json
import os
import re

EXPERIMENT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Every world is an 11 x 15 grid with one door on each of these three walls
# (src/world/*.py ENTRANCES); the main world's doors are the cafe's.
GRID_HEIGHT, GRID_WIDTH = 11, 15
CAFE_DOORS = {'left': (5, 0), 'bottom': (10, 7), 'right': (5, 14)}
SIDE_WORDS = {'left': r'left|west|western', 'right': r'right|east|eastern',
              'bottom': r'bottom|south|southern|lower'}

_EXIT = r'(?:exit|exits|exiting|leave|leaves|leaving|depart|departs|way out|go out|goes out)'


def door_rule_pattern(side):
    """"exit/leave ... <side> door" (or the cafe door's cell), either order, one clause."""
    r, c = CAFE_DOORS[side]
    door = (rf'(?:(?:{SIDE_WORDS[side]})(?:-side|-hand| side| hand)?\s+(?:door|exit|entrance|doorway)'
            rf'|\(\s*{r}\s*,\s*{c}\s*\))')
    return re.compile(rf'{_EXIT}[^.;]*?{door}|{door}[^.;]*?{_EXIT}', re.IGNORECASE)


# "move one step then wait", "alternate moving and waiting", "never two moves in a row", ...
RHYTHM_RULE = re.compile(
    r'alternat\w*[^.;]{0,40}(?:mov|step|wait|rest|paus)'
    r'|every (?:other|second) (?:step|turn|move)'
    r'|(?:mov\w*|step\w*)[^.;]{0,40}(?:then|and then|followed by)[^.;]{0,15}(?:wait|rest|paus)'
    r'|(?:wait|rest|paus)\w*[^.;]{0,25}(?:after|between) (?:each|every|one|a single) (?:move|step)'
    r'|(?:never|not|no)[^.;]{0,25}(?:two|2|twice|consecutive|back-to-back)[^.;]{0,20}(?:mov|step)'
    r'|(?:two|2|consecutive)[^.;]{0,15}moves? in a row',
    re.IGNORECASE)


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def side_of(position):
    r, c = position
    if c <= 0:
        return 'left'
    if c >= GRID_WIDTH - 1:
        return 'right'
    if r >= GRID_HEIGHT - 1:
        return 'bottom'
    return None


def exit_counts(norm_dir):
    """'L/B/R' door counts of the subject's finished lives: (interactive, tests)."""
    counts = {'interactive': dict(left=0, bottom=0, right=0),
              'tests': dict(left=0, bottom=0, right=0)}
    for path in glob.glob(os.path.join(norm_dir, 'raw_runs', '*.json')):
        run = load(path)
        stage = 'interactive' if run.get('stage') == 'interactive' else 'tests'
        last = {}
        for entry in run.get('trace', []):
            last[entry['agent_id']] = entry
        for entry in last.values():
            side = side_of(entry['position']) if entry.get('done') else None
            if side:
                counts[stage][side] += 1
    return tuple('/'.join(str(counts[s][k]) for k in ('left', 'bottom', 'right'))
                 for s in ('interactive', 'tests'))


def interactive_violations(norm_dir):
    """The subject's norm violations over the interactive stage, counted from
    its raw run log (None when the log is missing)."""
    run = load(os.path.join(norm_dir, 'raw_runs', 'interactive.json'))
    if run is None:
        return None
    return sum(r.get('step_norm_violations', 0) for r in run.get('trace', []))


def norm_row(norm_dir, norm, pattern):
    metrics = load(os.path.join(norm_dir, f'metrics_summary_{norm}.json'))
    memory = load(os.path.join(norm_dir, 'simulation_log', 'interactive_stage.json'))
    if metrics is None or memory is None:
        return None
    beliefs = [(s.get('step'), s.get('belief') or '') for s in memory.get('steps', [])]
    flagged = [step for step, belief in beliefs if pattern.search(belief)]
    final_belief = memory.get('belief') or ''
    judge = metrics.get('norm_belief_judge') or {}
    interactive_exits, test_exits = exit_counts(norm_dir)
    return {
        'test_vfs': metrics['test']['summary']['ViolationFreeSuccessRate'],
        'new_map_vfs': metrics['new_map_test']['summary']['ViolationFreeSuccessRate'],
        'test_violations': metrics['test']['summary']['ViolationsPerEpisode'],
        'new_map_violations': metrics['new_map_test']['summary']['ViolationsPerEpisode'],
        'interactive_violations': interactive_violations(norm_dir),
        'judge_score': judge.get('score'),
        'final_belief_distractor_rule': bool(pattern.search(final_belief)),
        'first_distractor_rule_step': flagged[0] if flagged else None,
        'distractor_rule_step_share': round(len(flagged) / len(beliefs), 3) if beliefs else None,
        'exits_LBR_interactive': interactive_exits,
        'exits_LBR_tests': test_exits,
        'final_belief': final_belief,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_type', default='memory_llm')
    parser.add_argument('--model', default='openai_gpt-5.6-sol')
    parser.add_argument('--world_type', default='cafe')
    parser.add_argument('--baseline_root', default='results')
    parser.add_argument('--distractor_root', default='results_distractor')
    parser.add_argument('--csv', default=None)
    parser.add_argument('--show_beliefs', action='store_true',
                        help='print both final beliefs for every norm')
    args = parser.parse_args()

    def norm_dir(root, norm):
        return os.path.join(EXPERIMENT_ROOT, root, args.agent_type, args.model,
                            args.world_type, norm)

    distractor_dir = norm_dir(args.distractor_root, '')
    norms = sorted(os.listdir(distractor_dir)) if os.path.isdir(distractor_dir) else []

    rows = []
    for norm in norms:
        setup = load(os.path.join(norm_dir(args.distractor_root, norm), 'distractor.json'))
        if setup is None:
            print(f"[skip] {norm}: no distractor.json")
            continue
        pattern = (door_rule_pattern(setup['exit_side']) if setup.get('exit_side')
                   else RHYTHM_RULE)
        what = setup['distractor'] + ('+free_exit' if setup.get('subject_free_exit') else '')
        pair = {'baseline': norm_row(norm_dir(args.baseline_root, norm), norm, pattern),
                'distractor': norm_row(norm_dir(args.distractor_root, norm), norm, pattern)}
        if pair['distractor'] is None:
            print(f"[skip] {norm}: distractor run not finished")
            continue
        for cond, row in pair.items():
            if row is not None:
                rows.append({'norm': norm, 'distractor': what, 'condition': cond, **row})

    shown = [k for k in (rows[0] if rows else {}) if k != 'final_belief']
    print('\t'.join(shown))
    for row in rows:
        print('\t'.join(str(row[k]) for k in shown))
        if args.show_beliefs:
            print(f"    belief: {row['final_belief']}")

    if args.csv and rows:
        with open(args.csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[OK] wrote {args.csv}")


if __name__ == "__main__":
    main()
