"""SPIRAL v2 — belief objective: verifiable norm identification, no judge.

After each episode the subject answers ONE multiple-choice question: which of
the candidate norm descriptions (shuffled, unnamed; the training norms of the
group) was the crowd following? Reward 1 if the letter is right, 0 if wrong,
``unparsed_reward`` if the answer doesn't parse — an exact, judge-free 1/0
(see spiral/v2_belief/identify.py for why this replaces the vigal v2 judge).

Credit is trajectory-level: the identification turn AND the episode's
behavior turns all carry the episode advantage — behaving so as to see the
queue clearly is part of answering correctly. Episodes whose unparseable-turn
fraction exceeds ``max_unparsed_frac`` skip the question and score
``unparsed_reward`` (the format gate).

The action objective is deliberately absent (the vigal v2 decision:
belief-only) — the game outcome is logged but unrewarded.

adapter_best = best held-out ``eval_id_acc``. By default every norm plays at
its benchmark capacity from experiment_config.json (no ladder): more crowd =
more demonstrators = MORE identification evidence, so a capacity curriculum
does not obviously help here; pass --capacity_ladder to experiment.

Launchers retired 2026-09-11 (not maintained).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

from experiment.train.config import CKPT_DIR, NORM_GROUP, TRAIN_NORMS
from experiment.train.spiral import episode_trainer
from experiment.train.spiral.common import generate_many
from experiment.train.spiral.v2_belief.identify import (
    build_id_turn,
    load_descriptions,
    parse_choice,
)

V2_OUTPUT_DIR = os.path.join(CKPT_DIR, 'spiral_v2_belief', NORM_GROUP, 'run1')

V2_DEFAULTS = dict(
    episode_trainer.ENGINE_DEFAULTS,
    correct_reward=1.0,
    wrong_reward=0.0,
    unparsed_reward=-0.1,     # unparseable identification answer / format gate
    id_history_stride=3,      # position-log lines: every 3rd step
    id_desc_chars=0,          # 0 = full candidate descriptions
)


class BeliefObjective:
    metric_key = 'eval_id_acc'

    def __init__(self):
        self.descriptions = load_descriptions()

    def score(self, episodes, backend, args, rng, temperature):
        candidates = args.norms or list(TRAIN_NORMS)
        prepared = []
        for ep in episodes:
            unparsed = 1.0 - ep.turns_parsed / max(1, ep.turns_total)
            if unparsed > args.max_unparsed_frac:
                prepared.append(('gated', None, None))
                continue
            built = build_id_turn(ep, candidates, self.descriptions, rng,
                                  stride=args.id_history_stride,
                                  desc_chars=args.id_desc_chars)
            if built is None:
                prepared.append(('skip', None, None))
                continue
            prepared.append(('ask', *built))
        responses = generate_many(
            backend,
            [turn.messages() if kind == 'ask' else None
             for kind, turn, _ in prepared],
            temperature, args.max_new_tokens, args.num_workers)
        results = []
        for (kind, turn, answer), resp in zip(prepared, responses):
            if kind == 'skip':
                results.append(None)
                continue
            if kind == 'gated' or resp is None:
                results.append({'reward': args.unparsed_reward, 'success': 0.0,
                                'extra_turn': None,
                                'log': {'id_choice': None, 'id_correct': False,
                                        'format_gated': kind == 'gated'}})
                continue
            turn.response = resp
            choice = parse_choice(resp)
            if choice is None:
                reward, correct = args.unparsed_reward, False
            elif choice == answer:
                reward, correct = args.correct_reward, True
            else:
                reward, correct = args.wrong_reward, False
            results.append({'reward': reward, 'success': 1.0 if correct else 0.0,
                            'extra_turn': turn,
                            'log': {'id_choice': choice, 'id_answer': answer,
                                    'id_correct': correct,
                                    'format_gated': False}})
        return results

    def summarize(self, results):
        n = max(len(results), 1)
        asked = [r for r in results if not r['log']['format_gated']]
        return {
            'id_acc': sum(r['log']['id_correct'] for r in results) / n,
            'id_parse': (sum(r['log'].get('id_choice') is not None for r in asked)
                         / max(1, len(asked))),
            'format_gated_frac': sum(r['log']['format_gated'] for r in results) / n,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    episode_trainer.add_engine_args(parser, V2_DEFAULTS)
    parser.set_defaults(
        out=V2_OUTPUT_DIR,
        # The identification prompt (position log + 11 full descriptions +
        # final frame) runs past the behavior turns' 8192-token gate.
        max_turn_tokens=12288,
    )
    args = parser.parse_args()
    episode_trainer.run(args, BeliefObjective())


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        os._exit(1)
