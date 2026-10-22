"""SPIRAL v1 — action objective: the game outcome IS the reward.

One episode -> one sparse terminal reward, SPIRAL's rho in {-1, 0, +1}:

    +1  violation-free full success (both goals, zero violations)
     0  full success with violations, OR >= 1 goal with zero violations
    -1  everything else

plus the format gate: an episode whose unparseable-turn fraction exceeds
``max_unparsed_frac`` scores -1 outright (SPIRAL: an invalid move loses the
game). No step rewards, no shaping, no budgets — the per-norm EMA baseline
(RAE) and the capacity curriculum carry the signal instead (see
spiral/common.py for why that combination removes the vigal lineage's
absorbing collapse state).

Every trained turn of the episode inherits the episode advantage
(trajectory-level credit): unlike the vigal single-snapshot trainers, the
policy is optimized on whole episodes where the queue evidence accrues in
context, matching how it is evaluated.

adapter_best = best held-out ``eval_vf_success`` (fixed scenarios at the
benchmark capacity 8).

Launchers retired 2026-09-11 (not maintained).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

from experiment.train.config import CKPT_DIR, NORM_GROUP
from experiment.train.spiral import episode_trainer

V1_OUTPUT_DIR = os.path.join(CKPT_DIR, 'spiral_v1_action', NORM_GROUP, 'run1')

V1_DEFAULTS = dict(
    episode_trainer.ENGINE_DEFAULTS,
    win_reward=1.0,           # violation-free full success
    mid_reward=0.0,           # success with violations / partial clean progress
    loss_reward=-1.0,         # everything else, and the format gate
)


class ActionObjective:
    metric_key = 'eval_vf_success'

    def score(self, episodes, backend, args, rng, temperature):
        results = []
        for ep in episodes:
            unparsed = 1.0 - ep.turns_parsed / max(1, ep.turns_total)
            gated = unparsed > args.max_unparsed_frac
            if gated:
                reward = args.loss_reward
            elif ep.task_success and ep.num_violations == 0:
                reward = args.win_reward
            elif ep.task_success or (ep.goals_reached >= 1
                                     and ep.num_violations == 0):
                reward = args.mid_reward
            else:
                reward = args.loss_reward
            results.append({
                'reward': reward,
                # Curriculum signal = the clean win rate (self-play's analogue
                # of the 50% win rate): a norm the policy beats cleanly more
                # than ctrl_hi of the time gets a bigger crowd.
                'success': 1.0 if reward >= args.win_reward else 0.0,
                'extra_turn': None,
                'log': {'format_gated': gated},
            })
        return results

    def summarize(self, results):
        n = max(len(results), 1)
        return {
            'win_frac': sum(r['reward'] >= 1.0 for r in results) / n,
            'mid_frac': sum(0.0 <= r['reward'] < 1.0 for r in results) / n,
            'format_gated_frac': sum(r['log']['format_gated'] for r in results) / n,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    episode_trainer.add_engine_args(parser, V1_DEFAULTS)
    parser.set_defaults(out=V1_OUTPUT_DIR)
    args = parser.parse_args()
    episode_trainer.run(args, ActionObjective())


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        os._exit(1)
