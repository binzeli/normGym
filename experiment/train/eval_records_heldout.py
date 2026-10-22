"""Records-level test of a trained adapter on HELD-OUT norms — the SAME task
each version trains on, only the norms change (2026-09-16).

The game-play test (eval_checkpoint.py / run_eval_trained.sh) plays the
model as the benchmark's memory_llm actor. This script instead keeps each
version's own training prompt, output schema and verifier and runs it
greedily on records harvested from the group's TEST_NORMS:

    v1  action snapshots (full-memory actor prompt, all steps)
        -> move accuracy under snapshot_verifier.grade_response
    v2  the same snapshots at step >= 50, identification tail with a lettered
        candidate list of the group's TEST norms only (user decision
        2026-09-16; chance = 1/len(TEST_NORMS))
        -> identification accuracy under identification.grade_choice
    v3  violation-judge records at step >= 50, class-balanced
        -> balanced accuracy under judge_task.grade

Nothing inside the version folders is modified: their loaders, prompt
builders and graders are imported and used as-is. The model is a vLLM
server on the base model; the adapter is hot-swapped in under --lora_name
with rollout_pool.sync_lora (``--adapter base`` = no adapter).

Usage (NORM_GROUP exported; a vLLM server up):
    python experiment/train/eval_records_heldout.py --version v1 \
        --adapter experiment/train/checkpoints/v1_action_rloo/queuing/run2/adapter_latest \
        --snapshots experiment/train/data/queuing/snapshots_test_fullmem \
        --rollout_url http://IP:PORT/v1 --out results.json
"""
import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from experiment.train.config import NORM_CATEGORY, NORM_GROUP, TEST_NORMS
from experiment.train.policy_client import OpenAIBackend
from experiment.train.rollout_pool import sync_lora, wait_for_server


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version', required=True, choices=['v1', 'v2', 'v3'])
    p.add_argument('--adapter', required=True, help='adapter dir, or "base" for the untrained base model')
    p.add_argument('--rollout_url', required=True)
    p.add_argument('--lora_name', default='policy')
    p.add_argument('--base_model_name', default='base', help='served name of the base model')
    p.add_argument('--snapshots', default=None, help='v1/v2: action snapshot set of the test norms')
    p.add_argument('--records', default=None, help='v3: judge record set of the test norms')
    p.add_argument('--norms', nargs='*', default=None, help='default: TEST_NORMS of $NORM_GROUP')
    p.add_argument('--n', type=int, default=800, help='v1/v2: snapshots evaluated (stratified per norm)')
    p.add_argument('--n_judge', type=int, default=400, help='v3: class-balanced records evaluated')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--num_workers', type=int, default=32)
    p.add_argument('--max_new_tokens', type=int, default=0, help='0 = the version default (1024 / 768 / 1024)')
    p.add_argument('--v2_min_step', type=int, default=50)
    p.add_argument('--v3_min_step', type=int, default=50)
    p.add_argument('--category', default=NORM_CATEGORY)
    p.add_argument('--out', required=True, help='summary JSON')
    p.add_argument('--responses_out', default=None, help='per-record JSONL (default: next to --out)')
    p.add_argument('--dry_run', action='store_true', help='load and sample the records, print counts, no server')
    return p.parse_args()


def stratified(records, n, norms, rng):
    """~n/len(norms) records per norm (all of a norm if it has fewer)."""
    per = max(1, n // max(1, len(norms)))
    out = []
    for norm in norms:
        pool = [r for r in records if r['norm'] == norm]
        rng.shuffle(pool)
        out.extend(pool[:per])
    rng.shuffle(out)
    return out


def greedy(backend, jobs, max_new_tokens, num_workers):
    def one(messages):
        try:
            return backend.generate(messages, max_new_tokens=max_new_tokens, temperature=0.0)
        except Exception as exc:
            print(f'[gen] failed: {exc}', flush=True)
            return None
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as ex:
        return list(ex.map(one, jobs))


def per_norm(pairs):
    by = {}
    for norm, ok in pairs:
        by.setdefault(norm, []).append(bool(ok))
    return {k: {'acc': round(sum(v) / len(v), 3), 'n': len(v)} for k, v in sorted(by.items())}


# ---------------------------------------------------------------------------

def eval_v1(args, backend, norms, rng):
    from experiment.train.vigal.v1_action_rloo.step_rloo_train import load_snapshots, make_turn, summarize
    from experiment.train.vigal.v1_action_rloo.snapshot_verifier import REWARD_DEFAULTS, grade_response
    recs = stratified(load_snapshots(args.snapshots, norms, min_step=0), args.n, norms, rng)
    info = {'task': 'action', 'n': len(recs), 'chance': None, 'min_step': 0}
    if args.dry_run:
        return info, [], []
    responses = greedy(backend, [make_turn(r).messages() for r in recs],
                       args.max_new_tokens or 1024, args.num_workers)
    grades = [grade_response(r['label'], resp or '', **REWARD_DEFAULTS) for r, resp in zip(recs, responses)]
    summ = summarize(grades)
    summ['per_norm'] = per_norm([(r['norm'], g['category'] == 'good') for r, g in zip(recs, grades)])
    summ['headline'] = summ['acc']
    rows = [{'id': r['id'], 'norm': r['norm'], 'step': r['step'], 'good': r['label'].get('good'),
             'action': g.get('action'), 'category': g['category'], 'reward': g['reward'], 'response': resp}
            for r, g, resp in zip(recs, grades, responses)]
    return {**info, **summ}, rows, grades


def eval_v2(args, backend, norms, rng):
    from experiment.train.vigal.v2_belief_rloo.step_rloo_belief_train import load_snapshots, make_turn, summarize_id
    from experiment.train.vigal.v2_belief_rloo.identification import grade_choice, prepare_identification
    recs = [r for r in load_snapshots(args.snapshots, norms) if r['step'] >= args.v2_min_step]
    recs = stratified(recs, args.n, norms, rng)
    # Candidate list = the TEST norms only (user decision 2026-09-16).
    prepare_identification(recs, norms, args.seed, category=args.category)
    info = {'task': 'identification', 'n': len(recs), 'chance': round(1 / len(norms), 3),
            'min_step': args.v2_min_step, 'candidates': sorted(norms)}
    if args.dry_run:
        return info, [], []
    responses = greedy(backend, [make_turn(r).messages() for r in recs],
                       args.max_new_tokens or 768, args.num_workers)
    grades = [grade_choice(r, resp) for r, resp in zip(recs, responses)]
    summ = summarize_id(grades)
    summ['per_norm'] = per_norm([(r['norm'], g['correct']) for r, g in zip(recs, grades)])
    summ['headline'] = summ['id_acc']
    rows = [{'id': r['id'], 'norm': r['norm'], 'step': r['step'], 'answer': r['id_answer'],
             'choice': g.get('choice'), 'correct': g['correct'], 'reward': g['reward'], 'response': resp}
            for r, g, resp in zip(recs, grades, responses)]
    return {**info, **summ}, rows, grades


def eval_v3(args, backend, norms, rng):
    from experiment.train.vigal.v3_violation_judge.step_rloo_judge_train import (
        balanced_subset, grade_rec, load_records, messages_for, summarize,
    )
    recs = balanced_subset(load_records(args.records, norms, min_step=args.v3_min_step), args.n_judge, rng)
    info = {'task': 'judge', 'n': len(recs), 'chance': 0.5, 'min_step': args.v3_min_step,
            'positives': sum(r['label'] for r in recs)}
    if args.dry_run:
        return info, [], []
    ns = SimpleNamespace(show_feedback=1, max_new_tokens=args.max_new_tokens or 1024,
                         correct_reward=1.0, format_reward=0.1)
    responses = greedy(backend, [messages_for(r, ns) for r in recs], ns.max_new_tokens, args.num_workers)
    grades = [grade_rec(r, resp or '', ns) for r, resp in zip(recs, responses)]
    summ = summarize(grades, recs)
    summ['per_norm'] = per_norm([(r['norm'], g['correct']) for r, g in zip(recs, grades)])
    summ['headline'] = summ['balanced_acc']
    rows = [{'id': r['id'], 'norm': r['norm'], 'step': r['step'], 'label': r['label'],
             'pred': g.get('pred'), 'correct': g['correct'], 'reward': g['reward'], 'response': resp}
            for r, g, resp in zip(recs, grades, responses)]
    return {**info, **summ}, rows, grades


# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    norms = args.norms or list(TEST_NORMS)
    rng = random.Random(args.seed)
    if args.version in ('v1', 'v2') and not args.snapshots:
        sys.exit('[FATAL] --snapshots is required for v1/v2')
    if args.version == 'v3' and not args.records:
        sys.exit('[FATAL] --records is required for v3')

    backend = None
    if not args.dry_run:
        wait_for_server(args.rollout_url)
        if args.adapter == 'base':
            model_name = args.base_model_name
        else:
            if not os.path.exists(os.path.join(args.adapter, 'adapter_config.json')):
                sys.exit(f'[FATAL] not an adapter dir: {args.adapter}')
            sync_lora(args.rollout_url, os.path.abspath(args.adapter), args.lora_name)
            model_name = args.lora_name
        backend = OpenAIBackend(args.rollout_url, model_name, top_p=1.0)

    t0 = time.time()
    fn = {'v1': eval_v1, 'v2': eval_v2, 'v3': eval_v3}[args.version]
    summary, rows, _ = fn(args, backend, norms, rng)
    summary = {'version': args.version, 'group': NORM_GROUP, 'adapter': args.adapter, 'norms': norms,
               'seed': args.seed, 'seconds': round(time.time() - t0, 1), **summary}
    if args.dry_run:
        print('[dry-run] ' + json.dumps(summary))
        return
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    resp_path = args.responses_out or os.path.splitext(args.out)[0] + '_responses.jsonl'
    with open(resp_path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    print('[records-test] ' + json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                          for k, v in summary.items() if k not in ('per_norm', 'candidates')}))
    print('[records-test] per norm: ' + json.dumps(summary['per_norm']))


if __name__ == '__main__':
    main()
