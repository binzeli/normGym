"""Prepare and run DQN training, validation, and final evaluation.

No command submits jobs. Training/validation and final tests are separate;
final checkpoints are never selected using test outcomes.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
from pathlib import Path

import numpy as np

from experiment.run.common import WORLD_TYPE_CHOICES, _random_subject_positions
from experiment.run.rl.dqn import load_dqn_model, train_or_load_dqn
from experiment.run.rl.provenance import ROOT, check_storage, digest, source_identity, versions, write_json
from experiment.run.rl.td_loss import load_probe, prepare_probes, score_td_loss
from experiment.run.rl.validation import VALIDATION_LAYOUTS, SeededTrainingEnv, evaluate_episode, layout_spec

BASELINE_DQN = dict(
    learning_rate=1e-4, gamma=0.99, buffer_size=100000, learning_starts=1000,
    batch_size=32, train_freq=4, gradient_steps=1, target_update_interval=10000,
    tau=1.0, exploration_initial_eps=1.0, exploration_final_eps=0.05,
    exploration_fraction=0.30, max_grad_norm=10.0, net_arch=[64, 64],
    features_dim=256, mask_td_target=True,
)
CONDITIONS = {
    'sparse_penalty': ('sparse', True),
    'sparse_no_penalty': ('sparse', False),
    'dense_penalty': ('dense', True),
    'dense_no_penalty': ('dense', False),
}


def build_manifest(tag, *, timesteps=1000000, eval_episodes=12, norms=None,
                   conditions=None, policies=('mlp', 'cnn')):
    """Freeze source hashes, layouts, disjoint seeds, and training settings."""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', tag):
        raise ValueError('Use a nonempty alphanumeric tag with hyphens/underscores')
    if not 1 <= timesteps <= 1000000 or not 1 <= eval_episodes <= 1000:
        raise ValueError('Budget must be 1..1M transitions and validation count 1..1000')
    config = json.loads((ROOT / 'experiment/run/experiment_config.json').read_text())
    all_norms = [n['name'] for n in json.loads((ROOT / 'src/norms/norm_descriptions.json').read_text())['norms']]
    selected = list(norms) if norms is not None else all_norms
    conditions = ['sparse_penalty', 'sparse_no_penalty', 'dense_penalty'] if conditions is None else list(conditions)
    if (not selected or len(selected) != len(set(selected)) or set(selected) - set(all_norms)
            or not conditions or len(conditions) != len(set(conditions)) or set(conditions) - set(CONDITIONS)
            or not policies or len(policies) != len(set(policies)) or set(policies) - {'mlp', 'cnn'}):
        raise ValueError('Unknown or duplicate norms, conditions, or policies')
    layouts = {name: layout_spec(name) for name in WORLD_TYPE_CHOICES}
    validation = {name: layout_spec(**spec) for name, spec in VALIDATION_LAYOUTS.items()}
    splits, locked = {}, {}
    for index, norm in enumerate(all_norms):
        if norm not in selected:
            continue
        base = config['norm_seeds'][norm]
        locked[norm] = {
            'same_layout': [{'seed': base+i, 'world_type': 'cafe', 'layout_sha256': layouts['cafe']['layout_sha256']}
                            for i in range(1, 4)],
            'held_out_layout': [{'seed': base+3+i, 'world_type': world, 'layout_sha256': layouts[world]['layout_sha256']}
                                for i, world in enumerate(config['norm_new_map_worlds'][norm], 1)],
        }
        start = 1000000 + index*100000
        splits[norm] = {
            'training_seed_starts': [start], 'training_episode_seed_count': 10000,
            'same_layout_validation': list(range(start+50000, start+50000+eval_episodes)),
            'train_protocol_validation': list(range(start+55000, start+55000+min(3, eval_episodes))),
            'held_out_layout_validation': {
                name: list(range(start+60000+j*1000, start+60000+j*1000+eval_episodes))
                for j, name in enumerate(validation)},
        }
    tasks = []
    for condition in conditions:
        reward, penalties = CONDITIONS[condition]
        for policy in policies:
            for norm in selected:
                capacity = config['norm_capacity'].get(norm, config['capacity'])
                seed = splits[norm]['training_seed_starts'][0]
                # Private RNG state: preparing a manifest must not perturb callers.
                state = random.getstate()
                try:
                    random.seed(seed)
                    position = min(_random_subject_positions(capacity, norm))
                finally:
                    random.setstate(state)
                tasks.append(dict(
                    index=len(tasks), norm=norm, policy=policy, reward=reward,
                    condition='penalty' if penalties else 'no_penalty', candidate=condition,
                    training_seed=seed, replicate=0, capacity=capacity, subject_position=position,
                    dqn=copy.deepcopy(BASELINE_DQN), total_timesteps=timesteps,
                    training_horizon=200, evaluation_horizon=100, training_no_norm_agents=1,
                    evaluation_no_norm_agents=0, goal_progress_magnitude=6.0 if reward=='dense' else 0.0,
                    checkpoints=[x for x in (100000, 250000, 500000, 750000) if x < timesteps],
                    norm_penalty=penalties, collision_penalty=penalties,
                    task_goal_mode='category', tensorboard_log_every=100,
                ))
    manifest = dict(
        schema_version=1, tag=tag, parent_tag=None, stage='paper',
        source=source_identity(), preparation_versions=versions(),
        experiment_config_sha256=digest(config), norms=selected, tasks=tasks,
        training_layout=layouts['cafe'], layouts=layouts, validation_layouts=validation,
        locked_test=locked, splits=splits,
        checkpoint_rule='Fixed final-budget checkpoint; no checkpoint or hyperparameter selection from final tests.',
        evaluation_protocol='shared_test_v1 placement and traffic; seeded subject class, no forced class rotation.',
        td_loss_diagnostics={'episodes_per_split': min(3, eval_episodes), 'selection_use': 'diagnostic only'},
    )
    manifest['manifest_sha256'] = digest(manifest)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest):
    if manifest['manifest_sha256'] != digest({k:v for k,v in manifest.items() if k!='manifest_sha256'}):
        raise ValueError('Manifest digest mismatch')
    final_seeds = [x['seed'] for split in manifest['locked_test'].values() for rows in split.values() for x in rows]
    allocated = set(final_seeds)
    if len(allocated) != len(final_seeds):
        raise ValueError('Final-test seed collision')
    val_hashes = {v['layout_sha256'] for v in manifest['validation_layouts'].values()}
    final_hashes = {v['layout_sha256'] for v in manifest['layouts'].values()}
    if val_hashes & final_hashes:
        raise ValueError('Validation geometry overlaps benchmark layouts')
    for split in manifest['splits'].values():
        groups = [split['same_layout_validation'], split['train_protocol_validation'],
                  *split['held_out_layout_validation'].values()]
        groups += [range(s, s+split['training_episode_seed_count']) for s in split['training_seed_starts']]
        for group in groups:
            values = set(group)
            if len(values) != len(group) or values & allocated:
                raise ValueError('Training/validation/final seed overlap')
            allocated.update(values)
    for index, task in enumerate(manifest['tasks']):
        if task['index'] != index or task['candidate'] not in CONDITIONS:
            raise ValueError('Invalid task index or reward condition')


def check_source(manifest):
    validate_manifest(manifest)
    if source_identity()['content_sha256'] != manifest['source']['content_sha256']:
        raise ValueError('Source differs from manifest; prepare a new manifest after edits')


def run_directory(manifest, task, output_root):
    return Path(output_root)/manifest['tag']/f'{task["index"]:03d}_{task["norm"]}_{task["policy"]}_{task["reward"]}_{task["candidate"]}'


def run_task(manifest, index, output_root, device="cpu"):
    validate_manifest(manifest)
    if not 0 <= index < len(manifest["tasks"]):
        raise ValueError("Task index outside manifest")
    check_source(manifest)
    task = manifest["tasks"][index]
    root = Path(output_root) / manifest["tag"]
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    try:
        write_json(path, manifest, exclusive=True)
    except FileExistsError:
        # Full comparison avoids a same-tag/different-config collision.
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Tag already has a different manifest")
    run = run_directory(manifest, task, output_root)
    run.mkdir(exist_ok=False)  # never resume/overwrite a partially completed run implicitly
    check_storage(run)
    write_json(run / "task.json", {"task": task, "manifest_sha256": manifest["manifest_sha256"],
                                    "split": manifest["splits"][task["norm"]]})
    split = manifest["splits"][task["norm"]]
    env = SeededTrainingEnv(
        episode_seed_start=task["training_seed"], episode_seed_count=split["training_episode_seed_count"],
        norm_type=task["norm"], capacity=task["capacity"], max_steps=task["training_horizon"],
        subject_position=task["subject_position"], num_no_norm_agents=task["training_no_norm_agents"],
        goal_progress_reward=task["reward"] == "dense", goal_progress_magnitude=task["goal_progress_magnitude"],
        norm_penalty=task.get("norm_penalty"), collision_penalty=task.get("collision_penalty"),
        task_goal_mode=task.get("task_goal_mode", "benchmark"),
    )
    try:
        model = train_or_load_dqn(
            env=env, model_path=run / "dqn_final.zip", total_timesteps=task["total_timesteps"],
            seed=task["training_seed"], checkpoint_dir=run / "checkpoints", checkpoint_freq=50000,
            checkpoint_steps=task["checkpoints"], policy=task["policy"], device=device,
            save_replay_buffer=False, run_context={"tag": manifest["tag"], "parent_tag": manifest["parent_tag"],
                                                  "manifest_sha256": manifest["manifest_sha256"], "task": task},
            tensorboard_log_every=task.get("tensorboard_log_every", 1),
            **task["dqn"],
        )
        del model
    finally:
        env.close()
    checkpoints = [(n, run / "checkpoints" / f"dqn_{n}_steps.zip") for n in task["checkpoints"]]
    checkpoints.append((task["total_timesteps"], run / "dqn_final.zip"))
    probe_paths = prepare_probes(manifest, task, run)
    summaries = []
    for step, model_path in checkpoints:
        check_storage(run)
        model = load_dqn_model(model_path, device=device)
        td_losses = {}
        for label, probe_path in probe_paths.items():
            arrays, metadata = load_probe(probe_path)
            td_losses[label] = {**score_td_loss(model, arrays),
                                "dataset_sha256": metadata["dataset_sha256"]}
        checkpoint_metadata = (run / "training_complete.json" if model_path.name == "dqn_final.zip"
                               else model_path.with_suffix(".json"))
        td_losses["optimizer_training_loss_last_logged"] = json.loads(checkpoint_metadata.read_text()).get("training_td_loss_last_logged")
        write_json(run / "evaluation" / str(step) / "td_loss.json", td_losses, exclusive=True)
        batches = [("train_diagnostic", manifest["training_layout"],
                    list(range(task["training_seed"], task["training_seed"] + min(3, max(1, task["total_timesteps"] // 200)))), True),
                   ("train_protocol_validation", manifest["training_layout"], split["train_protocol_validation"], True),
                   ("same_layout_validation", manifest["training_layout"], split["same_layout_validation"], False)]
        batches += [(key, manifest["validation_layouts"][key], seeds, False)
                    for key, seeds in split["held_out_layout_validation"].items()]
        means = {}
        for label, layout, seeds, training in batches:
            metrics = []
            for seed in seeds:
                result = evaluate_episode(model, norm=task["norm"], capacity=task["capacity"],
                                          layout=layout, seed=seed,
                                          horizon=task["training_horizon"] if training else task["evaluation_horizon"],
                                          training_distribution=training, subject_position=task["subject_position"],
                                          task_goal_mode=task.get("task_goal_mode", "benchmark"))
                result.update(checkpoint_timestep=step, model_path=str(model_path), split=label,
                              manifest_sha256=manifest["manifest_sha256"], training_seed=task["training_seed"],
                              training_exploration_rate=float(model.exploration_rate))
                write_json(run / "evaluation" / str(step) / label / f"seed_{seed}.json", result, exclusive=True)
                metrics.append(result["metrics"])
            means[label] = {"n": len(metrics), **{key: float(np.mean([m[key] for m in metrics]))
                                                for key in ("violation_free_success", "success", "comply", "full_task_completion")}}
        validation_joint = 0.5 * means["same_layout_validation"]["violation_free_success"] + 0.5 * float(np.mean([
            means[k]["violation_free_success"] for k in manifest["validation_layouts"]]))
        summaries.append({"timestep": step, "splits": means, "validation_joint": validation_joint,
                          "td_loss": td_losses})
    write_json(run / "validation_summary.json", {"task": task, "checkpoints": summaries,
                                                  "manifest_sha256": manifest["manifest_sha256"]}, exclusive=True)
    return run


def run_final_tests(manifest, output_root, *, norm):
    """Evaluate each final checkpoint once on the six predefined test episodes."""
    from experiment.run.rl.validation import LifecycleRecorder

    check_source(manifest)
    tasks = [t for t in manifest['tasks'] if t['norm'] == norm]
    if not tasks:
        raise ValueError('Norm is not present in this manifest')
    # Resolve every required checkpoint before creating any final-test outputs.
    models = [(t, run_directory(manifest, t, output_root)/'dqn_final.zip') for t in tasks]
    for task, path in models:
        complete = path.parent/'training_complete.json'
        if not path.is_file() or not complete.is_file():
            raise FileNotFoundError(f'Incomplete training run: {path.parent}')
        if json.loads(complete.read_text())['environment_transitions'] != task['total_timesteps']:
            raise ValueError(f'Final training budget differs: {path}')
    destination = Path(output_root)/manifest['tag']/'final_test'/norm
    destination.mkdir(parents=True, exist_ok=False)
    check_storage(destination)
    specs = manifest['locked_test'][norm]
    task = tasks[0]

    def emit(model, name):
        for split, tests in specs.items():
            check_storage(destination)
            stage = 'test' if split == 'same_layout' else 'new_map_test'
            for ordinal, spec in enumerate(tests, 1):
                layout = manifest['layouts'][spec['world_type']]
                result = evaluate_episode(
                    model, norm=norm, capacity=task['capacity'], layout=layout, seed=spec['seed'],
                    horizon=task['evaluation_horizon'], subject_position=task['subject_position'],
                    oracle=model is None, task_goal_mode='category', recorder_class=LifecycleRecorder,
                )
                run_id = f'test_{ordinal:03d}' if stage == 'test' else f'new_map_test_{ordinal:03d}_{spec["world_type"]}'
                result['raw_run'].update(run_id=run_id, stage=stage)
                result['metrics'].update(run_id=run_id, stage=stage)
                result.update(norm=norm, split=stage, seed=spec['seed'], variant=name,
                              manifest_sha256=manifest['manifest_sha256'])
                write_json(destination/name/f'{run_id}.json', result, exclusive=True)

    emit(None, 'oracle_with_norm')
    for task, path in models:
        model = load_dqn_model(path)
        before = (model.num_timesteps, model._n_updates)
        emit(model, f'{task["policy"]}_{task["candidate"]}')
        if before != (model.num_timesteps, model._n_updates):
            raise RuntimeError('Evaluation modified training counters')
        del model
    write_json(destination/'complete.json', {'models': len(models), 'episodes': 6*(len(models)+1),
               'manifest_sha256': manifest['manifest_sha256']}, exclusive=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='Write a frozen run manifest; does not train')
    prepare.add_argument('--tag', required=True)
    prepare.add_argument('--manifest', type=Path, required=True)
    prepare.add_argument('--timesteps', type=int, default=1000000)
    prepare.add_argument('--eval-episodes', type=int, default=12)
    prepare.add_argument('--norms', nargs='+')
    prepare.add_argument('--conditions', nargs='+', choices=list(CONDITIONS))
    prepare.add_argument('--policies', nargs='+', choices=['mlp', 'cnn'], default=['mlp', 'cnn'])
    inspect = commands.add_parser('inspect')
    inspect.add_argument('--manifest', type=Path, required=True)
    train = commands.add_parser('train', help='Train one manifest row and record validation')
    train.add_argument('--manifest', type=Path, required=True)
    train.add_argument('--index', type=int, required=True)
    train.add_argument('--output-root', type=Path, required=True)
    train.add_argument('--device', default='cpu')
    final = commands.add_parser('final-test', help='Evaluate final checkpoints for one norm')
    final.add_argument('--manifest', type=Path, required=True)
    final.add_argument('--output-root', type=Path, required=True)
    final.add_argument('--norm', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        manifest = build_manifest(args.tag, timesteps=args.timesteps, eval_episodes=args.eval_episodes,
                                  norms=args.norms, conditions=args.conditions, policies=args.policies)
        write_json(args.manifest, manifest, exclusive=True)
        print(json.dumps({'manifest': str(args.manifest), 'tasks': len(manifest['tasks']),
                          'total_transitions': sum(t['total_timesteps'] for t in manifest['tasks'])}))
        return
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    if args.command == 'inspect':
        print(json.dumps({'tag': manifest['tag'], 'tasks': len(manifest['tasks']),
                          'checkpoint_rule': manifest['checkpoint_rule'],
                          'manifest_sha256': manifest['manifest_sha256']}, indent=2))
    elif args.command == 'train':
        print(run_task(manifest, args.index, args.output_root, device=args.device))
    elif args.command == 'final-test':
        print(run_final_tests(manifest, args.output_root, norm=args.norm))


if __name__ == '__main__':
    main()
