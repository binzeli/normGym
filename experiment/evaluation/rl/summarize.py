"""Aggregate DQN test episodes into Joint/Task tables."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from experiment.evaluation.calculate_metrics import norm_family
from experiment.evaluation.rl.tables import build_tables
from experiment.run.rl.run_rl import validate_manifest
from experiment.run.rl.provenance import write_json

FAMILY_NAMES = {'Region':'Access','Trajectory':'Procedure','Queue':'Coordination','Allocation':'Allocation'}


def summarize(manifest, root):
    validate_manifest(manifest)
    groups = defaultdict(list)
    for norm in manifest['norms']:
        base = Path(root)/manifest['tag']/'final_test'/norm
        complete = json.loads((base/'complete.json').read_text())
        if complete['manifest_sha256'] != manifest['manifest_sha256']:
            raise ValueError(f'Final-test manifest differs: {norm}')
        tasks = [t for t in manifest['tasks'] if t['norm']==norm]
        for task in tasks:
            name = f'{task["policy"]}_{task["candidate"]}'
            for split in ['test','new_map_test']:
                paths = sorted((base/name).glob(split+'_*.json'))
                if len(paths) != 3:
                    raise ValueError(f'Expected three {split} episodes: {norm}/{name}')
                for path in paths:
                    episode = json.loads(path.read_text())
                    if episode['manifest_sha256'] != manifest['manifest_sha256']:
                        raise ValueError(f'Episode manifest differs: {path}')
                    metrics = episode['metrics']
                    score = {'norm':norm, 'task':metrics['success'],
                             'joint':float(metrics['success']==1 and metrics['norm_violation']==0)}
                    family = FAMILY_NAMES[norm_family(norm)]
                    for label in ['All', family]:
                        groups[task['condition'],task['policy']+'_'+task['reward'],split,label].append(score)
    tables = []
    for (condition,variant,split,family), records in groups.items():
        tables.append(dict(condition=condition,variant=variant,split=split,family=family,
                           episodes=len(records),norms=len({r['norm'] for r in records}),
                           joint=statistics.mean(r['joint'] for r in records),
                           task=statistics.mean(r['task'] for r in records)))
    return {'tables':tables, 'manifest_sha256':manifest['manifest_sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args = parser.parse_args()
    result = summarize(json.loads(args.manifest.read_text()),args.output_root)
    args.report.mkdir(parents=True,exist_ok=True)
    write_json(args.report/'metrics.json',result)
    build_tables(result,args.report)
    print(args.report.resolve())


if __name__=='__main__':
    main()
