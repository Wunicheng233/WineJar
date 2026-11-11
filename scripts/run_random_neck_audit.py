#!/usr/bin/env python3
"""Generate random jars and evaluate frozen methods; report every attempted case."""
import argparse
from dataclasses import asdict
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MUJOCO_GL','egl')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from neck_localization.random_audit import random_audit_config
from neck_localization.scene_preparation import prepare_scene_bundle, write_json
from neck_localization.support_feedback import estimate_support_feedback, FeedbackSettings
from neck_localization.joint_profile import JointSettings
from neck_localization.evaluation import score,summarize
from scripts.run_depth_profile_baseline import load_frames
from scripts.run_support_feedback import source_hashes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=2026102327)
    parser.add_argument('--nominal',type=int,default=24)
    parser.add_argument('--stress',type=int,default=12)
    args=parser.parse_args()
    if args.nominal<1 or args.stress<0:raise ValueError('Invalid group sizes')
    frozen=json.loads((ROOT/'experiments/neck_localization/results/support_feedback_20261023/freeze.json').read_text())
    assert frozen['sources']==source_hashes(), 'Frozen methods/generator changed'
    assert frozen['feedback_settings']==asdict(FeedbackSettings())
    assert frozen['joint_settings']==asdict(JointSettings())
    args.output.mkdir(parents=True,exist_ok=False)
    template=json.loads((ROOT/'experiments/neck_localization/configs/support_feedback_test_v1.json').read_text())
    config=random_audit_config(template,args.seed,args.nominal,args.stress)
    write_json(args.output/'config.json',config)
    locked=source_hashes() | {n:sha256((ROOT/n).read_bytes()).hexdigest() for n in
        ['neck_localization/random_audit.py','scripts/run_random_neck_audit.py']}
    write_json(args.output/'plan.json',dict(seed=args.seed, config_sha256=sha256((args.output/'config.json').read_bytes()).hexdigest(),
        locked_sources=locked, protocol='Before scene generation/evaluation; no tuning after results; all sampled jars retained',
        modes=['fixed_equal_trust','region_equal_trust'],camera_sets=['cam_a','cam_a,cam_b','all'],
        assumptions=config['audit_assumptions']))
    print('Locked random geometry and frozen sources; preparing scenes',flush=True)
    prepare_scene_bundle(config,args.output/'scenes')
    result_dir=args.output/'results';result_dir.mkdir()
    write_json(result_dir/'settings.json',dict(**asdict(JointSettings()),feedback=asdict(FeedbackSettings())))
    rows=[]
    with (result_dir/'records.jsonl').open('w') as journal:
        for case in config['cases']:
            folder=args.output/'scenes'/case['name']
            for camera_set in ['cam_a','cam_a,cam_b','all']:
                frames=load_frames(folder/'observations','rgb',None if camera_set=='all' else camera_set.split(','))
                for mode in ['fixed','region']:
                    estimate=estimate_support_feedback(frames,mode=mode,camera_mode='equal_trust')
                    # Evaluation truth first opened AFTER estimation.
                    truth=json.loads((folder/'evaluation_only/target.json').read_text())
                    row=dict(case=case['name'],group=case['group'],split='random_audit_'+case['group'],
                        geometry='photo_short_neck',evidence='rgb',camera_set=camera_set,
                        factor='ideal_true_cameras/'+camera_set,method=mode+'_equal_trust',**score(estimate,truth))
                    row.setdefault('diagnostics',estimate.diagnostics)
                    rows.append(row)
                    journal.write(json.dumps(row)+'\n');journal.flush()
            print(case['name'],[(r['camera_set'],r['method'],r['accepted'],r['normal_deg'],r['center_mm']) for r in rows[-6:]],flush=True)
    current=source_hashes() | {n:sha256((ROOT/n).read_bytes()).hexdigest() for n in
        ['neck_localization/random_audit.py','scripts/run_random_neck_audit.py']}
    assert locked==current,'Sources changed during audit'
    (result_dir/'records.json.gz').write_bytes(gzip.compress(json.dumps(rows).encode(),mtime=0))
    write_json(result_dir/'summary.json',summarize(rows))
    write_json(result_dir/'provenance.json',dict(locked_sources=locked,seed=args.seed,scene_root=str((args.output/'scenes').resolve()),
        config_sha256=sha256((args.output/'config.json').read_bytes()).hexdigest(),scope=config['audit_assumptions']))
    print('Completed',len(rows),'attempts',flush=True)


if __name__=='__main__':main()
