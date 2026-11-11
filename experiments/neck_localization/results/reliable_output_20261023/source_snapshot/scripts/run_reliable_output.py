#!/usr/bin/env python3
"""Compare frozen baseline and success-rate improvements, offline only."""
import argparse
from dataclasses import asdict
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from neck_localization.reliable_output import estimate_reliable_output,ReliabilitySettings
from neck_localization.support_feedback import estimate_support_feedback
from neck_localization.joint_profile import JointSettings
from neck_localization.evaluation import score,summarize
from neck_localization.scene_preparation import write_json
from scripts.run_depth_profile_baseline import load_frames
from scripts.run_support_feedback import source_hashes as frozen_hashes


def source_hashes():
    return frozen_hashes()|{n:sha256((ROOT/n).read_bytes()).hexdigest() for n in
        ['neck_localization/reliable_output.py','scripts/run_reliable_output.py']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenes',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=['development','test'],default='development')
    p.add_argument('--freeze',type=Path)
    args=p.parse_args();locked=source_hashes()
    old=json.loads((ROOT/'experiments/neck_localization/results/support_feedback_20261023/freeze.json').read_text())
    assert old['sources']==frozen_hashes()
    if args.split=='test':
        assert args.freeze
        frozen=json.loads(args.freeze.read_text())
        assert frozen['sources']==locked
        assert frozen['settings']==json.loads(json.dumps(asdict(ReliabilitySettings())))
        assert frozen['test_config_sha256']==sha256((args.scenes/'effective_config.json').read_bytes()).hexdigest()
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'settings.json',dict(**asdict(JointSettings()),reliability=asdict(ReliabilitySettings())))
    config=json.loads((args.scenes/'effective_config.json').read_text())
    group_by_case={c['name']:c.get('group','development') for c in config['cases']}
    rows=[]
    with (args.output/'records.jsonl').open('w') as journal:
        for cf in sorted(args.scenes.glob('*/observations/camera_true_parameters.json')):
            folder=cf.parent.parent
            for cs in ['cam_a','cam_a,cam_b','all']:
                frames=load_frames(folder/'observations','rgb',None if cs=='all' else cs.split(','))
                for method in ['frozen_region','stable_local','adaptive_validated']:
                    est=(estimate_support_feedback(frames,mode='region') if method=='frozen_region' else
                         estimate_reliable_output(frames,adaptive=method=='adaptive_validated'))
                    truth=json.loads((folder/'evaluation_only/target.json').read_text())
                    row=dict(case=folder.name,group=group_by_case[folder.name],split=args.split+'_'+group_by_case[folder.name],
                        geometry='photo_short_neck',evidence='rgb',camera_set=cs,factor='ideal_true_cameras/'+cs,
                        method=method,**score(est,truth))
                    row.setdefault('diagnostics',est.diagnostics)
                    row['over_illustrative_1000ms']=row['runtime_ms']>1000
                    rows.append(row);journal.write(json.dumps(row)+'\n');journal.flush()
            print(folder.name,[(r['method'],r['accepted'],r['center_mm'],r['normal_deg'],r['reasons']) for r in rows[-3:]],flush=True)
    assert locked==source_hashes(),'Sources changed during batch'
    (args.output/'records.json.gz').write_bytes(gzip.compress(json.dumps(rows).encode(),mtime=0))
    write_json(args.output/'summary.json',summarize(rows))
    write_json(args.output/'provenance.json',dict(sources=locked,settings=asdict(ReliabilitySettings()),split=args.split,
        scene_root=str(args.scenes.resolve()),scope='ideal depth and true cameras, no band/transport/robot simulation',
        deadline='factory unknown; 1000ms is an illustrative offline analysis line only'))


if __name__=='__main__':main()
