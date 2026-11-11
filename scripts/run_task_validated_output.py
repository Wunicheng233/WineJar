#!/usr/bin/env python3
"""Second independent success-rate audit, primary THREE-camera condition only."""
import argparse
from dataclasses import asdict
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from neck_localization.task_validated_output import estimate_task_validated_output,TaskValidationSettings
from neck_localization.reliable_output import estimate_reliable_output
from neck_localization.support_feedback import estimate_support_feedback
from neck_localization.joint_profile import JointSettings
from neck_localization.evaluation import score,summarize
from neck_localization.scene_preparation import write_json
from scripts.run_depth_profile_baseline import load_frames
from scripts.run_reliable_output import source_hashes as v1_hashes


def source_hashes():
    return v1_hashes()|{n:sha256((ROOT/n).read_bytes()).hexdigest() for n in
        ['neck_localization/task_validated_output.py','scripts/run_task_validated_output.py']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenes',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=['development','test'],default='development')
    p.add_argument('--freeze',type=Path)
    p.add_argument('--write-freeze',type=Path)
    args=p.parse_args();locked=source_hashes()
    old=json.loads((ROOT/'experiments/neck_localization/results/reliable_output_20261023/freeze.json').read_text())
    assert old['sources']==v1_hashes(),'Frozen v1 changed'
    if args.split=='test':
        assert args.freeze and not args.write_freeze
        freeze=json.loads(args.freeze.read_text());assert freeze['sources']==locked
        assert freeze['settings']==asdict(TaskValidationSettings())
        assert freeze['test_config_sha256']==sha256((args.scenes/'effective_config.json').read_bytes()).hexdigest()
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'settings.json',dict(**asdict(JointSettings()),task_validation=asdict(TaskValidationSettings())))
    cfg=json.loads((args.scenes/'effective_config.json').read_text());groups={c['name']:c.get('group','unknown') for c in cfg['cases']}
    rows=[]
    with (args.output/'records.jsonl').open('w') as journal:
        for cf in sorted(args.scenes.glob('*/observations/camera_true_parameters.json')):
            folder=cf.parent.parent;frames=load_frames(folder/'observations','rgb')
            for method in ['frozen_region','adaptive_validated','task_validated_output']:
                est=(estimate_support_feedback(frames,mode='region') if method=='frozen_region' else
                     estimate_reliable_output(frames) if method=='adaptive_validated' else estimate_task_validated_output(frames))
                truth=json.loads((folder/'evaluation_only/target.json').read_text())
                row=dict(case=folder.name,group=groups[folder.name],split=args.split+'_'+groups[folder.name],
                    geometry='photo_short_neck',evidence='rgb',camera_set='all',factor='ideal_true_cameras/all',method=method,**score(est,truth))
                row.setdefault('diagnostics',est.diagnostics);row['over_illustrative_1000ms']=row['runtime_ms']>1000
                rows.append(row);journal.write(json.dumps(row)+'\n');journal.flush()
            print(folder.name,[(r['method'],r['accepted'],r['center_mm'],r['normal_deg'],r['reasons']) for r in rows[-3:]],flush=True)
    assert locked==source_hashes()
    (args.output/'records.json.gz').write_bytes(gzip.compress(json.dumps(rows).encode(),mtime=0))
    write_json(args.output/'summary.json',summarize(rows))
    write_json(args.output/'provenance.json',dict(sources=locked,split=args.split,settings=asdict(TaskValidationSettings()),
        scope='three cameras only, ideal depth/true cameras; no band, robot or transfer simulation'))
    if args.write_freeze:
        assert args.split=='development' and not args.write_freeze.exists()
        plan_path=ROOT/'experiments/neck_localization/results/reliable_output_20261023/v2_plan.json'
        plan=json.loads(plan_path.read_text())
        config_path=ROOT/'experiments/neck_localization/configs/reliable_task_fallback_test_v2.json'
        assert sha256(config_path.read_bytes()).hexdigest()==plan['test_config_sha256']
        write_json(args.write_freeze,dict(sources=locked,settings=asdict(TaskValidationSettings()),
            test_config_sha256=plan['test_config_sha256'],plan_sha256=sha256(plan_path.read_bytes()).hexdigest(),
            development_sha256=sha256((args.output/'records.json.gz').read_bytes()).hexdigest(),policy='v1 test is development now; no v2 tuning after fresh test'))


if __name__=='__main__':main()
