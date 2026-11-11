#!/usr/bin/env python3
"""Frozen-plan comparison of observation-only additive depth-bias compensation."""
import argparse
from dataclasses import asdict
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from neck_localization.bias_compensation import BiasCompensationSettings,estimate_bias_compensated
from neck_localization.band_geometry import plan_band,evaluate_band
from neck_localization.depth_perturbation import axial_depth_bias
from neck_localization.evaluation import score
from neck_localization.joint_profile import JointSettings
from scripts.run_depth_profile_baseline import load_frames
from scripts.run_depth_bias_stage1 import export_observations,summarize,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cases',nargs='*',help='Only allowed for explicitly labelled limited reproduction')
    args=p.parse_args()
    plan=json.loads(args.plan.read_text())
    for name,digest in plan['estimator_sources'].items():
        assert sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'plan.json',plan)
    run=ROOT/plan['input_run']
    old=json.load(gzip.open(run/'records.json.gz','rt'))
    cases=[c for c in plan['cases'] if not args.cases or c['name'] in args.cases]
    rows=[]
    with (args.output/'records.jsonl').open('w') as journal:
        for method in plan['methods']:
            settings=BiasCompensationSettings(sigma_bias_mm=method['sigma_bias_mm'])
            for condition in plan['conditions']:
                condition_dir=args.output/method['name']/condition['name']
                write_json(condition_dir/'settings.json',dict(**asdict(JointSettings()),band_width_mm=plan['band_width_mm'],
                    sensor_perturbation=condition,preprocessing=dict(kind='estimated_bias_corrected_depth',
                    raw_inputs=str((run/condition['name']/'scenes').resolve())),bias_compensation=asdict(settings)))
                selected=[]
                for case in cases:
                    source=run/condition['name']/'scenes'/case['name']
                    frames=load_frames(source/'observations','rgb')
                    est=estimate_bias_compensated(frames,settings=settings)
                    diag=est.diagnostics
                    correction=diag['bias_compensation']['estimated_bias_by_camera_mm'] if diag['bias_compensation']['applied'] else {}
                    processed=axial_depth_bias(frames,{name:-value/1000 for name,value in correction.items()})
                    export_observations(source,condition_dir/'scenes'/case['name'],processed,
                        {name:-value/1000 for name,value in correction.items()})
                    band=(plan_band(dict(center_m=est.center,normal=est.normal,radius_m=est.radius),diag,plan['band_width_mm']/1000)
                          if est.center is not None and 'near_min_interval_from_axis_origin_m' in diag else None)
                    truth=json.loads((source/'evaluation_only/target.json').read_text())
                    row=dict(case=case['name'],group=case['group'],split='development' if plan['phase']=='development' else 'locked_test',
                        evidence='rgb',geometry='photo_short_neck',camera_set='all',factor=condition['name'],condition_name=condition['name'],
                        method=method['name'],**score(est,truth))
                    row.setdefault('diagnostics',diag)
                    row['band']=band
                    row['band_evaluation']=({name:evaluate_band(band,truth,truth[field]) for name,field in
                        [('near_min_proxy','near_min_interval_s_m'),('full_annotated_neck','search_interval_s_m')]} if band else None)
                    # Injection values enter only scorer, after estimator and band proposal.
                    bhat=diag['bias_compensation']['estimated_bias_by_camera_mm']
                    row['evaluation_only_bias_error_mm']={name:value-condition['bias_by_camera_mm'].get(name,0.) for name,value in bhat.items()}
                    row['signed_diameter_error_mm']=2000*(est.radius-truth['radius_m']) if est.radius is not None else None
                    selected.append(row);rows.append(row)
                    journal.write(json.dumps(row)+'\n');journal.flush()
                    print(method['name'],condition['name'],case['name'],est.accepted,row['center_mm'],row['diameter_mm'],
                          'bhat',bhat,'fit',diag['bias_compensation'].get('compensation_reasons',[]),flush=True)
                (condition_dir/'records.json.gz').write_bytes(gzip.compress(json.dumps(selected).encode(),mtime=0))
    baseline=[]
    for row in old:
        if row['case'] in {c['name'] for c in cases} and row['condition_name'] in {c['name'] for c in plan['conditions']}:
            item=dict(row);item['method']='frozen_v2';baseline.append(item)
    rows+=baseline
    (args.output/'records.json.gz').write_bytes(gzip.compress(json.dumps(rows).encode(),mtime=0))
    summaries=[]
    for name in ['frozen_v2']+[m['name'] for m in plan['methods']]:
        method_rows=[r for r in rows if r['method']==name]
        for group in summarize(method_rows,plan['conditions']):
            group['method']=name
            local=[r for r in method_rows if r['condition_name']==group['condition_name'] and (group['group']=='all' or r['group']==group['group'])]
            errors=[abs(e) for r in local for e in r.get('evaluation_only_bias_error_mm',{}).values()]
            group['bias_error_mm_max']=float(max(errors)) if errors else None
            group['bias_error_mm_median']=float(np.median(errors)) if errors else None
            summaries.append(group)
    write_json(args.output/'summary.json',summaries)
    for name,digest in plan['estimator_sources'].items():
        assert sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    write_json(args.output/'provenance.json',dict(plan_sha256=sha256(args.plan.read_bytes()).hexdigest(),
        estimator_sources=plan['estimator_sources'],limited_reproduction=bool(args.cases),
        baseline_reused=True,baseline_runtime_is_previous_measurement=True,
        whole_pipeline_inputs='raw biased depth; exported viewer depths are estimated-corrected internal final-fit inputs',
        test_protocol=plan['phase'],uncertainty='not_calibrated; prior/noise scales assumed'))


if __name__=='__main__':main()
