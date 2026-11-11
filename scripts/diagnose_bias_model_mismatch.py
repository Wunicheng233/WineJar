#!/usr/bin/env python3
"""Developer diagnostics: errors not explained by additive depth bias alone.

These are observation/parameter sensitivity recipes, NOT a full physical
camera calibration simulator. Original camera pose/shape truth only scores.
"""
import argparse
from dataclasses import replace
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from neck_localization.bias_compensation import estimate_bias_compensated,BiasCompensationSettings
from neck_localization.task_validated_output import estimate_task_validated_output
from neck_localization.band_geometry import plan_band,evaluate_band
from neck_localization.evaluation import score
from scripts.run_depth_profile_baseline import load_frames
from scripts.run_depth_bias_stage1 import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    root=ROOT/'experiments/neck_localization/runs/reliable_output_v2_test_scenes'
    cases=['nominal_001','nominal_002','stress_000','stress_003']
    conditions=['zero','depth_scale_plus0p3pct','focal_estimate_plus0p5pct','extrinsic_a_x2mm','extrinsic_shared_x5mm','disparity_sigma0p01px']
    seed=2026102461
    plan=dict(cases=cases,conditions=conditions,seed=seed,assumed_stereo_baseline_m=.05,
              scope='developer observation/parameter diagnostics, not end-to-end calibration or camera performance; no tuning',
              inputs=str(root),bias_model='unchanged frozen primary no-prior',
              noise='last factor only: independent disparity Gaussian sigma .01px, true synthetic fx, assumed B=.05m; no spatial correlation')
    sources={name:sha256((ROOT/name).read_bytes()).hexdigest() for name in
             ['neck_localization/bias_compensation.py','neck_localization/bias_profile.py',str(Path(__file__).relative_to(ROOT))]}
    write_json(args.output/'plan.json',plan)
    rows=[]
    for ci,case in enumerate(cases):
        original=load_frames(root/case/'observations','rgb')
        for ki,condition in enumerate(conditions):
            observed=[]
            output=args.output/'inputs'/condition/case
            output.mkdir(parents=True)
            cameras=[]
            for j,frame in enumerate(original):
                K=frame.camera.K.copy();T=frame.camera.T_world_from_camera.copy()
                depth=frame.depth_z_m.copy()
                valid=np.isfinite(depth)&(depth>0)
                if condition=='depth_scale_plus0p3pct':
                    depth=np.array(depth,float);depth[valid]*=1.003
                elif condition=='focal_estimate_plus0p5pct':
                    K[0,0]*=1.005;K[1,1]*=1.005
                elif condition=='extrinsic_a_x2mm' and j==0:
                    T[0,3]+=.002
                elif condition=='extrinsic_shared_x5mm':
                    T[0,3]+=.005
                elif condition=='disparity_sigma0p01px':
                    rng=np.random.default_rng(np.random.SeedSequence([seed,ci,ki,j]))
                    depth=np.array(depth,float)
                    fB=frame.camera.K[0,0]*.05
                    disparity=fB/depth[valid]+rng.normal(0,.01,valid.sum())
                    depth[valid]=np.where(disparity>0,fB/disparity,0)
                camera=replace(frame.camera,K=K,T_world_from_camera=T)
                observed.append(replace(frame,camera=camera,depth_z_m=depth))
                np.savez_compressed(output/(camera.name+'_depth_z_m.npz'),depth_z_m=depth)
                (output/(camera.name+'_rgb.png')).symlink_to((root/case/'observations'/(camera.name+'_rgb.png')).resolve())
                cameras.append(dict(name=camera.name,K_est=K.tolist(),T_world_from_camera_est=T.tolist(),
                                    K_true=frame.camera.K.tolist(),T_world_from_camera_true=frame.camera.T_world_from_camera.tolist(),
                                    width=camera.width,height=camera.height))
            write_json(output/'camera_parameters_diagnostic.json',dict(cameras=cameras,warning='estimated camera parameters may differ from true renderer parameters; do not label as ideal inputs'))
            for method in ['frozen_v2','bias_no_prior']:
                est=(estimate_task_validated_output(observed) if method=='frozen_v2' else
                     estimate_bias_compensated(observed,settings=BiasCompensationSettings(sigma_bias_mm=None)))
                d=est.diagnostics
                band=(plan_band(dict(center_m=est.center,normal=est.normal,radius_m=est.radius),d,.003)
                      if est.center is not None and 'near_min_interval_from_axis_origin_m' in d else None)
                truth=json.loads((root/case/'evaluation_only/target.json').read_text())
                row=dict(case=case,group=case.split('_')[0],condition_name=condition,method=method,**score(est,truth))
                row.setdefault('diagnostics',d);row['band']=band
                row['band_evaluation']=evaluate_band(band,truth,truth['search_interval_s_m']) if band else None
                rows.append(row)
                print(case,condition,method,est.accepted,row['center_mm'],row['diameter_mm'],row['reasons'],flush=True)
    (args.output/'records.json.gz').write_bytes(gzip.compress(json.dumps(rows).encode(),mtime=0))
    summaries=[]
    for condition in conditions:
        for method in ['frozen_v2','bias_no_prior']:
            rr=[r for r in rows if r['condition_name']==condition and r['method']==method]
            item=dict(condition_name=condition,method=method,n=len(rr),correct=sum(r['accepted'] and r['truth_pass'] for r in rr),
                      rejected=sum(not r['accepted'] for r in rr),wrong_accepted=sum(r['accepted'] and not r['truth_pass'] for r in rr))
            for metric in ['center_mm','normal_deg','diameter_mm']:
                values=[r[metric] for r in rr if r[metric] is not None]
                item[metric+'_max']=max(values) if values else None
            summaries.append(item)
    write_json(args.output/'summary.json',summaries)
    assert all(sha256((ROOT/name).read_bytes()).hexdigest()==h for name,h in sources.items())
    write_json(args.output/'provenance.json',dict(sources=sources,notes='all factors are assumed sensitivity recipes; shared world shift is exact gauge counterexample'))


if __name__=='__main__':main()
