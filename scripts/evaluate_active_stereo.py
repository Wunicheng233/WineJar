#!/usr/bin/env python3
"""Evaluate saved RGB-D with frozen geometry and optional image-only masks.

Also supports RGB camera parameters estimated from rendered calibration boards.
No target truth is opened until after inference and the band proposal.
"""
import argparse
from dataclasses import asdict, replace
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys
import traceback
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from neck_localization.band_geometry import evaluate_band, plan_band
from neck_localization.evaluation import score
from neck_localization.geometry import Camera, Estimate
from neck_localization.joint_profile import JointSettings
from neck_localization.scene_preparation import write_json
from neck_localization.stereo_robust import estimate_stereo_robust, extract_stereo_masks
from neck_localization.task_validated_output import estimate_task_validated_output
from scripts.run_active_stereo_stage1 import sources as sensor_sources
from scripts.run_depth_profile_baseline import load_frames


def source_hashes():
    return sensor_sources() | {n:sha256((ROOT/n).read_bytes()).hexdigest() for n in
        ['neck_localization/stereo_robust.py', 'scripts/evaluate_active_stereo.py']}


def summaries(rows):
    result = []
    for method, condition in sorted({(r['method'], r['condition_name']) for r in rows}):
        rr=[r for r in rows if r['method']==method and r['condition_name']==condition]
        out=dict(method=method, condition=condition, n=len(rr), accepted=sum(r['accepted'] for r in rr),
                 correct_circle=sum(r['accepted'] and r['truth_pass'] for r in rr),
                 rejected=sum(not r['accepted'] for r in rr),
                 wrong_accepted=sum(r['accepted'] and not r['truth_pass'] for r in rr),
                 band_success=sum(bool(r['accepted'] and r['truth_pass'] and r.get('band')
                    and r['band']['nominal_compatible'] and r['band_evaluation']['near_min_proxy']['contained']) for r in rr))
        for metric in ['center_mm','normal_deg','diameter_mm','normal_component_mm','tangent_mm','runtime_ms']:
            values=[abs(r[metric]) for r in rr if r.get(metric) is not None]
            for suffix, p in [('median',50),('p95',95),('max',100)]:
                out[metric+'_'+suffix]=float(np.percentile(values,p)) if values else None
        out['reasons']={s:sum(s in r['reasons'] for r in rr) for s in sorted({s for r in rr for s in r['reasons']})}
        result.append(out)
    return result


def export_input(source, destination, frames, camera_mode, masks=None):
    destination.mkdir(parents=True,exist_ok=False)
    for name in ['assets','evaluation_only','model_manifest.json']:
        (destination/name).symlink_to((source/name).resolve(),target_is_directory=(source/name).is_dir())
    tree=ET.parse(source/'scene.xml')
    if camera_mode != 'true':
        for frame in frames:
            node=tree.getroot().find(f"worldbody/camera[@name='{frame.camera.name}']")
            rot=frame.camera.T_world_from_camera[:3,:3] @ np.diag([1,-1,-1])
            node.set('pos',' '.join(map(str,frame.camera.T_world_from_camera[:3,3])))
            node.set('xyaxes',' '.join(map(str,rot[:,:2].T.ravel())))
    tree.write(destination/'scene.xml',encoding='unicode')
    obs=destination/'observations';obs.mkdir()
    write_json(obs/'camera_true_parameters.json',dict(kind=('true_simulation_parameters' if camera_mode=='true' else
        'estimated_rendered_board_parameters_legacy_filename'),cameras=[dict(name=f.camera.name,
        K=f.camera.K,T_world_from_camera=f.camera.T_world_from_camera,width=f.camera.width,height=f.camera.height) for f in frames]))
    for frame in frames:
        for suffix in ['_rgb.png','_depth_z_m.npz']:
            (obs/(frame.camera.name+suffix)).symlink_to((source/'observations'/(frame.camera.name+suffix)).resolve())
    if masks is not None:
        mask_dir=destination/'foreground_masks';mask_dir.mkdir()
        for frame,mask in zip(frames,masks):
            cv2.imwrite(str(mask_dir/(frame.camera.name+'_foreground.png')),mask.astype(np.uint8)*255)


def run(source, output, *, methods=('stereo_rgb_adapted',), conditions=None, freeze=None, estimated_parameters=None):
    source,output=Path(source),Path(output)
    config=json.loads((source/'effective_config.json').read_text())
    locked=source_hashes()
    if config['dataset_split'].endswith('_test'):
        if not freeze:raise ValueError('Test evaluation requires the pre-test freeze')
        plan=json.loads(Path(freeze).read_text())
        assert plan['evaluation_sources']==locked,'Evaluator changed after test freeze'
        assert plan['test_config_sha256']==sha256((source/'effective_config.json').read_bytes()).hexdigest()
    camera_mode='true'
    estimates={}
    if estimated_parameters:
        estimates={c['name']:c for c in json.loads(Path(estimated_parameters).read_text())['cameras']}
        camera_mode='rendered_board_estimated'
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'effective_config.json',config)
    write_json(output/'provenance.json',dict(sources=locked,input=str(source.resolve()),
        camera_mode=camera_mode,estimated_parameters=str(estimated_parameters) if estimated_parameters else None,
        estimated_parameters_sha256=sha256(Path(estimated_parameters).read_bytes()).hexdigest() if estimated_parameters else None,
        methods=list(methods),condition_selection=conditions,truth='evaluation after inference only'))
    conditions=conditions or sorted(p.name for p in source.iterdir() if (p/'scenes').is_dir())
    rows=[]
    for method in methods:
        for condition in conditions:
            target=output/method/condition
            target.mkdir(parents=True)
            settings=json.loads((source/condition/'settings.json').read_text())
            settings.update(asdict(JointSettings()),camera_parameter_mode=camera_mode)
            write_json(target/'settings.json',settings)
            local=[]
            for case in config['cases']:
                folder=source/condition/'scenes'/case['name']
                frames=load_frames(folder/'observations','rgb')
                if estimates:
                    frames=[replace(f,camera=Camera(f.camera.name,np.array(estimates[f.camera.name]['K']),
                        np.array(estimates[f.camera.name]['T_world_from_camera']),f.camera.width,f.camera.height)) for f in frames]
                masks=extract_stereo_masks(frames)[0] if method=='stereo_rgb_adapted' else None
                export_input(folder,target/'scenes'/case['name'],frames,camera_mode,masks)
                try:
                    estimate=(estimate_stereo_robust(frames) if method=='stereo_rgb_adapted' else estimate_task_validated_output(frames))
                except Exception as e:
                    estimate=Estimate(reasons=['estimator_exception'],diagnostics=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc()))
                d=estimate.diagnostics
                band=(plan_band(dict(center_m=estimate.center,normal=estimate.normal,radius_m=estimate.radius),d)
                      if estimate.center is not None and 'near_min_interval_from_axis_origin_m' in d else None)
                truth=json.loads((folder/'evaluation_only/target.json').read_text())
                row=dict(case=case['name'],group=case['group'],split=config['dataset_split'],geometry='photo_short_neck',
                    evidence='rgb',camera_set='all',method=method,factor=condition,condition_name=condition,
                    camera_parameter_mode=camera_mode,**score(estimate,truth))
                row.setdefault('diagnostics',d);row['band']=band
                row['band_evaluation']=({name:evaluate_band(band,truth,truth[field]) for name,field in
                    [('near_min_proxy','near_min_interval_s_m'),('full_annotated_neck','search_interval_s_m')]} if band else None)
                rows.append(row);local.append(row)
                with (output/'records.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print(method,condition,case['name'],row['accepted'],row['center_mm'],row['normal_deg'],row['reasons'],flush=True)
            (target/'records.json.gz').write_bytes(gzip.compress(json.dumps(local).encode(),mtime=0))
    (output/'records.json.gz').write_bytes(gzip.compress(json.dumps(rows).encode(),mtime=0))
    write_json(output/'summary.json',summaries(rows))
    assert source_hashes()==locked,'Source changed while evaluating'
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--methods',nargs='+',choices=['frozen_v2','stereo_rgb_adapted'],default=['stereo_rgb_adapted'])
    p.add_argument('--conditions',nargs='+');p.add_argument('--freeze',type=Path)
    p.add_argument('--estimated-parameters',type=Path)
    args=p.parse_args()
    run(args.source,args.output,methods=args.methods,conditions=args.conditions,freeze=args.freeze,estimated_parameters=args.estimated_parameters)
