#!/usr/bin/env python3
"""Ancillary representation/quality-scale diagnosis on opened 12-jar tests.

Frozen native estimator and measurement inputs. This is NOT a fresh unseen
test after the main results were inspected. Three explicit residual scales are
reported together; no truth-based choice is made inside the estimator.
"""
import argparse
from dataclasses import asdict, replace
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from neck_localization.active_stereo import quantize_depth
from neck_localization.band_geometry import plan_band,evaluate_band
from neck_localization.evaluation import score
from neck_localization.geometry import Camera,Estimate
from neck_localization.native_stereo import estimate_native_stereo,map_rgb_onto_native
from neck_localization.task_validated_output import TaskValidationSettings
from scripts.report_structural_stage import render_root,aggregate


def run(source,output,layouts):
    output.mkdir(parents=True,exist_ok=False);all_rows=[];summaries=[]
    for layout in layouts:
        folder=render_root(source,layout);config=json.loads((folder/'effective_config.json').read_text())
        destination=output/layout;destination.mkdir()
        (destination/'effective_config.json').write_bytes((folder/'effective_config.json').read_bytes())
        (destination/'provenance.json').write_text(json.dumps(dict(input=str(folder.resolve()),
            ancillary_after_main_tests_opened=True,true_K_T=True,source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
            solver='unchanged native_stereo; supplied task residual scales .4/.8/1.2mm; not calibrated confidence'),indent=2)+'\n')
        rows=[]
        for case in config['cases']:
            name=case['name'];sensor=json.loads((folder/'geometry'/name/'sensor.json').read_text())
            for mode in ['native_ideal_z16','native_raw','native_narrow','native_phase']:
                frames=[];masks=[];mappings=[];camera_meta=[]
                obs=destination/mode/'scenes'/name/'observations';obs.mkdir(parents=True)
                mask_dir=obs.parent/'foreground_masks';mask_dir.mkdir()
                for item in sensor['cameras']:
                    camera_name=item['name'];rgb_path=folder/'stereo_raw/scenes'/name/'observations'/(camera_name+'_rgb.png')
                    image=cv2.cvtColor(cv2.imread(str(rgb_path)),cv2.COLOR_BGR2RGB)
                    rgb_camera=Camera(camera_name,np.array(item['K_rgb']),np.array(item['T_world_from_rgb']),image.shape[1],image.shape[0])
                    if mode in ['native_ideal_z16','native_raw']:
                        with np.load(folder/'sensor_diagnostics'/name/(camera_name+'_native.npz')) as native:
                            depth=native['ideal_left_z' if mode=='native_ideal_z16' else 'native_z16_m'].copy()
                        if mode=='native_ideal_z16':depth=quantize_depth(depth,.001).astype(np.float32)*.001
                    elif mode=='native_narrow':
                        with np.load(source/layout/'rematched/sensor_diagnostics'/name/(camera_name+'_narrow5.npz')) as native:depth=native['native_z16_m'].copy()
                    else:
                        with np.load(source/layout/'phase/sensor_diagnostics'/name/(camera_name+'_phase.npz')) as native:depth=native['native_z16_m'].copy()
                    camera=Camera(camera_name,np.array(item['K_ir']),np.array(item['T_world_from_left_ir']),depth.shape[1],depth.shape[0])
                    frame,mask,valid,mapping=map_rgb_onto_native(SimpleNamespace(camera=rgb_camera,rgb=image),camera,depth)
                    frames.append(frame);masks.append(mask);mappings.append(mapping)
                    cv2.imwrite(str(obs/(camera_name+'_rgb.png')),cv2.cvtColor(frame.rgb,cv2.COLOR_RGB2BGR))
                    np.savez_compressed(obs/(camera_name+'_depth_z_m.npz'),depth_z_m=depth)
                    cv2.imwrite(str(mask_dir/(camera_name+'_foreground.png')),mask.astype(np.uint8)*255)
                    camera_meta.append(dict(name=camera_name,K=camera.K.tolist(),T_world_from_camera=camera.T_world_from_camera.tolist(),width=camera.width,height=camera.height))
                (obs/'camera_true_parameters.json').write_text(json.dumps(dict(cameras=camera_meta),indent=2)+'\n')
                for sigma in [.4,.8,1.2]:
                    settings=replace(TaskValidationSettings(),residual_rms_mm=sigma,heldout_signed_mean_mm=sigma/2)
                    started=perf_counter()
                    try:estimate=estimate_native_stereo(frames,masks,mappings,settings=settings)
                    except (ValueError,IndexError) as error:
                        estimate=Estimate(reasons=['estimator_exception:'+type(error).__name__+':'+str(error)],diagnostics=dict(traceback=traceback.format_exc()))
                    estimate.elapsed_ms=(perf_counter()-started)*1000
                    truth=json.loads((folder/'geometry'/name/'evaluation_only/target.json').read_text())
                    method='native_'+str(sigma).replace('.','p')
                    row=dict(case=name,group=case['group'],layout=layout,condition_name=mode,method=method,
                        evidence='rgb',camera_set='all',diagnostic_after_test_opened=True,settings=asdict(settings),**score(estimate,truth))
                    row.setdefault('diagnostics',estimate.diagnostics)
                    band=plan_band(row['estimate'],estimate.diagnostics) if estimate.center is not None and 'near_min_interval_from_axis_origin_m' in estimate.diagnostics else None
                    row['band']=band
                    row['band_evaluation']=({label:evaluate_band(band,truth,truth[field]) for label,field in
                        [('near_min_proxy','near_min_interval_s_m'),('full_annotated_neck','search_interval_s_m')]} if band else None)
                    rows.append(row)
                    with (destination/'records.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                    print(layout,name,mode,method,row['accepted'],row['center_mm'],row['normal_deg'],row['diameter_mm'],row['reasons'],flush=True)
        for mode in sorted({r['condition_name'] for r in rows}):
            selected=[r for r in rows if r['condition_name']==mode]
            (destination/mode/'records.json.gz').write_bytes(gzip.compress(json.dumps(selected).encode(),mtime=0))
            for method in sorted({r['method'] for r in selected}):
                summaries.append(dict(layout=layout,mode=mode,method=method,**aggregate([r for r in selected if r['method']==method])))
        (destination/'records.json.gz').write_bytes(gzip.compress(json.dumps(rows).encode(),mtime=0));all_rows.extend(rows)
    (output/'records.json.gz').write_bytes(gzip.compress(json.dumps(all_rows).encode(),mtime=0))
    (output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--layouts',nargs='+',default=['far','close']);a=p.parse_args();run(a.run,a.output,a.layouts)
