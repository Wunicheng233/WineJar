#!/usr/bin/env python3
"""Render-only counterpart of frozen active-stereo study; estimator-free.

Uses identical core scene/capture/export functions and settings. An estimator
exception must not prevent the remaining independent observations being made.
"""
import argparse
from dataclasses import asdict
import gc
from hashlib import sha256
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from neck_localization.active_stereo import ActiveStereoRig,StereoSettings,initialize_renderer,add_jar_assets
from neck_localization.scene_preparation import prepare_scene_bundle,scene_cameras,write_json
from scripts.run_active_stereo_stage1 import export_case,sources,MODES


def run(config_path,freeze_path,output):
    import sapien
    config=json.loads(config_path.read_text());freeze=json.loads(freeze_path.read_text())
    assert freeze['sources']==sources()
    assert freeze['test_config_sha256']==sha256(config_path.read_bytes()).hexdigest()
    assert freeze['settings']==asdict(StereoSettings())
    output.mkdir(parents=True,exist_ok=False)
    (output/'effective_config.json').write_bytes(config_path.read_bytes())
    prepare_scene_bundle(config,output/'geometry');initialize_renderer();rig_cameras=scene_cameras(config['scene'])
    for mode in MODES:(output/mode).mkdir()
    checks=[]
    for index,case in enumerate(config['cases']):
        name=case['name'];source=output/'geometry'/name
        scene=sapien.Scene();scene.set_ambient_light([.3,.3,.3])
        scene.add_directional_light([1,-1,-2],[.6,.6,.6]);scene.add_directional_light([-1,1,-1],[.25,.25,.25])
        add_jar_assets(scene,source)
        rig=ActiveStereoRig(scene,rig_cameras,StereoSettings(seed=StereoSettings().seed+index*10))
        try:
            write_json(source/'sensor.json',rig.metadata());cameras=[e['rgb'] for e in rig.entries];captured=[]
            diag=output/'sensor_diagnostics'/name;diag.mkdir(parents=True)
            for i,camera in enumerate(cameras):
                data=rig.capture(i);captured.append(data)
                for j,image in enumerate(data['ir_clean']):
                    cv2.imwrite(str(diag/(camera.name+f'_ir_clean_{j}.png')),np.clip(image*255,0,255).astype(np.uint8))
                np.savez_compressed(diag/(camera.name+'_native.npz'),native_float_z=data['native_float_z'],
                    native_z16_m=data['native_z16_m'],native_filtered_m=data['native_filtered_m'],ideal_left_z=data['ideal_left_z'])
                np.savez_compressed(diag/(camera.name+'_ideal_rgb.npz'),ideal_rgb_z=data['depths']['ideal_rgb'])
            for mode in MODES:export_case(source,output/mode/'scenes'/name,cameras,captured,mode)
            checks.append(dict(case=name,cameras=len(cameras),modes=list(MODES)))
            print(name,'rendered; no estimator invoked',flush=True)
        finally:rig.close()
        del rig,scene,captured;gc.collect()
    write_json(output/'render_completed.json',dict(cases=checks,estimator_invoked=False,
        frozen_core_sources=sources(),scaffold_sha256=sha256(Path(__file__).read_bytes()).hexdigest()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--freeze',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.config,a.freeze,a.output)
