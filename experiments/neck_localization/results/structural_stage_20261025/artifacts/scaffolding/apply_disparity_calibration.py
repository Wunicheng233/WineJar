#!/usr/bin/env python3
"""Apply independently learned disparity correction before quantization/align."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from neck_localization.active_stereo import OfflineSpatialFilter,StereoSettings
from neck_localization.disparity_calibration import DisparityCorrection
from neck_localization.geometry import Camera
from scripts.run_active_stereo_stage1 import export_case


def run(source, model_path, output):
    model=DisparityCorrection.load(model_path)
    config=json.loads((source/'effective_config.json').read_text())
    output.mkdir(parents=True,exist_ok=False)
    (output/'effective_config.json').write_bytes((source/'effective_config.json').read_bytes())
    (output/'geometry').symlink_to((source/'geometry').resolve(),target_is_directory=True)
    for case in config['cases']:
        name=case['name']; geometry=source/'geometry'/name
        sensor=json.loads((geometry/'sensor.json').read_text());results=[];cameras=[]
        for item in sensor['cameras']:
            camera_name=item['name']
            rgb=cv2.cvtColor(cv2.imread(str(source/'stereo_raw/scenes'/name/'observations'/(camera_name+'_rgb.png'))),cv2.COLOR_BGR2RGB)
            with np.load(source/'sensor_diagnostics'/name/(camera_name+'_native.npz')) as data:
                original=data['native_float_z'].copy()
            corrected=model.correct(original)
            np.testing.assert_array_equal(corrected>0,original>0)
            assert np.isclose(model.fb,item['K_ir'][0][0]*item['baseline_m'])
            left=Camera(camera_name,np.array(item['K_ir']),np.array(item['T_world_from_left_ir']),original.shape[1],original.shape[0])
            camera=Camera(camera_name,np.array(item['K_rgb']),np.array(item['T_world_from_rgb']),rgb.shape[1],rgb.shape[0]);cameras.append(camera)
            processor=OfflineSpatialFilter(left,item['baseline_m'],StereoSettings(),rgb_camera=camera)
            try:
                z16,_=processor.process(corrected);aligned=processor.register(rgb)
            finally:processor.close()
            diag=output/'sensor_diagnostics'/name;diag.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(diag/(camera_name+'_phase.npz'),native_float_z=corrected,native_z16_m=z16)
            results.append(dict(rgb=rgb,depths={'phase_calibrated':aligned}))
        export_case(geometry,output/'phase_calibrated/scenes'/name,cameras,results,'phase_calibrated')
        print(name,'phase calibration exported',flush=True)
    (output/'provenance.json').write_text(json.dumps(dict(source=str(source.resolve()),model=str(model_path.resolve()),
        model_sha256=sha256(model_path.read_bytes()).hexdigest(),truth_read=False,point_validity_unchanged_before_registration=True,
        note='default matcher only; same native IR observations, correction then Z16 and official RGB align'),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.run,a.model,a.output)
