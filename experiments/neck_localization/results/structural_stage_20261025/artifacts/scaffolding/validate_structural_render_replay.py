#!/usr/bin/env python3
"""Compare decoded RGB, depth and native arrays in a bounded rerender."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def run(source,replay,output):
    config=json.loads((replay/'effective_config.json').read_text());checks=[]
    for case in config['cases']:
        name=case['name']
        for camera in ['cam_a','cam_b','cam_c']:
            for mode in ['ideal_rgb','ideal_ir_registered','stereo_raw','stereo_spatial']:
                old=source/mode/'scenes'/name/'observations';new=replay/mode/'scenes'/name/'observations'
                np.testing.assert_array_equal(cv2.imread(str(old/(camera+'_rgb.png'))),cv2.imread(str(new/(camera+'_rgb.png'))))
                with np.load(old/(camera+'_depth_z_m.npz')) as data:a=data['depth_z_m'].copy()
                with np.load(new/(camera+'_depth_z_m.npz')) as data:b=data['depth_z_m'].copy()
                np.testing.assert_array_equal(a,b)
                checks.append(dict(case=name,camera=camera,condition=mode,rgb_identical=True,depth_max_difference_m=0.))
            with np.load(source/'sensor_diagnostics'/name/(camera+'_native.npz')) as a, np.load(replay/'sensor_diagnostics'/name/(camera+'_native.npz')) as b:
                for key in a.files:np.testing.assert_array_equal(a[key],b[key])
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(dict(scope='bounded opened-case rerender, not complete second test',
        cases=len(config['cases']),checks=checks,native_arrays_identical=True),f,indent=2)
    print('matched',len(checks),'RGB/depth groups',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--replay',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.source,a.replay,a.output)
