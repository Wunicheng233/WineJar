#!/usr/bin/env python3
"""Portable original MuJoCo viewer for the saved 1/2/3-camera comparisons."""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ.pop('MUJOCO_GL',None)
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from neck_localization.inspection_viewer import InspectionViewer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,default=ROOT/'experiments/presentation_assets_20261111/viewer')
    p.add_argument('--case',default='nominal_014')
    p.add_argument('--camera-set',choices=['cam_a','cam_a,cam_b','all'],default='all')
    p.add_argument('--point-cloud',action='store_true');p.add_argument('--snapshot',type=Path)
    a=p.parse_args()
    v=InspectionViewer(a.bundle/'scenes',a.bundle/'records.json.gz',a.case,a.camera_set,
        method='region_equal_trust',visible=a.snapshot is None,ground_truth=True,overlay=True,
        large_neck=True,point_cloud=a.point_cloud)
    try:
        truth=json.loads((a.bundle/'scenes'/a.case/'evaluation_only/target.json').read_text())
        v.neck_camera.lookat[:]=truth['center_m']
        if a.snapshot:v.snapshot(a.snapshot)
        else:v.run()
    finally:v.close()


if __name__=='__main__':main()
