#!/usr/bin/env python3
"""Exercise new condition/method/rejected-candidate buttons on actual GL UI."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
os.environ.pop('MUJOCO_GL',None)
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from neck_localization.inspection_viewer import InspectionViewer,read_records
from scripts.verify_viewer_navigation import click


def run(manifest,output):
    output.mkdir(parents=True,exist_ok=False)
    datasets=json.loads(manifest.read_text());raw=next(d for d in datasets if d['condition']=='far/stereo_raw')
    protected={Path(d['records']) for d in datasets}
    for dataset in datasets:
        for p in (Path(dataset['scenes'])/'structural_test_nominal_000'/'observations').iterdir():
            if p.is_file():protected.add(p.resolve())
    hashes={str(p):sha256(p.read_bytes()).hexdigest() for p in protected};actions=[]
    viewer=InspectionViewer(raw['scenes'],raw['records'],'structural_test_nominal_000',method='portfolio',
        visible=False,point_cloud=True,large_neck=True,overlay=True,datasets=datasets)
    def act(action):click(viewer,action);actions.append(action)
    def select_case(case):
        for _ in range(len(viewer.cases)):
            if viewer.cases[viewer.case_index]==case:return
            act('next')
        raise AssertionError('Case missing')
    try:
        act('toggle_rejected');act('toggle_truth');act('view:fit')
        assert viewer.show_rejected_candidate and viewer.show_truth and viewer.cloud_local
        pose=np.array(viewer.camera.lookat).copy()
        act('dataset:close/stereo_raw')
        np.testing.assert_array_equal(viewer.camera.lookat,pose)
        assert viewer.method=='portfolio' and not viewer.row['accepted'] and viewer.row.get('estimate')
        viewer.nav_scroll=0;viewer.snapshot(output/'close_rejected_fit.png')
        act('method:portfolio_fixed');assert viewer.method=='portfolio_fixed'
        act('method:baseline');assert viewer.row.get('estimate') is None
        act('dataset:far/narrow5');assert viewer.method=='baseline'
        act('method:portfolio_single')
        candidates=[r for r in viewer.rows if r['method']=='portfolio_single' and r['accepted'] and r['truth_pass']]
        select_case(candidates[0]['case']);assert viewer.row['accepted'] and viewer.row['truth_pass']
        viewer.nav_scroll=0;viewer.snapshot(output/'far_narrow_correct.png')
        act('method:portfolio')
        wrong=[r for r in viewer.rows if r['method']=='portfolio' and r['accepted'] and not r['truth_pass']]
        worst=max(wrong,key=lambda r:r['center_mm']/4+r['normal_deg']/2+r['diameter_mm']/2)
        select_case(worst['case']);assert viewer.row['accepted'] and not viewer.row['truth_pass']
        act('view:cloud');act('input:depth')
        viewer.nav_scroll=0;viewer.snapshot(output/'far_narrow_wrong_accept.png')
        report=dict(scope='new-viewer mouse callback integration; post-hoc selected examples are visualization only',
            actions=actions,protected_files=len(hashes),inputs_unchanged=True,method_preserved_between_conditions=True,
            same_case_view_pose_preserved=True,rejected_orange_candidate=True,no_circle_fabricated_for_baseline=True,
            wrong_accept_example=dict(case=worst['case'],condition='far/narrow5',method='portfolio',
                center_mm=worst['center_mm'],normal_deg=worst['normal_deg'],diameter_mm=worst['diameter_mm']),
            presented_GL_backbuffer_snapshots=True)
    finally:viewer.close()
    assert all(sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    (output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.manifest,a.output)
