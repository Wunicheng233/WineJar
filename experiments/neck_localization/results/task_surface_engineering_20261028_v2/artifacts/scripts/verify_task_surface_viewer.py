#!/usr/bin/env python3
"""Exercise the existing MuJoCo viewer; do not run estimators or models."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
os.environ.pop('MUJOCO_GL',None)
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from neck_localization.inspection_viewer import InspectionViewer
from scripts.verify_viewer_navigation import click


def run(bundle,output):
    output.mkdir(parents=True,exist_ok=False)
    datasets=json.loads((bundle/'datasets.json').read_text());validation=json.loads((bundle/'validation.json').read_text())
    protected=validation['input_sha256'];assert validation['passed']
    assert all(sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protected.items())
    selected=next(d for d in datasets if d['condition']=='close/fs_old8/clean')
    viewer=InspectionViewer(selected['scenes'],selected['records'],'confirmation_nominal_006',
        method='visible_profile',visible=False,point_cloud=False,large_neck=True,
        overlay=True,ground_truth=True,datasets=datasets)
    checks=[]
    try:
        viewer.show_rejected_candidate=True;viewer.nav_scroll=0
        viewer.snapshot(output/'old_full_context_rejection.png')
        click(viewer,'method:task_surface');viewer.nav_scroll=0
        viewer.snapshot(output/'old_task_surface_accepted.png')
        for dataset in datasets:
            click(viewer,'dataset:'+dataset['condition'])
            for method in viewer.methods:
                click(viewer,'method:'+method)
                checks.append(dict(condition=dataset['condition'],method=method,case=viewer.row['case']))
        click(viewer,'dataset:close/fs_test8/shared_noise');click(viewer,'method:task_surface')
        viewer.show_band=True;viewer.nav_scroll=0
        viewer.snapshot(output/'new_candidate_band.png')
        viewer.point_cloud=viewer.cloud_local=True;viewer.cloud_weights=True;viewer._cloud_cache_key=None
        viewer.nav_scroll=0;viewer.snapshot(output/'new_actual_fit_weights.png')
    finally:viewer.close()
    assert all(sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protected.items())
    (output/'validation.json').write_text(json.dumps(dict(passed=True,datasets=len(datasets),routes=checks,
        sources_unchanged=len(protected),actual_GL=True,mouse_callbacks_injected=True,all_windows_closed=True,
        no_estimation=True,no_hardware=True,default_viewer_bundle_unchanged=True),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.bundle.resolve(),a.output.resolve())
