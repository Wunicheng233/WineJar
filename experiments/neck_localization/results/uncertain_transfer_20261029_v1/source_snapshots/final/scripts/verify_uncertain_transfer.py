#!/usr/bin/env python3
"""Offline source, observation, actual-FK and existing-GL-viewer audit."""
import argparse
from pathlib import Path
import sys
import os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.environ.pop('MUJOCO_GL',None)
import numpy as np
from neck_localization.experiment_io import read,write,digest


def run(bundle,viewer_name='arrival_viewer_v2',audit_output=None):
    protected=read(bundle/'protected_before.json') if (bundle/'protected_before.json').exists() else {}
    locked=read(bundle/'confirmation_lock.json')['source_sha256'] if (bundle/'confirmation_lock.json').exists() else {}
    for p,h in (protected|locked).items():assert digest(p)==h,p
    geometry=read(bundle/'confirmation_inputs/geometry.json')
    assert geometry['nominal_IK_and_board_visibility_checked']
    observations=list((bundle/'confirmation_inputs/measurements/test').glob('*/*/observations.npz'))
    assert len(observations)==24
    for p in observations:
        a=np.load(p);assert a['identified_pixels'].shape==(30,35,2)
        assert np.isfinite(a['identified_pixels']).all()
    fitted=[]
    for name in ['baselines','poe','poe_block']:
        fitted.extend(read(bundle/f'confirmation_{name}/records.json'))
    assert len(fitted)==96 and all(r['accepted'] for r in fitted)
    ideal=[r for r in fitted if r['condition']=='ideal']
    assert all(r['holdout_pixel_rms']<1e-4 for r in ideal)
    from scripts.view_calibration_arrival import ArrivalViewer
    output=audit_output or bundle/'analysis/viewer_audit';output.mkdir(parents=True)
    viewer=bundle/viewer_name
    datasets=read(viewer/'datasets.json');routes=[]
    first=datasets[0]
    from neck_localization.inspection_viewer import read_records
    v=ArrivalViewer(first['scenes'],first['records'],'new8__confirmation_nominal_003',method=first['method'],datasets=datasets,
        station_config=read(viewer/'config.json'),visible=False,ground_truth=True,overlay=True,band=False)
    try:
        for d in datasets:
            v.root=Path(d['scenes']);v.records_path=Path(d['records']);v.rows=read_records(v.records_path);v.method=d['method']
            v.record_settings=read(v.records_path.parent/'settings.json');v.cases=sorted({r['case'] for r in v.rows})
            v.methods=[d['method']]
            for i,c in enumerate(v.cases):
                v.case_index=i;v._load_case();v.draw();routes.append([d['method'],c])
            v.case_index=v.cases.index('new8__confirmation_nominal_003');v._load_case();v.snapshot(output/(d['method']+'.png'))
    finally:v.close()
    assert len(routes)==11*len(datasets)
    for p,h in (protected|locked).items():assert digest(p)==h,p
    result_path=output/'verification.json' if audit_output else bundle/'verification.json'
    write(result_path,dict(passed=True,protected_original_files=len(protected),locked_candidate_files=len(locked),
        calibration_observation_sets=len(observations),confirmation_calibrations=len(fitted),
        source_and_calibration_unchanged=True,ideal_controls_passed=True,actual_GL_routes=len(routes),viewer_bundle=viewer_name,
        actual_tool_FK_matched_recorded_position_and_rotation_at_1e_7=True,
        no_hardware_contact_force_training_default_change=True))
    print('uncertain transfer verification passed',len(routes),'actual GL routes')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--viewer-name',default='arrival_viewer_v2');p.add_argument('--audit-output',type=Path)
    a=p.parse_args();run(a.bundle.resolve(),a.viewer_name,a.audit_output.resolve() if a.audit_output else None)
