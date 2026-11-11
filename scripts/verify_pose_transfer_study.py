#!/usr/bin/env python3
"""Saved-data/chain/ROI audit and actual existing-viewer GLFW route checks."""
import argparse
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.pop('MUJOCO_GL',None)
import numpy as np
from scipy.spatial.transform import Rotation
from neck_localization.experiment_io import read,write,digest
from neck_localization.station_tying import pose,move_circle


def run(bundle,calibration,preview_dir=None):
    tracked={}
    for p in [calibration/'provenance.json',bundle/'body/provenance.json',bundle/'body/partial_view_ablation/provenance.json',bundle/'charuco/provenance.json']:
        for name,h in read(p)['source_sha256'].items():
            assert digest(name)==h,name
            tracked[name]=h
    config=read(bundle/'body/protocol.json');base=ROOT/config['baseline']
    for name,h in read(base/'viewer/provenance.json')['input_and_source_sha256'].items():
        assert digest(name)==h,name;tracked[name]=h
    old=ROOT/read(base/'protocol.json')['source_bundle']
    cal=read(calibration/'records.json')
    ideal=[r for r in cal if r['condition'] in ('ideal','gauge_relabel') and r['layout']=='diverse']
    assert all(r['physical_query_max_mm']<1e-4 for r in ideal)
    assert all(not r['accepted'] and r['reason']=='insufficient_nonparallel_rotation' for r in cal if r['layout']=='one_axis')
    # Positive and negative controls have identical nominal transport outputs.
    rows=read(bundle/'body/records.json')
    for source in config['body']['source_controls']:
        for cohort,case in sorted({(r['cohort'],r['case']) for r in rows}):
            for delta in config['body']['stop_errors_m']:
                group=[r for r in rows if (r['cohort'],r['case'],r['source_control'],r['stop_error_mm'],r['method'])==(cohort,case,source,delta*1000,'nominal')
                    and r['condition'] not in ('upstream_frame_bias_2mm','rotation_mismatch_1deg')]
                assert all(r['estimate']==group[0]['estimate'] for r in group)
    # Geometric evaluation proves the fixed observation-derived body ROI does
    # not contain the neck. This check never feeds a label back to registration.
    gaps=[]
    for folder in sorted((bundle/'body/scenes').glob('*/*')):
        truth=read(old/folder.parent.name/'ss/clean/scenes'/folder.name/'evaluation_only/target.json')
        C=np.array(read(folder/'secondary_cameras.json')[0]['T_world_from_camera'])
        for p in folder.glob('target_*.npz'):
            parts=p.stem.split('_');stop_id=int(parts[1]);condition='_'.join(parts[2:])
            R=Rotation.from_euler('y',1,degrees=True).as_matrix() if condition=='rotation_mismatch_1deg' else np.eye(3)
            M=pose(R,[2.4+config['body']['stop_errors_m'][stop_id],0,0]);t=move_circle(truth,M)
            q=np.load(p)['points_C'];world=q @C[:3,:3].T+C[:3,3]
            axial=(world-t['center_m']) @t['normal']
            gap=truth['search_interval_s_m'][0]-truth['s_m']-float(axial.max())
            gaps.append(gap)
            assert gap>0,p
    # Actual GL, with the original scene/sidebar/camera/close-up components.
    from scripts.view_body_transfer import BodyTransferViewer
    datasets=read(bundle/'viewer/datasets.json');routes=[];method_paths=[]
    previews=preview_dir or bundle/'analysis';previews.mkdir(parents=True,exist_ok=True)
    first=datasets[0]
    viewer=BodyTransferViewer(first['scenes'],first['records'],'new8__confirmation_nominal_003',
        method='observed_subset_axis',datasets=datasets,visible=False,ground_truth=True,overlay=True,band=True)
    try:
        for dataset in datasets:
            viewer.root=Path(dataset['scenes']);viewer.records_path=Path(dataset['records']);viewer.rows=read(dataset['records'])
            viewer.record_settings=read(viewer.records_path.parent/'settings.json')
            viewer.cases=sorted({r['case'] for r in viewer.rows});viewer.methods=sorted({r['method'] for r in viewer.rows})
            viewer.method='observed_subset_axis'
            for i,case in enumerate(viewer.cases):
                viewer.case_index=i;viewer._load_case();assert len(viewer.frames)==2
                for method in viewer.methods:
                    viewer.method=method
                    assert viewer.row['secondary_body_observation'] and viewer.row['neck_has_not_been_reobserved']
                    # One GL draw for each saved condition/case/method route.
                    viewer.draw();method_paths.append([dataset['condition'],case,method])
                routes.append([dataset['condition'],case])
            viewer.case_index=viewer.cases.index('new8__confirmation_nominal_003');viewer.method='reciprocal_axis';viewer._load_case()
            viewer.snapshot(previews/(dataset['condition']+'_preview.png'))
        viewer.point_cloud=True;viewer.snapshot(previews/'body_point_cloud_preview.png')
    finally:viewer.close()
    for name,h in tracked.items():assert digest(name)==h,name
    write(bundle/'verification.json',dict(passed=True,protected_input_source_hashes=len(tracked),
        calibration_records=len(cal),body_records=len(rows)+len(read(bundle/'body/partial_view_ablation/records.json')),
        ideal_eye_to_hand_and_gauge_controls=True,one_axis_calibration_degeneracy_detected=True,
        nominal_controls_identical=True,minimum_observed_body_to_true_neck_axial_gap_mm=min(gaps)*1000,
        gl_routes=len(routes),gl_method_paths=len(method_paths),old_viewer_sources_unchanged=True,
        no_hardware_training_real_calibration_or_physical_fastening=True,verification_script_sha256=digest(__file__)))
    print('pose-transfer audit passed',len(routes),'GL scene routes',len(method_paths),'GL method paths')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--preview-dir',type=Path,help='New preview folder for a repeat audit; existing images are preserved')
    a=p.parse_args();run(a.bundle.resolve(),a.calibration.resolve(),a.preview_dir.resolve() if a.preview_dir else None)
