#!/usr/bin/env python3
"""Frozen-input correspondence-direction ablation; no renderer, no device."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from neck_localization.station_tying import pose,move_circle
from neck_localization.body_partial_registration import register_observed_subset
from neck_localization.cross_station_uncertainty import evaluate_arrays
from neck_localization.experiment_io import read,write,digest,quantiles


def run(source):
    output=source/'partial_view_ablation'
    if output.exists():raise FileExistsError(output)
    output.mkdir()
    original=read(source/'records.json');protocol=read(source/'protocol.json')
    baseline=ROOT/protocol['baseline']
    priors=read(baseline/'band_controls.json')['rows']
    before=read(source/'provenance.json')['source_sha256']
    for p,h in before.items():assert digest(p)==h
    tracked=dict(before)
    for p in [Path(__file__),ROOT/'neck_localization/body_partial_registration.py',source/'records.json']:
        tracked[str(p.resolve())]=digest(p)
    rows=[]
    for control in [r for r in original if r['method']=='nominal']:
        cohort,case=control['cohort'],control['case'];condition=control['condition']
        folder=source/'scenes'/cohort/case
        ids=protocol['body']['stop_errors_m'].index(control['stop_error_mm']/1000)
        targetpath=folder/f'target_{ids}_{condition}.npz'
        cached=np.load(targetpath);q=cached['points_C'];normals=cached['normals_C']
        p=np.load(folder/'recognition_body_templates.npz')[control['source_control']]
        if condition=='lower_strip':
            roi=read(folder/'roi_from_observation.json')['narrow_roi_world_z_m'];p=p[(p[:,2]>roi[0])&(p[:,2]<roi[1])]
        if condition=='upstream_frame_bias_2mm':p=p+[0,0,.002]
        W_T_C=np.array(read(folder/'secondary_cameras.json')[0]['T_world_from_camera'])
        Mnom=pose(translation=[2.4,0,0]);initial=np.array(control['transform'])
        # Reconstruct the actual observed first circle, including the explicitly
        # injected frame error; never replace it with a target truth circle.
        circle=move_circle(control['estimate'],np.linalg.inv(Mnom))
        predictions=[]
        for name,reciprocal in [('observed_subset_axis',False),('reciprocal_axis',True)]:
            start=perf_counter()
            fit=register_observed_subset(p,q,normals,initial,W_T_C[:3,:3].T @ [1,0,0],reciprocal=reciprocal)
            elapsed=(perf_counter()-start)*1000
            world=W_T_C @ fit['transform']
            if condition=='local_extrinsic_1mm':world[:3,3]+=[.001,0,0]
            predictions.append((name,fit,move_circle(circle,world),elapsed))
        # Evaluation begins after both candidate predictions have been frozen.
        base_source=ROOT/read(baseline/'protocol.json')['source_bundle']
        truth=read(base_source/cohort/'ss/clean/scenes'/case/'evaluation_only/target.json')
        spec=read(base_source/cohort/'ss/clean/scenes'/case/'model_manifest.json')['jar_spec']
        R=np.eye(3)
        if condition=='rotation_mismatch_1deg':
            from scipy.spatial.transform import Rotation
            R=Rotation.from_euler('y',1,degrees=True).as_matrix()
        actualM=pose(R,[2.4+control['stop_error_mm']/1000,0,0]);actualtruth=move_circle(truth,actualM)
        for name,fit,estimate,elapsed in predictions:
            value=evaluate_arrays(np.array(estimate['center_m'])[None],np.array(estimate['normal'])[None],np.array([estimate['radius_m']]),
                np.array(actualtruth['center_m'])[None],np.array(actualtruth['normal'])[None],np.array([spec['neck_length_m']]))
            rows.append({k:control[k] for k in ['cohort','case','condition','source_control','stop_error_mm','source_points','target_points','diameter_error_mm']}
                |fit|dict(method=name,estimate=estimate,elapsed_ms=elapsed,**{k:v[0] for k,v in value.items()},final_fastening_success=None))
    write(output/'records.json',rows)
    for p,h in tracked.items():assert digest(p)==h,p
    write(output/'provenance.json',dict(source_sha256=tracked,no_ground_truth_correspondence_or_mask=True,
        frozen_body_inputs=True,change='correspondence_direction_and_optional_reciprocity_only',
        no_method_threshold_change=True,not_independent_jar_test=True))
    print('partial-view ablation complete',len(rows),'records')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    run(p.parse_args().source.resolve())
