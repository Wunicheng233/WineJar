#!/usr/bin/env python3
"""Local-camera-specific image calibration composed with body relocation.

Known board-to-gun fixture is an explicit ideal diagnostic metrology prior.
No independent X/Y errors are fabricated: both come from the same images.
"""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from neck_localization.station_tying import pose,frame,move_circle,GunSpec
from neck_localization.fixed_camera_calibration import projection,solve_eye_to_hand
from neck_localization.cross_station_uncertainty import evaluate_arrays
from neck_localization.experiment_io import read,write,digest,quantiles
from scripts.run_fixed_camera_calibration_study import dataset


def run(bundle):
    output=bundle/'combined_chain'
    if output.exists():raise FileExistsError(output)
    output.mkdir()
    protocol=read(bundle/'body/protocol.json');base=ROOT/protocol['baseline'];source=ROOT/read(base/'protocol.json')['source_bundle']
    station=read(source/'config.json');B=np.array(station['B_T_W']);mount=GunSpec(**station['gun']).T_F_G
    camera=read(bundle/'body/scenes/new8/confirmation_nominal_003/secondary_cameras.json')[0]
    K=np.array(camera['K']);W_T_C=np.array(camera['T_world_from_camera']);trueX=B @W_T_C
    inputs=[r for r in read(bundle/'body/partial_view_ablation/records.json') if r['source_control']=='saved_simsense_body' and r['condition'] in ('ideal','depth_bias_3mm') and r['method']=='reciprocal_axis']
    fits=[];rows=[]
    for condition in ('nominal','one_bad_image'):
        for session in range(protocol['calibration']['sessions']):
            exact=dataset(protocol,session,'diverse','ideal');measured=dataset(protocol,session,'diverse',condition)
            # Regenerate corner images for the actual local camera K/resolution.
            U=trueX @ np.linalg.inv(exact['truth_X'])
            robot=U @ measured['robot']
            pixels=np.array([projection(np.linalg.inv(exact['truth_X']) @ b @ exact['truth_Y'],exact['board'],K)
                for b in exact['robot']])+(measured['pixels']-exact['pixels'])
            nominal_jig=np.linalg.inv(exact['truth_Y']) @mount
            for method in ('park','pixel_huber'):
                fit=solve_eye_to_hand(robot[:18],pixels[:18],exact['board'],K,method=method)
                fits.append(fit|dict(condition=condition,session=session,method=method))
                X,Y=fit['B_T_C'],fit['F_T_P']
                for tool_policy,Fhat in [('exact_tool_diagnostic',mount),('joint_board_gun_jig_diagnostic',Y @nominal_jig)]:
                    for r in inputs:
                        desired_C=np.linalg.inv(W_T_C) @pose(frame(r['estimate']['normal'],station['preferred_guide_x_W']),r['estimate']['center_m'])
                        command=X @desired_C @np.linalg.inv(Fhat)
                        actual=np.linalg.inv(B) @command @mount
                        folder=source/r['cohort']/'ss/clean/scenes'/r['case']
                        truth=read(folder/'evaluation_only/target.json');spec=read(folder/'model_manifest.json')['jar_spec']
                        moved=move_circle(truth,pose(translation=[2.4+r['stop_error_mm']/1000,0,0]))
                        v=evaluate_arrays(actual[None,:3,3],actual[None,:3,2],np.array([r['estimate']['radius_m']]),
                            np.array(moved['center_m'])[None],np.array(moved['normal'])[None],np.array([spec['neck_length_m']]))
                        rows.append(dict(calibration_condition=condition,calibration_method=method,session=session,tool_policy=tool_policy,
                            cohort=r['cohort'],case=r['case'],stop_error_mm=r['stop_error_mm'],body_condition=r['condition'],
                            **{k:x[0] for k,x in v.items()},final_fastening_success=None))
    write(output/'calibration_estimates.json',fits);write(output/'records.json',rows)
    summary=[]
    for c in ('nominal','one_bad_image'):
        for m in ('park','pixel_huber'):
            for policy in ('exact_tool_diagnostic','joint_board_gun_jig_diagnostic'):
                for body in ('ideal','depth_bias_3mm'):
                    for level in (.5,2.):
                        a=[r for r in rows if (r['calibration_condition'],r['calibration_method'],r['tool_policy'],r['body_condition'],abs(r['stop_error_mm']))==(c,m,policy,body,level)]
                        summary.append(dict(calibration_condition=c,calibration_method=m,tool_policy=policy,body_condition=body,stop_half_amplitude_mm=level,
                            conditional_draws=len(a),center_mm=quantiles([r['center_mm'] for r in a]),normal_deg=quantiles([r['normal_deg'] for r in a]),
                            band_contained_fraction=float(np.mean([r['band_contained'] for r in a])),center_over_4mm_fraction=float(np.mean([r['center_mm']>4 for r in a])),
                            independent_physical_trials=0,final_fastening_success=None))
    write(output/'summary.json',summary)
    write(output/'provenance.json',dict(script_sha256=digest(__file__),same_local_camera_intrinsics_and_geometry=True,
        X_Y_correlation_preserved=True,exact_rigid_board_to_gun_metrology_prior_not_measured=True,
        no_real_kinematics_tracking_contact_or_hardware_validation=True,
        body_input_records_sha256=digest(bundle/'body/partial_view_ablation/records.json')))
    print('combined image calibration/body/tool chain complete:',len(rows),'conditional records')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True)
    run(p.parse_args().bundle.resolve())
