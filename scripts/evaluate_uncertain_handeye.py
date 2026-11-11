#!/usr/bin/env python3
"""Evaluate held-out calibration and actual perturbed-FK tool arrival.

Truth is read only by this evaluator, never by the calibration solver.
No dynamics/contact/hardware success is inferred from these calculations.
"""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation
from neck_localization.robot_error_model import VirtualRobot
from neck_localization.tying_scene import OfflineArm
from neck_localization.station_tying import pose,frame
from neck_localization.cross_station_uncertainty import evaluate_arrays
from neck_localization.experiment_io import read,write,digest,quantiles


def run(inputs,fits,output):
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    geometry=read(inputs/'geometry.json');protocol=read(inputs/'protocol.json')
    source=ROOT/protocol['source_viewer'];C=np.array(geometry['W_T_C']);B=np.array(geometry['B_T_W'])
    nominal_mount=np.array(geometry['nominal_F_T_G']);xml=(inputs/'robot_nominal.xml').read_text()
    nominal=VirtualRobot(xml);arm=OfflineArm(nominal.model)
    candidates=read(source.parent/'band_controls.json')['rows'];cases=sorted({(r['cohort'],r['case']) for r in candidates})
    tasks=[]
    for cohort,case in cases:
        e=next(r['estimate'] for r in candidates if (r['cohort'],r['case'],r['method'],r['condition'])==(cohort,case,'pose_fs_radius','clean'))
        folder=source/cohort/'fusion/clean/scenes'/case
        truth=read(folder/'evaluation_only/target.json');length=read(folder/'model_manifest.json')['jar_spec']['neck_length_m']
        desired=pose(frame(e['normal'],[0,1,0]),np.array(e['center_m'])+[2.4,0,0])
        seed=np.array(read(folder/'ik_path.json')['samples'][-1]['q_rad'])
        tasks.append(dict(cohort=cohort,case=case,desired=desired,seed=seed,estimate=e,truth=truth,length=length))
    rows=[];camera_rows=[]
    for fit in read(fits/'records.json'):
        if 'B_T_C' not in fit:continue
        true=read(inputs/'evaluation_only'/fit['phase']/str(fit['seed'])/fit['condition']/'truth.json')
        actual=VirtualRobot(xml,true['actual_robot_parameters']);Gtrue=np.array(true['F_T_G_true'])
        data=np.load(fit['source_observations']);jig=data['measured_P_T_G']
        X,Y=np.array(fit['B_T_C']),np.array(fit['F_T_P']);TX=np.array(true['B_T_C'])
        camera_mapping=None
        if 'camera_poe' in fit:
            from neck_localization.poe_camera_calibration import CameraPOE,solve_camera_target
            camera_mapping=CameraPOE(**fit['camera_poe'])
        query=np.array([[0,0,.5],[.1,.1,.8],[-.1,-.1,.5],[0,0,.8]])
        a=query @X[:3,:3].T+X[:3,3];b=query @TX[:3,:3].T+TX[:3,3]
        camera_rows.append({k:fit[k] for k in ('phase','seed','condition','method','accepted','pixel_rms','holdout_pixel_rms','elapsed_ms')}|
            dict(query_max_mm=float(np.linalg.norm(a-b,axis=1).max()*1000) if camera_mapping is None else None,
                initial_X_Y_not_runtime_mapping=camera_mapping is not None,estimated_sigmas=fit.get('estimated_sigmas')))
        for task in tasks:
            for policy,Fhat in [('nominal_mount',nominal_mount),('measured_board_jig',Y @jig)]:
                if camera_mapping is not None and policy=='nominal_mount':continue
                command=X @np.linalg.inv(C) @task['desired'] @np.linalg.inv(Fhat)
                target=np.linalg.inv(B) @command @nominal_mount
                if camera_mapping is None:
                    solved=arm.solve(target,task['seed'])
                else:
                    target_C_P=np.linalg.inv(C) @task['desired'] @np.linalg.inv(jig)
                    solved=solve_camera_target(camera_mapping,target_C_P,task['seed'],arm.limits)
                q=np.array(solved['q_rad'])
                arrived=actual.forward(q) @Gtrue
                residual=arrived[:3,3]-task['desired'][:3,3]
                tv=task['truth'];tc=np.array(tv['center_m'])+[2.4,0,0]
                value=evaluate_arrays(arrived[None,:3,3],arrived[None,:3,2],np.array([task['estimate']['radius_m']]),
                    tc[None],np.array(tv['normal'])[None],np.array([task['length']]))
                row={k:fit[k] for k in ('phase','seed','condition','method','accepted')}
                row.update(cohort=task['cohort'],case=task['case'],tool_policy=policy,
                    nominal_IK_passed=solved['passed'],q_command=q,
                    actual_guide_W=arrived,calibration_and_execution_only_mm=float(np.linalg.norm(residual)*1000),
                    **{k:v[0] for k,v in value.items()},final_fastening_success=None)
                if 'joint_X_Y_covariance' in fit and policy=='measured_board_jig':
                    cov=np.array(fit['joint_X_Y_covariance']);direction=np.linalg.inv(C) @task['desired']
                    def prediction(d):
                        x=pose(Rotation.from_rotvec(d[3:6]).as_matrix() @X[:3,:3],X[:3,3]+d[:3])
                        y=pose(Rotation.from_rotvec(d[9:12]).as_matrix() @Y[:3,:3],Y[:3,3]+d[6:9])
                        j=pose(Rotation.from_rotvec(d[15:18]).as_matrix() @jig[:3,:3],jig[:3,3]+d[12:15])
                        return (np.linalg.inv(B) @x @direction @np.linalg.inv(y @j) @Fhat)[:3,3]
                    Jac=np.empty((3,18));step=1e-6
                    for k in range(18):
                        d=np.zeros(18);d[k]=step;Jac[:,k]=(prediction(d)-prediction(-d))/(2*step)
                    allcov=np.zeros((18,18));allcov[:12,:12]=cov
                    allcov[12:,12:]=np.diag([.0001**2]*3+[np.deg2rad(.03)**2]*3)
                    pcov=Jac @allcov @Jac.T
                    row.update(conditional_center_covariance_m2=pcov,
                        conditional_95_ellipsoid_contains_error=bool(residual @np.linalg.pinv(pcov,hermitian=True) @residual<=7.814727903),
                        conditional_radius95_upper_mm=float(np.sqrt(np.linalg.eigvalsh(pcov).max()*7.814727903)*1000),
                        covariance_excludes_structured_robot_bias_and_upstream_localization_error=True)
                rows.append(row)
        write(output/'records.partial.json',rows)
        print('evaluated',fit['phase'],fit['seed'],fit['condition'],fit['method'],flush=True)
    write(output/'records.json',rows);write(output/'camera_records.json',camera_rows)
    summary=[]
    for cond in protocol['conditions']:
        for method in sorted({r['method'] for r in rows}):
            for policy in ('nominal_mount','measured_board_jig'):
                g=[r for r in rows if (r['condition'],r['method'],r['tool_policy'])==(cond,method,policy)]
                if not g:continue
                conditional=[r for r in g if 'conditional_95_ellipsoid_contains_error' in r]
                summary.append(dict(condition=cond,method=method,tool_policy=policy,total=len(g),
                    IK_success=sum(r['nominal_IK_passed'] for r in g),calibration_candidate_accepted=sum(r['accepted'] for r in g),
                    **{k:quantiles([r[k] for r in g]) for k in ('center_mm','normal_deg','calibration_and_execution_only_mm','band_margin_mm')},
                    band_contained=sum(r['band_contained'] for r in g),
                    covariance_coverage95=None if not conditional else float(np.mean([r['conditional_95_ellipsoid_contains_error'] for r in conditional])),
                    final_fastening_success=None))
    write(output/'summary.json',summary)
    write(output/'provenance.json',dict(source_sha256={str(p.resolve()):digest(p) for p in [Path(__file__),fits/'records.json',inputs/'geometry.json',ROOT/'neck_localization/robot_error_model.py']},
        actual_forward_kinematics_evaluated_after_nominal_IK=True,truth_only_for_scoring=True,no_hardware_or_dynamics=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--fits',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.inputs.resolve(),a.fits.resolve(),a.output.resolve())
