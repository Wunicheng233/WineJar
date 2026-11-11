#!/usr/bin/env python3
"""Projected-corner fixed-camera calibration; not real calibration or motion."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from neck_localization.fixed_camera_calibration import solve_eye_to_hand, projection, transform_points
from neck_localization.geometry import look_at
from neck_localization.station_tying import pose
from neck_localization.experiment_io import read, write, digest, quantiles


def dataset(protocol, session, layout, condition):
    c = protocol['calibration']
    rng = np.random.default_rng(np.random.SeedSequence([protocol['seed'], session]))
    n = c['training_poses']+c['holdout_poses']
    # Local camera near the tying robot, NOT a board seen from 2.4m away.
    X = look_at('local', [.6,-.35,.6], [0,.05,.3], 848, 480, 45).T_world_from_camera
    Y = pose(Rotation.from_euler('xyz', [5,-8,12], degrees=True).as_matrix(), [.03,.01,.08])
    K = np.array([[700.,0,423.5],[0,700.,239.5],[0,0,1.]])
    a, b = np.meshgrid(np.arange(7)-3, np.arange(5)-2)
    points = np.c_[a.ravel(), b.ravel(), np.zeros(a.size)]*.020
    physical = points.copy()
    if condition == 'board_scale': physical *= 1+c['assumed_board_scale_error']
    if condition == 'board_warp':
        physical[:,2] = c['assumed_board_warp_m']*np.cos(physical[:,0]/.06*np.pi/2)*np.cos(physical[:,1]/.04*np.pi/2)
    angles = rng.uniform(-25,25,(n,3))
    if layout == 'small_rotation': angles *= .04
    if layout == 'one_axis': angles[:,:2] = 0
    shifts = rng.uniform([-.07,-.05,.52],[.07,.05,.78],(n,3))
    camera_board = [pose(Rotation.from_euler('xyz', angle, degrees=True).as_matrix() @ np.diag([1.,-1.,-1.]), t) for angle,t in zip(angles,shifts)]
    actual_robot = np.array([X @ t @ np.linalg.inv(Y) for t in camera_board])
    eps_pixel = rng.normal(size=(n,len(points),2))
    eps_t, eps_r = rng.normal(size=(n,3)), rng.normal(size=(n,3))
    noisy = condition not in ('ideal','gauge_relabel')
    reported, pixels = [], []
    relabel = pose(Rotation.from_euler('z', .2, degrees=True).as_matrix(), [.001,-.002,.0005]) if condition == 'gauge_relabel' else np.eye(4)
    for i,(T,B) in enumerate(zip(camera_board, actual_robot)):
        uv = projection(T, physical, K)+eps_pixel[i]*c['assumed_pixel_std']*noisy
        if condition == 'one_bad_image' and i == 5: uv += c['assumed_bad_image_shift_px']
        obs = B.copy()
        obs[:3,3] += eps_t[i]*c['assumed_robot_translation_std_m']*noisy
        obs[:3,:3] = Rotation.from_rotvec(eps_r[i]*np.deg2rad(c['assumed_robot_rotation_std_deg'])*noisy).as_matrix() @ obs[:3,:3]
        if condition == 'robot_pose_dependence':
            obs[:3,3] += c['assumed_robot_pose_dependent_amplitude_m']*np.sin(np.deg2rad(angles[i])*5)
        reported.append(relabel @ obs)
        pixels.append(uv)
    K_est = K.copy()
    if condition == 'wrong_focal': K_est[0,0] *= 1+c['assumed_focal_relative_error']
    return dict(robot=np.array(reported), pixels=np.array(pixels), board=points, K=K_est,
        truth_X=X, truth_Y=Y, physical_board=physical, gauge_relabel=relabel,
        query_points_camera=np.array([[-.09,-.05,.5],[.08,.04,.7],[0,0,.6],[.04,-.03,.8]]))


def run(config, output):
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    protocol = read(config)
    tracked = {str(p.resolve()):digest(p) for p in (config, Path(__file__), ROOT/'neck_localization/fixed_camera_calibration.py')}
    write(output/'protocol.json', protocol)
    records = []
    c = protocol['calibration']
    for layout in c['layouts']:
        for condition in c['conditions']:
            for session in range(c['sessions']):
                d = dataset(protocol, session, layout, condition)
                # Estimator receives only nominal board, pixels, K, robot data.
                n = c['training_poses']
                folder = output/'measurements'/layout/condition
                folder.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(folder/f'{session:02}.npz', B_T_F_reported=d['robot'], identified_pixels=d['pixels'],
                                    nominal_board=d['board'], K_est=d['K'])
                for method in c['methods']:
                    started=perf_counter()
                    try:
                        result = solve_eye_to_hand(d['robot'][:n],d['pixels'][:n],d['board'],d['K'],method=method)
                        row = dict(layout=layout,condition=condition,session=session,method=method,**result)
                        if 'B_T_C' in result:
                            X,Y = result['B_T_C'],result['F_T_P']
                            # Independent physical points, not fitted board corners.
                            physical_X = np.linalg.inv(d['gauge_relabel']) @ X
                            difference=transform_points(physical_X,d['query_points_camera'])-transform_points(d['truth_X'],d['query_points_camera'])
                            row['physical_query_max_mm']=float(np.linalg.norm(difference,axis=1).max()*1000)
                            row['physical_rotation_error_deg']=float(np.rad2deg(Rotation.from_matrix(physical_X[:3,:3] @ d['truth_X'][:3,:3].T).magnitude()))
                            row['raw_parameter_translation_difference_mm']=float(np.linalg.norm(X[:3,3]-d['truth_X'][:3,3])*1000)
                            row['holdout_pixel_rms']=float(np.sqrt(np.mean([(projection(np.linalg.inv(X) @ b @ Y,d['board'],d['K'])-uv)**2 for b,uv in zip(d['robot'][n:],d['pixels'][n:])])))
                        row['elapsed_ms']=(perf_counter()-started)*1000
                    except (ValueError,cv2.error,np.linalg.LinAlgError) as exc:
                        row=dict(layout=layout,condition=condition,session=session,method=method,accepted=False,reason=str(exc),elapsed_ms=(perf_counter()-started)*1000)
                    records.append(row)
            print('fixed-camera',layout,condition,'complete',flush=True)
    write(output/'records.json',records)
    summary=[]
    for layout in c['layouts']:
        for condition in c['conditions']:
            for method in c['methods']:
                rows=[r for r in records if (r['layout'],r['condition'],r['method'])==(layout,condition,method)]
                solved=[r for r in rows if 'physical_query_max_mm' in r]
                summary.append(dict(layout=layout,condition=condition,method=method,total=len(rows),accepted=sum(r['accepted'] for r in rows),
                    **{key:quantiles([r[key] for r in solved]) if solved else None for key in ('physical_query_max_mm','physical_rotation_error_deg','holdout_pixel_rms','pixel_rms','elapsed_ms')}))
    write(output/'summary.json',summary)
    for p,h in tracked.items(): assert digest(p)==h
    write(output/'provenance.json',dict(source_sha256=tracked,opencv_version=cv2.__version__,identified_corner_diagnostic=True,no_devices_or_calibration_change=True,
        robot_pose_candidates_not_IK_checked=True,all_error_amplitudes_assumed=True,training_and_holdout_robot_poses_separate=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'experiments/neck_localization/configs/pose_transfer_20261028_v1.json')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.config.resolve(),a.output.resolve())
