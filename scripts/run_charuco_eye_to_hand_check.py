#!/usr/bin/env python3
"""Actual OpenCV ChArUco extraction on homographically rendered board images.

An image-extraction diagnostic, not photorealistic optics, MuJoCo rendering,
physical board metrology, or an IK-approved robot calibration trajectory.
"""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from neck_localization.fixed_camera_calibration import solve_eye_to_hand,projection,transform_points
from neck_localization.experiment_io import read,write,digest,quantiles
from scripts.run_fixed_camera_calibration_study import dataset


def run(config,output):
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    protocol=read(config)
    d=dataset(protocol,0,'diverse','ideal')
    board=cv2.aruco.CharucoBoard((8,6),.020,.014,cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    texture=board.generateImage((1024,768))
    detector=cv2.aruco.CharucoDetector(board)
    # P uses +Y up, +Z towards printed front. Convert known ChArUco coordinates,
    # whose image convention has +Y down, by a proper 180deg X rotation.
    nominal=(board.getChessboardCorners().astype(float)-[.08,.06,0]) @ np.diag([1.,-1.,-1.])
    outer=np.array([[-.08,.06,0],[.08,.06,0],[.08,-.06,0],[-.08,-.06,0]])
    image_corners=np.float32([[-.5,-.5],[1023.5,-.5],[1023.5,767.5],[-.5,767.5]])
    records=[];robots=[];pixels=[];corner_errors=[]
    for i,B in enumerate(d['robot']):
        T=np.linalg.inv(d['truth_X']) @ B @ d['truth_Y']
        hi=d['K'].copy();hi[:2,:]*=4;hi[:2,2]+=1.5
        projected=projection(T,outer,hi).astype(np.float32)
        H=cv2.getPerspectiveTransform(image_corners,projected)
        high=cv2.warpPerspective(texture,H,(848*4,480*4),flags=cv2.INTER_LINEAR,borderValue=180)
        gray=cv2.resize(high,(848,480),interpolation=cv2.INTER_AREA)
        corners,ids,markers,markerids=detector.detectBoard(gray)
        cv2.imwrite(str(output/f'{i:02}_image.png'),gray)
        found=0 if ids is None else len(ids)
        records.append(dict(frame=i,identified_corners=found,used=found==35))
        if found!=35:continue
        order=np.argsort(ids.ravel());detected=corners.reshape(-1,2)[order]
        evaluation_pixels=projection(T,nominal,d['K'])
        corner_errors.extend(np.linalg.norm(detected-evaluation_pixels,axis=1))
        robots.append(B);pixels.append(detected)
        preview=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
        cv2.aruco.drawDetectedCornersCharuco(preview,corners,ids)
        cv2.imwrite(str(output/f'{i:02}_detected.png'),preview)
    if len(pixels)<22:raise RuntimeError(f'Only {len(pixels)}/26 fully detected views; retained raw images for diagnosis')
    np.savez_compressed(output/'measurements.npz',B_T_F_reported=robots,detected_pixels=pixels,nominal_board=nominal,K=d['K'])
    results=[]
    for method in ['park','horaud','pixel_ls','pixel_huber']:
        r=solve_eye_to_hand(robots[:18],pixels[:18],nominal,d['K'],method=method)
        X,Y=r['B_T_C'],r['F_T_P']
        e=transform_points(X,d['query_points_camera'])-transform_points(d['truth_X'],d['query_points_camera'])
        r.update(method=method,physical_query_max_mm=float(np.linalg.norm(e,axis=1).max()*1000),
            holdout_pixel_rms=float(np.sqrt(np.mean([(projection(np.linalg.inv(X) @ B @ Y,nominal,d['K'])-uv)**2 for B,uv in zip(robots[18:],pixels[18:])]))))
        results.append(r)
    write(output/'result.json',dict(frames=records,full_detection_frames=len(pixels),corner_error_px=quantiles(corner_errors),
        methods=results,renderer='analytic_planar_homography_4x_supersampling_not_sensor_simulation',
        hardware_or_training=False,nominal_intrinsics_and_robot_poses_exact=True))
    write(output/'provenance.json',dict(source_sha256={str(p.resolve()):digest(p) for p in [config,Path(__file__),ROOT/'neck_localization/fixed_camera_calibration.py',ROOT/'scripts/run_fixed_camera_calibration_study.py']},opencv=cv2.__version__))
    print('ChArUco actual image extraction complete:',len(pixels),'frames')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'experiments/neck_localization/configs/pose_transfer_20261028_v1.json')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.config.resolve(),a.output.resolve())
