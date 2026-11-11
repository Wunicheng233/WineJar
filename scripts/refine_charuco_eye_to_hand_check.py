#!/usr/bin/env python3
"""Paired image-gradient refinement, not a fixed truth-derived pixel offset."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from neck_localization.experiment_io import read,write,digest,quantiles
from neck_localization.fixed_camera_calibration import solve_eye_to_hand,projection,transform_points
from scripts.run_fixed_camera_calibration_study import dataset


def run(source):
    raw=np.load(source/'measurements.npz');meta=read(source/'result.json')
    used=[r['frame'] for r in meta['frames'] if r['used']]
    refined=[]
    for frame,points in zip(used,raw['detected_pixels']):
        image=cv2.imread(str(source/f'{frame:02}_image.png'),cv2.IMREAD_GRAYSCALE)
        out=cv2.cornerSubPix(image,points.astype(np.float32).reshape(-1,1,2).copy(),(3,3),(-1,-1),
                            (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER,50,1e-4))
        refined.append(out.reshape(-1,2))
    # All estimates are frozen before any reference pixel or pose is evaluated.
    results=[]
    for method in ['park','horaud','pixel_ls','pixel_huber']:
        r=solve_eye_to_hand(raw['B_T_F_reported'][:18],refined[:18],raw['nominal_board'],raw['K'],method=method)
        results.append(r|dict(method=method))
    config=ROOT/'experiments/neck_localization/configs/pose_transfer_20261028_v1.json'
    d=dataset(read(config),0,'diverse','ideal')
    errors=[]
    for i,pixels in enumerate(refined):
        T=np.linalg.inv(d['truth_X']) @ raw['B_T_F_reported'][i] @ d['truth_Y']
        errors.extend(np.linalg.norm(pixels-projection(T,raw['nominal_board'],raw['K']),axis=1))
    for r in results:
        X,Y=r['B_T_C'],r['F_T_P']
        e=transform_points(X,d['query_points_camera'])-transform_points(d['truth_X'],d['query_points_camera'])
        r['physical_query_max_mm']=float(np.linalg.norm(e,axis=1).max()*1000)
        r['holdout_pixel_rms']=float(np.sqrt(np.mean([(projection(np.linalg.inv(X) @ B @ Y,raw['nominal_board'],raw['K'])-uv)**2 for B,uv in zip(raw['B_T_F_reported'][18:],refined[18:])])) )
    np.savez_compressed(source/'gradient_refined_pixels.npz',pixels=refined)
    write(source/'gradient_refinement.json',dict(methods=results,corner_error_px=quantiles(errors),
        operation='cv2.cornerSubPix_3x3_image_gradients_no_constant_pixel_shift',
        same_images_ids_poses_intrinsics_and_split=True,not_real_sensor_accuracy=True,
        script_sha256=digest(__file__),input_measurements_sha256=digest(source/'measurements.npz')))
    print('paired gradient refinement complete')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    run(p.parse_args().source.resolve())
