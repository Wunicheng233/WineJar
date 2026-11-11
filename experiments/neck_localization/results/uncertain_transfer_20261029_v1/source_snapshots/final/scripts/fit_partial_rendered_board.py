#!/usr/bin/env python3
"""Detect partial ChArUco IDs in saved RGB and fit two frozen method families."""
import argparse
from pathlib import Path
from time import perf_counter
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from neck_localization.partial_board_calibration import fit_partial_board
from neck_localization.fixed_camera_calibration import projection
from neck_localization.poe_camera_calibration import CameraPOE
from neck_localization.experiment_io import read,write,digest
from scripts.check_rendered_robot_board import board_definition


def run(inputs,rendered,output):
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);detector=cv2.aruco.CharucoDetector(board_definition())
    source=read(rendered/'rendered/observations.json');points=np.array(source['nominal_points']);K=np.array(source['K']);frames=[];exclusions=[]
    for f in source['frames']:
        image=cv2.imread(str(rendered/'rendered'/f"{f['frame']:02}.png"));corners,ids,_,_=detector.detectBoard(image)
        if ids is None or len(ids)<8:
            exclusions.append(dict(frame=f['frame'],reason='fewer_than_8_identified_corners'));continue
        ids=ids.ravel();uv=corners.reshape(-1,2).copy();board=points[ids]
        if np.linalg.matrix_rank(board-board.mean(0),tol=1e-9)<2:
            exclusions.append(dict(frame=f['frame'],reason='collinear_detected_corners'));continue
        cv2.cornerSubPix(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),uv,(3,3),(-1,-1),
            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,50,.0001))
        frames.append(f|dict(points_P_m=board,identified_pixels=uv,ids=ids))
        cv2.aruco.drawDetectedCornersCharuco(image,uv[:,None],ids[:,None])
        cv2.imwrite(str(output/f"{f['frame']:02}_detected.png"),image)
    if len(frames)<16:raise RuntimeError(f'only {len(frames)} usable observations, preserve failure')
    # Fixed chronological 12-frame calibration, remaining views never optimized.
    n=12;write(output/'observations.json',dict(training=frames[:n],holdout=frames[n:],excluded=exclusions,K=K))
    nominal=read(inputs/'nominal_poe.json');records=[]
    for method,kinematic in [('pixel_huber',False),('camera_poe_block',True)]:
        started=perf_counter();r=fit_partial_board(frames[:n],K,nominal,kinematic=kinematic)
        T=CameraPOE(**r['camera_poe']).forward(np.array([f['q_reported'] for f in frames[n:]])) if kinematic else np.linalg.inv(r['B_T_C']) @np.array([f['B_T_F_reported'] for f in frames[n:]]) @r['F_T_P']
        errors=np.concatenate([(projection(t,f['points_P_m'],K)-f['identified_pixels']).ravel() for t,f in zip(T,frames[n:])])
        records.append(dict(phase='test',seed=2026102981,condition='kinematic_5',method=method,**r,
            source_observations=str((inputs/'measurements/test/2026102981/kinematic_5/observations.npz').resolve()),
            source_rendered_board=True,holdout_pixel_rms=float(np.sqrt(np.mean(errors**2))),elapsed_ms=(perf_counter()-started)*1000))
    write(output/'records.json',records)
    errors=[]
    for f in frames:
        true=np.array(read(rendered/'rendered'/f"{f['frame']:02}_evaluation_only.json")['projected_true_corners'])
        errors.extend(np.linalg.norm(f['identified_pixels']-true[f['ids']],axis=1))
    write(output/'summary.json',dict(training_frames=n,holdout_frames=len(frames)-n,excluded_frames=len(exclusions),
        corner_error_px=dict(median=float(np.median(errors)),p95=float(np.percentile(errors,95)),max=float(np.max(errors))),
        fit_summary=[{k:r[k] for k in ('method','accepted','pixel_rms','holdout_pixel_rms','elapsed_ms')} for r in records],
        no_added_pixel_Gaussian_noise_or_gt_pixel_replacement=True))
    paths=[Path(__file__),ROOT/'neck_localization/partial_board_calibration.py',rendered/'rendered/observations.json',rendered/'board_points.json']+list((rendered/'rendered').glob('*.png'))
    write(output/'provenance.json',dict(source_sha256={str(p.resolve()):digest(p) for p in paths},
        truth_not_loaded_until_models_frozen=True,missing_corners_never_filled=True,scene_and_intrinsics_still_ideal_rendered=True))
    print('partial RGB board:',len(frames),'views',len(exclusions),'excluded',[(r['method'],r['holdout_pixel_rms']) for r in records])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--rendered',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.inputs.resolve(),a.rendered.resolve(),a.output.resolve())
