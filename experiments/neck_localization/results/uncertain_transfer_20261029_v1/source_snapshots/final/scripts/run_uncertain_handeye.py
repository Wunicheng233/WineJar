#!/usr/bin/env python3
"""Fit observation-only hand-eye alternatives; truth is not imported or read."""
import argparse
from pathlib import Path
import sys
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from neck_localization.fixed_camera_calibration import solve_eye_to_hand,projection
from neck_localization.uncertain_handeye import solve_uncertain_eye_to_hand
from neck_localization.experiment_io import read,write,digest


def run(inputs,output,phase):
    if output.exists():raise FileExistsError(output)
    if not hasattr(cv2,'calibrateHandEye'):raise RuntimeError('Use winejar-vision: this stage needs OpenCV 4 hand-eye API and no renderer')
    output.mkdir(parents=True)
    protocol=read(inputs/'protocol.json');records=[];tracked={}
    for path in sorted((inputs/'measurements'/phase).glob('*/*/observations.npz')):
        tracked[str(path.resolve())]=digest(path)
        data=np.load(path);n=int(data['training_count']);robot=data['reported_B_T_F'];pixels=data['identified_pixels'];points=data['nominal_board_points'];K=data['K']
        for method in protocol['methods']:
            start=perf_counter()
            try:
                if method.startswith('uncertain_'):
                    result=solve_uncertain_eye_to_hand(robot[:n],pixels[:n],points,K,variance_components=method.endswith('vce'))
                else:result=solve_eye_to_hand(robot[:n],pixels[:n],points,K,method=method)
                if 'B_T_C' in result:
                    result['holdout_pixel_rms']=float(np.sqrt(np.mean([(projection(np.linalg.inv(result['B_T_C']) @a @result['F_T_P'],points,K)-uv)**2 for a,uv in zip(robot[n:],pixels[n:])])))
                row=dict(phase=phase,seed=int(path.parent.parent.name),condition=path.parent.name,method=method,
                    **result,elapsed_ms=(perf_counter()-start)*1000,source_observations=str(path.resolve()))
            except (ValueError,cv2.error,np.linalg.LinAlgError) as exc:
                row=dict(phase=phase,seed=int(path.parent.parent.name),condition=path.parent.name,method=method,
                    accepted=False,reason=str(exc),elapsed_ms=(perf_counter()-start)*1000,source_observations=str(path.resolve()))
            records.append(row)
            write(output/'records.partial.json',records)
        print('fit',phase,path.parent.parent.name,path.parent.name,flush=True)
    write(output/'records.json',records)
    tracked.update({str(p.resolve()):digest(p) for p in [Path(__file__),ROOT/'neck_localization/uncertain_handeye.py',inputs/'protocol.json']})
    write(output/'provenance.json',dict(input_and_source_sha256=tracked,opencv_version=cv2.__version__,truth_not_read=True,
        methods_frozen_for_this_run=protocol['methods'],fitted_robot_poses_not_used_for_holdout_pose_predictions=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['development','test'],required=True)
    a=p.parse_args();run(a.inputs.resolve(),a.output.resolve(),a.phase)
