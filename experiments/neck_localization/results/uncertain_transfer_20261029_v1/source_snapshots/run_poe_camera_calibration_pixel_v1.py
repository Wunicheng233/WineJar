#!/usr/bin/env python3
"""Observation-only PoE camera/encoder fit, preserving matrix baselines."""
import argparse
from pathlib import Path
from time import perf_counter
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from neck_localization.poe_camera_calibration import fit_camera_poe,CameraPOE
from neck_localization.fixed_camera_calibration import solve_eye_to_hand,projection
from neck_localization.experiment_io import read,write,digest

def run(inputs,output,phase):
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);geometry=read(inputs/'nominal_poe.json');records=[]
    for path in sorted((inputs/'measurements'/phase).glob('*/*/observations.npz')):
        if path.parent.name in ('random_pose',):continue
        a=np.load(path);n=int(a['training_count']);start=perf_counter()
        initial=solve_eye_to_hand(a['reported_B_T_F'][:n],a['identified_pixels'][:n],a['nominal_board_points'],a['K'],method='pixel_huber')
        result=fit_camera_poe(a['q_reported'][:n],a['identified_pixels'][:n],a['nominal_board_points'],a['K'],
            np.array(geometry['axes_B']),np.array(geometry['points_B']),np.array(geometry['B_T_F0']),initial['B_T_C'],initial['F_T_P'])
        model=CameraPOE(**result['camera_poe']);T=model.forward(a['q_reported'][n:])
        result['holdout_pixel_rms']=float(np.sqrt(np.mean([(projection(t,a['nominal_board_points'],a['K'])-uv)**2 for t,uv in zip(T,a['identified_pixels'][n:])])) )
        # X/Y are initialization diagnostics only; runtime uses camera_poe.
        result.update(B_T_C=initial['B_T_C'],F_T_P=initial['F_T_P'],initial_X_Y_are_not_runtime_calibration=True)
        records.append(dict(phase=phase,seed=int(path.parent.parent.name),condition=path.parent.name,method='camera_poe',**result,
            elapsed_ms=(perf_counter()-start)*1000,source_observations=str(path.resolve())))
        write(output/'records.partial.json',records);print('PoE',phase,path.parent.parent.name,path.parent.name,'evals',result['evaluations'],flush=True)
    write(output/'records.json',records)
    write(output/'provenance.json',dict(source_sha256={str(p.resolve()):digest(p) for p in [Path(__file__),ROOT/'neck_localization/poe_camera_calibration.py',inputs/'nominal_poe.json']},
        evaluation_truth_not_read=True,encoder_inputs_available_to_all_paired_geometric_conditions=True,
        random_Cartesian_pose_noise_condition_excluded_due_to_different_encoder_information=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['development','test'],required=True)
    a=p.parse_args();run(a.inputs.resolve(),a.output.resolve(),a.phase)
