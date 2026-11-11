#!/usr/bin/env python3
"""Supplementary simple controls on frozen observations; no truth inputs."""
import argparse
from pathlib import Path
from time import perf_counter
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from neck_localization.fixed_camera_calibration import solve_eye_to_hand, projection
from neck_localization.block_handeye import solve_block_handeye
from neck_localization.experiment_io import read, write, digest


def run(inputs, output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    rows=[]
    for path in sorted((inputs/'measurements/test').glob('*/*/observations.npz')):
        data=np.load(path)
        n=int(data['training_count'])
        A, uv, points, K=(data[k] for k in ('reported_B_T_F', 'identified_pixels', 'nominal_board_points', 'K'))
        for method in ('park', 'fixed_block'):
            started=perf_counter()
            r=(solve_eye_to_hand(A[:n], uv[:n], points, K, method='park') if method=='park'
               else solve_block_handeye(A[:n], uv[:n], points, K))
            X, Y=np.array(r['B_T_C']), np.array(r['F_T_P'])
            errors=[(projection(np.linalg.inv(X) @ a @ Y, points, K)-pixels)**2
                    for a, pixels in zip(A[n:], uv[n:])]
            rows.append(dict(phase='test', seed=int(path.parent.parent.name), condition=path.parent.name,
                             method=method, **r, holdout_pixel_rms=float(np.sqrt(np.mean(errors))),
                             elapsed_ms=(perf_counter()-started)*1000, source_observations=str(path.resolve())))
        print('simple controls', path.parent.parent.name, path.parent.name, flush=True)
    write(output/'records.json', rows)
    write(output/'provenance.json', dict(source_sha256={str(p.resolve()):digest(p) for p in
          [Path(__file__),ROOT/'neck_localization/block_handeye.py',ROOT/'neck_localization/fixed_camera_calibration.py']},
          truth_not_read=True, supplementary_after_main_results=True,
          no_retuning_of_frozen_main_candidates=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    run(a.inputs.resolve(), a.output.resolve())
