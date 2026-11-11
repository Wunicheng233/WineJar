#!/usr/bin/env python3
"""Fit per-camera author-adapted GP on calibration planes, evaluate held-out planes."""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from neck_localization.depth_correction import (GPSettings, GaussianProcessDepthCorrection,
    calibration_samples, load_calibration_manifest, UPSTREAM_COMMIT)


def error_statistics(error_m):
    if not len(error_m):
        return None
    e = np.asarray(error_m)*1000
    return dict(count=len(e), signed_mean_mm=float(e.mean()), rmse_mm=float(np.sqrt(np.mean(e*e))),
                abs_median_mm=float(np.median(abs(e))), abs_p95_mm=float(np.percentile(abs(e), 95)),
                abs_max_mm=float(np.max(abs(e))))


def evaluate_frames(frames, models, *, evaluation_splits=('train', 'validation', 'test')):
    """Evaluate saved/frozen models; no fitting or model selection happens here."""
    rows = []
    for f in frames:
        if f['split'] not in evaluation_splits:
            continue
        model = models[f['camera_id']]
        before = f['measured_depth']
        start = time.perf_counter()
        after, diagnostic = model.correct_depth(before, f['K'], f['temperature_c'], return_diagnostics=True)
        seconds = time.perf_counter()-start
        ref = f['reference_depth']
        reference_valid = np.isfinite(ref) & (ref > 0)
        if 'valid_mask' in f:
            reference_valid &= f['valid_mask'].astype(bool)
        good = np.isfinite(before) & (before > 0) & reference_valid
        corrected_good = good & np.isfinite(after) & (after > 0)
        # Same measured-valid pixels for paired error, invalidation separately.
        paired = corrected_good
        row = dict(camera_id=f['camera_id'], frame_id=f['frame_id'], split=f['split'],
                   raw=error_statistics(before[paired]-ref[paired]),
                   corrected=error_statistics(after[paired]-ref[paired]),
                   paired_valid_pixels=int(paired.sum()), raw_valid_pixels=int(good.sum()),
                   reference_valid_pixels=int(reference_valid.sum()),
                   raw_valid_fraction=float(good.sum()/reference_valid.sum()) if reference_valid.any() else None,
                   corrected_valid_fraction=float(corrected_good.sum()/reference_valid.sum()) if reference_valid.any() else None,
                   inference_seconds=seconds, diagnostics=diagnostic,
                   measured_sha256=sha256(Path(f['measured_depth_path']).read_bytes()).hexdigest(),
                   reference_sha256=sha256(Path(f['reference_depth_path']).read_bytes()).hexdigest())
        rows.append(row)
        print(f['camera_id'], f['split'], f['frame_id'], 'rmse_mm',
              row['raw']['rmse_mm'] if row['raw'] else None,
              row['corrected']['rmse_mm'] if row['corrected'] else None, flush=True)
    return rows


def fit_manifest(manifest_path, output, settings=GPSettings(), *, evaluation_splits=('train', 'validation', 'test')):
    manifest_path, output = Path(manifest_path), Path(output)
    manifest, frames = load_calibration_manifest(manifest_path)
    output.mkdir(parents=True, exist_ok=False)
    (output/'models').mkdir()
    report = dict(upstream='https://github.com/cheind/rgbd-correction', upstream_commit=UPSTREAM_COMMIT,
        implementation='BSD-author-adapted exact ARD-RBF GP; not Basso full calibration',
        manifest_sha256=sha256(manifest_path.read_bytes()).hexdigest(),
        reference_source=manifest['reference_source'], settings=asdict(settings), cameras={}, frames=[],
        evaluation_splits=list(evaluation_splits))
    models = {}
    for camera_id in sorted({f['camera_id'] for f in frames}):
        local = [f for f in frames if f['camera_id'] == camera_id]
        if not camera_id or any(x not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for x in camera_id):
            raise ValueError('camera_id must be a safe nonempty filename stem')
        train = [f for f in local if f['split'] == 'train']
        if not train:
            raise ValueError(f'No calibration training frames for {camera_id}')
        if any(not np.allclose(f['K'], train[0]['K'], rtol=0, atol=1e-9) for f in local):
            raise ValueError('Per-camera intrinsics must remain constant across calibration splits')
        samples = [calibration_samples(f['measured_depth'], f['reference_depth'], f['K'],
            temperature_c=f['temperature_c'], mask=f.get('valid_mask'), count=settings.samples_per_frame) for f in train]
        X, y = np.concatenate([s[0] for s in samples]), np.concatenate([s[1] for s in samples])
        if len(y) > settings.max_training_points:
            selected = np.random.default_rng(settings.seed).choice(len(y), settings.max_training_points, replace=False)
            X, y = X[selected], y[selected]
        start = time.perf_counter()
        model = GaussianProcessDepthCorrection(settings).fit(X, y, K=train[0]['K'],
            metadata=dict(camera_id=camera_id, training_frame_ids=[f['frame_id'] for f in train],
                training_reference_source=manifest['reference_source'],
                training_inputs_sha256={f['frame_id']:sha256(Path(f['measured_depth_path']).read_bytes()).hexdigest() for f in train}))
        model.save(output/'models'/f'{camera_id}.npz')
        models[camera_id] = model
        report['cameras'][camera_id] = dict(training_points=len(y), training_frames=len(train),
            fit_seconds=time.perf_counter()-start, length_scale=model.length_scale.tolist(),
            signal_std_mm=model.signal_std_m*1000, noise_std_mm=model.noise_std_m*1000, **model.metadata)
        print(camera_id, 'fit', len(y), 'points', report['cameras'][camera_id]['optimizer'], flush=True)
    report['frames'] = evaluate_frames(frames, models, evaluation_splits=evaluation_splits)
    (output/'calibration_report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-training-points', type=int, default=512)
    p.add_argument('--samples-per-frame', type=int, default=64)
    p.add_argument('--max-iterations', type=int, default=70)
    p.add_argument('--seed', type=int, default=20261024)
    p.add_argument('--fixed-hyperparameters', action='store_true')
    p.add_argument('--evaluation-splits', nargs='+', choices=['train', 'validation', 'test'],
                   default=['train', 'validation', 'test'])
    p.add_argument('--evaluate-models', type=Path,
                   help='Evaluate existing frozen model directory; do not refit')
    args = p.parse_args()
    if args.evaluate_models:
        manifest, frames = load_calibration_manifest(args.manifest)
        models = {name:GaussianProcessDepthCorrection.load(args.evaluate_models/f'{name}.npz')
                  for name in {f['camera_id'] for f in frames}}
        args.output.mkdir(parents=True, exist_ok=False)
        report = dict(phase='frozen_model_evaluation', manifest_sha256=sha256(args.manifest.read_bytes()).hexdigest(),
                      reference_source=manifest['reference_source'],
                      models_sha256={name:sha256((args.evaluate_models/f'{name}.npz').read_bytes()).hexdigest() for name in models},
                      frames=evaluate_frames(frames, models, evaluation_splits=args.evaluation_splits))
        (args.output/'calibration_report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    else:
        fit_manifest(args.manifest, args.output, GPSettings(max_training_points=args.max_training_points,
            samples_per_frame=args.samples_per_frame, max_iterations=args.max_iterations, seed=args.seed,
            optimize=not args.fixed_hyperparameters), evaluation_splits=args.evaluation_splits)


if __name__ == '__main__':
    main()
