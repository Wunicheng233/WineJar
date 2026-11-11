#!/usr/bin/env python3
"""Explicit oracle depth control for diagnosing the frozen geometry backend.

This command reads ideal simulated IR depth. Its outputs are evaluation-only
upper-bound inputs, never a deployable method or a fair learned prediction.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from neck_localization.baseline_capability import file_sha256, verify_pair


def prepare(manifest_path, output):
    manifest = json.loads(manifest_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    references = {}
    with (output / 'predictions.jsonl').open('w') as stream:
        for pair in manifest['pairs']:
            verify_pair(pair)
            source = (Path(pair['render_root']) / 'sensor_diagnostics' /
                      pair['case'] / (pair['camera'] + '_native.npz'))
            references[str(source.resolve())] = file_sha256(source)
            with np.load(source, allow_pickle=False) as data:
                depth = data['ideal_left_z'].astype(np.float32).copy()
            valid = np.isfinite(depth) & (depth >= .2) & (depth < 10.)
            depth[~valid] = 0
            disparity = np.zeros_like(depth)
            np.divide(pair['K_ir'][0][0] * pair['baseline_m'], depth,
                      out=disparity, where=valid)
            destination = output / 'predictions' / (pair['pair_id'] + '.npz')
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination, native_float_z=depth,
                                disparity_px=disparity, valid_mask=valid)
            stream.write(json.dumps(dict(pair_id=pair['pair_id'], status='ok',
                input_sha256=pair['input_sha256'],
                prediction_sha256=file_sha256(destination),
                evaluation_only_oracle=True)) + '\n')
    provenance = dict(method='EVALUATION ONLY ideal native IR depth oracle',
        oracle=True, deployable=False, no_hardware=True,
        no_ground_truth_input=False,
        interpretation='Backend positive control with identical quantization and registration; not a method success rate',
        manifest_sha256=file_sha256(manifest_path), reference_sha256=references,
        script_sha256=file_sha256(__file__))
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(dict(reference_pairs=len(manifest['pairs']), output=str(output))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.manifest, args.output)
