#!/usr/bin/env python3
"""Freeze identical clean/shared-noise IR for SimSense and FoundationStereo."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from neck_localization.baseline_capability import file_sha256, shared_ir_noise


def prepare(source, development, output, layouts):
    output.mkdir(parents=True, exist_ok=False)
    sources = [('development', development)] if development else []
    for layout in layouts:
        mapping = source / layout / 'input_source.json'
        name = json.loads(mapping.read_text())['render_directory'] if mapping.exists() else 'render'
        sources.append((layout, source / layout / name))
    pairs, source_hashes = [], {}
    for layout_idx, (layout, rendered) in enumerate(sources):
        rendered = rendered.resolve()
        config = json.loads((rendered / 'effective_config.json').read_text())
        source_hashes[str(rendered / 'effective_config.json')] = file_sha256(rendered / 'effective_config.json')
        for case_idx, case in enumerate(config['cases']):
            name = case['name']
            sensor_path = rendered / 'geometry' / name / 'sensor.json'
            sensor = json.loads(sensor_path.read_text())
            source_hashes[str(sensor_path)] = file_sha256(sensor_path)
            for camera_idx, camera in enumerate(sensor['cameras']):
                cam = camera['name']
                paths = [rendered / 'sensor_diagnostics' / name / f'{cam}_ir_clean_{j}.png' for j in range(2)]
                images = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths]
                if any(im is None for im in images) or images[0].shape != images[1].shape:
                    raise ValueError(f'Missing or mismatched IR pair: {layout}/{name}/{cam}')
                for condition in ('clean', 'shared_noise'):
                    seed = 2026102601 + layout_idx * 10000 + case_idx * 100 + camera_idx * 2
                    selected = paths
                    if condition == 'shared_noise':
                        folder = output / 'ir' / layout / name / cam
                        folder.mkdir(parents=True)
                        selected = [folder / f'{j}.png' for j in range(2)]
                        for j in range(2):
                            if not cv2.imwrite(str(selected[j]), shared_ir_noise(images[j], seed + j)):
                                raise OSError('IR write failed')
                    pairs.append(dict(pair_id=f'{layout}/{name}/{cam}/{condition}', layout=layout,
                        case=name, camera=cam, condition=condition,
                        split='development' if layout == 'development' else 'previously_viewed_capability_audit',
                        render_root=str(rendered), scene_dir=str(rendered / 'stereo_raw' / 'scenes' / name),
                        sensor_path=str(sensor_path), left_path=str(selected[0].resolve()),
                        right_path=str(selected[1].resolve()), input_sha256=[file_sha256(p) for p in selected],
                        baseline_m=camera['baseline_m'], K_ir=camera['K_ir'],
                        T_world_from_left_ir=camera['T_world_from_left_ir'],
                        width=images[0].shape[1], height=images[0].shape[0], seed=seed))
    manifest = dict(schema=1, pairs=pairs, source_sha256=source_hashes, no_hardware=True,
        no_ground_truth_read=True, noise=dict(distribution='clip(round(I*Gamma(1333.33,1/1333.33)+Normal(0,.25)),0,255)',
            rng='NumPy default_rng per eye; not historical CUDA samples',
            source='https://github.com/haosulab/simsense/blob/9340b6069ce78ecf0432cd3336ac9b4d296e09e7/src/camera.cu',
            interpretation='shared synthetic photometric noise, not fitted physical D435i'),
        planned_comparison='both matchers receive byte-identical IR; SimSense internal noise OFF',
        audit='12 previously viewed jars in two fixed layouts, not 24 independent jars or a new heldout test')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(pairs=len(pairs), manifest=str(output / 'manifest.json'))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--development', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--layouts', nargs='*', choices=['far', 'close'], default=['far', 'close'])
    a = p.parse_args()
    prepare(a.source, a.development, a.output, a.layouts)
