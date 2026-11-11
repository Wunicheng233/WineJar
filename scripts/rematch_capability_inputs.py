#!/usr/bin/env python3
"""SimSense official default matcher on the shared, explicit audit IR inputs."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from neck_localization.baseline_capability import verify_pair, file_sha256


def run(manifest_path, output, layouts):
    import sapien
    from sapien.sensor import StereoDepthSensorConfig
    from sapien.sensor.simsense_component import SimSenseComponent
    manifest = json.loads(manifest_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    cfg = StereoDepthSensorConfig('D435')
    keys = ['min_depth','max_depth','rectified','census_width','census_height','max_disp',
        'block_width','block_height','p1_penalty','p2_penalty','uniqueness_ratio','lr_max_diff','median_filter_size']
    options = {key:getattr(cfg,key) for key in keys}
    options.update(ir_speckle_noise=0., ir_thermal_noise=0., ir_noise_seed=0)
    for rec in manifest['pairs']:
        if layouts and rec['layout'] not in layouts:
            continue
        verify_pair(rec)
        images = [np.ascontiguousarray(cv2.imread(rec[k], cv2.IMREAD_GRAYSCALE)) for k in ('left_path','right_path')]
        scene, entity = sapien.Scene(), sapien.Entity()
        scene.add_entity(entity)
        matcher = SimSenseComponent(rgb_resolution=cfg.ir_resolution, ir_resolution=cfg.ir_resolution,
            rgb_intrinsic=cfg.ir_intrinsic, ir_intrinsic=cfg.ir_intrinsic,
            trans_pose_l=sapien.Pose(), trans_pose_r=cfg.trans_pose_l.inv()*cfg.trans_pose_r,
            depth_dilation=False, **options)
        entity.add_component(matcher)
        started = time.perf_counter()
        try:
            matcher.compute(*images)
            depth = matcher.get_ndarray().copy()
            elapsed = (time.perf_counter() - started) * 1000
            valid = np.isfinite(depth) & (depth >= cfg.min_depth) & (depth < cfg.max_depth)
            depth[~valid] = 0
            disparity = np.zeros_like(depth)
            np.divide(rec['K_ir'][0][0] * rec['baseline_m'], depth, out=disparity, where=valid)
            path = output / 'predictions' / (rec['pair_id'] + '.npz')
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, native_float_z=depth, disparity_px=disparity, valid_mask=valid)
            row = dict(pair_id=rec['pair_id'], status='ok', inference_ms=elapsed,
                input_sha256=rec['input_sha256'], prediction_sha256=file_sha256(path))
        except Exception as error:
            row = dict(pair_id=rec['pair_id'], status='error', error=repr(error))
        finally:
            entity.remove_component(matcher)
            scene.remove_entity(entity)
            del matcher, entity, scene
            gc.collect()
        rows.append(row)
        with (output / 'predictions.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        print(row['pair_id'], row['status'], flush=True)
    (output / 'provenance.json').write_text(json.dumps(dict(method='SimSense default D435',
        options=options, manifest_sha256=file_sha256(manifest_path), no_hardware=True,
        script_sha256=file_sha256(__file__)), indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--layouts', nargs='*', default=[])
    a = p.parse_args()
    run(a.manifest, a.output, a.layouts)
