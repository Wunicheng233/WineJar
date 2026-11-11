#!/usr/bin/env python3
"""Apply a locked plane-trained model to native stereo depth, then register.

Reads no target labels. Raw and corrected observations remain separate and the
RGB images are unchanged. Does not recalibrate K/T or claim calibrated output
uncertainty. The shared model is for identical simulated D435 presets only.
"""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from neck_localization.active_stereo import OfflineSpatialFilter, StereoSettings
from neck_localization.depth_correction import GaussianProcessDepthCorrection
from neck_localization.geometry import Camera
from neck_localization.scene_preparation import write_json
from scripts.run_active_stereo_stage1 import export_case


def run(source, model_path, output):
    source, model_path, output = Path(source), Path(model_path), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((source/'effective_config.json').read_text())
    write_json(output/'effective_config.json', config)
    model = GaussianProcessDepthCorrection.load(model_path)
    locked_model = sha256(model_path.read_bytes()).hexdigest()
    all_diagnostics = []
    for mode in ['stereo_gp', 'stereo_gp_spatial']:
        (output/mode).mkdir()
        write_json(output/mode/'effective_config.json', config)
        write_json(output/mode/'settings.json', dict(band_width_mm=3, preprocessing=dict(
            kind='plane_trained_gp_depth_correction', mode=mode, model_sha256=locked_model,
            model_path=str(model_path.resolve()), reference_data='independent known planar fixtures, no jars',
            spatial_filter=mode.endswith('_spatial'), temperature_c=25.)))
    for case in config['cases']:
        geometry = source/'geometry'/case['name']
        metadata = json.loads((geometry/'sensor.json').read_text())
        captured, cameras = [], []
        for c in metadata['cameras']:
            name = c['name']
            frame_dir = source/'stereo_raw/scenes'/case['name']/'observations'
            camera_metadata = json.loads((frame_dir/'camera_true_parameters.json').read_text())
            original = next(x for x in camera_metadata['cameras'] if x['name'] == name)
            rgb_camera = Camera(name, np.array(original['K']), np.array(original['T_world_from_camera']),
                                original['width'], original['height'])
            rgb = cv2.cvtColor(cv2.imread(str(frame_dir/(name+'_rgb.png'))), cv2.COLOR_BGR2RGB)
            native_path = source/'sensor_diagnostics'/case['name']/(name+'_native.npz')
            with np.load(native_path, allow_pickle=False) as data:
                native = data['native_float_z'].copy()
            left = Camera(name+'_left_ir', np.array(c['K_ir']), np.array(c['T_world_from_left_ir']),
                          native.shape[1], native.shape[0])
            before_hash = sha256(native.tobytes()).hexdigest()
            start = time.perf_counter()
            corrected, diag = model.correct_depth(native, left.K, temperature_c=25., return_diagnostics=True)
            assert sha256(native.tobytes()).hexdigest() == before_hash
            processor = OfflineSpatialFilter(left, c['baseline_m'], rgb_camera=rgb_camera)
            try:
                processor.process(corrected)
                depths = dict(stereo_gp=processor.register(rgb),
                              stereo_gp_spatial=processor.register(rgb, filtered=True))
            finally:
                processor.close()
            diag.update(case=case['name'], camera=name, correction_and_registration_ms=(time.perf_counter()-start)*1000,
                        native_input_sha256=sha256(native_path.read_bytes()).hexdigest(),
                        model_sha256=locked_model)
            all_diagnostics.append(diag)
            cameras.append(rgb_camera)
            captured.append(dict(rgb=rgb, depths=depths))
        for mode in ['stereo_gp', 'stereo_gp_spatial']:
            export_case(geometry, output/mode/'scenes'/case['name'], cameras, captured, mode)
        print(case['name'], 'corrected', flush=True)
    write_json(output/'correction_diagnostics.json', all_diagnostics)
    write_json(output/'provenance.json', dict(source=str(source.resolve()), model=str(model_path.resolve()),
        model_sha256=locked_model, source_config_sha256=sha256((source/'effective_config.json').read_bytes()).hexdigest(),
        source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), no_target_truth_read=True,
        pixel_and_depth_semantics='same native optical-Z/true K as calibration; correction before Z16 and RGB alignment',
        caveat='posterior and fitted correction are not validated D435i or absolute output confidence'))
    assert sha256(model_path.read_bytes()).hexdigest() == locked_model


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args()
    run(args.source, args.model, args.output)
