#!/usr/bin/env python3
"""Same saved rendered IR pairs/noise seeds, different SimSense support sizes.

No geometry or ideal depth enters matching. Default replay is checked against
the saved native output before any alternative is interpreted. Noise remains
ON; resolution/camera layout/rendered appearance do not change.
"""
import argparse
import gc
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from neck_localization.active_stereo import OfflineSpatialFilter, StereoSettings
from neck_localization.geometry import Camera
from scripts.run_active_stereo_stage1 import export_case

PRESETS = dict(default7=dict(), local3=dict(census_width=3, census_height=3, block_width=3, block_height=3, median_filter_size=1),
    narrow5=dict(census_width=5, census_height=3, block_width=3, block_height=1, median_filter_size=1))


def run(source, output, presets):
    import sapien
    from sapien.sensor import StereoDepthSensorConfig
    from sapien.sensor.simsense_component import SimSenseComponent
    config = json.loads((source/'effective_config.json').read_text())
    output.mkdir(parents=True, exist_ok=False)
    (output/'effective_config.json').write_bytes((source/'effective_config.json').read_bytes())
    (output/'geometry').symlink_to((source/'geometry').resolve(), target_is_directory=True)
    records = []
    for mode in presets: (output/mode).mkdir()
    for case in config['cases']:
        name = case['name']; geometry = source/'geometry'/name
        sensor = json.loads((geometry/'sensor.json').read_text())
        captured = {p:[] for p in presets}; cameras = []
        for i, item in enumerate(sensor['cameras']):
            rgb = cv2.cvtColor(cv2.imread(str(source/'stereo_raw/scenes'/name/'observations'/(item['name']+'_rgb.png'))), cv2.COLOR_BGR2RGB)
            images = [cv2.imread(str(source/'sensor_diagnostics'/name/(item['name']+f'_ir_clean_{j}.png')), cv2.IMREAD_GRAYSCALE) for j in range(2)]
            if any(a is None for a in images): raise FileNotFoundError('Missing observed IR pair')
            cfg = StereoDepthSensorConfig('D435')
            left = Camera(item['name'], np.array(item['K_ir']), np.array(item['T_world_from_left_ir']), *cfg.ir_resolution)
            camera = Camera(item['name'], np.array(item['K_rgb']), np.array(item['T_world_from_rgb']), *cfg.rgb_resolution)
            cameras.append(camera)
            for preset in presets:
                scene, entity = sapien.Scene(), sapien.Entity(); scene.add_entity(entity)
                keys = ['min_depth','max_depth','ir_noise_seed','ir_speckle_noise','ir_thermal_noise',
                    'rectified','census_width','census_height','max_disp','block_width','block_height',
                    'p1_penalty','p2_penalty','uniqueness_ratio','lr_max_diff','median_filter_size']
                options = {key:getattr(cfg,key) for key in keys}
                options.update(ir_noise_seed=sensor['settings']['seed']+i,
                    ir_speckle_noise=sensor['settings']['ir_speckle_noise'], ir_thermal_noise=sensor['settings']['ir_thermal_noise'])
                options.update(PRESETS[preset])
                matcher = SimSenseComponent(rgb_resolution=cfg.ir_resolution, ir_resolution=cfg.ir_resolution,
                    rgb_intrinsic=cfg.ir_intrinsic, ir_intrinsic=cfg.ir_intrinsic,
                    trans_pose_l=sapien.Pose(), trans_pose_r=cfg.trans_pose_l.inv()*cfg.trans_pose_r,
                    depth_dilation=False, **options)
                entity.add_component(matcher)
                try:
                    matcher.compute(*images); native = matcher.get_ndarray().copy()
                    difference = None
                    if preset == 'default7':
                        with np.load(source/'sensor_diagnostics'/name/(item['name']+'_native.npz')) as saved:
                            difference = float(np.max(abs(saved['native_float_z']-native)))
                        if difference > 1e-6: raise ValueError(f'Default replay differs: {difference}')
                    processor = OfflineSpatialFilter(left, item['baseline_m'], StereoSettings(), rgb_camera=camera)
                    try:
                        raw, _ = processor.process(native)
                        aligned = processor.register(rgb)
                    finally: processor.close()
                    diag = output/'sensor_diagnostics'/name; diag.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(diag/(item['name']+'_'+preset+'.npz'), native_float_z=native, native_z16_m=raw)
                    captured[preset].append(dict(rgb=rgb, depths={preset:aligned}))
                    records.append(dict(case=name, camera=item['name'], preset=preset,
                        replay_max_difference_m=difference, options=options,
                        valid_depth_fraction=float((native > 0).mean()),
                        input_sha256=[sha256(a.tobytes()).hexdigest() for a in images]))
                    print(name, item['name'], preset, 'replay', difference, 'valid', (native > 0).mean(), flush=True)
                finally:
                    entity.remove_component(matcher); scene.remove_entity(entity)
                    del matcher, entity, scene; gc.collect()
        for preset in presets:
            export_case(geometry, output/preset/'scenes'/name, cameras, captured[preset], preset)
    (output/'replay.json').write_text(json.dumps(records, indent=2)+'\n')
    (output/'provenance.json').write_text(json.dumps(dict(source=str(source.resolve()), presets=PRESETS,
        note='same pre-noise uint8 IR and upstream seeded internal gamma/Gaussian; NO noise reduction, no GT input',
        sources={str(Path(__file__).relative_to(ROOT)):sha256(Path(__file__).read_bytes()).hexdigest()}), indent=2)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--presets', nargs='+', choices=list(PRESETS), default=list(PRESETS))
    a=p.parse_args(); run(a.run,a.output,a.presets)
