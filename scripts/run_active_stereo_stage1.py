#!/usr/bin/env python3
"""Paired MuJoCo-geometry/SAPIEN-observation study with a frozen estimator.

No cameras or robots are opened. The existing estimator only sees exported
RGB-D and declared true camera parameters, not meshes/diagnostic depth/truth.
"""
import argparse
from dataclasses import asdict
import gc
import gzip
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from neck_localization.active_stereo import (ActiveStereoRig, StereoSettings,
                                            initialize_renderer, add_jar_assets)
from neck_localization.band_geometry import plan_band, evaluate_band
from neck_localization.depth_profile import rgb_foreground
from neck_localization.evaluation import score
from neck_localization.joint_profile import JointSettings
from neck_localization.scene_preparation import prepare_scene_bundle, scene_cameras, write_json
from neck_localization.task_validated_output import estimate_task_validated_output
from scripts.run_depth_profile_baseline import load_frames
from scripts.run_task_validated_output import source_hashes as frozen_sources

MODES = ('ideal_rgb', 'ideal_ir_registered', 'stereo_raw', 'stereo_spatial')


def sources():
    names = ['neck_localization/active_stereo.py', 'scripts/run_active_stereo_stage1.py',
             'neck_localization/scene_preparation.py', 'neck_localization/jar_model.py',
             'neck_localization/band_geometry.py']
    return frozen_sources() | {n:sha256((ROOT/n).read_bytes()).hexdigest() for n in names}


def export_case(source, destination, cameras, results, mode):
    destination.mkdir(parents=True, exist_ok=False)
    for name in ['assets', 'evaluation_only', 'model_manifest.json']:
        (destination/name).symlink_to((source/name).resolve(), target_is_directory=(source/name).is_dir())
    tree = ET.parse(source/'scene.xml')
    visual = tree.getroot().find('visual/global')
    visual.set('offwidth', str(cameras[0].width))
    visual.set('offheight', str(cameras[0].height))
    # Main/close-up MuJoCo views share the exact mesh and camera poses. Camera
    # image panels always read SAPIEN's saved input; off-centre K is in metadata.
    for camera in cameras:
        node = tree.getroot().find(f"worldbody/camera[@name='{camera.name}']")
        node.set('fovy', str(np.degrees(2*np.arctan(camera.height/(2*camera.K[1, 1])))))
    tree.write(destination/'scene.xml', encoding='unicode')
    obs = destination/'observations'
    obs.mkdir()
    write_json(obs/'camera_true_parameters.json', dict(
        kind='true_simulation_cameras_not_virtual_calibration',
        depth='RGB optical-axis Z in metres; distinct left/right IR and RGB viewpoints; SDK native-Z conversion explicit',
        cameras=[dict(name=c.name, K=c.K, T_world_from_camera=c.T_world_from_camera,
                      width=c.width, height=c.height) for c in cameras]))
    for camera, data in zip(cameras, results):
        cv2.imwrite(str(obs/(camera.name+'_rgb.png')), cv2.cvtColor(data['rgb'], cv2.COLOR_RGB2BGR))
        np.savez_compressed(obs/(camera.name+'_depth_z_m.npz'), depth_z_m=data['depths'][mode])
    write_json(obs/'measurement_model.json', dict(mode=mode, ideal_diagnostic=mode.startswith('ideal'),
        semantics='actual saved estimator input; no GT mask used', no_optional_hole_filling=True,
        note='SDK alignment pixel footprints can spread a measured pixel over multiple RGB pixels'))


def summarize(rows):
    result = []
    for mode in MODES:
        rr = [r for r in rows if r['condition_name'] == mode]
        n = len(rr)
        if not n:
            continue
        out = dict(mode=mode, n=n, accepted=sum(r['accepted'] for r in rr),
                   correct_circle=sum(r['accepted'] and r['truth_pass'] for r in rr),
                   rejected=sum(not r['accepted'] for r in rr),
                   wrong_accepted=sum(r['accepted'] and not r['truth_pass'] for r in rr),
                   band_success=sum(bool(r['accepted'] and r['truth_pass'] and r.get('band')
                       and r['band']['nominal_compatible'] and r['band_evaluation']['near_min_proxy']['contained']) for r in rr))
        out['reasons'] = {s:sum(s in r['reasons'] for r in rr) for s in sorted({s for r in rr for s in r['reasons']})}
        for metric in ['center_mm', 'normal_deg', 'diameter_mm', 'normal_component_mm', 'tangent_mm', 'runtime_ms']:
            values = [abs(r[metric]) for r in rr if r.get(metric) is not None]
            for suffix, percentile in [('median', 50), ('p95', 95), ('max', 100)]:
                out[metric+'_'+suffix] = float(np.percentile(values, percentile)) if values else None
        result.append(out)
    return result


def run(config_path, output, freeze_path=None):
    import sapien
    config_path, output = Path(config_path), Path(output)
    config = json.loads(config_path.read_text())
    locked_sources = sources()
    if config['dataset_split'].endswith('_test'):
        if freeze_path is None:
            raise ValueError('Independent test requires the pre-test source/config freeze')
        freeze = json.loads(Path(freeze_path).read_text())
        assert freeze['sources'] == locked_sources, 'Implementation changed after freeze'
        assert freeze['test_config_sha256'] == sha256(config_path.read_bytes()).hexdigest()
        assert freeze['settings'] == asdict(StereoSettings())
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output/'effective_config.json')
    write_json(output/'provenance.json', dict(sources=locked_sources, settings=asdict(StereoSettings()),
        split=config['dataset_split'], true_cameras=True, no_virtual_calibration_yet=True,
        limitations=['D435 preset, not fitted to current D435i', 'assumed shared visible/IR material',
                     'isolated emission, no multi-projector interaction', 'fixed circular-section model family']))
    prepare_scene_bundle(config, output/'geometry')
    initialize_renderer()
    rig_cameras = scene_cameras(config['scene'])
    rows, measurement_rows = [], []
    for mode in MODES:
        (output/mode).mkdir()
        shutil.copy2(config_path, output/mode/'effective_config.json')
        write_json(output/mode/'settings.json', dict(**asdict(JointSettings()), band_width_mm=3,
                   preprocessing=dict(kind='active_stereo_measurement', condition=mode,
                                      ideal_diagnostic=mode.startswith('ideal'))))
    with (output/'records.jsonl').open('w') as journal:
        for index, case in enumerate(config['cases']):
            source = output/'geometry'/case['name']
            scene = sapien.Scene()
            scene.set_ambient_light([.3, .3, .3])
            scene.add_directional_light([1, -1, -2], [.6, .6, .6])
            scene.add_directional_light([-1, 1, -1], [.25, .25, .25])
            add_jar_assets(scene, source)
            # Case and camera seeds are fixed without inspecting any result.
            rig = ActiveStereoRig(scene, rig_cameras, StereoSettings(seed=StereoSettings().seed+index*10))
            try:
                metadata = rig.metadata()
                write_json(source/'sensor.json', metadata)
                cameras = [e['rgb'] for e in rig.entries]
                captured = []
                diag = output/'sensor_diagnostics'/case['name']
                diag.mkdir(parents=True)
                for i, camera in enumerate(cameras):
                    data = rig.capture(i)
                    captured.append(data)
                    for j, image in enumerate(data['ir_clean']):
                        cv2.imwrite(str(diag/(camera.name+f'_ir_clean_{j}.png')), np.clip(image*255, 0, 255).astype(np.uint8))
                    np.savez_compressed(diag/(camera.name+'_native.npz'),
                        native_float_z=data['native_float_z'], native_z16_m=data['native_z16_m'],
                        native_filtered_m=data['native_filtered_m'], ideal_left_z=data['ideal_left_z'])
                    np.savez_compressed(diag/(camera.name+'_ideal_rgb.npz'), ideal_rgb_z=data['depths']['ideal_rgb'])
                    mask = rgb_foreground(data['rgb'])
                    for mode in MODES:
                        z, gt = data['depths'][mode], data['depths']['ideal_rgb']
                        valid = mask & (z > 0) & (gt > 0)
                        error = (z[valid]-gt[valid])*1000
                        measurement_rows.append(dict(case=case['name'], camera=camera.name, mode=mode,
                            foreground_pixels=int(mask.sum()), valid_fraction=float(valid.sum()/max(1, mask.sum())),
                            mean_z_error_mm=float(np.mean(error)) if len(error) else None,
                            p95_abs_z_error_mm=float(np.percentile(abs(error), 95)) if len(error) else None,
                            simulation_capture_processing_ms=data['capture_processing_ms']))
                for mode in MODES:
                    folder = output/mode/'scenes'/case['name']
                    export_case(source, folder, cameras, captured, mode)
                    # Only observations are opened here. No mesh, dimensions,
                    # profile, ideal depth or masks enter this frozen function.
                    frames = load_frames(folder/'observations', 'rgb')
                    estimate = estimate_task_validated_output(frames)
                    diagnostics = estimate.diagnostics
                    band = (plan_band(dict(center_m=estimate.center, normal=estimate.normal, radius_m=estimate.radius), diagnostics)
                            if estimate.center is not None and 'near_min_interval_from_axis_origin_m' in diagnostics else None)
                    truth = json.loads((source/'evaluation_only/target.json').read_text())
                    row = dict(case=case['name'], group=case['group'], split=config['dataset_split'],
                               geometry='photo_short_neck', evidence='rgb', camera_set='all', factor=mode,
                               condition_name=mode, method='task_validated_output', **score(estimate, truth))
                    row.setdefault('diagnostics', diagnostics)
                    row['band'] = band
                    row['band_evaluation'] = ({name:evaluate_band(band, truth, truth[field]) for name, field in
                        [('near_min_proxy', 'near_min_interval_s_m'), ('full_annotated_neck', 'search_interval_s_m')]} if band else None)
                    rows.append(row)
                    journal.write(json.dumps(row)+'\n')
                    journal.flush()
                    print(case['name'], mode, row['accepted'], row['center_mm'], row['normal_deg'], row['diameter_mm'], row['reasons'], flush=True)
            finally:
                rig.close()
            del rig, scene, captured
            gc.collect()
    for mode in MODES:
        selected = [r for r in rows if r['condition_name'] == mode]
        (output/mode/'records.json.gz').write_bytes(gzip.compress(json.dumps(selected).encode(), mtime=0))
    (output/'records.json.gz').write_bytes(gzip.compress(json.dumps(rows).encode(), mtime=0))
    write_json(output/'summary.json', summarize(rows))
    write_json(output/'measurement_metrics.json', measurement_rows)
    assert sources() == locked_sources, 'Source changed during experiment'
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--freeze', type=Path)
    args = p.parse_args()
    run(args.config, args.output, args.freeze)
