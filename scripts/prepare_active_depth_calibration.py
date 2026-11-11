#!/usr/bin/env python3
"""Independent known-plane data for depth correction, not camera calibration.

Reference Z is analytically intersected with the known planar fixture. Camera
K/T are still simulator truth. No jar or target-circle label is involved.
"""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from neck_localization.active_stereo import ActiveStereoRig, StereoSettings, initialize_renderer
from neck_localization.geometry import look_at
from neck_localization.scene_preparation import write_json


def make_plan():
    frames = []
    # Predeclared disjoint fixture distances/tilts. The same unknown jar is
    # never used for training or model selection.
    groups = [('train', np.round(np.arange(.50, 1.1001, .02), 3), [(-15, 0), (0, 0), (15, 0)]),
              ('validation', [.51, .61, .71, .81, .91, 1.01, 1.09], [(-10, 7), (10, -7)]),
              ('test', [.555, .655, .755, .855, .955, 1.055], [(-20, -5), (20, 5)])]
    for split, distances, tilts in groups:
        for distance in distances:
            for rz, ry in tilts:
                frames.append(dict(split=split, distance_m=float(distance), tilt_z_deg=rz, tilt_y_deg=ry,
                                   frame_id=f'{split}_{distance:.3f}_{rz:+03d}_{ry:+03d}'.replace('.', 'p')))
    return dict(schema_version=1, frames=frames, settings=asdict(StereoSettings()),
                reference='analytic known-plane fixture with true camera K/T; not a solved calibration board')


def run(output):
    import sapien
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plan = make_plan()
    write_json(output/'plan.json', plan)
    initialize_renderer()
    scene = sapien.Scene()
    scene.set_ambient_light([.12, .12, .12])
    scene.add_directional_light([1, 0, -1], [.5, .5, .5])
    builder = scene.create_actor_builder()
    builder.add_box_visual(half_size=[.02, .65, .65], material=[.6, .6, .6, 1])
    plane = builder.build_static(name='known_planar_fixture')
    rig = ActiveStereoRig(scene, [look_at('calibration_sensor', [0, 0, 0], [1, 0, 0])])
    camera = rig.entries[0]['left']
    yy, xx = np.indices((camera.height, camera.width))
    ray_camera = np.stack([xx, yy, np.ones_like(xx)], axis=-1) @ np.linalg.inv(camera.K).T
    ray_world = ray_camera @ camera.T_world_from_camera[:3, :3].T
    origin = camera.T_world_from_camera[:3, 3]
    manifest = dict(schema_version='winejar-depth-calibration-v1',
        reference_source='known_virtual_planar_fixture_analytic_ray_plane_intersection_true_K_T', frames=[])
    try:
        write_json(output/'sensor.json', rig.metadata())
        for index, item in enumerate(plan['frames']):
            distance = item['distance_m']
            rotation = Rotation.from_euler('zy', [item['tilt_z_deg'], item['tilt_y_deg']], degrees=True)
            normal = rotation.apply([1., 0, 0])
            face_origin = np.array([distance, 0, 0])
            box_center = face_origin+normal*.02
            quat = rotation.as_quat()
            plane.set_pose(sapien.Pose(box_center, np.r_[quat[3], quat[:3]]))
            data = rig.capture(0)
            reference = (face_origin-origin) @ normal / (ray_world @ normal)
            point = origin+ray_world*reference[..., None]
            local = (point-box_center) @ rotation.as_matrix()
            valid = ((reference > .2) & (reference < 2) & (abs(local[..., 1]) < .55)
                     & (abs(local[..., 2]) < .55) & (data['native_float_z'] > 0))
            # Keep central sensor area to avoid unsupported fixture corners;
            # this mask is known-fixture calibration information only.
            valid[:40] = valid[-40:] = False
            valid[:, :80] = valid[:, -80:] = False
            reference[~valid] = 0
            folder = output/item['frame_id']
            folder.mkdir()
            np.save(folder/'measured.npy', data['native_float_z'])
            np.save(folder/'reference.npy', reference.astype(np.float32))
            np.save(folder/'valid.npy', valid)
            for j, image in enumerate(data['ir_clean']):
                cv2.imwrite(str(folder/f'ir_clean_{j}.png'), np.clip(image*255, 0, 255).astype(np.uint8))
            entry = dict(frame_id=item['frame_id'], camera_id='sim_d435_preset', split=item['split'],
                measured_depth=str(folder.relative_to(output)/'measured.npy'),
                reference_depth=str(folder.relative_to(output)/'reference.npy'),
                valid_mask=str(folder.relative_to(output)/'valid.npy'), K=camera.K.tolist(), temperature_c=25.,
                fixture=item, known_plane_normal_world=normal.tolist(), known_plane_point_world=face_origin.tolist())
            manifest['frames'].append(entry)
            write_json(output/'manifest.json', manifest)
            diff=(data['native_float_z'][valid]-reference[valid])*1000
            print(index+1, '/', len(plan['frames']), item['frame_id'], 'mean_mm', round(float(np.mean(diff)), 3), flush=True)
        write_json(output/'provenance.json', dict(plan_sha256=sha256((output/'plan.json').read_bytes()).hexdigest(),
            generator_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
            no_jar_data=True, assumed_material='uniform grey .6; NOT measured ceramic IR response',
            temperature='fixed arbitrary25; no thermal effect simulated',
            scope='depth-correction calibration with perfect planar references; camera-parameter calibration NOT performed'))
    finally:
        rig.close()
    return manifest


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args()
    run(args.output)
