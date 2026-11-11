#!/usr/bin/env python3
"""Known-plane diagnostics of coordinates, depth semantics and stereo noise."""
import argparse
from dataclasses import asdict
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


def run(output, distances=(.6, .8, 1.), tilts=(0., 20.)):
    import sapien
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    settings = StereoSettings()
    initialize_renderer(settings)
    scene = sapien.Scene()
    scene.set_ambient_light([.12, .12, .12])
    scene.add_directional_light([1, 0, -1], [.5, .5, .5])
    builder = scene.create_actor_builder()
    builder.add_box_visual(half_size=[.02, .5, .5], material=[.6, .6, .6, 1])
    plane = builder.build_static(name='known_plane')
    rig = ActiveStereoRig(scene, [look_at('cam_a', [0, 0, 0], [1, 0, 0])], settings)
    rows = []
    try:
        write_json(output/'sensor.json', rig.metadata())
        for distance in distances:
            for tilt in tilts:
                rotation = Rotation.from_euler('z', tilt, degrees=True)
                normal = rotation.apply([1., 0, 0])
                center = np.array([distance, 0, 0])+normal*.02
                quat = rotation.as_quat()
                plane.set_pose(sapien.Pose(center, np.r_[quat[3], quat[:3]]))
                data = rig.capture(0)
                name = f'plane_{distance:.2f}_tilt_{tilt:.0f}'
                np.savez_compressed(output/(name+'.npz'), **data['depths'],
                                    native_float_z=data['native_float_z'], ideal_left_z=data['ideal_left_z'])
                for i, ir in enumerate(data['ir_clean']):
                    cv2.imwrite(str(output/(name+f'_ir_{i}.png')), np.clip(ir*255, 0, 255).astype(np.uint8))
                rgb = rig.entries[0]['rgb']
                for mode, depth in data['depths'].items():
                    # Central ROI avoids plane-edge silhouette effects, declared
                    # diagnostic geometry only. No estimation is scored here.
                    mask = np.zeros(depth.shape, bool)
                    mask[160:320, 300:550] = True
                    y, x = np.nonzero(mask & (depth > 0))
                    points = rgb.unproject(np.c_[x, y], depth[y, x])
                    error = (points-np.array([distance, 0, 0])) @ normal*1000
                    row = dict(case=name, distance_m=distance, tilt_deg=tilt, mode=mode,
                               valid_fraction=float(len(x)/mask.sum()), n=int(len(x)),
                               mean_plane_mm=float(np.mean(error)), rms_plane_mm=float(np.sqrt(np.mean(error**2))),
                               p95_abs_plane_mm=float(np.percentile(abs(error), 95)), max_abs_plane_mm=float(max(abs(error))))
                    rows.append(row)
                    print(json.dumps(row), flush=True)
        write_json(output/'records.json', rows)
        # Semantic checks use ideal renderer only. Stereo accuracy is reported,
        # never forced to pass an invented real-camera tolerance.
        assert max(r['max_abs_plane_mm'] for r in rows if r['mode'] == 'ideal_rgb') < .01
        assert min(r['valid_fraction'] for r in rows if r['mode'] == 'stereo_raw') > .3
        write_json(output/'validation.json', dict(coordinates_and_ideal_depth_passed=True,
                   minimum_stereo_roi_valid_fraction=.3, scope='known-plane diagnostic, not D435i validation'))
    finally:
        rig.close()
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    run(args.output)
