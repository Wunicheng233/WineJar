#!/usr/bin/env python3
"""Offline SAPIEN/SimSense runtime probe; never opens a hardware device."""
import argparse
import importlib.metadata
import inspect
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--ray-tracing', action='store_true')
    args = p.parse_args()
    import sapien
    import sapien.render
    from sapien.sensor import StereoDepthSensor, StereoDepthSensorConfig
    # 3.0.3 renders headlessly without creating a Viewer. The newer docs'
    # offscreen_only keyword is absent from this pinned wheel.
    if args.ray_tracing:
        sapien.render.set_camera_shader_dir('rt')
        sapien.render.set_ray_tracing_samples_per_pixel(32)
        sapien.render.set_ray_tracing_path_depth(4)
        sapien.render.set_ray_tracing_denoiser('optix')
    scene = sapien.Scene()
    scene.set_ambient_light([.1, .1, .1])
    scene.add_directional_light([1, 0, -1], [.5, .5, .5])
    builder = scene.create_actor_builder()
    builder.add_box_visual(half_size=[.02, .5, .5], material=[.6, .6, .6, 1])
    plane = builder.build_static(name='plane')
    plane.set_pose(sapien.Pose([.82, 0, 0]))  # front surface exactly x=.8m
    mount = scene.create_actor_builder().build_kinematic(name='sensor_mount')
    cfg = StereoDepthSensorConfig('D435')
    cfg.depth_dilation = False
    sensor = StereoDepthSensor(cfg, mount)
    scene.update_render()
    sensor.take_picture()
    sensor.compute_depth()
    depth = sensor.get_depth()
    rgb = sensor.get_rgb()
    roi = depth[180:300, 300:550]
    valid = roi[np.isfinite(roi) & (roi > 0)]
    result = dict(versions={n:importlib.metadata.version(n) for n in ['sapien', 'numpy', 'pyrealsense2']},
                  sensor_source=inspect.getfile(StereoDepthSensor), ray_tracing=args.ray_tracing,
                  rgb_shape=list(rgb.shape), depth_shape=list(depth.shape),
                  valid_roi_fraction=float(len(valid)/roi.size),
                  plane_expected_m=.8, plane_median_m=float(np.median(valid)) if len(valid) else None,
                  rgb_intrinsic=cfg.rgb_intrinsic.tolist(), ir_intrinsic=cfg.ir_intrinsic.tolist())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    if len(valid) < .5*roi.size or abs(np.median(valid)-.8) > .01:
        raise RuntimeError('Runtime probe did not recover the known plane')


if __name__ == '__main__':
    main()
