"""Pinned SAPIEN active stereo + offline librealsense single-frame processing.

No hardware API, geometric target label or true jar dimension enters inference.
World uses metres/+Z up; observation cameras use OpenCV right/down/forward.
SAPIEN camera poses use forward/left/up. Position buffers use OpenGL axes.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
import importlib.metadata
import inspect
from pathlib import Path

import cv2
import numpy as np

from .geometry import Camera

# p_cv = CV_FROM_ROS @ p_sapien_camera
CV_FROM_ROS = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])


@dataclass(frozen=True)
class StereoSettings:
    model: str = 'D435'
    seed: int = 2026102451
    samples_per_pixel: int = 32
    path_depth: int = 4
    denoiser: str = 'optix'
    depth_unit_m: float = .001  # SDK default assumption, not measured on our unit
    ir_speckle_noise: float = 1.0  # package defaults, not fitted D435i values
    ir_thermal_noise: float = 1.0
    spatial_magnitude: int = 2
    spatial_alpha: float = .5
    spatial_delta: int = 20


def pose_cv_to_sapien(t_world_cv):
    result = np.array(t_world_cv, dtype=float, copy=True)
    result[:3, :3] = result[:3, :3] @ CV_FROM_ROS
    return result


def pose_sapien_to_cv(t_world_ros):
    result = np.array(t_world_ros, dtype=float, copy=True)
    result[:3, :3] = result[:3, :3] @ CV_FROM_ROS.T
    return result


def quantize_depth(depth_m, unit_m):
    if not np.isfinite(unit_m) or unit_m <= 0:
        raise ValueError('Positive finite depth unit required')
    depth = np.asarray(depth_m)
    good = np.isfinite(depth) & (depth > 0) & (depth < 65535*unit_m)
    out = np.zeros(depth.shape, dtype=np.uint16)
    out[good] = np.maximum(1, np.rint(depth[good]/unit_m)).astype(np.uint16)
    return out


def aligned_source_z_to_rgb_z(aligned_z_left_m, left: Camera, rgb: Camera):
    """SDK align copies source Z; convert it to the RGB optical-axis Z.

    z_left = (R_left_rgb ray_rgb)_z*z_rgb+t_left_rgb,z. Does not use geometry
    truth; native pixel-footprint resampling is still an approximation.
    """
    out = np.asarray(aligned_z_left_m)
    reverse = np.linalg.inv(left.T_world_from_camera) @ rgb.T_world_from_camera
    yy, xx = np.indices(out.shape)
    rays_rgb = np.stack([xx, yy, np.ones_like(xx)], axis=-1) @ np.linalg.inv(rgb.K).T
    denominator = rays_rgb @ reverse[2, :3]
    good = np.isfinite(out) & (out > 0) & (denominator > 0)
    result = np.zeros(out.shape, np.float32)
    result[good] = (out[good]-reverse[2, 3])/denominator[good]
    result[~np.isfinite(result) | (result <= 0)] = 0
    return result


class OfflineSpatialFilter:
    """Official depth->disparity->spatial->depth blocks on ONE synthetic frame.

    software_device is an in-memory source only: no context, pipeline, physical
    device discovery, calibration writes, temporal averaging or hole filling.
    """
    def __init__(self, camera: Camera, baseline_m, settings=StereoSettings(), rgb_camera=None):
        import pyrealsense2 as rs
        self.rs, self.camera, self.settings = rs, camera, settings
        self.device = rs.software_device()
        self.sensor = self.device.add_sensor('Synthetic depth only')
        self.sensor.add_read_only_option(rs.option.depth_units, settings.depth_unit_m)
        self.sensor.add_read_only_option(rs.option.stereo_baseline, float(baseline_m*1000))
        intr = rs.intrinsics()
        intr.width, intr.height = camera.width, camera.height
        intr.fx, intr.fy = float(camera.K[0, 0]), float(camera.K[1, 1])
        intr.ppx, intr.ppy = float(camera.K[0, 2]), float(camera.K[1, 2])
        intr.model, intr.coeffs = rs.distortion.none, [0.]*5
        stream = rs.video_stream()
        stream.type, stream.index, stream.uid = rs.stream.depth, 0, 0
        stream.width, stream.height, stream.fps = camera.width, camera.height, 30
        stream.bpp, stream.fmt, stream.intrinsics = 2, rs.format.z16, intr
        self.profile = self.sensor.add_video_stream(stream).as_video_stream_profile()
        self.queue = rs.frame_queue(1)
        self.sensor.open(self.profile)
        self.sensor.start(self.queue)
        self.forward, self.reverse = rs.disparity_transform(True), rs.disparity_transform(False)
        self.spatial = rs.spatial_filter()
        for key, val in [(rs.option.filter_magnitude, settings.spatial_magnitude),
                         (rs.option.filter_smooth_alpha, settings.spatial_alpha),
                         (rs.option.filter_smooth_delta, settings.spatial_delta),
                         (rs.option.holes_fill, 0)]:
            self.spatial.set_option(key, val)
        self.number = 0
        self.rgb_camera = rgb_camera
        if rgb_camera is not None:
            other = rgb_camera
            ci = rs.intrinsics()
            ci.width, ci.height = other.width, other.height
            ci.fx, ci.fy = float(other.K[0, 0]), float(other.K[1, 1])
            ci.ppx, ci.ppy = float(other.K[0, 2]), float(other.K[1, 2])
            ci.model, ci.coeffs = rs.distortion.none, [0.]*5
            stream = rs.video_stream()
            stream.type, stream.index, stream.uid = rs.stream.color, 0, 1
            stream.width, stream.height, stream.fps = other.width, other.height, 30
            stream.bpp, stream.fmt, stream.intrinsics = 3, rs.format.rgb8, ci
            self.color_sensor = self.device.add_sensor('Synthetic color only')
            self.color_profile = self.color_sensor.add_video_stream(stream).as_video_stream_profile()
            self.color_queue = rs.frame_queue(1)
            self.color_sensor.open(self.color_profile)
            self.color_sensor.start(self.color_queue)
            transform = np.linalg.inv(other.T_world_from_camera) @ camera.T_world_from_camera
            extr = rs.extrinsics()
            extr.rotation = transform[:3, :3].flatten(order='F').tolist()
            extr.translation = transform[:3, 3].tolist()
            self.profile.register_extrinsics_to(self.color_profile, extr)
            self.composite_queue = rs.frame_queue(1)
            self.composite = rs.processing_block(self._composite_callback)
            self.composite.start(self.composite_queue)
            self.align = rs.align(rs.stream.color)

    def _composite_callback(self, frame, source):
        source.frame_ready(source.allocate_composite_frame([frame, self.pending_color]))

    def process(self, depth_m):
        rs = self.rs
        raw = quantize_depth(depth_m, self.settings.depth_unit_m)
        self.number += 1
        frame = rs.software_video_frame()
        frame.pixels, frame.stride, frame.bpp = raw, self.camera.width*2, 2
        frame.timestamp, frame.domain = self.number*1000/30, rs.timestamp_domain.hardware_clock
        frame.frame_number, frame.profile = self.number, self.profile
        frame.depth_units = self.settings.depth_unit_m
        self.sensor.on_video_frame(frame)
        incoming = self.queue.wait_for_frame(1000)
        output = self.reverse.process(self.spatial.process(self.forward.process(incoming)))
        self.raw_frame, self.filtered_frame, self.raw_buffer = incoming, output, raw
        filtered = np.asanyarray(output.get_data()).copy()
        return raw.astype(np.float32)*self.settings.depth_unit_m, filtered.astype(np.float32)*self.settings.depth_unit_m

    def register(self, rgb, *, filtered=False):
        rs = self.rs
        if self.rgb_camera is None:
            raise ValueError('RGB camera required for registration')
        self.color_buffer = np.ascontiguousarray(rgb)
        frame = rs.software_video_frame()
        frame.pixels, frame.stride, frame.bpp = self.color_buffer, self.rgb_camera.width*3, 3
        frame.timestamp, frame.domain = self.number*1000/30, rs.timestamp_domain.hardware_clock
        frame.frame_number, frame.profile = self.number, self.color_profile
        self.color_sensor.on_video_frame(frame)
        self.pending_color = self.color_queue.wait_for_frame(1000)
        self.composite.invoke(self.filtered_frame if filtered else self.raw_frame)
        pair = self.composite_queue.wait_for_frame(1000).as_frameset()
        aligned = self.align.process(pair).as_frameset().get_depth_frame()
        source_z = np.asanyarray(aligned.get_data()).copy().astype(np.float32)*self.settings.depth_unit_m
        return aligned_source_z_to_rgb_z(source_z, self.camera, self.rgb_camera)

    def close(self):
        self.sensor.stop()
        self.sensor.close()
        if self.rgb_camera is not None:
            self.color_sensor.stop()
            self.color_sensor.close()


def initialize_renderer(settings=StereoSettings()):
    import sapien.render
    sapien.render.set_camera_shader_dir('rt')
    sapien.render.set_ray_tracing_samples_per_pixel(settings.samples_per_pixel)
    sapien.render.set_ray_tracing_path_depth(settings.path_depth)
    sapien.render.set_ray_tracing_denoiser(settings.denoiser)


class ActiveStereoRig:
    """One emitter at a time, common geometry; no multi-device IR interaction."""
    def __init__(self, scene, cameras, settings=StereoSettings()):
        import sapien
        from sapien.sensor import StereoDepthSensor, StereoDepthSensorConfig
        from sapien.sensor.simsense_component import SimSenseComponent
        self.scene, self.settings, self.entries = scene, settings, []
        for i, camera in enumerate(cameras):
            cfg = StereoDepthSensorConfig(settings.model)
            cfg.ir_noise_seed = settings.seed+i
            cfg.ir_speckle_noise = settings.ir_speckle_noise
            cfg.ir_thermal_noise = settings.ir_thermal_noise
            cfg.depth_dilation = False
            mount = scene.create_actor_builder().build_kinematic(name=camera.name+'_mount')
            mount.set_pose(sapien.Pose(pose_cv_to_sapien(camera.T_world_from_camera)))
            sensor = StereoDepthSensor(cfg, mount)
            # Pinned raster shaders support one textured projector per scene.
            # Emitters are already explicitly isolated; attach only the one
            # being captured instead of keeping three zero-intensity lights.
            mount.remove_component(sensor._alight)
            # Keep the upstream camera/projector implementation; replace its
            # RGB-registered matcher output by a native-IR matcher so official
            # filters operate before RGB registration. Release the unused GPU
            # engine before constructing the native one.
            mount.remove_component(sensor._ss)
            # SAPIEN shader samples pixel (x+.5,y+.5); our OpenCV-style arrays
            # address centres at integer (x,y). Convert the principal point,
            # without modifying focal length or re-estimating it from a jar.
            rgb_k, ir_k = cfg.rgb_intrinsic.copy(), cfg.ir_intrinsic.copy()
            rgb_k[:2, 2] -= .5
            ir_k[:2, 2] -= .5
            rgb = Camera(camera.name, rgb_k, camera.T_world_from_camera.copy(), *cfg.rgb_resolution)
            tleft = pose_cv_to_sapien(camera.T_world_from_camera) @ cfg.trans_pose_l.to_transformation_matrix()
            left = Camera(camera.name+'_left_ir', ir_k, pose_sapien_to_cv(tleft), *cfg.ir_resolution)
            # RT Position is overwritten at every subpixel sample and stores
            # the LAST Halton ray, not the centre ray or the averaged RGB hit.
            # Obtain ideal depth/labels from separate standard raster cameras.
            raster_dir = str(Path(sapien.__file__).parent/'vulkan_shader/default')
            ideal_cameras = []
            for resolution, k, pose in [(cfg.rgb_resolution, cfg.rgb_intrinsic, sapien.Pose()),
                                         (cfg.ir_resolution, cfg.ir_intrinsic, cfg.trans_pose_l)]:
                cam = sapien.render.RenderCameraComponent(*resolution, raster_dir)
                cam.local_pose = pose
                cam.set_perspective_parameters(.1, 100, k[0, 0], k[1, 1], k[0, 2], k[1, 2], 0)
                mount.add_component(cam)
                ideal_cameras.append(cam)
            # Native left-IR output. The image matcher is the vendor's existing
            # SimSense component; only its output registration is configured.
            native_cfg = dict(rgb_resolution=cfg.ir_resolution, ir_resolution=cfg.ir_resolution,
                              rgb_intrinsic=cfg.ir_intrinsic, ir_intrinsic=cfg.ir_intrinsic,
                              trans_pose_l=sapien.Pose(), trans_pose_r=cfg.trans_pose_l.inv()*cfg.trans_pose_r)
            keys = ['min_depth', 'max_depth', 'ir_noise_seed', 'ir_speckle_noise', 'ir_thermal_noise',
                    'rectified', 'census_width', 'census_height', 'max_disp', 'block_width', 'block_height',
                    'p1_penalty', 'p2_penalty', 'uniqueness_ratio', 'lr_max_diff', 'median_filter_size']
            native_cfg.update({key:getattr(cfg, key) for key in keys})
            native_cfg['depth_dilation'] = False
            native = SimSenseComponent(**native_cfg)
            mount.add_component(native)
            baseline = np.linalg.norm(cfg.trans_pose_r.p-cfg.trans_pose_l.p)
            processor = OfflineSpatialFilter(left, baseline, settings, rgb_camera=rgb)
            self.entries.append(dict(sensor=sensor, native=native, rgb=rgb, left=left, ideal_cameras=ideal_cameras,
                                     config=cfg, processor=processor, baseline_m=float(baseline)))

    def capture(self, index):
        sensor = self.entries[index]['sensor']
        sensor._mount.add_component(sensor._alight)
        try:
            return self._capture_one(index)
        finally:
            sensor._normal_mode()
            sensor._mount.remove_component(sensor._alight)
            self.scene.update_render()

    def _capture_one(self, index):
        import time
        entry = self.entries[index]
        sensor, native = entry['sensor'], entry['native']
        start = time.perf_counter()
        # Capture ideal RGB/position before lighting mode changes. Position and
        # segmentation are evaluation diagnostics, never estimator masks.
        self.scene.update_render()
        sensor._cam_rgb.take_picture()
        rgb = np.clip(sensor._cam_rgb.get_picture('Color')[..., :3]*255, 0, 255).astype(np.uint8)
        ideal_rgb, ideal_left = entry['ideal_cameras']
        ideal_rgb.take_picture()
        ideal_left.take_picture()
        pos = ideal_rgb.get_picture('Position').copy()
        ir_pos = ideal_left.get_picture('Position').copy()
        ideal_rgb_z = np.where(pos[..., 3] < 1, -pos[..., 2], 0).astype(np.float32)
        seg = ideal_rgb.get_picture('Segmentation').copy()
        self.scene.update_render()
        model_matrix = sensor._cam_rgb.get_model_matrix()
        expected_gl = entry['rgb'].T_world_from_camera.copy()
        expected_gl[:3, :3] = expected_gl[:3, :3] @ np.diag([1., -1., -1.])
        np.testing.assert_allclose(model_matrix, expected_gl, atol=2e-6)
        sensor._ir_mode()
        self.scene.update_render()
        sensor._cam_ir_l.take_picture()
        sensor._cam_ir_r.take_picture()
        ir = [sensor._cam_ir_l.get_picture('Color')[..., 0].copy(),
              sensor._cam_ir_r.get_picture('Color')[..., 0].copy()]
        ideal_left_z = np.where(ir_pos[..., 3] < 1, -ir_pos[..., 2], 0).astype(np.float32)
        native.compute(sensor._cam_ir_l.get_picture_cuda('Color'), sensor._cam_ir_r.get_picture_cuda('Color'))
        native_z = native.get_ndarray().copy()
        sensor._normal_mode()
        self.scene.update_render()
        raw, filtered = entry['processor'].process(native_z)
        depths = {'ideal_rgb':ideal_rgb_z,
                  'stereo_raw':entry['processor'].register(rgb),
                  'stereo_spatial':entry['processor'].register(rgb, filtered=True)}
        # Explicit IDEAL diagnostic: how much IR sampling/Z16/registration alone
        # changes the downstream result. Never presented as realistic sensing.
        entry['processor'].process(ideal_left_z)
        depths['ideal_ir_registered'] = entry['processor'].register(rgb)
        return dict(rgb=rgb, depths=depths, ir_clean=ir, ideal_left_z=ideal_left_z,
                    native_float_z=native_z, native_z16_m=raw, native_filtered_m=filtered,
                    segmentation=seg, position_rgb_diagnostic=pos,
                    capture_processing_ms=(time.perf_counter()-start)*1000)

    def metadata(self):
        from sapien.sensor import StereoDepthSensor
        from sapien.sensor.simsense_component import SimSenseComponent
        paths = [Path(inspect.getfile(StereoDepthSensor)), Path(inspect.getfile(SimSenseComponent))]
        return dict(settings=asdict(self.settings),
                    versions={n:importlib.metadata.version(n) for n in ['sapien', 'pyrealsense2', 'numpy', 'opencv-python']},
                    upstream_source_sha256={str(p):sha256(p.read_bytes()).hexdigest() for p in paths},
                    ir_pattern_sha256=sha256(Path(self.entries[0]['config'].light_pattern).read_bytes()).hexdigest(),
                    cameras=[dict(name=e['rgb'].name, K_rgb=e['rgb'].K.tolist(), K_ir=e['left'].K.tolist(),
                                  T_world_from_rgb=e['rgb'].T_world_from_camera.tolist(),
                                  T_world_from_left_ir=e['left'].T_world_from_camera.tolist(), baseline_m=e['baseline_m'])
                             for e in self.entries],
                    ir_saved_semantics='rendered IR before matcher-internal speckle/thermal noise',
                    registration='official SDK align pixel footprints; explicit source-Z to RGB-Z conversion; no optional hole completion',
                    pixel_convention='renderer corner-origin K -> integer-centre K: cx,cy minus .5; ideal depth from raster centre rays, not RT last Halton hit',
                    emitter_mode='isolated per-camera emission, not simultaneous multi-projector simulation')

    def close(self):
        for entry in self.entries:
            entry['processor'].close()
            entry['sensor']._mount.remove_component(entry['native'])


def add_jar_assets(scene, folder, *, roughness=.65, specular=.12):
    """Reuse the exact MuJoCo OBJ/texture; IR optical properties are assumed."""
    import sapien
    folder = Path(folder)
    # The frozen RGB extractor assumes grey ceramic against a blue surround,
    # exactly as the MuJoCo baseline scene. SAPIEN's default untextured RT sky
    # is grey and would be incorrectly selected as foreground. Set a physical
    # scene background, never paste a truth mask into the rendered RGB image.
    background = folder/'sensor_environment_blue.png'
    cv2.imwrite(str(background), np.full((32, 32, 3), [55, 28, 12], np.uint8))
    scene.set_environment_map(sapien.render.RenderCubemap(*([str(background)]*6)))
    material = sapien.render.RenderMaterial(base_color=[1, 1, 1, 1], roughness=roughness, specular=specular)
    material.base_color_texture = sapien.render.RenderTexture2D(str(folder/'assets/ceramic.png'))
    inner = sapien.render.RenderMaterial(base_color=[.055, .05, .043, 1], roughness=.9)
    builder = scene.create_actor_builder()
    builder.add_visual_from_file(str(folder/'assets/jar.obj'), material=material)
    builder.add_visual_from_file(str(folder/'assets/jar_inside.obj'), material=inner)
    jar = builder.build_static(name='jar')
    support = scene.create_actor_builder()
    support.add_box_visual(half_size=[.65, .42, .025], material=[.06, .10, .15, 1])
    support.build_static(name='conveyor').set_pose(sapien.Pose([0, 0, -.025]))
    return jar
