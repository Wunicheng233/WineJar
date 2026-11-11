"""Native left-IR point geometry, with original RGB sampled onto those points.

Only the color/mask is reprojected. Native optical-axis depth and K/T are kept;
depth is never resampled onto the RGB grid. The measured native point z-buffer
resolves collisions in RGB pixels, but cannot establish visibility against RGB
surfaces absent from the native cloud. No target geometry is consulted.
"""
from dataclasses import asdict
from hashlib import sha256
from time import perf_counter

import numpy as np
from scipy.spatial import QhullError

from .depth_profile import DepthFrame, ProfileRejection
from .reliable_output import ReliabilitySettings, _initial
from .stereo_robust import (
    BlueScreenSettings, _attach_rejected_preparation, _prepare, _reliable_output,
    blue_screen_foreground,
)
from .task_validated_output import TaskValidationSettings, _validate_task


def map_rgb_onto_native(rgb_frame, native_camera, depth_z_m, *, foreground_settings=BlueScreenSettings()):
    """Nearest original RGB color/mask for each observed native 3D point.

    Nonfinite/nonpositive depths, points behind RGB, out-of-frame projections,
    and points behind another native point at the same nearest RGB pixel are
    excluded by an explicit mask. Unmapped RGB-array entries are zero/unknown;
    they are never interpreted as foreground. No image colors are fabricated
    to drive the frozen HSV extractor.
    """
    depth = np.asarray(depth_z_m)
    if depth.shape != (native_camera.height, native_camera.width):
        raise ValueError("Native depth dimensions differ from camera dimensions")
    original_rgb = np.asarray(rgb_frame.rgb)
    camera = rgb_frame.camera
    if original_rgb.shape != (camera.height, camera.width, 3) or original_rgb.dtype != np.uint8:
        raise ValueError("RGB image dimensions/type differ from RGB camera")
    source_mask = blue_screen_foreground(original_rgb, foreground_settings)
    native_rgb = np.zeros((*depth.shape, 3), np.uint8)
    native_mask = np.zeros(depth.shape, bool)
    mapping_valid = np.zeros(depth.shape, bool)
    yy, xx = np.nonzero(np.isfinite(depth) & (depth > 0))
    world = native_camera.unproject(np.c_[xx, yy], depth[yy, xx])
    rgb_points = (world-camera.T_world_from_camera[:3, 3]) @ camera.T_world_from_camera[:3, :3]
    front = np.isfinite(rgb_points).all(axis=1) & (rgb_points[:, 2] > 0)
    indices = np.flatnonzero(front)
    projected = rgb_points[front] @ camera.K.T
    uv = np.rint(projected[:, :2]/projected[:, 2:3]).astype(np.int64)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < camera.width) & (uv[:, 1] >= 0) & (uv[:, 1] < camera.height)
    indices, uv = indices[inside], uv[inside]
    z_buffer = np.full(camera.width*camera.height, np.inf)
    addresses = uv[:, 1]*camera.width+uv[:, 0]
    np.minimum.at(z_buffer, addresses, rgb_points[indices, 2])
    visible = rgb_points[indices, 2] <= z_buffer[addresses]
    # Colors from original RGB are retained even for measured-point occlusions;
    # the explicit validity/foreground masks decide which ones enter geometry.
    native_rgb[yy[indices], xx[indices]] = original_rgb[uv[:, 1], uv[:, 0]]
    mapping_valid[yy[indices[visible]], xx[indices[visible]]] = True
    selected = visible & source_mask[uv[:, 1], uv[:, 0]]
    native_mask[yy[indices[selected]], xx[indices[selected]]] = True
    metadata = dict(
        method="native_3D_to_original_RGB_nearest_color_and_blue_key_mask",
        depth_semantics="native left-IR optical-axis Z in metres; not RGB-grid depth",
        color_semantics="original RGB sampled by measured 3D reprojection; not native RGB exposure",
        visibility="nearest measured-native RGB-Z per rounded RGB pixel; no RGB visibility oracle",
        limitations="RGB/native disocclusion and erroneous depths can misassign color/mask",
        foreground_settings=asdict(foreground_settings), valid_depth_points=len(xx),
        projected_inside_rgb_points=len(indices), native_visible_points=int(visible.sum()),
        foreground_points=int(native_mask.sum()),
        source_rgb_sha256=sha256(np.ascontiguousarray(original_rgb).tobytes()).hexdigest(),
        mask_sha256=sha256(native_mask.astype(np.uint8).tobytes()).hexdigest())
    return DepthFrame(native_camera, native_rgb, depth), native_mask, mapping_valid, metadata


def estimate_native_stereo(frames, masks, mapping_metadata, *, settings=TaskValidationSettings()):
    """Frozen portfolio with explicitly mapped original-RGB foreground masks."""
    if len(frames) != len(masks) or len(frames) != len(mapping_metadata):
        raise ValueError("One explicit mask and mapping record required per frame")
    started = perf_counter()
    primary = _reliable_output(frames, masks)
    attempts = list(primary.diagnostics.get("attempts", []))
    best, last_alternative_preparation = None, None
    if not primary.accepted:
        for depth in ReliabilitySettings().candidate_depths_mm:
            try:
                prep = _prepare(frames, masks, depth)
                last_alternative_preparation = prep
                fit = _initial(prep["problem"], prep["joint"])
            except (ProfileRejection, np.linalg.LinAlgError, QhullError) as error:
                attempts.append(dict(depth_mm=depth, path="task_alternative_preparation", accepted=False, reasons=[str(error)]))
                continue
            candidates = []
            for label, x, ok in [("task_validated_joint_alternative", fit.x, bool(fit.success)),
                                 ("task_validated_observed_axis", np.zeros(4), True)]:
                try:
                    candidate = _validate_task(prep, x, frames, settings, label, ok)
                    attempts.append(dict(depth_mm=depth, path=label, accepted=candidate.accepted, reasons=candidate.reasons))
                    if candidate.accepted:
                        candidates.append(candidate)
                except (ProfileRejection, np.linalg.LinAlgError, QhullError) as error:
                    attempts.append(dict(depth_mm=depth, path=label, accepted=False, reasons=[str(error)]))
            if not candidates:
                continue
            if len(candidates) > 1:
                a, b = candidates[:2]
                gap_center = float(np.linalg.norm(a.center-b.center)*1000)
                gap_normal = float(np.degrees(np.arctan2(np.linalg.norm(np.cross(a.normal, b.normal)), a.normal @ b.normal)))
                if gap_center > settings.agreement_center_mm or gap_normal > settings.agreement_normal_deg:
                    attempts.append(dict(depth_mm=depth, path="task_alternative_agreement", accepted=False,
                                         reasons=["ambiguous_task_alternatives"], center_gap_mm=gap_center, normal_gap_deg=gap_normal))
                    continue
            best = min(candidates, key=lambda c:c.diagnostics["heldout_context_rms_mm"])
            break
    result = best if best is not None else primary
    if result.center is None and last_alternative_preparation is not None:
        _attach_rejected_preparation(result, last_alternative_preparation, frames)
    result.diagnostics.update(method="native_ir_rgb_mapped", attempts=attempts,
                              primary_rejection=[] if primary.accepted else primary.reasons,
                              task_validation_settings=asdict(settings), native_color_mapping=mapping_metadata,
                              geometry_change="none: frozen mouth plane, axis/profile solvers and acceptance thresholds",
                              uncertainty="not_estimated")
    result.elapsed_ms = (perf_counter()-started)*1000
    return result
