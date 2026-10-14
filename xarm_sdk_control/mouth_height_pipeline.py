"""Offline data loading, quality checks and visualization for rim localization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from xarm_sdk_control.mouth_height import (
    estimate_camera_top_height,
    transform_reference_points_to_workcell,
)
from xarm_sdk_control.multiview_rig import (
    load_fixed_camera_rig,
    rig_geometry_fingerprint,
)
from xarm_sdk_control.neck_observation import (
    detect_aperture,
    fit_shoulder_transition,
    refine_horizontal_edges,
    silhouette_edges,
    top_arc,
)
from xarm_sdk_control.rim_geometry import (
    RimObservation,
    circle_points,
    fit_multiview_rim,
    intersect_height,
    project_workcell,
    surface_point_at_silhouette,
)
from xarm_sdk_control.silhouette_geometry import segment_fixed_background_foreground
from xarm_sdk_control.workcell_frame import deproject_depth_roi_to_reference


@dataclass(frozen=True)
class LocalizationSettings:
    minimum_height_mm: float = 30.0
    maximum_height_mm: float = 600.0
    profile_window_mm: float = 26.0
    edge_search_px: float = 1.5
    maximum_edge_fit_error_px: float = 1.2
    maximum_tangent_radius_error_mm: float = 4.0
    maximum_neck_spread_mm: float = 5.0
    maximum_rim_rmse_px: float = 2.0
    maximum_depth_disagreement_mm: float = 8.0
    bootstrap_samples: int = 20
    random_seed: int = 7

    def __post_init__(self):
        values = [
            value
            for key, value in vars(self).items()
            if key not in {"bootstrap_samples", "random_seed"}
        ]
        if not all(np.isfinite(value) and value > 0 for value in values):
            raise ValueError("Localization thresholds must be positive finite numbers")
        if (
            self.maximum_height_mm <= self.minimum_height_mm
            or self.bootstrap_samples < 0
        ):
            raise ValueError("Invalid height range or bootstrap count")


def _read_frame(directory, camera):
    metadata = json.loads((directory / "frame.json").read_text())
    if metadata.get("camera", {}).get("serial") != camera.serial:
        raise ValueError(
            f"{camera.camera_id}: capture serial does not match calibration"
        )
    color = cv2.imread(str(directory / "color.png"))
    depth = cv2.imread(str(directory / "depth_raw.png"), cv2.IMREAD_UNCHANGED)
    expected = (camera.image_size_px[1], camera.image_size_px[0])
    if (
        color is None
        or depth is None
        or color.shape[:2] != expected
        or depth.shape != expected
    ):
        raise ValueError(
            f"{camera.camera_id}: missing images or resolution differs from calibration"
        )
    if metadata.get("stream", {}).get("depth_aligned_to") != "color":
        raise ValueError(f"{camera.camera_id}: depth must be aligned to color")
    return color, depth, metadata


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def localize_jar(
    rig_path: Path,
    workcell_path: Path,
    capture: Path,
    rois: dict[str, tuple[int, int, int, int]],
    *,
    masks_dir: Path | None = None,
    background_capture: Path | None = None,
    settings: LocalizationSettings = LocalizationSettings(),
) -> tuple[dict, dict[str, np.ndarray], dict[str, np.ndarray]]:
    rig = load_fixed_camera_rig(rig_path)
    workcell = json.loads(workcell_path.read_text())
    capture_metadata = json.loads((capture / "capture.json").read_text())
    if (
        workcell.get("type") != "fixed_workcell_support_plane"
        or workcell.get("status") != "ok"
    ):
        raise ValueError("Expected a successful workcell support-plane calibration")
    if capture_metadata.get("type") != "fixed_multiview_rgbd_capture":
        raise ValueError("Expected a static multiview RGB-D capture")
    fingerprint = rig_geometry_fingerprint(rig)
    for label, payload in [("capture", capture_metadata), ("workcell", workcell)]:
        if payload.get("rig_geometry_fingerprint", fingerprint) != fingerprint:
            raise ValueError(
                f"{label}: recorded calibration geometry differs from the selected rig"
            )
    if workcell.get("reference_frame") != rig.reference_camera_id:
        raise ValueError("Workcell and rig reference camera differ")
    if set(rois) != set(rig.cameras) or len(rig.cameras) != 3:
        raise ValueError(
            "Exactly three calibrated cameras and matching ROIs are required"
        )
    if masks_dir is None and background_capture is None:
        raise ValueError("Supply learned masks or an empty background capture")
    if masks_dir is not None:
        mask_meta = json.loads((masks_dir / "segmentation.json").read_text())
        if Path(mask_meta["capture_dir"]).resolve() != capture.resolve():
            raise ValueError("Learned masks belong to a different capture")
    transform = np.asarray(workcell["transform_workcell_from_reference"], float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("Invalid workcell transform")
    inverse = np.linalg.inv(transform)
    images, depths, metadata, masks = {}, {}, {}, {}
    diagnostics = {}
    for cid, camera in rig.cameras.items():
        images[cid], depths[cid], metadata[cid] = _read_frame(capture / cid, camera)
        x, y, width, height = rois[cid]
        if (
            min(x, y) < 0
            or min(width, height) <= 0
            or x + width > camera.image_size_px[0]
            or y + height > camera.image_size_px[1]
        ):
            raise ValueError(f"{cid}: invalid object ROI")
        if masks_dir is not None:
            mask = cv2.imread(str(masks_dir / cid / "mask.png"), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != depths[cid].shape:
                raise ValueError(f"{cid}: missing or mismatched learned mask")
        else:
            background, _, _ = _read_frame(background_capture / cid, camera)
            mask = segment_fixed_background_foreground(
                images[cid], background, roi_xywh=rois[cid], close_kernel=5
            ).silhouette_mask
        keep = np.zeros_like(mask)
        keep[y : y + height, x : x + width] = mask[y : y + height, x : x + width]
        masks[cid] = keep
        diagnostics[cid] = {
            "roi_xywh": list(rois[cid]),
            "mask_area_px": int(np.count_nonzero(keep)),
        }

    ref = rig.reference_camera_id
    camera = rig.camera(ref)
    points, pixels, _ = deproject_depth_roi_to_reference(
        depths[ref],
        metadata[ref]["stream"]["depth_scale_m_per_unit"],
        camera.camera_matrix,
        camera.distortion_coefficients,
        camera.transform_reference_from_camera,
        rois[ref],
    )
    selected = masks[ref][pixels[:, 1], pixels[:, 0]] != 0
    workcell_points = transform_reference_points_to_workcell(
        points[selected], transform
    )
    depth_top = estimate_camera_top_height(
        workcell_points,
        planar_radius_m=1.0,
        minimum_object_height_m=settings.minimum_height_mm / 1000,
        maximum_object_height_m=settings.maximum_height_mm / 1000,
    )
    aperture = detect_aperture(images[ref], masks[ref])
    aperture_3d = intersect_height(
        rig, ref, transform, aperture.pixels, depth_top.height_m
    )
    # The fitted ellipse center is not biased by uneven contour point density.
    center = intersect_height(
        rig, ref, transform, aperture.center_px[None], depth_top.height_m
    )[0]
    radius = float(np.median(np.linalg.norm(aperture_3d[:, :2] - center[:2], axis=1)))
    observations = [RimObservation(ref, aperture.pixels, True)]
    transitions = {}
    profile_errors = []
    for cid in rig.cameras:
        if cid == ref:
            continue
        projected = project_workcell(
            rig, cid, inverse, np.array([center, center + [0, 0, 0.01]])
        )
        upward_row_change = projected[0, 1] - projected[1, 1]
        if upward_row_change <= 1e-3:
            raise ValueError(
                f"{cid}: side profiles require an upright camera with observable vertical extent"
            )
        mm_per_row = 10 / upward_row_change
        cx, cy = projected[0]
        first_row = round(cy + 1)
        last_row = first_row + max(12, round(settings.profile_window_mm / mm_per_row))
        left, right = silhouette_edges(masks[cid], first_row, last_row, cx)
        span = (float(np.median(left[:3, 0])), float(np.median(right[:3, 0])))
        arc = top_arc(
            masks[cid],
            (span[0] + 2, span[1] - 2),
            (cy - 30 / mm_per_row, cy + 10 / mm_per_row),
        )
        observations.append(RimObservation(cid, arc, False, span))
        diagnostics[cid].update(
            {
                "rim_observation_count": len(arc),
                "rim_span_px": list(span),
                "profile_rows": [first_row, last_row],
            }
        )
        transitions[cid] = []
        for name, edge, sign in [("left", left, -1), ("right", right, 1)]:
            refined = refine_horizontal_edges(
                images[cid], edge, settings.edge_search_px
            )
            try:
                transition = fit_shoulder_transition(refined, sign)
                transitions[cid].append((name, transition))
            except ValueError as error:
                profile_errors.append(f"{cid}/{name}: {error}")

    rim = fit_multiview_rim(rig, inverse, observations, center, radius)
    reasons = list(profile_errors)
    for cid, errors in rim.residuals_px.items():
        rmse = float(np.sqrt(np.mean(errors**2)))
        diagnostics[cid]["rim_rmse_px"] = rmse
        diagnostics[cid]["rim_p95_px"] = float(np.percentile(np.abs(errors), 95))
        if rmse > settings.maximum_rim_rmse_px:
            reasons.append(
                f"{cid}: rim reprojection RMSE {rmse:.2f}px exceeds threshold"
            )
    if rim.at_bound:
        reasons.append("Rim optimizer reached a parameter bound")
    if rim.condition_number > 1e4:
        reasons.append("Rim geometry is poorly observable in these views")
    depth_disagreement = (depth_top.height_m - rim.center_m[2]) * 1000
    if abs(depth_disagreement) > settings.maximum_depth_disagreement_mm:
        reasons.append(
            f"Overhead depth and RGB rim plane disagree by {depth_disagreement:.1f}mm"
        )

    corner_diagnostics, accepted_lengths = [], {}
    for cid, entries in transitions.items():
        for side, transition in entries:
            points, misses = surface_point_at_silhouette(
                rig, cid, transform, transition.pixel[None], rim
            )
            length = float(-(points[0] - rim.center_m) @ rim.normal)
            accepted = bool(
                transition.fit_error_px <= settings.maximum_edge_fit_error_px
                and abs(misses[0]) * 1000 <= settings.maximum_tangent_radius_error_mm
                and 0 < length < 0.08
            )
            record = {
                "camera_id": cid,
                "side": side,
                "pixel": transition.pixel.tolist(),
                "surface_point_workcell_mm": (points[0] * 1000).tolist(),
                "distance_below_rim_plane_mm": length * 1000,
                "edge_fit_error_px": float(transition.fit_error_px),
                "tangent_radius_error_mm": float(misses[0] * 1000),
                "accepted": accepted,
            }
            corner_diagnostics.append(record)
            if accepted:
                accepted_lengths.setdefault(cid, []).append(length)
    per_camera = {
        cid: float(np.median(values)) for cid, values in accepted_lengths.items()
    }
    neck_length = None
    spread = None
    if len(per_camera) < 2:
        reasons.append("Shoulder boundary lacks support from both side cameras")
    else:
        spread = float(np.ptp(list(per_camera.values())) * 1000)
        neck_length = float(np.median(list(per_camera.values())))
        if spread > settings.maximum_neck_spread_mm:
            reasons.append(f"Side shoulder boundaries disagree by {spread:.1f}mm")

    bootstrap = []
    rng = np.random.default_rng(settings.random_seed)
    for _ in range(settings.bootstrap_samples):
        sampled = [
            RimObservation(
                o.camera_id,
                o.pixels[rng.integers(0, len(o.pixels), len(o.pixels))],
                o.inner,
                o.horizontal_extent_px,
            )
            for o in observations
        ]
        try:
            resampled = fit_multiview_rim(
                rig,
                inverse,
                sampled,
                rim.center_m,
                rim.inner_radius_m,
                rim.outer_radius_m - rim.inner_radius_m,
            )
            if not resampled.at_bound:
                bootstrap.append(np.r_[resampled.center_m, resampled.outer_radius_m])
        except ValueError:
            continue
    bootstrap_intervals = (
        np.percentile(bootstrap, [2.5, 97.5], axis=0) * 1000
        if len(bootstrap) >= 5
        else None
    )
    if (
        settings.bootstrap_samples >= 5
        and len(bootstrap) < settings.bootstrap_samples * 0.8
    ):
        reasons.append(
            "Too many resampled rim fits failed; conditional stability is unreliable"
        )

    top_points = circle_points(rim.center_m, rim.normal, rim.outer_radius_m)
    lower_center = (
        None if neck_length is None else rim.center_m - neck_length * rim.normal
    )
    overlays, closeups = {}, {}
    for cid in rig.cameras:
        overlay = images[cid].copy()
        top = project_workcell(rig, cid, inverse, top_points)
        inner = project_workcell(
            rig,
            cid,
            inverse,
            circle_points(rim.center_m, rim.normal, rim.inner_radius_m),
        )
        cv2.polylines(
            overlay,
            [np.rint(top).astype(np.int32)],
            True,
            (0, 190, 255),
            1,
            cv2.LINE_AA,
        )
        if cid == ref:
            cv2.polylines(
                overlay,
                [np.rint(inner).astype(np.int32)],
                True,
                (0, 220, 0),
                1,
                cv2.LINE_AA,
            )
        if lower_center is not None:
            lower = project_workcell(
                rig,
                cid,
                inverse,
                circle_points(lower_center, rim.normal, rim.outer_radius_m),
            )
            cv2.polylines(
                overlay,
                [np.rint(lower).astype(np.int32)],
                True,
                (255, 120, 0),
                1,
                cv2.LINE_AA,
            )
        center_pixel = project_workcell(rig, cid, inverse, rim.center_m[None])[0]
        cv2.drawMarker(
            overlay,
            tuple(np.rint(center_pixel).astype(int)),
            (255, 0, 255),
            cv2.MARKER_CROSS,
            12,
            1,
        )
        for corner in corner_diagnostics:
            if corner["camera_id"] == cid:
                color = (0, 255, 0) if corner["accepted"] else (0, 0, 255)
                cv2.circle(
                    overlay, tuple(np.rint(corner["pixel"]).astype(int)), 3, color, 1
                )
        label = f"{cid} | rim plane {rim.center_m[2] * 1000:.1f} mm | RMSE {diagnostics[cid]['rim_rmse_px']:.2f} px"
        cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 28), (25, 25, 25), -1)
        cv2.putText(
            overlay,
            label,
            (8, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        overlays[cid] = overlay
        margin = 28
        lo = np.maximum(0, np.floor(top.min(axis=0)).astype(int) - margin)
        hi = np.minimum(
            [overlay.shape[1], overlay.shape[0]],
            np.ceil(top.max(axis=0)).astype(int) + margin + [0, 12],
        )
        closeups[cid] = cv2.resize(
            overlay[lo[1] : hi[1], lo[0] : hi[0]],
            None,
            fx=4,
            fy=4,
            interpolation=cv2.INTER_NEAREST,
        )

    payload = {
        "schema_version": 3,
        "type": "multiview_metric_rim_localization",
        "status": "needs_review" if reasons else "geometry_consistent",
        "absolute_accuracy_validated": False,
        "robot_motion_allowed": False,
        "quality_flags": reasons,
        "measurement_definitions": {
            "rim_plane_center_height": "Z of the fitted mouth-rim plane at its center",
            "rim_z_extent": "Min/max Z of the tilted outer rim; not neck length",
            "rim_to_shoulder_axial_length": "Distance along the fitted plane normal from rim plane to shoulder transition",
            "binding_band": "Not inferred: usable groove below a raised lip needs separate boundary evidence",
        },
        "rim_center_workcell_mm": (rim.center_m * 1000).tolist(),
        "rim_plane_normal_workcell": rim.normal.tolist(),
        "rim_plane_equation_workcell_m": [
            *rim.normal.tolist(),
            -float(rim.normal @ rim.center_m),
        ],
        "rim_plane_tilt_deg": float(
            np.degrees(np.arccos(np.clip(rim.normal[2], -1, 1)))
        ),
        "inner_diameter_mm": float(rim.inner_radius_m * 2000),
        "outer_diameter_mm": float(rim.outer_radius_m * 2000),
        "rim_z_extent_mm": (
            np.array([top_points[:, 2].min(), top_points[:, 2].max()]) * 1000
        ).tolist(),
        "rim_to_shoulder_axial_length_mm": None
        if neck_length is None
        else neck_length * 1000,
        "shoulder_plane_center_workcell_mm": None
        if lower_center is None
        else (lower_center * 1000).tolist(),
        "side_shoulder_spread_mm": spread,
        "binding_band_mm": None,
        "depth_initialization_height_mm": depth_top.height_m * 1000,
        "depth_vs_rgb_plane_difference_mm": depth_disagreement,
        "conditional_uncertainty": {
            "definition": "Image-point resampling only; excludes calibration, segmentation bias, noncircular shape and support-plane error",
            "bootstrap_successes": len(bootstrap),
            "requested_samples": settings.bootstrap_samples,
            "center_xyz_outer_radius_95_interval_mm": None
            if bootstrap_intervals is None
            else bootstrap_intervals.tolist(),
            "scaled_jacobian_condition_number": rim.condition_number,
        },
        "cameras": diagnostics,
        "shoulder_observations": corner_diagnostics,
        "parameters": vars(settings),
        "provenance": {
            "rig_path": str(rig_path.resolve()),
            "rig_sha256": _sha256(rig_path),
            "workcell_path": str(workcell_path.resolve()),
            "workcell_sha256": _sha256(workcell_path),
            "capture_dir": str(capture.resolve()),
            "masks_dir": None if masks_dir is None else str(masks_dir.resolve()),
            "background_capture_dir": None
            if background_capture is None
            else str(background_capture.resolve()),
            "segmentation_method": "GroundingDINO+SAM2"
            if masks_dir
            else "fixed_background_difference",
        },
    }
    return payload, overlays, closeups
