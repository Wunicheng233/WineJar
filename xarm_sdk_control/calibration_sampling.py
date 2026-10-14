"""Image-space sample admission checks, separate from extrinsic estimation."""

from __future__ import annotations

import time

import cv2
import numpy as np

from xarm_sdk_control.multiview_calibration import (
    normalized_board_area_ratio,
    reprojection_error_px,
)


def corner_displacement(first, second):
    if first is None or second is None:
        return float("inf")
    common, a, b = np.intersect1d(
        first.corner_ids, second.corner_ids, return_indices=True
    )
    if len(common) < 8:
        return float("inf")
    return float(
        np.percentile(
            np.linalg.norm(
                first.corners_px.reshape(-1, 2)[a]
                - second.corners_px.reshape(-1, 2)[b],
                axis=1,
            ),
            90,
        )
    )


class StationaryBoardGate:
    """Require a continuous interval near a fixed anchor; slow drift must not pass."""

    def __init__(self, hold_seconds=0.6, motion_px=1.0):
        if (
            not np.isfinite(hold_seconds)
            or hold_seconds <= 0
            or not np.isfinite(motion_px)
            or motion_px <= 0
        ):
            raise ValueError("Stationary gate limits must be positive and finite")
        self.hold_seconds, self.motion_px = hold_seconds, motion_px
        self.anchor = None
        self.started = None
        self.last_time = None
        self.status = {}

    def update(self, pair, now):
        interval = None if self.last_time is None else now - self.last_time
        gap = interval is not None and (
            interval < 0 or interval > max(0.3, self.hold_seconds)
        )
        self.last_time = now
        self.status = {
            "update_interval_s": interval,
            "anchor_motion_px": None,
            "held_seconds": 0.0,
            "reason": "accumulating stationary time",
        }
        if gap:
            self.anchor, self.started = None, None
            self.status["reason"] = (
                "clock moved backwards" if interval < 0 else "processing interval too long"
            )
        if any(item is None for item in pair):
            self.anchor, self.started = None, None
            self.status["reason"] = "board detection missing"
            return False
        if self.anchor is None:
            self.anchor, self.started = pair, now
            return False
        motion = [corner_displacement(a, b) for a, b in zip(self.anchor, pair)]
        self.status["anchor_motion_px"] = [
            value if np.isfinite(value) else None for value in motion
        ]
        if any(value > self.motion_px for value in motion):
            self.anchor, self.started = pair, now
            self.status["reason"] = (
                "too few shared corners" if not all(np.isfinite(motion)) else "board moved"
            )
            return False
        self.status["held_seconds"] = now - self.started
        ready = now - self.started >= self.hold_seconds
        if ready:
            self.status["reason"] = None
        return ready



def duplicate_pose(pair, accepted_pairs, minimum_shift_px=12.0):
    return any(
        all(corner_displacement(a, b) < minimum_shift_px for a, b in zip(pair, old))
        for old in accepted_pairs
    )


def check_view(
    detection,
    image_size,
    camera_matrix,
    distortion,
    minimum_corners=20,
    minimum_area=0.03,
    maximum_pnp_error=1.0,
    minimum_spread_ratio=0.12,
):
    """Return rejection reason and measured quality; thresholds are image-resolution dependent."""
    if detection is None or len(detection.corner_ids) < minimum_corners:
        return "too few corners", {}
    pixels = detection.corners_px.reshape(-1, 2)
    if not np.all(np.isfinite(pixels)) or not np.all(
        np.isfinite(detection.object_points_m)
    ):
        return "nonfinite corners", {}
    area = normalized_board_area_ratio(pixels, image_size)
    singular = np.linalg.svd(pixels - pixels.mean(axis=0), compute_uv=False)
    spread = float(singular[1] / max(singular[0], 1e-9))
    metrics = {"corners": len(pixels), "area_ratio": area, "spread_ratio": spread}
    if area < minimum_area:
        return "board too small", metrics
    if spread < minimum_spread_ratio:
        return "corners too compressed", metrics
    ok, rvec, tvec = cv2.solvePnP(
        detection.object_points_m, detection.corners_px, camera_matrix, distortion
    )
    if not ok:
        return "board pose failed", metrics
    error = reprojection_error_px(
        detection.object_points_m,
        detection.corners_px,
        camera_matrix,
        distortion,
        rvec,
        tvec,
    )
    metrics["pnp_rmse_px"] = error
    if not np.isfinite(error) or error > maximum_pnp_error:
        return "single-view reprojection error", metrics
    return None, metrics


def start_color_pipeline(context, serial, width, height, fps):
    import pyrealsense2 as rs

    from xarm_sdk_control.realsense_capture import camera_info, choose_device

    info = camera_info(choose_device(context, serial))
    pipeline, config = rs.pipeline(context), rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipeline.start(config)
    return pipeline, info


def read_color_frame(pipeline):
    frame = pipeline.wait_for_frames(3000).get_color_frame()
    if not frame:
        raise RuntimeError("Missing RGB frame")
    received = time.monotonic_ns()
    metadata = {
        "frame_number": frame.get_frame_number(),
        "device_timestamp_ms": frame.get_timestamp(),
        "timestamp_domain": str(frame.get_frame_timestamp_domain()),
        "host_received_monotonic_ns": received,
    }
    return np.asanyarray(frame.get_data()).copy(), metadata


def read_color_pair(pipelines):
    frames = [read_color_frame(pipeline) for pipeline in pipelines]
    stamps = [item[1] for item in frames]
    host_skew = (
        abs(
            stamps[1]["host_received_monotonic_ns"]
            - stamps[0]["host_received_monotonic_ns"]
        )
        / 1e6
    )
    comparable = stamps[0]["timestamp_domain"] == stamps[1]["timestamp_domain"] and all(
        "global_time" in stamp["timestamp_domain"]
        or "system_time" in stamp["timestamp_domain"]
        for stamp in stamps
    )
    skew = (
        abs(stamps[1]["device_timestamp_ms"] - stamps[0]["device_timestamp_ms"])
        if comparable
        else host_skew
    )
    return [item[0] for item in frames], {
        "frames": stamps,
        "skew_ms": skew,
        "skew_basis": "device_common_clock" if comparable else "host_receipt_only",
        "hardware_synchronized": False,
    }


def pair_frame_rejection(metadata, previous, maximum_skew_ms=100.0):
    """Explain frame rejection without conflating it with board movement."""
    frames = metadata["frames"]
    if len(frames) != 2 or (previous is not None and len(previous["frames"]) != 2):
        return "invalid frame pair"
    skew = metadata["skew_ms"]
    if not np.isfinite(skew) or skew < 0:
        return "invalid frame skew"
    if skew > maximum_skew_ms:
        return "frame skew too large"
    if previous is not None and any(
        new["frame_number"] <= old["frame_number"]
        for new, old in zip(frames, previous["frames"])
    ):
        return "repeated or reversed frame"
    return None


def pair_frames_fresh(metadata, previous, maximum_skew_ms=100.0):
    return pair_frame_rejection(metadata, previous, maximum_skew_ms) is None
