"""Shared interactive capture for both fixed-camera calibration target types."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from xarm_sdk_control.calibration_data import (
    bind_intrinsics,
    current_scene_id,
    file_sha256,
    fit_quality,
    intrinsic_health,
    load_intrinsics,
    write_new_json,
)
from xarm_sdk_control.calibration_sampling import (
    StationaryBoardGate,
    check_view,
    duplicate_pose,
    pair_frame_rejection,
    read_color_pair,
    start_color_pipeline,
)
from xarm_sdk_control.calibration_target import CalibrationTarget
from xarm_sdk_control.multiview_calibration import (
    calibrate_relative_camera_pose_from_observations,
)
from xarm_sdk_control.run_output import new_timestamped_run_dir

ROOT = Path(__file__).resolve().parent
WINDOW = "WineJar paired board calibration"


class PendingSample:
    """One key press requests one future qualified pair, with time to return hands."""

    def __init__(self):
        self.requested_at = None

    def update(self, key, now, ready):
        if key in (ord("s"), ord("S")):
            if self.requested_at is not None:
                self.requested_at = None
                return False, "Save request cancelled"
            self.requested_at = now
            return False, "ARMED: steady board; S cancel (30s timeout)"
        if key in (ord("c"), ord("C"), ord("q"), ord("Q"), 27):
            self.requested_at = None
            return False, None
        if self.requested_at is None:
            return False, None
        elapsed = now - self.requested_at
        if elapsed >= 30.0:
            self.requested_at = None
            return False, "Save request timed out; S to try again"
        if elapsed >= 2.0 and ready:
            self.requested_at = None
            return True, None
        return False, None


def parse_args(kind):
    parser = argparse.ArgumentParser(
        description="Keep cameras fixed; hold board still. S arm/cancel one qualified sample, C solve, Q cancel. No robot motion."
    )
    for role in ("reference", "target"):
        parser.add_argument(f"--{role}-id", required=True)
        parser.add_argument(f"--{role}-serial", required=True)
        parser.add_argument(f"--{role}-intrinsics", type=Path, required=True)
    if kind == "chessboard":
        parser.add_argument("--pattern-cols", type=int, required=True)
        parser.add_argument("--pattern-rows", type=int, required=True)
    else:
        parser.add_argument("--squares-x", type=int, required=True)
        parser.add_argument("--squares-y", type=int, required=True)
        parser.add_argument("--marker-size-mm", type=float, required=True)
        parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--square-size-mm", type=float, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=45)
    parser.add_argument("--min-corners-per-camera", type=int, default=20)
    parser.add_argument("--min-samples", type=int, default=12)
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument("--min-board-area-ratio", type=float, default=0.03)
    parser.add_argument("--max-board-area-ratio", type=float, default=0.75)
    parser.add_argument("--max-pnp-reprojection-error-px", type=float, default=1.0)
    parser.add_argument("--min-corner-spread-ratio", type=float, default=0.12)
    parser.add_argument("--hold-still-seconds", type=float, default=0.6)
    parser.add_argument("--max-still-motion-px", type=float, default=1.0)
    parser.add_argument("--min-sample-shift-px", type=float, default=12.0)
    parser.add_argument("--max-pair-skew-ms", type=float, default=100.0)
    parser.add_argument(
        "--method", choices=("pose_average", "stereo"), default="pose_average"
    )
    parser.add_argument("--scene-id")
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "data/multiview_neck/paired_samples"
    )
    parser.add_argument(
        "--calibration-root", type=Path, default=ROOT / "calibration/multiview_neck"
    )
    args = parser.parse_args()
    if (
        args.reference_serial == args.target_serial
        or args.reference_id == args.target_id
    ):
        parser.error("Camera identities must differ")
    if (
        min(args.width, args.height, args.fps) <= 0
        or args.warmup < 0
        or not 4 <= args.min_samples <= args.max_samples
    ):
        parser.error("Invalid capture dimensions or sample counts")
    positive = (
        args.square_size_mm,
        args.max_pnp_reprojection_error_px,
        args.hold_still_seconds,
        args.max_still_motion_px,
        args.min_sample_shift_px,
        args.max_pair_skew_ms,
    )
    if not all(np.isfinite(v) and v > 0 for v in positive):
        parser.error("Thresholds must be finite and positive")
    if (
        not 0 < args.min_board_area_ratio < args.max_board_area_ratio < 1
        or not 0 < args.min_corner_spread_ratio < 1
        or args.min_corners_per_camera < 8
    ):
        parser.error("Invalid observation quality limits")
    return args


def main(kind):
    args = parse_args(kind)
    scene_id = args.scene_id or current_scene_id()
    if scene_id != current_scene_id():
        raise ValueError("Requested scene is not the current physical scene")
    spec = {"square_size_mm": args.square_size_mm}
    if kind == "chessboard":
        spec.update(
            inner_corners_cols=args.pattern_cols, inner_corners_rows=args.pattern_rows
        )
    else:
        spec.update(
            squares_x=args.squares_x,
            squares_y=args.squares_y,
            marker_size_mm=args.marker_size_mm,
            dictionary=args.dictionary,
        )
    target = CalibrationTarget(kind, spec)
    size = (args.width, args.height)
    matrices, distortions, camera_records = [], [], {}
    for role in ("reference", "target"):
        path = getattr(args, role + "_intrinsics").expanduser().resolve()
        serial = getattr(args, role + "_serial")
        matrix, distortion, intrinsic = load_intrinsics(path, serial, size)
        if not intrinsic_health(matrix, distortion, size)["passed"]:
            raise ValueError(
                f"{role}: intrinsic model fails full-image checks; evaluate/refit intrinsics before collecting new pairs"
            )
        matrices.append(matrix)
        distortions.append(distortion)
        record = {
            "camera_id": getattr(args, role + "_id"),
            "serial": serial,
            "intrinsics_path": str(path),
        }
        bind_intrinsics(record, intrinsic)
        camera_records[role + "_camera"] = record
    run = new_timestamped_run_dir(args.output_root.expanduser().resolve())
    payload = {
        "schema_version": 2,
        "type": f"fixed_camera_pair_{kind}_extrinsics",
        "calibration_status": "candidate",
        "scene_id": scene_id,
        kind: spec,
        **camera_records,
        "created_at": datetime.now().astimezone().isoformat(),
        "sample_run_dir": str(run),
        "capture": {
            "width": size[0],
            "height": size[1],
            "fps": args.fps,
            "color_format": "BGR8",
        },
        "detector_version": "board_observation_v2",
        "samples": [],
        "coordinate_convention": {
            "units": "metres",
            "transform": "p_reference = T_reference_from_target @ p_target",
        },
    }
    payload["admission_thresholds"] = {
        k: getattr(args, k)
        for k in (
            "min_corners_per_camera",
            "min_board_area_ratio",
            "max_board_area_ratio",
            "max_pnp_reprojection_error_px",
            "min_corner_spread_ratio",
            "hold_still_seconds",
            "max_still_motion_px",
            "min_sample_shift_px",
            "max_pair_skew_ms",
        )
    }
    write_new_json(run / "session.json", {**payload, "status": "collecting"})
    import pyrealsense2 as rs

    context, pipelines = rs.context(), []
    saved, previous = [], None
    pending = PendingSample()
    stability = StationaryBoardGate(args.hold_still_seconds, args.max_still_motion_px)
    diagnostics = (run / "sampling_diagnostics.jsonl").open("x", buffering=1)
    last_logged = float("-inf")
    try:
        for role in ("reference", "target"):
            pipeline, _ = start_color_pipeline(
                context, getattr(args, role + "_serial"), *size, args.fps
            )
            pipelines.append(pipeline)
        for _ in range(args.warmup):
            read_color_pair(pipelines)
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1280, 540)
        print(
            f"{kind}: {args.reference_id} <- {args.target_id}; no robot connection",
            flush=True,
        )
        message = "Hold board still in both cameras"
        while True:
            loop_started = time.monotonic()
            colors, timing = read_color_pair(pipelines)
            read_finished = time.monotonic()
            observations = tuple(target.detect(color) for color in colors)
            detection_finished = time.monotonic()
            frame_reason = pair_frame_rejection(timing, previous, args.max_pair_skew_ms)
            fresh = frame_reason is None
            previous = timing
            still = stability.update(
                observations if fresh else (None, None), detection_finished
            )
            quality, reasons = [], []
            for observation, matrix, distortion in zip(
                observations, matrices, distortions
            ):
                reason, metrics = check_view(
                    observation,
                    size,
                    matrix,
                    distortion,
                    args.min_corners_per_camera,
                    args.min_board_area_ratio,
                    args.max_pnp_reprojection_error_px,
                    args.min_corner_spread_ratio,
                )
                if metrics.get("area_ratio", 0) > args.max_board_area_ratio:
                    reason = "board too large"
                quality.append(metrics)
                reasons.append(reason)
            duplicate = all(o is not None for o in observations) and duplicate_pose(
                observations, saved, args.min_sample_shift_px
            )
            reason = next(
                (f"{label}: {r}" for label, r in zip(
                    (args.reference_id, args.target_id), reasons
                ) if r), None
            )
            reason = (
                frame_reason
                or reason
                or (stability.status["reason"] if not still else None)
                or ("repeat pose; move board" if duplicate else None)
            )
            if reason is None and args.method == "stereo":
                _, ir, _ = np.intersect1d(
                    observations[0].corner_ids,
                    observations[1].corner_ids,
                    return_indices=True,
                )
                shared = observations[0].object_points_m[ir]
                if (
                    len(shared) < 8
                    or np.linalg.matrix_rank(shared - shared.mean(axis=0), tol=1e-6) < 2
                ):
                    reason = "stereo requires 8 shared non-collinear corners"
            diagnostic = {
                "frame_pair": timing,
                "fresh": fresh,
                "frame_rejection": frame_reason,
                "stationary": still,
                "stationary_status": stability.status,
                "corner_counts": [len(o.corner_ids) if o is not None else 0 for o in observations],
                "quality": quality,
                "rejection": reason,
                "read_seconds": read_finished - loop_started,
                "detection_seconds": detection_finished - read_finished,
            }
            displays = []
            for color, obs, label in zip(
                colors, observations, (args.reference_id, args.target_id)
            ):
                display = color.copy()
                if obs is not None:
                    for p in obs.corners_px.reshape(-1, 2):
                        cv2.circle(
                            display, tuple(np.rint(p).astype(int)), 2, (0, 220, 0), 1
                        )
                cv2.putText(
                    display,
                    label,
                    (12, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                displays.append(display)
            combined = cv2.hconcat(displays)
            motion = stability.status["anchor_motion_px"]
            motion_text = "/".join("?" if v is None else f"{v:.2f}" for v in motion) if motion else "-"
            interval = stability.status["update_interval_s"]
            detail = (
                f"corners {diagnostic['corner_counts']} | skew {timing['skew_ms']:.1f}ms "
                f"({timing['skew_basis']}) | dt {interval or 0:.3f}s | "
                f"motion {motion_text}px | hold {stability.status['held_seconds']:.2f}/{args.hold_still_seconds:.2f}s"
            )
            cv2.putText(combined, detail, (12, size[1] - 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)
            cv2.putText(
                combined,
                f"{len(saved)}/{args.max_samples} | {reason or 'READY'} | S arm/cancel / C solve / Q quit",
                (12, size[1] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255) if reason else (0, 255, 0),
                1,
            )
            cv2.putText(
                combined,
                message,
                (12, size[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )
            cv2.imshow(WINDOW, combined)
            key = cv2.waitKey(1) & 0xFF
            if (
                key in (ord("q"), ord("Q"), 27)
                or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
            ):
                print(
                    f"Cancelled; raw samples remain in {run}; no calibration published"
                )
                return 1
            save_now, update_message = pending.update(key, detection_finished, reason is None)
            if update_message:
                message = update_message
                print(message + " | " + detail, flush=True)
            diagnostic["save_requested"] = key in (ord("s"), ord("S"))
            diagnostic["save_pending"] = pending.requested_at is not None
            diagnostic["save_triggered"] = save_now
            if (detection_finished - last_logged >= 1.0
                    or diagnostic["save_requested"] or save_now or update_message):
                diagnostics.write(json.dumps(diagnostic, allow_nan=False) + "\n")
                last_logged = detection_finished
            if save_now:
                index = len(saved) + 1
                record = {
                    "sample_index": index,
                    "captured_at": datetime.now().astimezone().isoformat(),
                    "frame_pair": timing,
                    "stationary_evidence": stability.status,
                    "admission_quality": dict(zip(("reference", "target"), quality)),
                }
                for prefix, color, obs, metrics in zip(
                    ("reference", "target"), colors, observations, quality
                ):
                    path = run / f"sample_{index:02d}_{prefix}.png"
                    if not cv2.imwrite(str(path), color):
                        raise OSError(f"Failed to save {path}")
                    record.update(
                        {
                            prefix + "_color_path": path.name,
                            prefix + "_image_sha256": file_sha256(path),
                            prefix + "_corner_points_px": obs.corners_px.reshape(
                                -1, 2
                            ).tolist(),
                            prefix + "_corner_ids": obs.corner_ids.tolist(),
                            prefix + "_board_area_ratio": metrics["area_ratio"],
                        }
                    )
                write_new_json(run / f"sample_{index:02d}.json", record)
                payload["samples"].append(record)
                saved.append(observations)
                message = f"Saved {index}; change board position and tilt"
                print(message, flush=True)
            if key in (ord("c"), ord("C")) and len(saved) < args.min_samples:
                message = f"Need {args.min_samples} samples"
                continue
            if len(saved) >= args.max_samples or (
                key in (ord("c"), ord("C")) and len(saved) >= args.min_samples
            ):
                break
    finally:
        diagnostics.close()
        for pipeline in pipelines:
            pipeline.stop()
        cv2.destroyAllWindows()
    fit = calibrate_relative_camera_pose_from_observations(
        [p[0].object_points_m for p in saved],
        [p[0].corners_px for p in saved],
        [p[1].object_points_m for p in saved],
        [p[1].corners_px for p in saved],
        matrices[0],
        distortions[0],
        matrices[1],
        distortions[1],
        args.max_pnp_reprojection_error_px,
        method=args.method,
        image_size_px=size,
    )
    payload["quality"] = fit_quality(fit, len(saved))
    payload["transform_reference_from_target"] = (
        fit.transform_reference_from_target.tolist()
    )
    path = (
        args.calibration_root.expanduser().resolve()
        / f"{args.reference_id}_from_{args.target_id}_{kind}_extrinsics_{run.name}.json"
    )
    write_new_json(path, payload)
    print(f"Saved unaccepted candidate: {path}")
    return 0
