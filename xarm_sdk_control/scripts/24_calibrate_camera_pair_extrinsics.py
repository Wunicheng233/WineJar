#!/usr/bin/env python3
"""Calibrate one fixed side camera relative to a fixed reference camera.

Keep both cameras fixed. Hold the ordinary chessboard completely visible in both
views, keep it still, then press S to save a paired observation. Move the board
to a new position and tilt before saving the next pair. This script never
connects to or moves the xArm.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xarm_sdk_control.multiview_calibration import (  # noqa: E402
    calibrate_relative_camera_pose,
    chessboard_object_points,
    find_chessboard_corners,
    normalized_board_area_ratio,
)
from xarm_sdk_control.realsense_capture import camera_info, choose_device  # noqa: E402


DEFAULT_DATA_ROOT = ROOT / "data" / "multiview_neck" / "extrinsics"
DEFAULT_CALIBRATION_ROOT = ROOT / "calibration" / "multiview_neck"
WINDOW_NAME = "WineJar paired chessboard extrinsic calibration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-id", required=True, help="Stable reference rig name, e.g. overhead.")
    parser.add_argument("--reference-serial", required=True, help="RealSense serial for the reference camera.")
    parser.add_argument("--reference-intrinsics", type=Path, required=True)
    parser.add_argument("--target-id", required=True, help="Stable target rig name, e.g. side_a.")
    parser.add_argument("--target-serial", required=True, help="RealSense serial for the target camera.")
    parser.add_argument("--target-intrinsics", type=Path, required=True)
    parser.add_argument("--pattern-cols", type=int, required=True, help="Chessboard INNER-corner columns.")
    parser.add_argument("--pattern-rows", type=int, required=True, help="Chessboard INNER-corner rows.")
    parser.add_argument("--square-size-mm", type=float, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--min-samples", type=int, default=12)
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument("--min-board-area-ratio", type=float, default=0.03)
    parser.add_argument("--max-board-area-ratio", type=float, default=0.70)
    parser.add_argument("--max-pnp-reprojection-error-px", type=float, default=1.0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--calibration-root", type=Path, default=DEFAULT_CALIBRATION_ROOT)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.reference_serial == args.target_serial:
        raise ValueError("Reference and target cameras must use different serial numbers")
    if args.pattern_cols < 2 or args.pattern_rows < 2 or args.square_size_mm <= 0:
        raise ValueError("Chessboard dimensions and square size must be positive")
    if args.width <= 0 or args.height <= 0 or args.fps <= 0 or args.warmup < 0:
        raise ValueError("Capture width, height, fps must be positive and warmup non-negative")
    if not 4 <= args.min_samples <= args.max_samples:
        raise ValueError("Require 4 <= --min-samples <= --max-samples")
    if not 0 < args.min_board_area_ratio < args.max_board_area_ratio < 1:
        raise ValueError("Board-area ratios must satisfy 0 < min < max < 1")
    if args.max_pnp_reprojection_error_px <= 0:
        raise ValueError("--max-pnp-reprojection-error-px must be positive")


def load_intrinsics(
    path: Path,
    expected_serial: str,
    pattern_cols: int,
    pattern_rows: int,
    square_size_mm: float,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load and validate an intrinsic result by serial, not by its display name."""
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("type") != "fixed_camera_chessboard_intrinsics":
        raise ValueError(f"{resolved} is not a chessboard intrinsic calibration")
    if payload.get("camera", {}).get("serial") != expected_serial:
        raise ValueError(f"{resolved} belongs to serial {payload.get('camera', {}).get('serial')}, not {expected_serial}")
    board = payload.get("chessboard", {})
    if (
        board.get("inner_corners_cols") != pattern_cols
        or board.get("inner_corners_rows") != pattern_rows
        or not np.isclose(board.get("square_size_mm", 0.0), square_size_mm)
    ):
        raise ValueError(f"{resolved} used a different chessboard specification")
    capture = payload.get("capture", {})
    if (capture.get("width"), capture.get("height")) != image_size:
        raise ValueError(f"{resolved} uses {capture.get('width')}x{capture.get('height')}, expected {image_size}")
    return (
        np.asarray(payload["camera_matrix"], dtype=np.float64),
        np.asarray(payload["distortion_coefficients"], dtype=np.float64),
        payload,
    )


def start_color_pipeline(context: rs.context, serial: str, width: int, height: int, fps: int) -> tuple[rs.pipeline, rs.pipeline_profile, dict[str, str]]:
    device = choose_device(context, serial)
    info = camera_info(device)
    pipeline = rs.pipeline(context)
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    return pipeline, pipeline.start(config), info


def read_color_frame(pipeline: rs.pipeline) -> np.ndarray:
    frames = pipeline.wait_for_frames(2000)
    color_frame = frames.get_color_frame()
    if not color_frame:
        raise RuntimeError("No color frame received")
    return np.asanyarray(color_frame.get_data()).copy()


def draw_camera_view(
    color: np.ndarray,
    label: str,
    pattern_size: tuple[int, int],
    found: bool,
    corners: np.ndarray | None,
    area_ratio: float,
    accepted: bool,
) -> np.ndarray:
    display = color.copy()
    if found and corners is not None:
        cv2.drawChessboardCorners(display, pattern_size, corners, True)
    colour = (0, 220, 0) if accepted else (0, 0, 255)
    cv2.putText(display, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    cv2.putText(
        display,
        f"full board={found} area={area_ratio:.1%} accepted={accepted}",
        (12, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        colour,
        2,
    )
    return display


def save_pair_sample(
    run_dir: Path,
    index: int,
    reference_color: np.ndarray,
    target_color: np.ndarray,
    reference_overlay: np.ndarray,
    target_overlay: np.ndarray,
    reference_corners: np.ndarray,
    target_corners: np.ndarray,
    reference_area_ratio: float,
    target_area_ratio: float,
) -> dict[str, object]:
    sample_dir = run_dir / "samples"
    sample_dir.mkdir(exist_ok=True)
    stem = f"sample_{index:02d}"
    paths = {
        "reference_color": sample_dir / f"{stem}_reference.png",
        "target_color": sample_dir / f"{stem}_target.png",
        "reference_overlay": sample_dir / f"{stem}_reference_corners.png",
        "target_overlay": sample_dir / f"{stem}_target_corners.png",
    }
    for path, image in (
        (paths["reference_color"], reference_color),
        (paths["target_color"], target_color),
        (paths["reference_overlay"], reference_overlay),
        (paths["target_overlay"], target_overlay),
    ):
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Failed to save {path}")
    return {
        "sample_index": index,
        "captured_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "reference_color_path": str(paths["reference_color"].relative_to(run_dir)),
        "target_color_path": str(paths["target_color"].relative_to(run_dir)),
        "reference_overlay_path": str(paths["reference_overlay"].relative_to(run_dir)),
        "target_overlay_path": str(paths["target_overlay"].relative_to(run_dir)),
        "reference_corner_points_px": np.asarray(reference_corners, dtype=float).reshape(-1, 2).tolist(),
        "target_corner_points_px": np.asarray(target_corners, dtype=float).reshape(-1, 2).tolist(),
        "reference_board_area_ratio": float(reference_area_ratio),
        "target_board_area_ratio": float(target_area_ratio),
    }


def summary(values: list[float], scale: float = 1.0) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "max": None}
    scaled = np.asarray(values, dtype=np.float64) * scale
    return {"mean": float(np.mean(scaled)), "max": float(np.max(scaled))}


def main() -> int:
    args = parse_args()
    validate_args(args)
    pattern_size = (args.pattern_cols, args.pattern_rows)
    image_size = (args.width, args.height)
    reference_matrix, reference_distortion, reference_payload = load_intrinsics(
        args.reference_intrinsics,
        args.reference_serial,
        *pattern_size,
        args.square_size_mm,
        image_size,
    )
    target_matrix, target_distortion, target_payload = load_intrinsics(
        args.target_intrinsics,
        args.target_serial,
        *pattern_size,
        args.square_size_mm,
        image_size,
    )
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    pair_id = f"{args.reference_id}_from_{args.target_id}"
    run_dir = args.output_root.expanduser().resolve() / pair_id / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    context = rs.context()
    reference_pipeline: rs.pipeline | None = None
    target_pipeline: rs.pipeline | None = None
    samples: list[dict[str, object]] = []
    solve_requested = False
    try:
        reference_pipeline, _, reference_info = start_color_pipeline(
            context, args.reference_serial, args.width, args.height, args.fps
        )
        target_pipeline, _, target_info = start_color_pipeline(
            context, args.target_serial, args.width, args.height, args.fps
        )
        for _ in range(args.warmup):
            read_color_frame(reference_pipeline)
            read_color_frame(target_pipeline)
        print(
            f"Pair calibration: {args.reference_id} ({reference_info['serial']}) <- "
            f"{args.target_id} ({target_info['serial']}); robot connection/motion: none"
        )
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        message = "Hold the chessboard still in both views. S=save pair, C=solve, Q=quit"
        while True:
            reference_color = read_color_frame(reference_pipeline)
            target_color = read_color_frame(target_pipeline)
            reference_found, reference_corners = find_chessboard_corners(reference_color, pattern_size)
            target_found, target_corners = find_chessboard_corners(target_color, pattern_size)
            reference_area = normalized_board_area_ratio(reference_corners, image_size) if reference_found and reference_corners is not None else 0.0
            target_area = normalized_board_area_ratio(target_corners, image_size) if target_found and target_corners is not None else 0.0
            accepted = (
                reference_found
                and target_found
                and args.min_board_area_ratio <= reference_area <= args.max_board_area_ratio
                and args.min_board_area_ratio <= target_area <= args.max_board_area_ratio
            )
            reference_overlay = draw_camera_view(
                reference_color,
                f"reference: {args.reference_id}",
                pattern_size,
                reference_found,
                reference_corners,
                reference_area,
                accepted,
            )
            target_overlay = draw_camera_view(
                target_color,
                f"target: {args.target_id}",
                pattern_size,
                target_found,
                target_corners,
                target_area,
                accepted,
            )
            combined = cv2.hconcat((reference_overlay, target_overlay))
            cv2.putText(combined, f"saved={len(samples)}/{args.max_samples} | {message}", (12, combined.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 255, 255), 2)
            cv2.imshow(WINDOW_NAME, combined)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print("Cancelled; no pair extrinsic calibration was solved.")
                return 1
            if key in (ord("s"), ord("S")):
                if not accepted or reference_corners is None or target_corners is None:
                    message = "Not saved: full chessboard must be detected at a usable size in both cameras"
                    continue
                sample = save_pair_sample(
                    run_dir,
                    len(samples) + 1,
                    reference_color,
                    target_color,
                    reference_overlay,
                    target_overlay,
                    reference_corners,
                    target_corners,
                    reference_area,
                    target_area,
                )
                samples.append(sample)
                message = "Saved. Move board position and tilt, then hold it still before next S"
                print(f"Saved pair {len(samples):02d}: reference={reference_area:.1%}, target={target_area:.1%}")
            if key in (ord("c"), ord("C")):
                if len(samples) < args.min_samples:
                    message = f"Need at least {args.min_samples} valid pairs before solving"
                    continue
                solve_requested = True
                break
            if len(samples) >= args.max_samples:
                solve_requested = True
                break
    finally:
        cv2.destroyAllWindows()
        if reference_pipeline is not None:
            reference_pipeline.stop()
        if target_pipeline is not None:
            target_pipeline.stop()

    if not solve_requested:
        return 1
    object_points = chessboard_object_points(
        args.pattern_cols,
        args.pattern_rows,
        args.square_size_mm / 1000.0,
    )
    calibration = calibrate_relative_camera_pose(
        [np.asarray(sample["reference_corner_points_px"], dtype=np.float64) for sample in samples],
        [np.asarray(sample["target_corner_points_px"], dtype=np.float64) for sample in samples],
        object_points,
        reference_matrix,
        reference_distortion,
        target_matrix,
        target_distortion,
        args.max_pnp_reprojection_error_px,
    )
    pnp_observations = []
    accepted_set = set(calibration.accepted_indices)
    for index, reference_error, target_error in zip(
        calibration.pnp_accepted_indices,
        calibration.reference_reprojection_errors_px,
        calibration.target_reprojection_errors_px,
    ):
        pnp_observations.append(
            {
                "sample_index": index,
                "reference_reprojection_error_px": reference_error,
                "target_reprojection_error_px": target_error,
                "accepted_in_final_transform": index in accepted_set,
            }
        )
    args.calibration_root.expanduser().resolve().mkdir(parents=True, exist_ok=True)
    calibration_path = args.calibration_root.expanduser().resolve() / f"{pair_id}_extrinsics_{timestamp}.json"
    payload = {
        "schema_version": 1,
        "type": "fixed_camera_pair_chessboard_extrinsics",
        "created_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "reference_camera": {
            "camera_id": args.reference_id,
            "serial": args.reference_serial,
            "intrinsics_path": os.path.relpath(
                args.reference_intrinsics.expanduser().resolve(), calibration_path.parent
            ),
            "intrinsics_created_at": reference_payload.get("created_at"),
        },
        "target_camera": {
            "camera_id": args.target_id,
            "serial": args.target_serial,
            "intrinsics_path": os.path.relpath(
                args.target_intrinsics.expanduser().resolve(), calibration_path.parent
            ),
            "intrinsics_created_at": target_payload.get("created_at"),
        },
        "capture": {"width": args.width, "height": args.height, "fps": args.fps, "color_format": "BGR8"},
        "chessboard": {
            "inner_corners_cols": args.pattern_cols,
            "inner_corners_rows": args.pattern_rows,
            "square_size_mm": args.square_size_mm,
        },
        "sample_run_dir": str(run_dir),
        "coordinate_convention": {
            "units": "metres",
            "camera_axes": "+x right, +y down, +z forward",
            "transform": "p_reference = T_reference_from_target @ p_target_homogeneous",
        },
        "transform_reference_from_target": calibration.transform_reference_from_target.tolist(),
        "quality": {
            "paired_sample_count": len(samples),
            "pnp_accepted_indices": calibration.pnp_accepted_indices,
            "accepted_indices": calibration.accepted_indices,
            "rejected_indices": calibration.rejected_indices,
            "pnp_observations": pnp_observations,
            "reference_pnp_reprojection_error_px": summary(calibration.reference_reprojection_errors_px),
            "target_pnp_reprojection_error_px": summary(calibration.target_reprojection_errors_px),
            "transform_translation_residual_mm": summary(calibration.translation_residuals_m, 1000.0),
            "transform_rotation_residual_deg": summary(calibration.rotation_residuals_deg),
        },
        "samples": samples,
    }
    calibration_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    quality = payload["quality"]
    print(f"Saved pair extrinsics: {calibration_path}")
    print(
        f"Accepted {len(calibration.accepted_indices)}/{len(samples)} pairs; "
        f"translation residual max={quality['transform_translation_residual_mm']['max']:.3f} mm, "
        f"rotation residual max={quality['transform_rotation_residual_deg']['max']:.3f} deg"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
