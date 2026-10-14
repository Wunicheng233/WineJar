#!/usr/bin/env python3
"""Independently check either board type at one stationary pose; never refit."""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from xarm_sdk_control.calibration_data import (
    current_scene_id,
    file_sha256,
    load_pair_intrinsics,
    pair_geometry_fingerprint,
    read_json,
    write_new_json,
)
from xarm_sdk_control.calibration_sampling import (
    StationaryBoardGate,
    corner_displacement,
    pair_frame_rejection,
    check_view,
    read_color_pair,
    start_color_pipeline,
)
from xarm_sdk_control.calibration_target import CalibrationTarget
from xarm_sdk_control.calibration_validation import evaluate_observations
from xarm_sdk_control.pair_calibration_session import PendingSample
from xarm_sdk_control.run_output import new_timestamped_run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--interactive", action="store_true", help="Preview and S arm one repeat-frame validation pose")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    parser.add_argument("--allow-legacy-input", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data/multiview_neck/stationary_validation",
    )
    args = parser.parse_args()
    if (
        args.samples < 1
        or not np.isfinite(args.timeout_seconds)
        or args.timeout_seconds <= 0
    ):
        parser.error("Invalid sample count or timeout")
    path = args.pair.expanduser().resolve()
    pair = read_json(path)
    matrices, distortions, _ = load_pair_intrinsics(
        pair, path, allow_legacy=args.allow_legacy_input
    )
    target = CalibrationTarget.from_pair(pair)
    scene_id = current_scene_id()
    if pair.get("scene_id") is not None and pair["scene_id"] != scene_id:
        raise ValueError("Pair belongs to another physical scene")
    bound = all(
        "intrinsics_fingerprint" in pair[role]
        for role in ("reference_camera", "target_camera")
    )
    run = new_timestamped_run_dir(args.output_root.expanduser().resolve())
    report = {
        "type": "independent_board_check",
        "pair": str(path),
        "pair_sha256": file_sha256(path),
        "scene_id": scene_id,
        "pair_geometry_fingerprint": pair_geometry_fingerprint(pair) if bound else None,
        "intrinsic_binding_verified": bound,
        "complete": False,
        "stationarity_checked": True,
        "absolute_accuracy_validated": False,
        "samples": [],
        "limitation": "One pose with repeat RGB frames; unsynchronized devices. Not a metric error bound.",
    }
    pipelines, previous, anchor = [], None, None
    import pyrealsense2 as rs

    context = rs.context()
    try:
        for role in ("reference_camera", "target_camera"):
            pipeline, _ = start_color_pipeline(
                context,
                pair[role]["serial"],
                pair["capture"]["width"],
                pair["capture"]["height"],
                pair["capture"]["fps"],
            )
            pipelines.append(pipeline)
        for _ in range(45):
            read_color_pair(pipelines)
        gate = StationaryBoardGate()
        pending = PendingSample()
        capturing = not args.interactive
        window = "WineJar independent A/B validation"
        message = "NEW pose: S arm, return hands and steady board; Q quit"
        if args.interactive:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, 1280, 540)
        deadline = time.monotonic() + args.timeout_seconds
        while len(report["samples"]) < args.samples:
            if (not args.interactive or capturing) and time.monotonic() > deadline:
                raise TimeoutError("No stable usable board before timeout")
            colors, timing = read_color_pair(pipelines)
            observations = tuple(target.detect(color) for color in colors)
            frame_reason = pair_frame_rejection(timing, previous)
            fresh = frame_reason is None
            previous = timing
            still = gate.update(
                observations if fresh else (None, None), time.monotonic()
            )
            reasons = [check_view(o, (pair["capture"]["width"], pair["capture"]["height"]), k, d)[0]
                       for o, k, d in zip(observations, matrices, distortions)]
            reason = frame_reason or next((r for r in reasons if r), None) or (
                gate.status["reason"] if not still else None)
            if args.interactive:
                displays = []
                for color, observation, role in zip(colors, observations, ("reference_camera", "target_camera")):
                    display = color.copy()
                    if observation is not None:
                        for pixel in observation.corners_px.reshape(-1, 2):
                            cv2.circle(display, tuple(np.rint(pixel).astype(int)), 2, (0, 220, 0), 1)
                    label = f"{pair[role]['camera_id']}: {len(observation.corner_ids) if observation else 0} corners"
                    cv2.putText(display, label, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 1)
                    displays.append(display)
                combined = cv2.hconcat(displays)
                status = f"CHECK {len(report['samples'])}/{args.samples} | {reason or 'READY'} | skew {timing['skew_ms']:.1f}ms | hold {gate.status['held_seconds']:.2f}s"
                cv2.putText(combined, status, (12, combined.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,0) if reason is None else (0,0,255), 1)
                cv2.putText(combined, message, (12, combined.shape[0]-8), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
                cv2.imshow(window, combined)
                key = cv2.waitKey(1) & 0xff
                if key in (ord('q'), ord('Q'), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    raise RuntimeError("User cancelled validation")
                if not capturing:
                    start, notice = pending.update(key, time.monotonic(), reason is None)
                    if notice:
                        message = notice
                        print(notice, flush=True)
                    if start:
                        capturing = True
                        deadline = time.monotonic() + args.timeout_seconds
                        message = "Capturing repeat frames; keep board still"
                    else:
                        continue
            if reason is not None:
                continue
            if anchor is not None and any(
                corner_displacement(a, b) > 1.0 for a, b in zip(anchor, observations)
            ):
                raise ValueError(
                    "Board moved during repeat capture; this run cannot be accepted"
                )
            anchor = observations if anchor is None else anchor
            index = len(report["samples"]) + 1
            record = {
                "sample": index,
                "frame_pair": timing,
                "stationary_evidence": gate.status,
                "image_sha256": [],
                "image_paths": [],
            }
            for role, color in zip(("reference_camera", "target_camera"), colors):
                image_path = run / f"{index:02d}_{pair[role]['camera_id']}.png"
                if not cv2.imwrite(str(image_path), color):
                    raise OSError("Failed to save independent raw image")
                record["image_sha256"].append(file_sha256(image_path))
                record["image_paths"].append(image_path.name)
            try:
                metrics, predictions = evaluate_observations(
                    observations,
                    matrices,
                    distortions,
                    np.array(pair["transform_reference_from_target"]),
                )
                record.update(metrics)
                for role, color, obs, projected in zip(
                    ("reference_camera", "target_camera"),
                    colors,
                    observations,
                    predictions,
                ):
                    for actual, predicted in zip(
                        obs.corners_px.reshape(-1, 2), projected
                    ):
                        cv2.circle(
                            color, tuple(np.rint(actual).astype(int)), 3, (0, 255, 0), 1
                        )
                        cv2.drawMarker(
                            color,
                            tuple(np.rint(predicted).astype(int)),
                            (0, 0, 255),
                            cv2.MARKER_CROSS,
                            7,
                            1,
                        )
                    if not cv2.imwrite(
                        str(run / f"{index:02d}_{pair[role]['camera_id']}_check.png"),
                        color,
                    ):
                        raise OSError("Failed to save independent overlay")
            except (ValueError, cv2.error) as error:
                record["failure"] = str(error)
            report["samples"].append(record)
            print(
                f"Sample {index}: {record.get('cross_errors_px', record.get('failure'))}",
                flush=True,
            )
        report["complete"] = not any("failure" in item for item in report["samples"])
    except (ValueError, RuntimeError, TimeoutError, OSError) as error:
        report["failure"] = str(error)
        print(f"Validation failed: {error}")
    finally:
        for pipeline in pipelines:
            pipeline.stop()
        if args.interactive:
            cv2.destroyAllWindows()
        write_new_json(run / "validation.json", report)
    print(f"Saved independent diagnostic: {run}")
    print("This single pose is not an acceptance certificate.")
    return 0 if report["complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
