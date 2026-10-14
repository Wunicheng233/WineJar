#!/usr/bin/env python3
"""Redetect saved chessboard/ChArUco pairs without altering source data or live rig."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from xarm_sdk_control.calibration_data import (
    bind_intrinsics,
    current_scene_id,
    file_sha256,
    fit_quality,
    load_intrinsics,
    load_pair_intrinsics,
    write_new_json,
)
from xarm_sdk_control.calibration_target import CalibrationTarget
from xarm_sdk_control.multiview_calibration import (
    calibrate_relative_camera_pose_from_observations,
    normalized_board_area_ratio,
)
from xarm_sdk_control.run_output import new_timestamped_run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--reference-intrinsics", type=Path)
    parser.add_argument("--target-intrinsics", type=Path)
    parser.add_argument(
        "--method", choices=("pose_average", "stereo"), default="pose_average"
    )
    parser.add_argument(
        "--scene-id",
        help="Explicitly attest the source images belong to this unchanged scene",
    )
    parser.add_argument(
        "--save-overlays",
        action="store_true",
        help="Optional derived previews; source RGB is always retained",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data/multiview_neck/reprocessed_pairs",
    )
    args = parser.parse_args()
    source = args.pair.expanduser().resolve()
    payload = json.loads(source.read_text())
    target = CalibrationTarget.from_pair(payload)
    if args.scene_id and args.scene_id != current_scene_id():
        raise ValueError("Explicit scene differs from current scene")
    original_run = Path(payload["sample_run_dir"])
    load_pair_intrinsics(payload, source, allow_legacy=True)
    matrices, distortions = [], []
    for role in ("reference_camera", "target_camera"):
        override = (
            args.reference_intrinsics
            if role == "reference_camera"
            else args.target_intrinsics
        )
        path = (
            override.expanduser().resolve()
            if override
            else (source.parent / payload[role]["intrinsics_path"]).resolve()
        )
        matrix, distortion, intrinsic = load_intrinsics(
            path,
            payload[role]["serial"],
            (payload["capture"]["width"], payload["capture"]["height"]),
        )
        matrices.append(matrix)
        distortions.append(distortion)
        payload[role]["intrinsics_path"] = str(path)
        payload[role].pop("intrinsics_created_at", None)
        bind_intrinsics(payload[role], intrinsic)
    image_size = (payload["capture"]["width"], payload["capture"]["height"])
    output = new_timestamped_run_dir(args.output_root.resolve())
    observations = {"reference": [], "target": []}
    for sample in payload["samples"]:
        sample.pop("admission_pnp_rmse_px", None)
        for prefix in ("reference", "target"):
            path = original_run / sample[prefix + "_color_path"]
            color = cv2.imread(str(path))
            if color is None or color.shape[:2] != image_size[::-1]:
                raise ValueError(f"Missing or mismatched source image: {path}")
            digest = file_sha256(path)
            if sample.get(prefix + "_image_sha256", digest) != digest:
                raise ValueError(f"Source image changed since capture: {path}")
            observation = target.detect(color)
            if observation is None:
                raise ValueError(f"Detection failed; no calibration published: {path}")
            observations[prefix].append(observation)
            corners = observation.corners_px
            sample[prefix + "_corner_points_px"] = corners.reshape(-1, 2).tolist()
            sample[prefix + "_board_area_ratio"] = normalized_board_area_ratio(
                corners, image_size
            )
            sample[prefix + "_color_path"] = str(path.resolve())
            sample[prefix + "_image_sha256"] = digest
            sample[prefix + "_corner_ids"] = observation.corner_ids.tolist()
            overlay = (
                output / f"sample_{sample['sample_index']:02d}_{prefix}_corners.png"
            )
            sample.pop(prefix + "_overlay_path", None)
            if args.save_overlays:
                for pixel in corners.reshape(-1, 2):
                    cv2.circle(
                        color, tuple(np.rint(pixel).astype(int)), 2, (0, 255, 0), 1
                    )
                if not cv2.imwrite(str(overlay), color):
                    raise OSError(f"Cannot save {overlay}")
                sample[prefix + "_overlay_path"] = str(overlay)
        print(f"Redetected pair {sample['sample_index']:02d}", flush=True)
    fit = calibrate_relative_camera_pose_from_observations(
        [s.object_points_m for s in observations["reference"]],
        [s.corners_px for s in observations["reference"]],
        [s.object_points_m for s in observations["target"]],
        [s.corners_px for s in observations["target"]],
        matrices[0],
        distortions[0],
        matrices[1],
        distortions[1],
        method=args.method,
        image_size_px=image_size,
    )
    payload["created_at"] = datetime.now().astimezone().isoformat()
    payload["schema_version"] = 2
    payload["calibration_status"] = "candidate"
    payload["scene_id"] = args.scene_id or payload.get("scene_id")
    payload["quality"] = fit_quality(fit, len(payload["samples"]))
    payload["sample_run_dir"] = str(output)
    payload["transform_reference_from_target"] = (
        fit.transform_reference_from_target.tolist()
    )
    payload["reprocessing"] = {
        "source_calibration": str(source),
        "source_sha256": file_sha256(source),
        "detector": "board_observation_v2",
        "independent_validation_passed": False,
        "live_rig_updated": False,
    }
    path = output / "pair.json"
    write_new_json(path, payload)
    print(f"Saved candidate: {path}")
    print(json.dumps(payload["quality"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
