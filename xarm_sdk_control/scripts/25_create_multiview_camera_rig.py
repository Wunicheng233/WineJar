#!/usr/bin/env python3
"""Create a candidate rig, or explicitly accept it for static image capture.

This is a calibration-data operation only. It never opens a camera, connects to
the xArm, or overwrites calibration results. Explicit acceptance updates the scene
registry after backing it up; no robot-motion approval is granted.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xarm_sdk_control.calibration_data import (
    SCENE_STATE_PATH,
    current_scene_id,
    file_sha256,
    intrinsic_health,
    pair_geometry_fingerprint,
    read_json,
    write_new_json,
)
from xarm_sdk_control.calibration_validation import check_acceptance_reports
from xarm_sdk_control.multiview_rig import (
    build_fixed_camera_rig,
    rig_geometry_fingerprint,
    write_fixed_camera_rig,
)

DEFAULT_CALIBRATION_ROOT = ROOT / "calibration" / "multiview_neck"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-intrinsics", type=Path, required=True)
    parser.add_argument(
        "--pair-extrinsics",
        type=Path,
        required=True,
        action="append",
        help="One reference-to-target pair extrinsic file; provide once per side camera.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scene-id")
    parser.add_argument(
        "--accept-for-capture",
        action="store_true",
        help="Explicit acceptance for static capture only, not robot use",
    )
    parser.add_argument("--validation-report", type=Path, action="append", default=[])
    parser.add_argument("--max-cross-rmse-px", type=float)
    parser.add_argument("--max-cross-p95-px", type=float)
    args = parser.parse_args()

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    output = (
        args.output or DEFAULT_CALIBRATION_ROOT / f"three_camera_rig_{timestamp}.json"
    )
    try:
        scene_id = args.scene_id or current_scene_id()
        if scene_id != current_scene_id():
            raise ValueError("Requested scene is not the current physical scene")
        rig = build_fixed_camera_rig(
            args.reference_intrinsics, args.pair_extrinsics, scene_id=scene_id
        )
        acceptance = None
        if args.accept_for_capture:
            if args.max_cross_rmse_px is None or args.max_cross_p95_px is None:
                raise ValueError(
                    "Acceptance requires explicit --max-cross-rmse-px and --max-cross-p95-px; these are not millimetre guarantees"
                )
            for camera in rig.cameras.values():
                if not intrinsic_health(
                    camera.camera_matrix,
                    camera.distortion_coefficients,
                    camera.image_size_px,
                )["passed"]:
                    raise ValueError(f"{camera.camera_id}: unhealthy intrinsic model")
            fitting_hashes = {}
            for source in rig.source_extrinsics_paths:
                pair = read_json(source)
                hashes = set()
                for sample in pair["samples"]:
                    for prefix in ("reference", "target"):
                        value = file_sha256(
                            Path(pair["sample_run_dir"])
                            / sample[prefix + "_color_path"]
                        )
                        if sample.get(prefix + "_image_sha256", value) != value:
                            raise ValueError("Fitting image changed after calibration")
                        hashes.add(value)
                fitting_hashes[pair_geometry_fingerprint(pair)] = hashes
            acceptance = check_acceptance_reports(
                args.validation_report,
                fitting_hashes,
                scene_id,
                args.max_cross_rmse_px,
                args.max_cross_p95_px,
            )
        path = write_fixed_camera_rig(rig, output)
        if acceptance is not None:
            state = read_json(SCENE_STATE_PATH)
            write_new_json(
                SCENE_STATE_PATH.parent
                / "history"
                / f"scene_before_acceptance_{timestamp}.json",
                state,
            )
            state.update(
                status="validated_for_capture",
                scene_id=scene_id,
                approved_rig_fingerprint=rig_geometry_fingerprint(rig),
                approved_rig_path=str(path),
                acceptance=acceptance,
                absolute_accuracy_validated=False,
            )
            temporary = SCENE_STATE_PATH.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
            temporary.replace(SCENE_STATE_PATH)
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(
        f"Saved {'accepted-for-static-capture' if args.accept_for_capture else 'unaccepted candidate'} rig: {path}"
    )
    for camera in rig.cameras.values():
        translation_mm = camera.transform_reference_from_camera[:3, 3] * 1000.0
        print(
            f"  {camera.camera_id}: serial={camera.serial}, "
            f"T_reference_from_camera translation_mm={translation_mm.tolist()}"
        )
    print("Robot connection/motion: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
