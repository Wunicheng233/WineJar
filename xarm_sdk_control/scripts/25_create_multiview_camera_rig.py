#!/usr/bin/env python3
"""Create one validated fixed-camera rig manifest from intrinsic/extrinsic files.

This is a calibration-data operation only. It never opens a camera, connects to
the xArm, or changes any existing calibration file.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xarm_sdk_control.multiview_rig import build_fixed_camera_rig, write_fixed_camera_rig  # noqa: E402


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
    args = parser.parse_args()

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    output = args.output or DEFAULT_CALIBRATION_ROOT / f"three_camera_rig_{timestamp}.json"
    try:
        rig = build_fixed_camera_rig(args.reference_intrinsics, args.pair_extrinsics)
        path = write_fixed_camera_rig(rig, output)
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(f"Saved fixed camera rig: {path}")
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
