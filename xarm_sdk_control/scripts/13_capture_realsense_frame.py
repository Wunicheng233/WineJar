#!/usr/bin/env python3
"""Capture one aligned RealSense RGB-D frame without connecting to the robot."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xarm_sdk_control.realsense_capture import RealSenseCaptureSession  # noqa: E402


DEFAULT_OUTPUT_ROOT = ROOT / "data" / "vision_frames"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Save one color frame, depth aligned to color, and camera metadata. "
            "This script never connects to or moves the xArm."
        )
    )
    parser.add_argument("--serial", help="Optional RealSense serial number.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    try:
        with RealSenseCaptureSession(
            serial=args.serial,
            width=args.width,
            height=args.height,
            fps=args.fps,
            warmup_frames=args.warmup,
        ) as camera:
            camera.capture(args.output_root)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 2
    print("Robot connection/motion: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
