#!/usr/bin/env python3
"""Estimate open-rim geometry from a saved, calibrated three-camera capture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from xarm_sdk_control.mouth_height_pipeline import LocalizationSettings, localize_jar
from xarm_sdk_control.run_output import new_timestamped_run_dir


def parse_roi(value):
    try:
        camera, raw = value.split(":", 1)
        roi = tuple(int(part) for part in raw.split(","))
        if not camera or len(roi) != 4:
            raise ValueError
        return camera, roi
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "ROI must be camera_id:x,y,width,height"
        ) from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, help="Saved offline experiment configuration"
    )
    parser.add_argument("--rig", type=Path)
    parser.add_argument("--workcell-frame", type=Path)
    parser.add_argument("--capture", type=Path)
    parser.add_argument(
        "--masks", type=Path, help="GroundingDINO/SAM2 mask run directory"
    )
    parser.add_argument(
        "--background-capture",
        type=Path,
        help="Used when learned masks are not supplied",
    )
    parser.add_argument("--roi", type=parse_roi, action="append")
    parser.add_argument("--bootstrap-samples", type=int)
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "data/multiview_neck/mouth_heights"
    )
    args = parser.parse_args()
    try:
        config = {} if args.config is None else json.loads(args.config.read_text())
        owner = Path.cwd() if args.config is None else args.config.resolve().parent

        def input_path(name, required=False):
            value = getattr(args, name) or config.get(name)
            if value is None:
                if required:
                    raise ValueError(f"Missing --{name.replace('_', '-')}")
                return None
            path = Path(value).expanduser()
            if getattr(args, name) is None and not path.is_absolute():
                path = owner / path
            return path.resolve()

        rois = dict(args.roi or config.get("rois", {}).items())
        if args.roi and len(rois) != len(args.roi):
            raise ValueError("Duplicate camera ROI")
        parameters = dict(config.get("parameters", {}))
        if args.bootstrap_samples is not None:
            parameters["bootstrap_samples"] = args.bootstrap_samples
        result, overlays, closeups = localize_jar(
            input_path("rig", True),
            input_path("workcell_frame", True),
            input_path("capture", True),
            rois,
            masks_dir=input_path("masks"),
            background_capture=input_path("background_capture"),
            settings=LocalizationSettings(**parameters),
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"ERROR: localization failed: {error}")
        return 2
    run = new_timestamped_run_dir(args.output_root.expanduser().resolve())
    for cid in overlays:
        output = run / cid
        output.mkdir()
        for name, image in [
            ("height_overlay.png", overlays[cid]),
            ("rim_closeup.png", closeups[cid]),
        ]:
            if not cv2.imwrite(str(output / name), image):
                raise OSError(f"Cannot save {output / name}")
    (run / "mouth_height.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(f"Saved localization: {run}")
    print(f"Status: {result['status']}; absolute accuracy validated: false")
    print(f"Rim plane center Z: {result['rim_center_workcell_mm'][2]:.2f} mm")
    print(
        f"Rim-to-shoulder axial length: {result['rim_to_shoulder_axial_length_mm']} mm"
    )
    print("Usable binding groove: not yet validated")
    for flag in result["quality_flags"]:
        print(f"REVIEW: {flag}")
    print("Offline only; no camera connection or robot motion")
    return 0 if result["status"] == "geometry_consistent" else 3


if __name__ == "__main__":
    raise SystemExit(main())
