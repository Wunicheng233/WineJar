#!/usr/bin/env python3
"""Fit mouth axis, radius profile, and binding-plane candidate from a visual hull."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xarm_sdk_control.neck_localization import fit_neck_from_visual_hull  # noqa: E402
from xarm_sdk_control.run_output import new_timestamped_run_dir  # noqa: E402


DEFAULT_OUTPUT_ROOT = ROOT / "data" / "multiview_neck" / "neck_localizations"


def parse_vector(value: str) -> np.ndarray:
    try:
        result = np.array([float(part) for part in value.split(",")], dtype=np.float64)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Vector must be x,y,z") from exc
    if result.shape != (3,):
        raise argparse.ArgumentTypeError("Vector must contain exactly x,y,z")
    return result


def write_failure_result(output_root: Path, hull_dir: Path, args: argparse.Namespace, error: Exception) -> Path:
    """Persist an auditable no-target result instead of only printing a terminal error."""
    run_dir = new_timestamped_run_dir(output_root.expanduser().resolve())
    payload = {
        "schema_version": 1,
        "type": "multiview_surface_of_revolution_neck_localization",
        "status": "failed",
        "visual_hull_dir": str(hull_dir),
        "parameters": {
            "slice_width_mm": args.slice_width_mm,
            "min_circularity": args.min_circularity,
        },
        "confidence": 0.0,
        "failure_reason": "INSUFFICIENT_VISUAL_HULL_SUPPORT_OR_CIRCULARITY",
        "detail": str(error),
    }
    (run_dir / "neck_localization.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-hull", type=Path, required=True, help="Directory containing visual_hull.json.")
    parser.add_argument("--slice-width-mm", type=float, default=8.0)
    parser.add_argument(
        "--min-circularity",
        type=float,
        default=0.55,
        help="Minimum min/max radial-envelope ratio for a stable profile slice.",
    )
    parser.add_argument("--mouth-to-body-hint", type=parse_vector, default=np.array([0.0, 0.0, 1.0]))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    if args.slice_width_mm <= 0 or not 0 < args.min_circularity <= 1:
        parser.error("--slice-width-mm must be positive and --min-circularity must be in (0, 1]")
    hull_dir = args.visual_hull.expanduser().resolve()
    try:
        report = json.loads((hull_dir / "visual_hull.json").read_text(encoding="utf-8"))
        if report.get("type") != "fixed_multiview_visual_hull":
            raise ValueError("visual_hull.json has an unexpected type")
        points = np.load(hull_dir / report["files"]["points_npz"])["points_reference_m"]
        result = fit_neck_from_visual_hull(
            points,
            mouth_to_body_hint=args.mouth_to_body_hint,
            slice_width_m=args.slice_width_mm / 1000.0,
            min_circularity=args.min_circularity,
        )
    except (OSError, ValueError, KeyError) as exc:
        failure_dir = write_failure_result(args.output_root, hull_dir, args, exc)
        print(f"ERROR: neck fitting failed: {exc}; saved failure result: {failure_dir}")
        return 2
    run_dir = new_timestamped_run_dir(args.output_root.expanduser().resolve())
    payload = {
        "schema_version": 1,
        "type": "multiview_surface_of_revolution_neck_localization",
        "status": "ok",
        "visual_hull_dir": str(hull_dir),
        "coordinate_convention": {
            "units": "metres",
            "frame": "reference camera optical frame",
            "axis": "mouth_to_body",
        },
        "parameters": {
            "slice_width_mm": args.slice_width_mm,
            "min_circularity": args.min_circularity,
        },
        "mouth_center_m": result.mouth_center_m.tolist(),
        "axis_mouth_to_body": result.axis_mouth_to_body.tolist(),
        "mouth_plane": {"point_m": result.mouth_center_m.tolist(), "normal": result.mouth_plane_normal.tolist()},
        "binding_plane": {
            "point_m": result.binding_center_m.tolist(),
            "normal": result.binding_plane_normal.tolist(),
            "min_diameter_mm": result.binding_min_diameter_m * 1000.0,
            "max_diameter_mm": result.binding_max_diameter_m * 1000.0,
            "constriction_ratio_to_body": result.binding_constriction_ratio,
        },
        "radius_profile": [
            {
                "axial_position_m": sample.axial_position_m,
                "mean_radius_mm": sample.radius_mean_m * 1000.0,
                "min_radius_mm": sample.radius_min_m * 1000.0,
                "max_radius_mm": sample.radius_max_m * 1000.0,
                "sector_count": sample.sector_count,
                "point_count": sample.point_count,
            }
            for sample in result.profile
        ],
        "confidence": result.confidence,
        "failure_reason": result.failure_reason,
    }
    (run_dir / "neck_localization.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved neck localization: {run_dir}")
    print(
        f"confidence={result.confidence:.3f} failure_reason={result.failure_reason}; "
        f"binding diameter={payload['binding_plane']['min_diameter_mm']:.1f}..{payload['binding_plane']['max_diameter_mm']:.1f} mm"
    )
    print("Robot connection/motion: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
