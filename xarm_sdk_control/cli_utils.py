"""Shared argparse plumbing and workspace-limit helpers for task scripts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

from xarm_sdk_control.visual_targeting import DEFAULT_MIN_SCORE, DEFAULT_VISION_ENV, safety_report


DEFAULT_WORKSPACE_LIMIT_MM = 1000.0


def add_workspace_limit_args(parser: argparse.ArgumentParser) -> None:
    for axis in ("x", "y", "z"):
        parser.add_argument(f"--min-{axis}-mm", type=float)
        parser.add_argument(f"--max-{axis}-mm", type=float)


def add_vision_common_args(
    parser: argparse.ArgumentParser,
    calibration_default: Path | None = None,
    min_score_default: float | None = DEFAULT_MIN_SCORE,
) -> None:
    parser.add_argument("--calibration", type=Path, default=calibration_default)
    parser.add_argument("--vision-env", default=DEFAULT_VISION_ENV)
    parser.add_argument("--serial")
    if min_score_default is None:
        parser.add_argument("--min-score", type=float)
    else:
        parser.add_argument("--min-score", type=float, default=min_score_default)


def add_motion_profile_args(parser: argparse.ArgumentParser, defaults: Any) -> None:
    for name, value in asdict(defaults).items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=value)


def workspace_limits_from_args(
    args: Any,
    default_limit_mm: float = DEFAULT_WORKSPACE_LIMIT_MM,
) -> dict[str, float]:
    return {
        "min_x_mm": args.min_x_mm if args.min_x_mm is not None else -default_limit_mm,
        "max_x_mm": args.max_x_mm if args.max_x_mm is not None else default_limit_mm,
        "min_y_mm": args.min_y_mm if args.min_y_mm is not None else -default_limit_mm,
        "max_y_mm": args.max_y_mm if args.max_y_mm is not None else default_limit_mm,
        "min_z_mm": args.min_z_mm if args.min_z_mm is not None else -default_limit_mm,
        "max_z_mm": args.max_z_mm if args.max_z_mm is not None else default_limit_mm,
    }


def explicit_workspace_limits(args: Any) -> bool:
    return all(
        value is not None
        for value in (
            args.min_x_mm,
            args.max_x_mm,
            args.min_y_mm,
            args.max_y_mm,
            args.min_z_mm,
            args.max_z_mm,
        )
    )


def route_safety_report(waypoints: list[list[float]], args: Any) -> dict[str, Any]:
    return safety_report(waypoints, **workspace_limits_from_args(args))
