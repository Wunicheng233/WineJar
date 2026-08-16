"""Object profiles and guarded routes for attended vacuum-pick tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from xarm_sdk_control.geometry import compose_pose_mm
from xarm_sdk_control.robot_motion import require_overhead_camera_safe_route


MAX_COMPRESSION_MM = 30.0


@dataclass(frozen=True)
class PickTestProfile:
    kind: str
    text_prompt: str
    default_min_score: float
    default_compression_mm: float
    test_lift_mm: float


@dataclass(frozen=True)
class PickTestStep:
    label: str
    pose_mm_deg: list[float]
    motion: str
    event: str | None = None


PROFILES = {
    "bamboo": PickTestProfile(
        kind="bamboo",
        text_prompt="bamboo leaf.",
        default_min_score=0.35,
        default_compression_mm=13.0,
        test_lift_mm=40.0,
    ),
    "ceramic": PickTestProfile(
        kind="ceramic",
        text_prompt="ceramic disk.",
        default_min_score=0.45,
        default_compression_mm=21.0,
        test_lift_mm=25.0,
    ),
}


def profile_for(target: str) -> PickTestProfile:
    try:
        return PROFILES[target]
    except KeyError as exc:
        raise ValueError(f"Unsupported pick-test target: {target}") from exc


def build_pick_test_route(
    *,
    home_pose_mm_deg: list[float],
    source_point_m: list[float],
    profile: PickTestProfile,
    compression_mm: float,
    approach_clearance_mm: float = 35.0,
    precontact_clearance_mm: float = 5.0,
    release_clearance_mm: float = 5.0,
    inspection_event: str = "hold",
) -> list[PickTestStep]:
    if not 0.0 <= compression_mm <= MAX_COMPRESSION_MM:
        raise ValueError(
            f"Compression must remain within the configured 0~{MAX_COMPRESSION_MM:g} mm spring travel"
        )
    if min(approach_clearance_mm, precontact_clearance_mm, release_clearance_mm) < 0:
        raise ValueError("Route clearances must be non-negative")
    if inspection_event not in {"hold", "verify"}:
        raise ValueError("Inspection event must be 'hold' or 'verify'")

    home = list(map(float, home_pose_mm_deg))
    if len(home) != 6:
        raise ValueError("Home pose must contain six values")
    source = np.asarray(source_point_m, dtype=float) * 1000.0
    if source.shape != (3,) or not np.all(np.isfinite(source)):
        raise ValueError("Source point must contain three finite values")
    orientation = home[3:6]
    steps = [
        PickTestStep(
            "move directly above detected target",
            compose_pose_mm(source, source[2] + approach_clearance_mm, orientation),
            "empty_transit",
        ),
        PickTestStep(
            "descend to target precontact height",
            compose_pose_mm(source, source[2] + precontact_clearance_mm, orientation),
            "precision",
        ),
        PickTestStep(
            "compress suction cup on target",
            compose_pose_mm(source, source[2] - compression_mm, orientation),
            "contact",
            "acquire",
        ),
        PickTestStep(
            "lift target vertically for operator inspection",
            compose_pose_mm(source, source[2] + profile.test_lift_mm, orientation),
            "loaded_precision",
            inspection_event,
        ),
        PickTestStep(
            "lower target back to its original source",
            compose_pose_mm(source, source[2] + release_clearance_mm, orientation),
            "loaded_precision",
            "release",
        ),
        PickTestStep(
            "retract vertically after release",
            compose_pose_mm(source, source[2] + approach_clearance_mm, orientation),
            "precision",
        ),
        PickTestStep("return to camera-clear home pose", home, "empty_transit"),
    ]
    require_overhead_camera_safe_route([home, *[step.pose_mm_deg for step in steps]])
    return steps
