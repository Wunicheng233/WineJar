"""Object profiles, guarded routes, and execution for attended vacuum-pick tests."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from xarm_sdk_control.geometry import compose_pose_mm
from xarm_sdk_control.robot_motion import require_code, require_overhead_camera_safe_route
from xarm_sdk_control.vacuum_control import VacuumController


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


def validate_pick_test_args(args: Any, profile: PickTestProfile) -> tuple[float, float]:
    min_score = profile.default_min_score if args.min_score is None else float(args.min_score)
    compression_mm = profile.default_compression_mm if args.compression_mm is None else float(args.compression_mm)
    if not 0.0 <= min_score <= 1.0:
        raise ValueError("--min-score must be within 0~1")
    if not 0.0 <= compression_mm <= MAX_COMPRESSION_MM:
        raise ValueError(
            f"--compression-mm must remain within the configured 0~{MAX_COMPRESSION_MM:g} mm spring travel"
        )
    if min(args.vacuum_settle_seconds, args.release_settle_seconds, args.inspection_hold_seconds) < 0.0:
        raise ValueError("Hold and settle times must be non-negative")
    if min(args.transit_speed, args.precision_speed, args.acc) <= 0.0:
        raise ValueError("Motion speeds and acceleration must be positive")
    limits = (args.min_x_mm, args.max_x_mm, args.min_y_mm, args.max_y_mm, args.min_z_mm, args.max_z_mm)
    if args.run and any(value is None for value in limits):
        raise ValueError("--run requires explicit min/max X, Y, and Z workspace limits")
    if args.run and not args.confirm_no_pressure_sensor:
        raise ValueError("--run requires --confirm-no-pressure-sensor")
    if not 1 <= args.max_attempts <= 3:
        raise ValueError("--max-attempts must be within 1~3")
    if args.verify_pick and not args.run:
        raise ValueError("--verify-pick requires --run")
    if args.max_attempts > 1 and not args.verify_pick:
        raise ValueError("Multiple attempts require --verify-pick")
    if args.simulate_first_pick_failure and not args.verify_pick:
        raise ValueError("--simulate-first-pick-failure requires --verify-pick")
    if args.simulate_first_pick_failure and args.max_attempts < 2:
        raise ValueError("Simulated first-pick failure requires --max-attempts 2 or greater")
    return min_score, compression_mm


def speed_for_step(step: PickTestStep, args: Any) -> float:
    return float(args.precision_speed if step.motion != "empty_transit" else args.transit_speed)


def execute_pick_test_route(
    arm: Any,
    vacuum: VacuumController,
    route: Sequence[PickTestStep],
    args: Any,
    holding_state: dict[str, bool],
    *,
    label_prefix: str = "",
    failure_injected: bool = False,
    event_callback: Callable[[PickTestStep], None] | None = None,
) -> None:
    vacuum.prepare_for_pick()
    for index, step in enumerate(route, start=1):
        require_code(
            f"{label_prefix}move {index:02d}: {step.label}",
            arm.set_position(
                *step.pose_mm_deg,
                speed=speed_for_step(step, args),
                mvacc=args.acc,
                wait=True,
                radius=-1,
                is_radian=False,
            ),
        )
        if step.event == "acquire":
            if failure_injected:
                print("CONTROLLED TEST: pump activation skipped; valve remains in the suction path state.")
            else:
                vacuum.acquire(args.vacuum_settle_seconds)
                holding_state["value"] = True
        elif step.event == "hold":
            print(f"Holding for operator inspection: {args.inspection_hold_seconds:.1f} s")
            time.sleep(args.inspection_hold_seconds)
        elif step.event == "verify" and event_callback is not None:
            event_callback(step)
        elif step.event == "release":
            vacuum.release(args.release_settle_seconds)
            holding_state["value"] = False
