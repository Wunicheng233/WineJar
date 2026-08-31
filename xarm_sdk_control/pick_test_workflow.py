"""Object profiles, guarded routes, and execution for attended vacuum-pick tests."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from xarm_sdk_control.end_effector_config import load_end_effector_config
from xarm_sdk_control.geometry import compose_pose_mm
from xarm_sdk_control.robot_motion import require_code, require_overhead_camera_safe_route
from xarm_sdk_control.suction_sensor import SuctionNotConfirmedError, SuctionSensor
from xarm_sdk_control.vacuum_control import VacuumController


MAX_COMPRESSION_MM = load_end_effector_config().max_compression_mm


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
        test_lift_mm=100.0,
    ),
    "ceramic": PickTestProfile(
        kind="ceramic",
        text_prompt="ceramic disk.",
        default_min_score=0.45,
        default_compression_mm=15.0,
        test_lift_mm=100.0,
    ),
    "lotus-leaf": PickTestProfile(
        kind="lotus",
        text_prompt="lotus leaf.",
        default_min_score=0.35,
        default_compression_mm=15.0,
        test_lift_mm=100.0,
    ),
    "paper": PickTestProfile(
        kind="paper",
        text_prompt="white paper.",
        default_min_score=0.35,
        default_compression_mm=15.0,
        test_lift_mm=100.0,
    ),
}


def profile_for(target: str) -> PickTestProfile:
    try:
        return PROFILES[target]
    except KeyError as exc:
        raise ValueError(f"Unsupported pick-test target: {target}") from exc


def resolve_pick_test_settings(
    profile: PickTestProfile,
    *,
    min_score: float | None = None,
    compression_mm: float | None = None,
    test_lift_mm: float | None = None,
) -> tuple[float, float, float]:
    """Resolve CLI-or-profile values for one pick-test target: (min_score, compression, lift)."""
    return (
        profile.default_min_score if min_score is None else float(min_score),
        profile.default_compression_mm if compression_mm is None else float(compression_mm),
        profile.test_lift_mm if test_lift_mm is None else float(test_lift_mm),
    )


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


def build_pick_place_test_route(
    *,
    home_pose_mm_deg: list[float],
    source_point_m: list[float],
    destination_point_m: list[float],
    compression_mm: float,
    transport_clearance_mm: float = 150.0,
    approach_clearance_mm: float = 35.0,
    precontact_clearance_mm: float = 5.0,
    placement_clearance_mm: float = 0.0,
) -> list[PickTestStep]:
    """Build a pressure-checked visual transfer from source to destination."""
    if not 0.0 <= compression_mm <= MAX_COMPRESSION_MM:
        raise ValueError(
            f"Compression must remain within the configured 0~{MAX_COMPRESSION_MM:g} mm spring travel"
        )
    if min(
        transport_clearance_mm,
        approach_clearance_mm,
        precontact_clearance_mm,
        placement_clearance_mm,
    ) < 0.0:
        raise ValueError("Route clearances must be non-negative")

    home = list(map(float, home_pose_mm_deg))
    source = np.asarray(source_point_m, dtype=float) * 1000.0
    destination = np.asarray(destination_point_m, dtype=float) * 1000.0
    if source.shape != (3,) or destination.shape != (3,):
        raise ValueError("Source and destination points must each contain three values")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(destination)):
        raise ValueError("Source and destination points must be finite")

    orientation = home[3:6]
    transport_z = float(max(source[2], destination[2]) + transport_clearance_mm)
    steps = [
        PickTestStep(
            "move directly above detected source",
            compose_pose_mm(source, source[2] + approach_clearance_mm, orientation),
            "empty_transit",
        ),
        PickTestStep(
            "descend to source precontact height",
            compose_pose_mm(source, source[2] + precontact_clearance_mm, orientation),
            "precision",
        ),
        PickTestStep(
            "compress suction cups on source",
            compose_pose_mm(source, source[2] - compression_mm, orientation),
            "contact",
            "acquire",
        ),
        PickTestStep(
            "lift material vertically to transport height",
            compose_pose_mm(source, transport_z, orientation),
            "loaded_precision",
        ),
        PickTestStep(
            "carry material above visual destination",
            compose_pose_mm(destination, transport_z, orientation),
            "loaded_transit",
        ),
        PickTestStep(
            "lower material onto visual destination",
            compose_pose_mm(destination, destination[2] + placement_clearance_mm, orientation),
            "loaded_precision",
            "release",
        ),
        PickTestStep(
            "retract vertically after release",
            compose_pose_mm(destination, destination[2] + approach_clearance_mm, orientation),
            "precision",
        ),
        PickTestStep("return to camera-clear home pose", home, "empty_transit"),
    ]
    require_overhead_camera_safe_route([home, *[step.pose_mm_deg for step in steps]])
    return steps


def speed_for_step(step: PickTestStep, args: Any) -> float:
    return float(
        args.transit_speed
        if step.motion in {"empty_transit", "loaded_transit"}
        else args.precision_speed
    )


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
    suction_sensor: SuctionSensor | None = None,
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
            if suction_sensor is not None:
                if not suction_sensor.wait_for_suction():
                    assert suction_sensor.last_wait is not None
                    raise SuctionNotConfirmedError(
                        suction_sensor.config.input_channel,
                        suction_sensor.last_wait,
                    )
        if suction_sensor is not None and holding_state["value"] and step.motion.startswith("loaded"):
            # Sustain check: the material can drop off the cup during the lift
            # even though suction was confirmed at the contact.
            if not suction_sensor.wait_for_suction():
                assert suction_sensor.last_wait is not None
                raise SuctionNotConfirmedError(
                    suction_sensor.config.input_channel,
                    suction_sensor.last_wait,
                )
        if step.event == "hold":
            print(f"Holding for operator inspection: {args.inspection_hold_seconds:.1f} s")
            time.sleep(args.inspection_hold_seconds)
        elif step.event == "verify" and event_callback is not None:
            event_callback(step)
        elif step.event == "release":
            vacuum.release(args.release_settle_seconds)
            holding_state["value"] = False


def recover_failed_pick_test(
    arm: Any,
    route: Sequence[PickTestStep],
    args: Any,
    *,
    return_home: bool,
    label_prefix: str = "",
) -> None:
    """Retract vertically after a failed seal, optionally returning home."""
    # Route waypoint 0 is always the vertically clear source approach. A
    # failed seal must retreat there, not toward a placement destination.
    retract = route[0]
    require_code(
        f"{label_prefix}retract after failed suction",
        arm.set_position(
            *retract.pose_mm_deg,
            speed=args.precision_speed,
            mvacc=args.acc,
            wait=True,
            radius=-1,
            is_radian=False,
        ),
    )
    if not return_home:
        return
    home = route[-1]
    require_code(
        f"{label_prefix}return home after final failed suction",
        arm.set_position(
            *home.pose_mm_deg,
            speed=args.transit_speed,
            mvacc=args.acc,
            wait=True,
            radius=-1,
            is_radian=False,
        ),
    )
