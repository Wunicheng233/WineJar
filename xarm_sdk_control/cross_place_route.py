"""Route construction and task geometry for the two-shell placement task."""

from __future__ import annotations

from typing import Any

import numpy as np

from xarm_sdk_control.geometry import compose_pose_mm, wrapped_angle_deg
from xarm_sdk_control.robot_motion import (
    CAMERA_TRANSIT_Z_MM,
    require_overhead_camera_safe_route,
)
from xarm_sdk_control.visual_targeting import DISTINCT_LEAF_DISTANCE_M, select_leaf_waypoints


MIN_PICK_PLACE_SEPARATION_MM = 100.0
BLENDABLE_TRANSITIONS = {
    ("empty_transit", "precision"),
    ("precision", "vertical"),
}


def offset_detection(item: dict[str, Any], bias_m: np.ndarray) -> dict[str, Any]:
    result = dict(item)
    result["point_base_m"] = (np.asarray(item["point_base_m"], dtype=float) + bias_m).tolist()
    return result


def offset_point_mm(
    point_m: list[float],
    x_mm: float = 0.0,
    y_mm: float = 0.0,
    z_mm: float = 0.0,
) -> list[float]:
    offset_m = np.asarray([x_mm, y_mm, z_mm], dtype=float) / 1000.0
    return (np.asarray(point_m, dtype=float) + offset_m).tolist()


def require_distinct_pick_and_place_targets(
    source_point_m: list[float],
    placement_point_m: list[float],
    minimum_xy_separation_mm: float = MIN_PICK_PLACE_SEPARATION_MM,
) -> None:
    separation_mm = float(
        np.linalg.norm(
            np.asarray(source_point_m, dtype=float)[:2]
            - np.asarray(placement_point_m, dtype=float)[:2]
        )
        * 1000.0
    )
    if separation_mm < minimum_xy_separation_mm:
        raise RuntimeError(
            f"Visual pick and placement targets are only {separation_mm:.1f} mm apart; "
            "the bamboo shell and ceramic disk detections may refer to the same object"
        )


def build_cycle_route(
    home_pose: list[float],
    leaf_point_m: list[float],
    ceramic_point_m: list[float],
    place_yaw_offset_deg: float,
    transport_clearance_mm: float,
    approach_clearance_mm: float,
    compression_mm: float,
    release_clearance_mm: float,
    layer_height_mm: float,
    start_pose: list[float] | None = None,
    return_home: bool = True,
    precontact_clearance_mm: float = 5.0,
    precision_lift_mm: float = 20.0,
) -> list[tuple[str, list[float], str, str | None]]:
    cycle_start = list(map(float, start_pose if start_pose is not None else home_pose))
    pickup_orientation = list(map(float, home_pose[3:6]))
    place_orientation = pickup_orientation.copy()
    place_orientation[2] = wrapped_angle_deg(place_orientation[2] + place_yaw_offset_deg)
    leaf_mm = np.asarray(leaf_point_m, dtype=float) * 1000.0
    ceramic_mm = np.asarray(ceramic_point_m, dtype=float) * 1000.0
    transport_z = float(max(leaf_mm[2], ceramic_mm[2]) + transport_clearance_mm)
    if transport_z > CAMERA_TRANSIT_Z_MM:
        raise RuntimeError(
            f"Transport Z {transport_z:.1f} mm exceeds the {CAMERA_TRANSIT_Z_MM:.1f} mm camera-safe corridor"
        )

    route: list[tuple[str, list[float], str, str | None]] = [
        (
            "move directly above source leaf while interpolating pickup yaw",
            compose_pose_mm(leaf_mm, float(leaf_mm[2] + approach_clearance_mm), pickup_orientation),
            "empty_transit",
            None,
        ),
        (
            "descend slowly to source precontact height",
            compose_pose_mm(leaf_mm, float(leaf_mm[2] + precontact_clearance_mm), pickup_orientation),
            "precision",
            None,
        ),
        (
            "compress suction cup on source leaf",
            compose_pose_mm(leaf_mm, float(leaf_mm[2] - compression_mm), pickup_orientation),
            "contact",
            "acquire",
        ),
        (
            "lift leaf slowly until suction is stable",
            compose_pose_mm(leaf_mm, float(leaf_mm[2] + precision_lift_mm), pickup_orientation),
            "loaded_precision",
            None,
        ),
        (
            "lift leaf vertically with pickup yaw fixed",
            compose_pose_mm(leaf_mm, transport_z, pickup_orientation),
            "loaded_vertical",
            None,
        ),
        (
            "move above ceramic while interpolating placement yaw",
            compose_pose_mm(
                ceramic_mm,
                float(ceramic_mm[2] + approach_clearance_mm + layer_height_mm),
                place_orientation,
            ),
            "loaded_transit",
            None,
        ),
        (
            "lower leaf to release height",
            compose_pose_mm(
                ceramic_mm,
                float(ceramic_mm[2] + release_clearance_mm + layer_height_mm),
                place_orientation,
            ),
            "contact",
            "release",
        ),
        (
            "retreat slowly after releasing leaf",
            compose_pose_mm(
                ceramic_mm,
                float(ceramic_mm[2] + approach_clearance_mm + layer_height_mm),
                place_orientation,
            ),
            "precision",
            None,
        ),
    ]
    if return_home:
        route.append(
            (
                "return home directly after safe release retreat",
                list(map(float, home_pose)),
                "empty_transit",
                None,
            )
        )
    else:
        route.append(
            (
                "retract vertically above ceramic",
                compose_pose_mm(ceramic_mm, transport_z, place_orientation),
                "vertical",
                None,
            )
        )
    require_overhead_camera_safe_route([cycle_start, *[item[1] for item in route]])
    return route


def choose_second_source_leaf(
    detections: list[dict[str, Any]],
    first_source_m: list[float],
    allow_clipping: bool,
    max_shift_m: float,
) -> dict[str, Any]:
    candidates, _ = select_leaf_waypoints(
        detections,
        distinct_distance_m=DISTINCT_LEAF_DISTANCE_M,
        allow_any_clipping=allow_clipping,
    )
    reference_xy = np.asarray(first_source_m, dtype=float)[:2]
    selected = min(
        candidates,
        key=lambda item: float(np.linalg.norm(np.asarray(item["point_base_m"], dtype=float)[:2] - reference_xy)),
    )
    shift_m = float(np.linalg.norm(np.asarray(selected["point_base_m"], dtype=float)[:2] - reference_xy))
    if shift_m > max_shift_m:
        raise RuntimeError(
            f"Second-scan leaf is {shift_m * 1000.0:.1f} mm from the original source stack; "
            "refusing to risk picking the leaf already placed on the ceramic disk"
        )
    return selected


def route_blend_radii(
    route: list[tuple[str, list[float], str, str | None]],
    blend_radius_mm: float,
) -> list[float]:
    radii: list[float] = []
    for index, (_label, _target_pose, motion, event) in enumerate(route):
        next_event = route[index + 1][3] if index + 1 < len(route) else None
        next_motion = route[index + 1][2] if index + 1 < len(route) else None
        blendable = (
            blend_radius_mm > 0
            and event is None
            and (
                (next_event == "acquire" and motion in ("empty_transit", "precision"))
                or (motion, next_motion) in BLENDABLE_TRANSITIONS
            )
        )
        radii.append(float(blend_radius_mm if blendable else -1.0))
    return radii
