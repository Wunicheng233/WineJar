"""Pure visual checks for confirming that a vacuum-picked object moved with the tool."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from xarm_sdk_control.visual_targeting import detection_object_kind


VerificationStatus = Literal["confirmed", "not_confirmed", "inconclusive"]


@dataclass(frozen=True)
class VerificationPolicy:
    minimum_lift_mm: float
    tool_xy_tolerance_mm: float
    height_tolerance_mm: float
    source_xy_tolerance_mm: float
    source_height_tolerance_mm: float


POLICIES = {
    "bamboo": VerificationPolicy(15.0, 55.0, 35.0, 55.0, 15.0),
    "ceramic": VerificationPolicy(12.0, 35.0, 25.0, 35.0, 10.0),
}


def policy_for(target_kind: str) -> VerificationPolicy:
    try:
        return POLICIES[target_kind]
    except KeyError as exc:
        raise ValueError(f"Unsupported target kind: {target_kind}") from exc


@dataclass(frozen=True)
class PickVerification:
    status: VerificationStatus
    reason: str
    target_kind: str
    candidate: dict[str, Any] | None
    observed_lift_mm: float | None
    tool_xy_error_mm: float | None

    @property
    def confirmed(self) -> bool:
        return self.status == "confirmed"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_carried_object(
    detections: list[dict[str, Any]],
    *,
    target_kind: str,
    source_point_m: list[float],
    expected_object_point_m: list[float],
    min_score: float,
    minimum_lift_mm: float,
    tool_xy_tolerance_mm: float,
    height_tolerance_mm: float,
    source_xy_tolerance_mm: float,
    source_height_tolerance_mm: float,
    allow_clipped_candidate: bool = False,
) -> PickVerification:
    if target_kind not in {"bamboo", "ceramic"}:
        raise ValueError(f"Unsupported target kind: {target_kind}")
    source = np.asarray(source_point_m, dtype=float)
    expected = np.asarray(expected_object_point_m, dtype=float)
    if source.shape != (3,) or expected.shape != (3,):
        raise ValueError("Source and expected object points must contain three values")

    candidates = []
    for item in detections:
        point_value = item.get("point_base_m")
        if (
            detection_object_kind(item) != target_kind
            or float(item.get("score", 0.0)) < min_score
            or not isinstance(point_value, list)
            or len(point_value) != 3
        ):
            continue
        point = np.asarray(point_value, dtype=float)
        xy_error_mm = float(np.linalg.norm(point[:2] - expected[:2]) * 1000.0)
        lift_mm = float((point[2] - source[2]) * 1000.0)
        height_error_mm = float(abs(point[2] - expected[2]) * 1000.0)
        candidates.append((item, point, xy_error_mm, lift_mm, height_error_mm))

    carried = [
        entry
        for entry in candidates
        if entry[2] <= tool_xy_tolerance_mm
        and entry[3] >= minimum_lift_mm
        and entry[4] <= height_tolerance_mm
        and (allow_clipped_candidate or not bool(entry[0].get("touches_image_edge", False)))
    ]
    if carried:
        item, _point, xy_error, lift, _height_error = min(carried, key=lambda entry: (entry[2], entry[4]))
        return PickVerification(
            "confirmed",
            "target detected at the lifted tool position",
            target_kind,
            item,
            lift,
            xy_error,
        )

    at_source = [
        entry
        for entry in candidates
        if float(np.linalg.norm(entry[1][:2] - source[:2]) * 1000.0) <= source_xy_tolerance_mm
        and abs(entry[3]) <= source_height_tolerance_mm
    ]
    if at_source:
        item, _point, xy_error, lift, _height_error = max(
            at_source,
            key=lambda entry: float(entry[0].get("score", 0.0)),
        )
        return PickVerification(
            "not_confirmed",
            "target remains near its original source height",
            target_kind,
            item,
            lift,
            xy_error,
        )

    return PickVerification(
        "inconclusive",
        "no unambiguous target was detected at the tool or source",
        target_kind,
        None,
        None,
        None,
    )
