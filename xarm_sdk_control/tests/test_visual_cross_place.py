from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE = SourceFileLoader(
    "visual_cross_place",
    str(ROOT / "xarm_sdk_control" / "scripts" / "19_visual_cross_place_two_leaves.py"),
).load_module()


class TimeoutAtTargetArm:
    def get_position(self, is_radian=False):
        return 0, [300.0, 150.0, -60.0, 180.0, 0.0, 90.0]

    def get_state(self):
        return 0, 2

    def get_err_warn_code(self):
        return 0, [0, 0]


class TimeoutAwayFromTargetArm(TimeoutAtTargetArm):
    def get_position(self, is_radian=False):
        return 0, [300.0, 150.0, -50.0, 180.0, 0.0, 90.0]


def test_first_leaf_interpolates_yaw_during_safe_height_transport() -> None:
    route = MODULE.build_cycle_route(
        home_pose=[219, 0, 408, 178, 0, 0],
        leaf_point_m=[0.287, 0.150, -0.044],
        ceramic_point_m=[0.400, -0.119, 0.049],
        place_yaw_offset_deg=90,
        transport_clearance_mm=200,
        approach_clearance_mm=35,
        compression_mm=15,
        release_clearance_mm=5,
        layer_height_mm=0,
    )
    poses = {label: pose for label, pose, _motion, _event in route}
    labels = [item[0] for item in route]
    assert "rotate leaf at safe height" not in labels
    assert poses["carry leaf above ceramic while interpolating yaw"][2] == 249.0
    assert poses["carry leaf above ceramic while interpolating yaw"][5] == 90.0
    assert poses["lower leaf to release height"][5] == 90.0
    assert poses["return home while restoring tool yaw"][5] == 0.0


def test_second_leaf_keeps_original_yaw_and_adds_layer_height() -> None:
    route = MODULE.build_cycle_route(
        home_pose=[219, 0, 408, 178, 0, 0],
        leaf_point_m=[0.287, 0.150, -0.045],
        ceramic_point_m=[0.400, -0.119, 0.049],
        place_yaw_offset_deg=0,
        transport_clearance_mm=200,
        approach_clearance_mm=35,
        compression_mm=15,
        release_clearance_mm=5,
        layer_height_mm=3,
    )
    labels = [item[0] for item in route]
    poses = {label: pose for label, pose, _motion, _event in route}
    assert "rotate leaf at safe height" not in labels
    assert poses["lower leaf to release height"][2] == 57.0
    assert poses["lower leaf to release height"][5] == 0.0


def test_feedback_timeout_is_accepted_only_when_target_is_verified() -> None:
    MODULE.require_motion_completed(
        TimeoutAtTargetArm(),
        "contact",
        3,
        [300.0, 150.0, -60.0, 180.0, 0.0, 90.0],
    )


def test_feedback_timeout_away_from_target_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(MODULE.time, "sleep", lambda _seconds: None)
    try:
        MODULE.require_motion_completed(
            TimeoutAwayFromTargetArm(),
            "contact",
            3,
            [300.0, 150.0, -60.0, 180.0, 0.0, 90.0],
        )
    except RuntimeError as exc:
        assert "could not be verified" in str(exc)
    else:
        raise AssertionError("A timed-out motion away from target must be rejected")
