from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE = SourceFileLoader(
    "visual_vacuum_pick",
    str(ROOT / "xarm_sdk_control" / "scripts" / "16_visual_vacuum_pick.py"),
).load_module()


def test_full_route_uses_vertical_contact_and_release_segments() -> None:
    route = MODULE.build_route(
        initial_pose=[220, 0, 405, 180, 0, 0],
        leaf_point_m=[0.50, 0.06, -0.015],
        ceramic_point_m=[0.30, -0.16, 0.018],
        stage="full",
        transport_clearance_mm=200,
        approach_clearance_mm=35,
        compression_mm=15,
        release_clearance_mm=5,
    )

    poses = {label: pose for label, pose, _kind in route}
    assert poses["compress suction cups on leaf"][:3] == [500.0, 60.0, -30.0]
    assert poses["lower leaf to release clearance"][:3] == [300.0, -160.0, 23.0]
    assert poses["lift leaf vertically"][:2] == poses["compress suction cups on leaf"][:2]
    assert poses["retract vertically above ceramic"][:2] == poses["lower leaf to release clearance"][:2]
    assert route[0][0] == "move directly above leaf"
    assert route[-1][0] == "return directly to initial pose"


def test_approach_route_never_reaches_leaf_contact_height() -> None:
    route = MODULE.build_route(
        initial_pose=[220, 0, 405, 180, 0, 0],
        leaf_point_m=[0.50, 0.06, -0.015],
        ceramic_point_m=[0.30, -0.16, 0.018],
        stage="approach",
        transport_clearance_mm=200,
        approach_clearance_mm=35,
        compression_mm=15,
        release_clearance_mm=5,
    )

    assert min(pose[2] for _label, pose, _kind in route) == 20.0
    assert all(label != "compress suction cups on leaf" for label, _pose, _kind in route)


def test_low_initial_pose_is_raised_before_horizontal_motion() -> None:
    route = MODULE.build_route(
        initial_pose=[400, -120, 85, 180, 0, 0],
        leaf_point_m=[0.50, 0.06, -0.015],
        ceramic_point_m=[0.50, 0.06, -0.015],
        stage="approach",
        transport_clearance_mm=200,
        approach_clearance_mm=35,
        compression_mm=15,
        release_clearance_mm=5,
    )

    assert route[0][0] == "raise vertically from low initial pose"
    assert route[0][1][:2] == [400.0, -120.0]
    assert route[1][0] == "move directly above leaf"
    assert route[-2][0] == "move above low initial pose"
