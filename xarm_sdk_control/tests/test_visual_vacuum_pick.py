from __future__ import annotations

from conftest import load_script_module


MODULE = load_script_module("visual_vacuum_pick", "xarm_sdk_control/scripts/16_visual_vacuum_pick.py")


def test_approach_route_never_reaches_leaf_contact_height() -> None:
    route = MODULE.build_route(
        initial_pose=[220, 0, 405, 180, 0, 0],
        leaf_point_m=[0.50, 0.06, -0.015],
        transport_clearance_mm=200,
        approach_clearance_mm=35,
    )

    assert min(pose[2] for _label, pose, _kind in route) == 20.0
    assert all(label != "contact source leaf" for label, _pose, _kind in route)


def test_low_initial_pose_is_raised_before_horizontal_motion() -> None:
    route = MODULE.build_route(
        initial_pose=[400, -120, 85, 180, 0, 0],
        leaf_point_m=[0.50, 0.06, -0.015],
        transport_clearance_mm=200,
        approach_clearance_mm=35,
    )

    assert route[0][0] == "move vertically to camera-safe transit height"
    assert route[0][1][:2] == [400.0, -120.0]
    assert route[1][0] == "move directly above leaf"
    assert route[-2][0] == "move to camera-safe initial column"
