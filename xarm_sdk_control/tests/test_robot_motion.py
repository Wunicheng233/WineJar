import pytest

from xarm_sdk_control.robot_motion import (
    build_safe_home_route,
    require_overhead_camera_safe_route,
)


def test_low_pose_is_raised_before_horizontal_reset() -> None:
    route = build_safe_home_route(
        [370.0, 60.0, 1.0, 178.0, 0.0, 0.0],
        [215.0, 0.0, 386.0, 178.0, 0.0, 0.0],
    )

    assert [label for label, _pose in route] == [
        "move vertically to camera-safe corridor",
        "leave overhead-camera area at safe height",
        "raise vertically at camera-safe home column",
    ]
    assert route[0][1][:3] == [370.0, 60.0, 280.0]
    assert route[1][1][:3] == [215.0, 0.0, 280.0]
    assert route[2][1][:3] == [215.0, 0.0, 386.0]


def test_pose_already_inside_camera_keepout_requires_manual_recovery() -> None:
    with pytest.raises(RuntimeError, match="manually lower Z"):
        build_safe_home_route(
            [400.0, 100.0, 450.0, 180.0, 0.0, 20.0],
            [215.0, 0.0, 386.0, 178.0, 0.0, 0.0],
        )


def test_diagonal_segment_through_camera_keepout_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="Segment"):
        require_overhead_camera_safe_route(
            [
                [215.0, 0.0, 386.0, 178.0, 0.0, 0.0],
                [500.0, 80.0, 250.0, 178.0, 0.0, 0.0],
            ]
        )
