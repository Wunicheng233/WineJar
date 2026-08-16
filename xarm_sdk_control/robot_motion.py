"""Shared real-robot preparation and safe reset motion."""

from __future__ import annotations

from typing import Any


CAMERA_ZONE_X_MIN_MM = 300.0
CAMERA_ZONE_Z_MAX_MM = 300.0
CAMERA_TRANSIT_Z_MM = 280.0


def require_code(label: str, result: Any) -> Any:
    code = result[0] if isinstance(result, tuple) else result
    print(f"{label}: {result}")
    if code not in (0, None):
        raise RuntimeError(f"{label} failed with code/result: {result}")
    return result


def prepare_robot_for_motion(arm: Any) -> None:
    status = require_code("get controller error/warn before motion", arm.get_err_warn_code())
    error_code, warning_code = status[1]
    if error_code != 0:
        raise RuntimeError(f"Controller has unresolved error C{error_code} (warning={warning_code})")
    require_code("clean warning", arm.clean_warn())
    require_code("motion enable", arm.motion_enable(enable=True))
    require_code("set mode 0", arm.set_mode(0))
    require_code("set state 0", arm.set_state(0))


def pose_in_overhead_camera_keepout(pose: list[float] | tuple[float, ...]) -> bool:
    return float(pose[0]) > CAMERA_ZONE_X_MIN_MM and float(pose[2]) > CAMERA_ZONE_Z_MAX_MM


def _strictly_above_interval(start: float, end: float, limit: float) -> tuple[float, float] | None:
    if start > limit and end > limit:
        return 0.0, 1.0
    if start <= limit and end <= limit:
        return None
    crossing = (limit - start) / (end - start)
    return (0.0, crossing) if start > limit else (crossing, 1.0)


def segment_enters_overhead_camera_keepout(start: list[float], end: list[float]) -> bool:
    x_interval = _strictly_above_interval(float(start[0]), float(end[0]), CAMERA_ZONE_X_MIN_MM)
    z_interval = _strictly_above_interval(float(start[2]), float(end[2]), CAMERA_ZONE_Z_MAX_MM)
    if x_interval is None or z_interval is None:
        return False
    return max(x_interval[0], z_interval[0]) < min(x_interval[1], z_interval[1]) - 1e-9


def require_overhead_camera_safe_route(poses: list[list[float]]) -> None:
    for index, pose in enumerate(poses):
        if pose_in_overhead_camera_keepout(pose):
            raise RuntimeError(f"Pose {index} enters the overhead-camera keepout: {pose[:3]}")
    for index, (start, end) in enumerate(zip(poses, poses[1:]), start=1):
        if segment_enters_overhead_camera_keepout(start, end):
            raise RuntimeError(
                f"Segment {index} enters the overhead-camera keepout: {start[:3]} -> {end[:3]}"
            )


def build_safe_home_route(
    current_pose: list[float],
    home_pose: list[float] | tuple[float, ...],
) -> list[tuple[str, list[float]]]:
    current = list(map(float, current_pose))
    home = list(map(float, home_pose))
    if len(current) != 6 or len(home) != 6:
        raise ValueError("Current and home poses must each contain six values")

    if pose_in_overhead_camera_keepout(current):
        raise RuntimeError(
            "Current TCP pose is already inside the overhead-camera keepout; "
            "inspect the hardware and manually lower Z before automatic reset"
        )

    if current[0] > CAMERA_ZONE_X_MIN_MM:
        corridor_pose = [current[0], current[1], CAMERA_TRANSIT_Z_MM, *current[3:6]]
        route: list[tuple[str, list[float]]] = []
        if abs(current[2] - CAMERA_TRANSIT_Z_MM) > 1.0:
            route.append(("move vertically to camera-safe corridor", corridor_pose))
        route.append(
            (
                "leave overhead-camera area at safe height",
                [home[0], home[1], CAMERA_TRANSIT_Z_MM, *home[3:6]],
            )
        )
        if abs(home[2] - CAMERA_TRANSIT_Z_MM) > 1.0:
            route.append(("raise vertically at camera-safe home column", home))
        require_overhead_camera_safe_route([current, *[pose for _label, pose in route]])
        return route

    clearance_z = max(current[2], home[2])
    route: list[tuple[str, list[float]]] = []
    if current[2] < clearance_z - 1.0:
        route.append(("raise vertically before reset", [current[0], current[1], clearance_z, *current[3:6]]))
    high_home = [home[0], home[1], clearance_z, *home[3:6]]
    route.append(("move horizontally to reset pose", high_home))
    if home[2] < clearance_z - 1.0:
        route.append(("descend vertically to reset pose", home))
    require_overhead_camera_safe_route([current, *[pose for _label, pose in route]])
    return route


def move_robot_home(
    arm: Any,
    home_pose: list[float] | tuple[float, ...],
    speed: float,
    acceleration: float,
) -> list[float]:
    pose_code, current_pose = arm.get_position(is_radian=False)
    require_code("get current TCP pose before reset", (pose_code, current_pose))
    for label, target in build_safe_home_route(current_pose, home_pose):
        require_code(
            label,
            arm.set_position(
                *target,
                speed=speed,
                mvacc=acceleration,
                wait=True,
                radius=-1,
                is_radian=False,
            ),
        )
    pose_code, final_pose = arm.get_position(is_radian=False)
    require_code("get reset TCP pose", (pose_code, final_pose))
    return list(map(float, final_pose))


def reset_robot_at_ip(
    robot_ip: str,
    tcp_offset_mm_deg: list[float] | tuple[float, ...],
    home_pose_mm_deg: list[float] | tuple[float, ...],
    speed: float,
    acceleration: float,
) -> list[float]:
    """Connect, apply the tool frame, move safely home, and disconnect."""
    from xarm.wrapper import XArmAPI

    arm = XArmAPI(do_not_open=True, is_radian=False)
    try:
        require_code("connect for task reset", arm.connect(port=robot_ip))
        require_code(
            "set task TCP offset",
            arm.set_tcp_offset(list(tcp_offset_mm_deg), is_radian=False, wait=True),
        )
        prepare_robot_for_motion(arm)
        return move_robot_home(arm, home_pose_mm_deg, speed, acceleration)
    finally:
        arm.disconnect()
