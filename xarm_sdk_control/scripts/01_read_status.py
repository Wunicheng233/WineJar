#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import yaml
from xarm.wrapper import XArmAPI


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "robot.yaml"


def call_and_print(label: str, func, *args, **kwargs):
    try:
        result = func(*args, **kwargs)
    except Exception as exc:
        print(f"{label}: EXCEPTION: {exc!r}")
        return None
    print(f"{label}: {result}")
    return result


def main():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    robot_ip = config["robot_ip"]
    print(f"Connecting to xArm at {robot_ip}")
    print("Read-only check: no motion_enable, no mode/state change, no gripper command, no motion command.")

    arm = XArmAPI(do_not_open=True, is_radian=False)
    try:
        connect_code = arm.connect(port=robot_ip)
        print(f"connect_code: {connect_code}")
        print(f"connected: {arm.connected}")

        print("\n=== Cached Properties ===")
        print(f"version: {arm.version}")
        print(f"state: {arm.state}")
        print(f"mode: {arm.mode}")
        print(f"error_code: {arm.error_code}")
        print(f"warn_code: {arm.warn_code}")
        print(f"position: {arm.position}")
        print(f"angles: {arm.angles}")

        print("\n=== Getter Calls ===")
        call_and_print("get_version", arm.get_version)
        call_and_print("get_state", arm.get_state)
        call_and_print("get_cmdnum", arm.get_cmdnum)
        call_and_print("get_err_warn_code", arm.get_err_warn_code)
        call_and_print("get_position(deg)", arm.get_position, is_radian=False)
        call_and_print("get_servo_angle(deg)", arm.get_servo_angle, is_radian=False)
    finally:
        arm.disconnect()
        print("\nDisconnected.")


if __name__ == "__main__":
    main()
