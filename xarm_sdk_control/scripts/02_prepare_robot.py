#!/usr/bin/env python3
from __future__ import annotations

import time
from pathlib import Path

import yaml
from xarm.wrapper import XArmAPI


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "robot.yaml"


def call(label: str, func, *args, **kwargs):
    result = func(*args, **kwargs)
    print(f"{label}: {result}")
    return result


def print_status(arm: XArmAPI, title: str):
    print(f"\n=== {title} ===")
    print(f"connected: {arm.connected}")
    print(f"state: {arm.state}")
    print(f"mode: {arm.mode}")
    print(f"error_code: {arm.error_code}")
    print(f"warn_code: {arm.warn_code}")
    print(f"position: {arm.position}")
    print(f"angles: {arm.angles}")
    print(f"get_state: {arm.get_state()}")
    print(f"get_err_warn_code: {arm.get_err_warn_code()}")


def main():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    robot_ip = config["robot_ip"]
    print(f"Connecting to xArm at {robot_ip}")
    print("Prepare check: clean warn/error, motion_enable, set_mode(0), set_state(0).")
    print("No motion command. No gripper command.")

    arm = XArmAPI(do_not_open=True, is_radian=False)
    try:
        call("connect", arm.connect, port=robot_ip)
        print_status(arm, "Before Prepare")

        call("clean_warn", arm.clean_warn)
        call("clean_error", arm.clean_error)
        call("motion_enable", arm.motion_enable, enable=True)
        call("set_mode(0)", arm.set_mode, 0)
        call("set_state(0)", arm.set_state, 0)
        time.sleep(1.0)

        print_status(arm, "After Prepare")
    finally:
        arm.disconnect()
        print("\nDisconnected.")


if __name__ == "__main__":
    main()
