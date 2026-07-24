#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def read_pose(arm: XArmAPI, label: str):
    code, pose = arm.get_position(is_radian=False)
    print(f"{label}: code={code}, pose={pose}")
    return code, pose


def main():
    parser = argparse.ArgumentParser(description="Move xArm TCP up 20 mm and back down. Requires --run.")
    parser.add_argument("--run", action="store_true", help="Actually execute the small motion.")
    parser.add_argument("--dz", type=float, default=20.0, help="Relative Z motion in mm.")
    parser.add_argument("--speed", type=float, default=20.0, help="TCP speed in mm/s.")
    parser.add_argument("--acc", type=float, default=100.0, help="TCP acceleration in mm/s^2.")
    args = parser.parse_args()

    if not args.run:
        print("Dry run only. Re-run with --run to execute motion.")
        return

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    robot_ip = config["robot_ip"]
    print(f"Connecting to xArm at {robot_ip}")
    print(f"Motion plan: relative Z +{args.dz} mm, wait, then relative Z -{args.dz} mm.")
    print(f"Limits: speed={args.speed} mm/s, acc={args.acc} mm/s^2.")

    arm = XArmAPI(do_not_open=True, is_radian=False)
    try:
        call("connect", arm.connect, port=robot_ip)
        call("clean_warn", arm.clean_warn)
        call("clean_error", arm.clean_error)
        call("motion_enable", arm.motion_enable, enable=True)
        call("set_mode(0)", arm.set_mode, 0)
        call("set_state(0)", arm.set_state, 0)
        time.sleep(0.5)

        read_pose(arm, "before")
        call(
            f"move relative z +{args.dz}",
            arm.set_position,
            z=args.dz,
            relative=True,
            speed=args.speed,
            mvacc=args.acc,
            wait=True,
        )
        read_pose(arm, "after up")
        time.sleep(0.5)
        call(
            f"move relative z -{args.dz}",
            arm.set_position,
            z=-args.dz,
            relative=True,
            speed=args.speed,
            mvacc=args.acc,
            wait=True,
        )
        read_pose(arm, "after return")
    finally:
        arm.disconnect()
        print("\nDisconnected.")


if __name__ == "__main__":
    main()
