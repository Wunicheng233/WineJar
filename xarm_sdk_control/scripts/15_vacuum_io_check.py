#!/usr/bin/env python3
"""Check the prototype pump/valve outputs without enabling robot motion."""

from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xarm_sdk_control.vacuum_control import VacuumController, VacuumIOConfig  # noqa: E402


def controller_port_reachable(robot_ip: str, timeout_seconds: float = 2.0) -> bool:
    try:
        with socket.create_connection((robot_ip, 502), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", choices=("status", "off", "pump", "vent", "cycle"), default="status")
    parser.add_argument("--pump-on-value", type=int, choices=(0, 1), default=1)
    parser.add_argument("--vent-on-value", type=int, choices=(0, 1), default=1)
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.hold_seconds < 0.0 or args.hold_seconds > 10.0:
        raise ValueError("--hold-seconds must be between 0 and 10 seconds")

    with (ROOT / "config" / "robot.yaml").open("r", encoding="utf-8") as file:
        robot_ip = str(yaml.safe_load(file)["robot_ip"])
    if not controller_port_reachable(robot_ip):
        print(f"ERROR: xArm controller {robot_ip}:502 is unreachable.", file=sys.stderr)
        print("Check controller power, Ethernet connection, and the computer's IPv4 subnet before retrying.", file=sys.stderr)
        return 2

    from xarm.wrapper import XArmAPI

    arm = XArmAPI(do_not_open=True, is_radian=False)
    vacuum: VacuumController | None = None
    try:
        try:
            connect_code = arm.connect(port=robot_ip)
        except Exception as exc:
            print(f"ERROR: SDK connection failed after the controller port responded: {exc}", file=sys.stderr)
            return 3
        print(f"connect: {connect_code}")
        if connect_code not in (0, None):
            raise RuntimeError(f"Robot connection failed: {connect_code}")
        print(f"controller error/warn: {arm.get_err_warn_code()}")
        print(f"controller GPIO state: {arm.get_cgpio_state()}")
        if args.test == "status":
            print("STATUS ONLY: no output was changed and robot motion was not enabled.")
            return 0

        config = VacuumIOConfig(
            pump_channel=0,
            vent_channel=1,
            pump_on_value=args.pump_on_value,
            vent_on_value=args.vent_on_value,
        )
        vacuum = VacuumController(arm, config)
        vacuum.configure()
        vacuum.safe_shutdown()
        if args.test == "off":
            print("Pump is OFF and vent valve is CLOSED. Robot motion was not enabled.")
            return 0
        time.sleep(0.5)

        if args.test == "pump":
            print(f"Pulsing the pump for {args.hold_seconds:.1f} seconds. Observe relay X1 and pump sound.")
            vacuum.set_pump(True)
            time.sleep(args.hold_seconds)
            vacuum.set_pump(False)
        elif args.test == "vent":
            print(f"Opening the vent valve for {args.hold_seconds:.1f} seconds. Observe relay X2 and valve click.")
            vacuum.set_vent(True)
            time.sleep(args.hold_seconds)
            vacuum.set_vent(False)
        else:
            print(f"Running one vacuum/vent cycle with a {args.hold_seconds:.1f}-second vacuum hold.")
            vacuum.acquire(args.hold_seconds)
            vacuum.release(1.0)
        print("I/O check complete. Robot motion was not enabled.")
        return 0
    finally:
        if vacuum is not None:
            try:
                vacuum.safe_shutdown()
            except Exception as exc:
                print(f"WARNING: output cleanup failed: {exc}", file=sys.stderr)
        arm.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
