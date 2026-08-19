#!/usr/bin/env python3
"""Read and print the vacuum pressure switch state on xArm CI0 in real time.

Read-only: never enables motion or changes any controller output. Use it to
determine which CI0 level (0/1) means "suction achieved": run with the cup
open to air, then block the cup mouth with a finger while the pump is on.

Pump control for the polarity test is manual: run 15_vacuum_io_check.py
--test cycle (or --test on) in another terminal.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xarm_sdk_control.robot_motion import load_robot_ip, require_code  # noqa: E402


CONFIG_PATH = ROOT / "config" / "robot.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-seconds", type=float, default=0.2)
    parser.add_argument("--samples", type=int, default=0, help="Stop after N samples; 0 means run until Ctrl+C.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.01 <= args.interval_seconds <= 2.0:
        raise ValueError("--interval-seconds must be within 0.01~2.0")
    robot_ip = load_robot_ip(CONFIG_PATH)
    from xarm.wrapper import XArmAPI

    arm = XArmAPI(do_not_open=True, is_radian=False)
    try:
        require_code("connect for CI0 read", arm.connect(port=robot_ip))
        print("READ-ONLY: printing CI0 every "
              f"{args.interval_seconds:.2f} s. Ctrl+C to stop.", flush=True)
        sample = 0
        while True:
            code, value = arm.get_cgpio_digital(0)
            if code != 0:
                print(f"ERROR: failed to read CI0: code={code}", file=sys.stderr)
                return 1
            print(f"CI0 = {int(value)}", flush=True)
            sample += 1
            if args.samples and sample >= args.samples:
                return 0
            time.sleep(args.interval_seconds)
    except KeyboardInterrupt:
        print("Stopped by user.")
        return 0
    finally:
        arm.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
