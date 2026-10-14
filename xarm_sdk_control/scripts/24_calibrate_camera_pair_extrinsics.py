#!/usr/bin/env python3
"""Calibrate a fixed camera pair using chessboard observations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from xarm_sdk_control.pair_calibration_session import main

if __name__ == "__main__":
    raise SystemExit(main("chessboard"))
