"""Shared historical workcell paths and read-only controller checks."""
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE = REPO_ROOT / "xarm_sdk_control/data/recommissioning/2026-10-14"
CAL = REPO_ROOT / "xarm_sdk_control/calibration/experimental/20261014/global_camera_candidate.json"

def get(value):
    if value[0] != 0:
        raise RuntimeError(f'Controller read failed: {value}')
    return value[1]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def ready(arm):
    if get(arm.get_err_warn_code()) != [0, 0] or get(arm.get_state()) not in (0, 2):
        raise RuntimeError('Controller not ready')
    if get(arm.get_cmdnum()) != 0 or arm.mode != 0:
        raise RuntimeError('Controller queue/mode changed')
    return list(get(arm.get_position())), list(arm.tcp_offset)
