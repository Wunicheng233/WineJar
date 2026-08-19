"""Shared pytest fixtures: script loading, motion args, and fake robot objects."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xarm_sdk_control.cross_place_motion import DEFAULT_MOTION_PROFILE  # noqa: E402


def load_script_module(name: str, script_relative_path: str) -> ModuleType:
    """Import a scripts/*.py entry point as a module without executing its main()."""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / script_relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def motion_args(**overrides: Any) -> SimpleNamespace:
    values = asdict(DEFAULT_MOTION_PROFILE)
    values.update(overrides)
    return SimpleNamespace(**values)


class TimeoutAtTargetArm:
    def get_position(self, is_radian=False):
        return 0, [300.0, 150.0, -60.0, 180.0, 0.0, 90.0]

    def get_state(self):
        return 0, 2

    def get_err_warn_code(self):
        return 0, [0, 0]


class TimeoutAwayFromTargetArm(TimeoutAtTargetArm):
    def get_position(self, is_radian=False):
        return 0, [300.0, 150.0, -50.0, 180.0, 0.0, 90.0]


class RecordingArm:
    def __init__(self) -> None:
        self.commands = []

    def set_position(self, *pose, **kwargs):
        self.commands.append((pose, kwargs))
        return 0

    def get_state(self):
        return 0, 4

    def get_err_warn_code(self):
        return 0, [0, 0]

    def get_position(self, is_radian=False):
        return 0, [300.0, 100.0, 200.0, 178.0, 0.0, 0.0]

    def get_servo_angle(self, is_radian=False):
        return 0, [0.0] * 7

    def get_cgpio_digital(self, channel):
        return 0, 0


class RecordingVacuum:
    def __init__(self) -> None:
        self.actions = []

    def prepare_for_pick(self) -> None:
        self.actions.append("prepare")

    def acquire(self, settle_seconds: float) -> None:
        self.actions.append(("acquire", settle_seconds))

    def release(self, settle_seconds: float) -> None:
        self.actions.append(("release", settle_seconds))


class RecordingTimeline:
    def __init__(self) -> None:
        self.events = []

    def mark(self, name: str, **details) -> None:
        self.events.append((name, details))
