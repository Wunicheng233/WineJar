from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path

from xarm_sdk_control.vacuum_control import VacuumController, VacuumIOConfig


ROOT = Path(__file__).resolve().parents[2]
IO_CHECK = SourceFileLoader(
    "vacuum_io_check",
    str(ROOT / "xarm_sdk_control" / "scripts" / "15_vacuum_io_check.py"),
).load_module()


class FakeArm:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def set_cgpio_digital_output_function(self, channel: int, function: int) -> int:
        self.calls.append(("function", channel, function))
        return 0

    def set_cgpio_digital(self, channel: int, value: int) -> int:
        self.calls.append(("digital", channel, value))
        return 0


def test_acquire_and_release_use_co0_and_co1() -> None:
    arm = FakeArm()
    sleeps: list[float] = []
    vacuum = VacuumController(arm, VacuumIOConfig(), sleep=sleeps.append)

    vacuum.configure()
    vacuum.acquire(1.25)
    vacuum.release(0.5)

    assert arm.calls == [
        ("function", 0, 0),
        ("function", 1, 0),
        ("digital", 1, 0),
        ("digital", 0, 1),
        ("digital", 0, 0),
        ("digital", 1, 1),
        ("digital", 1, 0),
    ]
    assert sleeps == [1.25, 0.5]


def test_active_low_configuration_inverts_safe_outputs() -> None:
    arm = FakeArm()
    vacuum = VacuumController(
        arm,
        VacuumIOConfig(pump_on_value=0, vent_on_value=0),
        sleep=lambda _seconds: None,
    )

    vacuum.configure()
    vacuum.safe_shutdown()

    assert arm.calls[-2:] == [("digital", 0, 1), ("digital", 1, 1)]


def test_unreachable_controller_port_returns_false(monkeypatch) -> None:
    def fail_connection(*_args, **_kwargs):
        raise OSError("unreachable")

    monkeypatch.setattr(IO_CHECK.socket, "create_connection", fail_connection)
    assert not IO_CHECK.controller_port_reachable("192.0.2.1", timeout_seconds=0.01)
