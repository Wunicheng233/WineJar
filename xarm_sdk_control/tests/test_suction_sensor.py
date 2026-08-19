from __future__ import annotations

import pytest

from xarm_sdk_control.end_effector_config import load_end_effector_config
from xarm_sdk_control.suction_sensor import (
    SAMPLE_INTERVAL_SECONDS,
    SuctionNotConfirmedError,
    SuctionSensor,
    SuctionSensorConfig,
    SuctionWaitOutcome,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeArm:
    def __init__(self, values) -> None:
        self.values = list(values)

    def get_cgpio_digital(self, channel):
        return 0, self.values.pop(0) if self.values else 0


class ErrorArm:
    def get_cgpio_digital(self, channel):
        return 1, 0


def make_sensor(values, clock: FakeClock) -> SuctionSensor:
    arm = FakeArm(values)
    return SuctionSensor(
        arm,
        SuctionSensorConfig(input_channel=0, success_value=0, debounce_seconds=0.05, timeout_seconds=2.0),
        clock=clock,
        sleep=lambda _seconds: clock.advance(_seconds),
    ), arm


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="CI0~CI7"):
        SuctionSensorConfig(input_channel=8, success_value=0, debounce_seconds=0.05, timeout_seconds=2.0)
    with pytest.raises(ValueError, match="0 or 1"):
        SuctionSensorConfig(input_channel=0, success_value=2, debounce_seconds=0.05, timeout_seconds=2.0)
    with pytest.raises(ValueError, match="positive"):
        SuctionSensorConfig(input_channel=0, success_value=0, debounce_seconds=0.0, timeout_seconds=2.0)
    with pytest.raises(ValueError, match="at least debounce"):
        SuctionSensorConfig(input_channel=0, success_value=0, debounce_seconds=0.1, timeout_seconds=0.05)


def test_read_returns_value_and_raises_on_code_error() -> None:
    clock = FakeClock()
    sensor, _arm = make_sensor([0], clock)
    assert sensor.read() == 0
    sensor_fail = SuctionSensor(
        ErrorArm(),
        sensor.config,
        clock=clock,
        sleep=lambda s: clock.advance(s),
    )
    with pytest.raises(RuntimeError, match="Failed to read CI0"):
        sensor_fail.read()


def test_is_suction_ok_requires_continuous_success() -> None:
    clock = FakeClock()
    sensor, _arm = make_sensor([0, 0, 0, 0, 0, 0], clock)
    # First sample starts the streak; the sleep advances the clock so the
    # streak completes within the provided values.
    assert sensor.is_suction_ok() is True

    clock = FakeClock()
    sensor, _arm = make_sensor([0, 1], clock)
    assert sensor.is_suction_ok() is False


def test_wait_for_suction_confirms_within_timeout() -> None:
    clock = FakeClock()
    sensor, _arm = make_sensor([0, 0, 0, 0, 0, 0], clock)
    assert sensor.wait_for_suction() is True
    assert sensor.last_wait is not None and sensor.last_wait.ok is True
    assert sensor.last_wait.elapsed_seconds >= 0.05


def test_wait_for_suction_times_out_and_records_outcome() -> None:
    clock = FakeClock()
    sensor, _arm = make_sensor([1, 1, 1, 1], clock)
    assert sensor.wait_for_suction(timeout_seconds=0.03, debounce_seconds=0.01) is False
    assert sensor.last_wait is not None and sensor.last_wait.ok is False
    assert sensor.last_wait.last_value == 1


def test_wait_for_suction_rejects_timeout_below_debounce() -> None:
    clock = FakeClock()
    sensor, _arm = make_sensor([0], clock)
    with pytest.raises(ValueError, match="at least debounce"):
        sensor.wait_for_suction(timeout_seconds=0.01, debounce_seconds=0.05)


def test_suction_not_confirmed_error_carries_outcome() -> None:
    outcome = SuctionWaitOutcome(False, 2.0, 2.0, 1)
    error = SuctionNotConfirmedError(0, outcome)
    assert error.elapsed_seconds == 2.0
    assert error.timeout_seconds == 2.0
    assert error.last_value == 1
    assert "CI0" in str(error)


def test_end_effector_config_loads_suction_sensor() -> None:
    tool = load_end_effector_config()
    assert tool.suction_sensor is not None
    assert tool.suction_sensor.input_channel == 0
    assert tool.suction_sensor.success_value == 0
    assert tool.suction_sensor.debounce_seconds == 0.05
    assert tool.suction_sensor.timeout_seconds == 2.0


def test_end_effector_config_without_sensor_section_is_none(tmp_path) -> None:
    import yaml

    config = tmp_path / "tool.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "name": "test",
                "type": "single_vacuum_cup",
                "cup_count": 1,
                "tcp_offset_mm_deg": [0, 0, 120, 0, 0, 0],
                "home_pose_mm_deg": [215.0, -2.0, 300.0, 178.0, 0.0, 0.0],
                "vacuum_io": {
                    "pump_channel": 0,
                    "valve_channel": 1,
                    "pump_on_value": 1,
                    "valve_vacuum_value": 1,
                },
                "default_compression_mm": 13.0,
            }
        ),
        encoding="utf-8",
    )
    tool = load_end_effector_config(config)
    assert tool.suction_sensor is None
