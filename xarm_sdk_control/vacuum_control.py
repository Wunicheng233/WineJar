"""Controller-GPIO interface for the prototype vacuum end effector."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable


def _require_code(label: str, result: Any) -> Any:
    code = result[0] if isinstance(result, tuple) else result
    print(f"{label}: {result}")
    if code not in (0, None):
        raise RuntimeError(f"{label} failed with code/result: {result}")
    return result


@dataclass(frozen=True)
class VacuumIOConfig:
    pump_channel: int = 0
    valve_channel: int = 1
    pump_on_value: int = 1
    valve_vacuum_value: int = 1

    def __post_init__(self) -> None:
        for name, channel in (("pump_channel", self.pump_channel), ("valve_channel", self.valve_channel)):
            if not 0 <= channel <= 7:
                raise ValueError(f"{name} must select CO0~CO7, got {channel}")
        for name, value in (("pump_on_value", self.pump_on_value), ("valve_vacuum_value", self.valve_vacuum_value)):
            if value not in (0, 1):
                raise ValueError(f"{name} must be 0 or 1, got {value}")
        if self.pump_channel == self.valve_channel:
            raise ValueError("Pump and vent valve must use different controller outputs")

    @property
    def pump_off_value(self) -> int:
        return 1 - self.pump_on_value

    @property
    def valve_vent_value(self) -> int:
        return 1 - self.valve_vacuum_value


class VacuumController:
    """Drive the pump and vacuum/vent selector valve through controller outputs."""

    def __init__(
        self,
        arm: Any,
        config: VacuumIOConfig,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.arm = arm
        self.config = config
        self.sleep = sleep
        self.configured = False
        self.vacuum_path_prepared = False

    def configure(self) -> None:
        for channel, label in (
            (self.config.pump_channel, "pump"),
            (self.config.valve_channel, "vacuum selector valve"),
        ):
            _require_code(
                f"configure {label} CO{channel} as general output",
                self.arm.set_cgpio_digital_output_function(channel, 0),
            )
        self.configured = True

    def _write(self, channel: int, value: int, label: str) -> None:
        if not self.configured:
            raise RuntimeError("Vacuum outputs must be configured before use")
        _require_code(label, self.arm.set_cgpio_digital(channel, value))

    def set_pump(self, enabled: bool) -> None:
        value = self.config.pump_on_value if enabled else self.config.pump_off_value
        self._write(self.config.pump_channel, value, f"pump {'ON' if enabled else 'OFF'}: CO{self.config.pump_channel}={value}")

    def set_vacuum_path(self, enabled: bool) -> None:
        value = self.config.valve_vacuum_value if enabled else self.config.valve_vent_value
        state = "VACUUM" if enabled else "VENT (release)"
        self._write(
            self.config.valve_channel,
            value,
            f"selector valve {state}: CO{self.config.valve_channel}={value}",
        )
        self.vacuum_path_prepared = enabled

    def prepare_for_pick(self) -> None:
        """Select the vacuum path before approaching the workpiece."""
        self.set_pump(False)
        self.set_vacuum_path(True)

    def acquire(self, settle_seconds: float) -> None:
        if not self.vacuum_path_prepared:
            self.set_vacuum_path(True)
        self.set_pump(True)
        self.sleep(settle_seconds)

    def release(self, settle_seconds: float) -> None:
        self.set_pump(False)
        self.set_vacuum_path(False)
        self.sleep(settle_seconds)

    def safe_shutdown(self) -> None:
        """Best-effort de-energization: pump off and selector valve venting."""
        if not self.configured:
            return
        errors: list[Exception] = []
        for action in (lambda: self.set_pump(False), lambda: self.set_vacuum_path(False)):
            try:
                action()
            except Exception as exc:  # Preserve both cleanup attempts.
                errors.append(exc)
        if errors:
            raise RuntimeError("Failed to restore all vacuum outputs to their safe state") from errors[0]
