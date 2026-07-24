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
    vent_channel: int = 1
    pump_on_value: int = 1
    vent_on_value: int = 1

    def __post_init__(self) -> None:
        for name, channel in (("pump_channel", self.pump_channel), ("vent_channel", self.vent_channel)):
            if not 0 <= channel <= 7:
                raise ValueError(f"{name} must select CO0~CO7, got {channel}")
        for name, value in (("pump_on_value", self.pump_on_value), ("vent_on_value", self.vent_on_value)):
            if value not in (0, 1):
                raise ValueError(f"{name} must be 0 or 1, got {value}")
        if self.pump_channel == self.vent_channel:
            raise ValueError("Pump and vent valve must use different controller outputs")

    @property
    def pump_off_value(self) -> int:
        return 1 - self.pump_on_value

    @property
    def vent_off_value(self) -> int:
        return 1 - self.vent_on_value


class VacuumController:
    """Drive the pump and normally-closed vent valve through controller outputs."""

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

    def configure(self) -> None:
        for channel, label in (
            (self.config.pump_channel, "pump"),
            (self.config.vent_channel, "vent valve"),
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

    def set_vent(self, enabled: bool) -> None:
        value = self.config.vent_on_value if enabled else self.config.vent_off_value
        state = "OPEN (release)" if enabled else "CLOSED (vacuum hold)"
        self._write(self.config.vent_channel, value, f"vent valve {state}: CO{self.config.vent_channel}={value}")

    def acquire(self, settle_seconds: float) -> None:
        self.set_vent(False)
        self.set_pump(True)
        self.sleep(settle_seconds)

    def release(self, settle_seconds: float) -> None:
        self.set_pump(False)
        self.set_vent(True)
        self.sleep(settle_seconds)
        self.set_vent(False)

    def safe_shutdown(self) -> None:
        """Best-effort de-energization: pump off and vent valve closed."""
        if not self.configured:
            return
        errors: list[Exception] = []
        for action in (lambda: self.set_pump(False), lambda: self.set_vent(False)):
            try:
                action()
            except Exception as exc:  # Preserve both cleanup attempts.
                errors.append(exc)
        if errors:
            raise RuntimeError("Failed to restore all vacuum outputs to their safe state") from errors[0]
