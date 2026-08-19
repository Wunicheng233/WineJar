"""Vacuum pressure-switch confirmation through the xArm controller digital inputs."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable


SAMPLE_INTERVAL_SECONDS = 0.010


@dataclass(frozen=True)
class SuctionSensorConfig:
    input_channel: int
    success_value: int
    debounce_seconds: float
    timeout_seconds: float

    def __post_init__(self) -> None:
        if not 0 <= self.input_channel <= 7:
            raise ValueError(f"input_channel must select CI0~CI7, got {self.input_channel}")
        if self.success_value not in (0, 1):
            raise ValueError(f"success_value must be 0 or 1, got {self.success_value}")
        if self.debounce_seconds <= 0:
            raise ValueError("debounce_seconds must be positive")
        if self.timeout_seconds < self.debounce_seconds:
            raise ValueError("timeout_seconds must be at least debounce_seconds")


@dataclass(frozen=True)
class SuctionWaitOutcome:
    ok: bool
    elapsed_seconds: float
    timeout_seconds: float
    last_value: int


class SuctionNotConfirmedError(RuntimeError):
    """Raised when the pressure switch never reports suction within the timeout."""

    def __init__(self, input_channel: int, outcome: SuctionWaitOutcome) -> None:
        self.input_channel = input_channel
        self.outcome = outcome
        super().__init__(
            f"Suction was not confirmed on CI{input_channel} within {outcome.timeout_seconds:.2f}s "
            f"(elapsed {outcome.elapsed_seconds:.3f}s, last value {outcome.last_value})."
        )

    @property
    def elapsed_seconds(self) -> float:
        return self.outcome.elapsed_seconds

    @property
    def timeout_seconds(self) -> float:
        return self.outcome.timeout_seconds

    @property
    def last_value(self) -> int:
        return self.outcome.last_value


class SuctionSensor:
    """Poll a controller digital input for a debounced suction-confirmed level.

    ``wait_for_suction`` never raises unless reading the input fails (a
    hardware fault); callers raise ``SuctionNotConfirmedError`` using
    ``sensor.last_wait`` when it returns False.
    """

    def __init__(
        self,
        arm: Any,
        config: SuctionSensorConfig,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.arm = arm
        self.config = config
        self.clock = clock
        self.sleep = sleep
        self.last_wait: SuctionWaitOutcome | None = None

    def read(self) -> int:
        code, value = self.arm.get_cgpio_digital(self.config.input_channel)
        if code != 0:
            raise RuntimeError(f"Failed to read CI{self.config.input_channel}: code={code}")
        return int(value)

    def is_suction_ok(self, debounce_seconds: float | None = None) -> bool:
        """True when the success level holds continuously for the debounce window."""
        window = self.config.debounce_seconds if debounce_seconds is None else float(debounce_seconds)
        streak_start: float | None = None
        while True:
            value = self.read()
            now = self.clock()
            if value != self.config.success_value:
                return False
            if streak_start is None:
                streak_start = now
            elif now - streak_start >= window:
                return True
            self.sleep(SAMPLE_INTERVAL_SECONDS)

    def wait_for_suction(
        self,
        timeout_seconds: float | None = None,
        debounce_seconds: float | None = None,
    ) -> bool:
        """Poll until a debounced success streak appears or the timeout expires."""
        timeout = self.config.timeout_seconds if timeout_seconds is None else float(timeout_seconds)
        window = self.config.debounce_seconds if debounce_seconds is None else float(debounce_seconds)
        if timeout < window:
            raise ValueError("timeout_seconds must be at least debounce_seconds")
        started = self.clock()
        streak_start: float | None = None
        last_value: int = self.config.success_value
        while True:
            elapsed = self.clock() - started
            if elapsed >= timeout:
                self.last_wait = SuctionWaitOutcome(False, elapsed, timeout, last_value)
                return False
            value = self.read()
            last_value = value
            now = self.clock()
            if value == self.config.success_value:
                if streak_start is None:
                    streak_start = now
                elif now - streak_start >= window:
                    self.last_wait = SuctionWaitOutcome(True, now - started, timeout, value)
                    return True
            else:
                streak_start = None
            elapsed = self.clock() - started
            remaining = min(SAMPLE_INTERVAL_SECONDS, max(0.0, timeout - elapsed))
            self.sleep(remaining)
