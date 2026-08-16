"""Monotonic task timeline for robot-cycle performance measurements."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class TaskTimeline:
    clock: Callable[[], float] = time.perf_counter
    started_at: float = field(init=False)
    last_event_at: float = field(init=False)
    events: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.started_at = self.clock()
        self.last_event_at = self.started_at

    def mark(self, name: str, **details: Any) -> dict[str, Any]:
        now = self.clock()
        event = {
            "name": name,
            "elapsed_seconds": now - self.started_at,
            "delta_seconds": now - self.last_event_at,
            **details,
        }
        self.events.append(event)
        self.last_event_at = now
        print(
            f"TIMING {name}: elapsed={event['elapsed_seconds']:.3f}s "
            f"delta={event['delta_seconds']:.3f}s"
        )
        return event

    def elapsed_at(self, name: str) -> float:
        for event in reversed(self.events):
            if event["name"] == name:
                return float(event["elapsed_seconds"])
        raise KeyError(f"Timeline event not found: {name}")

    def to_dict(self, productive_start: str | None = None) -> dict[str, Any]:
        total = self.events[-1]["elapsed_seconds"] if self.events else 0.0
        result: dict[str, Any] = {
            "total_seconds": float(total),
            "events": list(self.events),
        }
        if productive_start is not None and self.events:
            result["productive_cycle_seconds"] = float(total) - self.elapsed_at(productive_start)
            result["productive_cycle_start_event"] = productive_start
        return result
