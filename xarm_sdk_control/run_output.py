"""Run-directory creation and JSON report serialization for task scripts."""

from __future__ import annotations

import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from xarm_sdk_control.task_timing import TaskTimeline


def new_timestamped_run_dir(root: Path) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    run_dir = root / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json_report(run_dir: Path, result: dict[str, Any], name: str = "result.json") -> Path:
    path = run_dir / name
    path.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return path


def write_failure_report(
    run_dir: Path,
    error: BaseException,
    *,
    holding: bool = False,
    timeline: TaskTimeline | None = None,
    productive_start: str | None = None,
    completed_records: list[dict[str, Any]] | None = None,
    robot_snapshot: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    report: dict[str, Any] = {
        "status": "failed",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
        "holding_leaf": holding,
        "robot_after_stop": robot_snapshot,
        "timing": timeline.to_dict(productive_start=productive_start) if timeline is not None else None,
        "completed_cycles": completed_records if completed_records is not None else [],
    }
    if extra:
        report.update(extra)
    return write_json_report(run_dir, report, name="failure.json")
