"""Offline accounting of recorded loading intervals, without hardware imports.

An interval is assigned to the event ending it. These are host-side elapsed
intervals, not pure device execution times. Nested diagnostics are never added
to these exclusive totals.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any


PHASE_LABELS = {
    "setup": "启动与复位",
    "planning": "规划区间",
    "capture": "采图区间",
    "vision": "视觉处理区间",
    "motion": "运动指令与等待区间",
    "vacuum": "真空开关与稳定等待区间",
    "suction": "显式吸附确认区间",
    "other": "其他与未细分区间",
}


def _seconds(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite nonnegative number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{field} must be a finite nonnegative number")
    return value


def _phase(name: str) -> str:
    suffix = name.rsplit(".", 1)[-1]
    if suffix == "route_planned":
        return "planning"
    if suffix in {"capture_complete", "capture_failed"}:
        return "capture"
    if suffix == "vision_complete":
        return "vision"
    if re.fullmatch(r"move_\d+", suffix) or name.endswith(".recovery.home_reached"):
        return "motion"
    if suffix.startswith("vacuum_"):
        return "vacuum"
    if suffix.startswith("suction_"):
        return "suction"
    return "other"


def summarize_loading_result(result: dict[str, Any], source: str = "") -> dict[str, Any]:
    timing = result["timing"]
    events = timing["events"]
    if not events:
        raise ValueError("No timeline events")
    start_name = timing.get("productive_cycle_start_event", "productive_cycle_started")
    starts = [i for i, event in enumerate(events) if event["name"] == start_name]
    if len(starts) > 1:
        raise ValueError("Ambiguous duplicate productive-cycle start events")
    start_index = starts[0] if starts else None
    totals = {phase: 0.0 for phase in PHASE_LABELS}
    elapsed = 0.0
    productive_start = None
    mismatches = []
    intervals = []
    for index, event in enumerate(events):
        delta = _seconds(event.get("delta_seconds"), f"event {index} delta_seconds")
        elapsed += delta
        recorded = _seconds(event.get("elapsed_seconds"), f"event {index} elapsed_seconds")
        if not math.isclose(elapsed, recorded, rel_tol=1e-9, abs_tol=1e-6):
            mismatches.append({"index": index, "name": event["name"],
                               "recorded_seconds": recorded, "reconstructed_seconds": elapsed})
        phase = "setup" if start_index is not None and index <= start_index else _phase(event["name"])
        totals[phase] += delta
        interval = {"name": event["name"], "phase": phase, "seconds": delta,
                    "end_seconds": elapsed}
        for field in ("planning_seconds", "planning_diagnostics", "inference_timing_seconds", "capture_diagnostics",
                      "suction_wait_seconds", "waited_for_completion"):
            if field in event:
                interval[field] = event[field]
        intervals.append(interval)
        if index == start_index:
            productive_start = elapsed
    # Do not silently repair arbitrary corrupt logs: delta reconstruction must
    # reconcile with the independently recorded run total.
    recorded_total = _seconds(timing["total_seconds"], "total_seconds")
    if not math.isclose(elapsed, recorded_total, rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError("Sum of event deltas does not match recorded total_seconds")
    productive = None if productive_start is None else elapsed - productive_start
    warnings = []
    if mismatches:
        warnings.append(f"{len(mismatches)} 个事件累计时间不一致；报告使用 delta_seconds 累加，原日志未修改。")
    if productive is None:
        warnings.append("缺少生产周期开始事件，无法分离启动耗时或计算生产周期。")
    elif "productive_cycle_seconds" in timing and not math.isclose(
        productive, _seconds(timing["productive_cycle_seconds"], "productive_cycle_seconds"),
        rel_tol=1e-9, abs_tol=1e-6,
    ):
        warnings.append("原生产周期时长与事件区间累加不一致，报告使用重建值。")
    planning = [item for item in intervals if item["phase"] == "planning"]
    if any(item.get("planning_diagnostics") is None for item in planning):
        warnings.append("部分规划事件缺少 IK/重试诊断，不能据此判定规划长等待的原因。")
    if any(item.get("capture_diagnostics") is None for item in intervals if item["phase"] == "capture"):
        warnings.append("部分采图事件缺少取帧/恢复/保存诊断，不能据此判定采图长等待的原因。")
    cycles = result.get("cycles", result.get("completed_cycles", []))
    return {
        "source": source, "status": result.get("status", "unknown"),
        "run_mode": result.get("run_mode"), "total_seconds": elapsed,
        "productive_cycle_seconds": productive, "phase_seconds": totals,
        "timestamp_mismatches": mismatches, "warnings": warnings,
        "completed_cycles": sum(c.get("completed") is True for c in cycles),
        "pick_attempts_by_cycle": [c.get("pick_attempts_used") for c in cycles],
        "performance_parameters": result.get("performance_parameters"),
        "intervals": intervals,
    }


def render_loading_report(reports: list[dict[str, Any]]) -> str:
    lines = ["# 上料节拍离线报告", "",
             "按事件结束边界划分互不重叠的耗时区间；细分诊断包含在对应区间内，不重复相加。",
             "运动区间可能包含排队、通信和传感器检查，不能作为单段实际运动时长。",
             "失败、未知状态和预演记录逐条保留；不同配置与现场条件应分别对照。", ""]
    counts: dict[str, int] = defaultdict(int)
    for report in reports:
        counts[report["status"]] += 1
    lines.append("输入记录：" + "，".join(f"{status} {count} 条" for status, count in sorted(counts.items())) + "。")
    for report in reports:
        lines += ["", f"## {report['source']}", "",
                  f"状态：{report['status']}；run_mode：{report['run_mode']}；已完成物料周期：{report['completed_cycles']}。",
                  f"总耗时：{report['total_seconds']:.3f} s。"]
        productive = report["productive_cycle_seconds"]
        lines += [f"生产周期：{productive:.3f} s。" if productive is not None else "生产周期：无法确定。",
                  f"各物料尝试次数：{report['pick_attempts_by_cycle']}（缺失值不当作首次成功）。", "",
                  "| 区间 | 耗时 / s | 占总耗时 |", "|---|---:|---:|"]
        for phase, seconds in report["phase_seconds"].items():
            share = seconds / report["total_seconds"] if report["total_seconds"] else 0.0
            lines.append(f"| {PHASE_LABELS[phase]} | {seconds:.3f} | {share:.1%} |")
        lines += ["", "规划、采图与视觉事件（子项不再计入总和）：", "",
                  "| 事件 | 区间 / s | 内部规划计时 / s |", "|---|---:|---:|"]
        for item in report["intervals"]:
            if item["phase"] in {"planning", "capture", "vision"}:
                inner = item.get("planning_seconds")
                formatted = f"{inner:.3f}" if isinstance(inner, (int, float)) else "—"
                lines.append(f"| {item['name']} | {item['seconds']:.3f} | {formatted} |")
        for item in report["intervals"]:
            capture = item.get("capture_diagnostics")
            if capture:
                lines += ["", f"采图细分：{item['name']}（{capture.get('status', 'unknown')}），"
                          f"读取尝试 {capture.get('read_attempts', '未知')} 次，"
                          f"恢复流程 {capture.get('restart_count', '未知')} 次。", "",
                          "| 内部阶段 | 耗时 / s |", "|---|---:|"]
                for stage, seconds in capture.get("timing_seconds", {}).items():
                    lines.append(f"| {stage} | {seconds:.6f} |")
                if capture.get("first_read_error"):
                    lines += ["", f"首次读取错误：{capture['first_read_error']}"]
                if capture.get("error"):
                    lines += ["", f"最终错误：{capture['error']}"]
        if report["warnings"]:
            lines += ["", *[f"- {warning}" for warning in report["warnings"]]]
    lines += ["", "该报告是测量与诊断结果，不代表已经缩短了实机节拍。", ""]
    return "\n".join(lines)
