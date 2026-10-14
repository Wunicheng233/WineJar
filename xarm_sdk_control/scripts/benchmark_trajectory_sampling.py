#!/usr/bin/env python3
"""Compare local planner CPU work and exact IK requests using a synthetic solver.

No controller is connected. Measured time does not include controller IK or
network latency and must not be reported as robot planning time saved.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from xarm_sdk_control import trajectory_speed_planner as current


class SyntheticIK:
    def __init__(self, failure=None, trace=False):
        self.failure = failure
        self.trace = trace
        self.calls = []
        self.count = 0

    def get_inverse_kinematics(self, pose, **kwargs):
        self.count += 1
        if self.trace:
            self.calls.append((pose, kwargs))
        if self.count == 3:
            if self.failure == "api":
                return -1, []
            if self.failure == "invalid":
                return 0, [float("nan")] * 6
            if self.failure == "jump":
                return 0, [90.0] * 6
        return 0, (np.asarray(pose) * .01).tolist()


def cases_from_result(result):
    params = result["performance_parameters"]
    speed_keys = {"empty_transit": "transit_speed_mm_s", "loaded_transit": "loaded_transit_speed_mm_s",
                  "vertical": "vertical_speed_mm_s", "loaded_vertical": "loaded_vertical_speed_mm_s",
                  "precision": "precision_speed_mm_s", "loaded_precision": "precision_speed_mm_s",
                  "contact": "contact_speed_mm_s"}
    # First-cycle start is a replay assumption, not a recovered joint/pose log.
    start = result["cycles"][-1]["route"][-1][1]
    for cycle in result["cycles"]:
        requests = []
        for label, target, motion, _ in cycle["route"]:
            acceleration_key = ("precision_acceleration_mm_s2" if motion in {"contact", "precision", "loaded_precision"}
                                else "loaded_acceleration_mm_s2" if motion.startswith("loaded") else "acceleration_mm_s2")
            requests.append([label, target, params[speed_keys[motion]], params[acceleration_key]])
        yield {"name": cycle["material"], "start": start, "requests": requests}
        start = requests[-1][1]
    yield {"name": "angle_wrap", "start": [0, 0, 0, 170, -170, 0],
           "requests": [["wrap", [100, 0, 0, -170, 170, 180], 50, 100]]}
    for failure in ("api", "invalid", "jump"):
        yield {"name": failure, "failure": failure, "start": [0.] * 6,
               "requests": [[failure, [100, 0, 0, 0, 0, 0], 50, 100]]}
    yield {"name": "pure_rotation", "start": [0.] * 6,
           "requests": [["rotation", [0, 0, 0, 0, 0, 30], 50, 100]]}
    yield {"name": "minimum_speed", "start": [0.] * 6,
           "requests": [["too slow", [100, 0, 0, 0, 0, 0], .01, 100]]}


def run(module, case, trace=False, endpoint=False):
    arm = SyntheticIK(case.get("failure"), trace)
    requests = [module.CartesianSegmentRequest(*request) for request in case["requests"]]
    config = module.JointConstraintConfig(velocity_utilization=.35, acceleration_utilization=.35)
    planner = module.plan_cartesian_route_speeds_endpoint if endpoint else module.plan_cartesian_route_speeds
    started = time.perf_counter()
    try:
        plans = planner(arm, case["start"], (np.asarray(case["start"]) * .01).tolist(),
                        requests, 180., 1000., config)
        outcome = {"plans": [plan.to_dict() for plan in plans]}
    except module.TrajectoryPlanningError as exc:
        outcome = {"error": str(exc)}
    return time.perf_counter() - started, outcome, arm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Trusted local Python snapshot")
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=51)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 3:
        parser.error("Output must not exist and repeats must be at least 3")
    spec = importlib.util.spec_from_file_location("trajectory_baseline", args.baseline.resolve())
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    modules = {"baseline": baseline, "optimized": current}
    report = {
        "python": platform.python_version(), "numpy": np.__version__, "repeats": args.repeats,
        "method": "synthetic linear IK, one warmup, alternating paired order; excludes hardware and network",
        "replay_assumptions": "First start = last cycle home target; subsequent starts = preceding endpoint; synthetic initial joints = pose * 0.01; joint limits 180 deg/s and 1000 deg/s²; utilization 0.35",
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "current_sha256": hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(args.result.read_bytes()).hexdigest(),
        "cases": [],
    }
    for case in cases_from_result(json.loads(args.result.read_text())):
        _, expected, old_arm = run(baseline, case, trace=True)
        _, actual, new_arm = run(current, case, trace=True)
        assert expected == actual and old_arm.calls == new_arm.calls, case["name"]
        if case["name"] == "pure_rotation":
            # Old fallback divides by zero here. The new explicit rejection is
            # intentional and not an equivalence claim for this invalid case.
            _, endpoint_actual, new_endpoint = run(current, case, trace=True, endpoint=True)
            assert "Pure orientation" in endpoint_actual["error"] and new_endpoint.count == 0
            endpoint_check = "unsupported_zero_translation_rejected_before_ik"
        else:
            _, endpoint_expected, old_endpoint = run(baseline, case, trace=True, endpoint=True)
            _, endpoint_actual, new_endpoint = run(current, case, trace=True, endpoint=True)
            assert endpoint_expected == endpoint_actual and old_endpoint.calls == new_endpoint.calls
            endpoint_check = "exact_ik_requests_and_plans"
        measurements = {mode: [] for mode in modules}
        for repeat in range(-1, args.repeats):
            order = list(modules) if repeat % 2 else list(reversed(modules))
            for mode in order:
                seconds, outcome, arm = run(modules[mode], case)
                assert outcome == expected and arm.count == old_arm.count
                if repeat >= 0:
                    measurements[mode].append(seconds)
        medians = {mode: statistics.median(values) for mode, values in measurements.items()}
        report["cases"].append({**case, "outcome": actual, "ik_calls": old_arm.count,
                                "exact_ik_requests_and_plans": True, "endpoint_check": endpoint_check,
                                "seconds": measurements, "median_seconds": medians})
        print(case["name"], "IK calls", old_arm.count,
              {mode: round(value * 1000, 3) for mode, value in medians.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
