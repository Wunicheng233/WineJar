#!/usr/bin/env python3
"""Offline numerical-equivalence and timing replay for multiview rim fitting."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import cv2
import numpy as np
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from xarm_sdk_control import rim_geometry as current
from xarm_sdk_control.multiview_rig import (
    FixedCameraModel, FixedCameraRig, load_fixed_camera_rig, rig_geometry_fingerprint,
)


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def synthetic_case(tilt, extents):
    poses = {}
    for cid, origin in (("overhead", [0, 0, 1.2]), ("side_a", [1.25, 0, .62]), ("side_b", [0, 1., .26])):
        origin = np.asarray(origin, float)
        z = np.array([0, 0, .48]) - origin
        z /= np.linalg.norm(z)
        x = np.cross(z, [0, 0, 1.]) if cid != "overhead" else np.array([1., 0, 0])
        x /= np.linalg.norm(x)
        poses[cid] = np.eye(4)
        poses[cid][:3, :3] = np.column_stack((x, np.cross(z, x), z))
        poses[cid][:3, 3] = origin
    transform = np.linalg.inv(poses["overhead"])
    cameras = {
        cid: FixedCameraModel(cid, cid, np.array([[610., 0, 320], [0, 610., 240], [0, 0, 1]]),
                             np.array([.02, -.03, .001, -.001, .005]), (640, 480),
                             transform @ pose, Path("synthetic.json"))
        for cid, pose in poses.items()
    }
    rig = FixedCameraRig("overhead", cameras, ())
    center = np.array([.012, -.018, .487])
    normal = np.array([tilt, -.07, 1.])
    normal /= np.linalg.norm(normal)
    rng = np.random.default_rng(74)
    observations = []
    for cid in cameras:
        inner = cid == "overhead"
        points = current.circle_points(center, normal, .05 if inner else .062, 200)
        pixels = current.project_workcell(rig, cid, transform, points)
        span = tuple(np.percentile(pixels[:, 0], [0, 100])) if cid in extents else None
        if not inner:
            pixels = pixels[pixels[:, 1] <= np.median(pixels[:, 1])]
        pixels += rng.normal(0, .15, pixels.shape)
        pixels[::24] += rng.normal(0, 5., pixels[::24].shape)
        observations.append(current.RimObservation(cid, pixels, inner, span))
    return rig, transform, observations, center + [.005, -.004, -.012], .052


def historical_case(path):
    payload = json.loads(path.read_text())
    rig = load_fixed_camera_rig(Path(payload["rig_path"]))
    ground_path = Path(payload["ground_path"])
    assert hashlib.sha256(ground_path.read_bytes()).hexdigest() == payload["ground_sha256"]
    ground = json.loads(ground_path.read_text())
    assert rig_geometry_fingerprint(rig) == payload["rig_fingerprint"] == ground["rig_geometry_fingerprint"]
    transform = np.linalg.inv(np.asarray(ground["transform_workcell_from_reference"]))
    observations = [current.RimObservation(cid, np.asarray(pixels), cid == "overhead")
                    for cid, pixels in payload["observations"].items()]
    # Reuse saved image observations, with an explicit common initial guess.
    # This is not a replay of depth initialization or an absolute-accuracy test.
    center = np.asarray(payload["rim_center_workcell_mm"]) / 1000 + [.005, -.004, -.012]
    return rig, transform, observations, center, payload["inner_radius_mm"] / 1000 * 1.04


def traced_fit(module, inputs):
    solver = module.least_squares
    trace = []

    def measured(fun, *args, **kwargs):
        calls = 0

        def residual(x):
            nonlocal calls
            calls += 1
            return fun(x)

        result = solver(residual, *args, **kwargs)
        trace.append(dict(residual_calls=calls, nfev=result.nfev, njev=result.njev,
                          status=result.status, cost=float(result.cost), x=result.x.tolist()))
        return result

    module.least_squares = measured
    try:
        return module.fit_multiview_rim(*inputs), trace
    finally:
        module.least_squares = solver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Trusted local rim_geometry.py snapshot to import.")
    parser.add_argument("--diagnostic-result", type=Path, help="Optional saved rgb_diagnostic_01/result.json.")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.output.exists():
        parser.error("Use positive repeats and a new output path")
    spec = importlib.util.spec_from_file_location("rim_baseline", args.baseline.resolve())
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    cv2.setNumThreads(1)
    groups = [
        ("low_tilt_two_extents", [synthetic_case(0., ("side_a", "side_b"))]),
        ("tilted_two_extents", [synthetic_case(.18, ("side_a", "side_b"))]),
        ("tilted_one_extent", [synthetic_case(.18, ("side_a",))]),
        ("tilted_no_extents", [synthetic_case(.18, ())]),
    ]
    seed_case = groups[1][1][0]
    seed_fit = baseline.fit_multiview_rim(*seed_case)
    rng = np.random.default_rng(19)
    bootstraps = []
    for _ in range(20):
        observations = [current.RimObservation(o.camera_id, o.pixels[rng.integers(0, len(o.pixels), len(o.pixels))],
                                               o.inner, o.horizontal_extent_px) for o in seed_case[2]]
        bootstraps.append((*seed_case[:2], observations, seed_fit.center_m, seed_fit.inner_radius_m,
                           seed_fit.outer_radius_m - seed_fit.inner_radius_m))
    groups.append(("bootstrap_20", bootstraps))
    if args.diagnostic_result:
        groups.append(("historical_observations_diagnostic_only", [historical_case(args.diagnostic_result)]))
    report = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                  opencv=cv2.__version__, opencv_threads=cv2.getNumThreads(), repeats=args.repeats,
                  baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
                  current_sha256=hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest(), groups=[])
    for name, cases in groups:
        fits, traces, fingerprints = [], [], []
        for inputs in cases:
            before, before_trace = traced_fit(baseline, inputs)
            after, after_trace = traced_fit(current, inputs)
            assert jsonable(asdict(before)) == jsonable(asdict(after)), f"Fit differs: {name}"
            assert before_trace == after_trace, f"Solver trajectory differs: {name}"
            fits.append(jsonable(asdict(after)))
            traces.append(after_trace)
            fingerprints.append(dict(
                rig=rig_geometry_fingerprint(inputs[0]), transform=jsonable(inputs[1]),
                observations_sha256=hashlib.sha256(json.dumps(jsonable([asdict(o) for o in inputs[2]])).encode()).hexdigest(),
                initialization=jsonable(inputs[3:]),
            ))
        times = {"before": [], "after": []}
        for repeat in range(args.repeats):
            order = [("before", baseline), ("after", current)]
            for label, module in (order if repeat % 2 == 0 else order[::-1]):
                start = time.perf_counter()
                for inputs in cases:
                    module.fit_multiview_rim(*inputs)
                times[label].append(time.perf_counter() - start)
        medians = {key: statistics.median(value) for key, value in times.items()}
        report["groups"].append(dict(name=name, fit_count=len(cases), exact_match=True,
                                     seconds=times, median_seconds=medians,
                                     reduction_percent=100 * (1 - medians["after"] / medians["before"]),
                                     solver_traces=traces, inputs=fingerprints, fits=fits))
        print(f"{name}: {medians['before']:.6f} -> {medians['after']:.6f}s, exact results and solver traces", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
