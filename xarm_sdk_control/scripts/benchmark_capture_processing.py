#!/usr/bin/env python3
"""Replay RGB-D preview and saving against a trusted local source snapshot; no hardware."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import platform
import statistics
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from xarm_sdk_control import realsense_capture as current


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=31)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 3:
        parser.error("Output must not exist and repeats must be at least 3")
    spec = importlib.util.spec_from_file_location("capture_baseline", args.baseline.resolve())
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    report = {
        "python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__,
        "opencv_threads": cv2.getNumThreads(), "repeats": args.repeats,
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "current_sha256": hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest(),
        "method": "1 warmup per mode; rotated paired order; filesystem writes use OS cache, no fsync",
        "cases": [],
    }
    modules = {"baseline": baseline, "optimized": current}
    with tempfile.TemporaryDirectory(prefix="winejar-capture-benchmark-") as directory:
        root = Path(directory)
        for index, item in enumerate(json.loads(args.manifest.read_text())):
            folder = Path(item["frame_dir"])
            metadata = json.loads((folder / "frame.json").read_text())
            color = cv2.imread(str(folder / metadata["files"]["color_bgr"]), -1)
            depth = cv2.imread(str(folder / metadata["files"]["depth_raw"]), -1)
            if color is None or depth is None:
                raise ValueError(f"Missing RGB-D files: {folder}")
            scale = metadata["stream"]["depth_scale_m_per_unit"]
            original_depth = depth.copy()
            sessions = {}
            for mode, module in modules.items():
                session = module.RealSenseCaptureSession(width=color.shape[1], height=color.shape[0])
                session.info = metadata["camera"]
                sensor = SimpleNamespace(get_depth_scale=lambda: scale)
                device = SimpleNamespace(first_depth_sensor=lambda: sensor)
                session.profile = SimpleNamespace(get_device=lambda: device)
                sessions[mode] = session
            measurements = {mode: {"preview_seconds": [], "save_seconds": []} for mode in modules}
            saved = {}
            previews = {}
            for repeat in range(-1, args.repeats):
                order = list(modules) if repeat % 2 else list(reversed(modules))
                for mode in order:
                    started = time.perf_counter()
                    previews[mode] = modules[mode].make_depth_preview(depth, scale)
                    preview_seconds = time.perf_counter() - started
                    timestamp = datetime(2026, 9, 19).astimezone() + timedelta(seconds=repeat + 1)
                    options = {"timing": {}} if mode == "optimized" else {}
                    with contextlib.redirect_stdout(io.StringIO()):
                        started = time.perf_counter()
                        saved[mode] = sessions[mode]._save_frame(
                            root / str(index) / mode, color, depth, timestamp, metadata["intrinsics"], **options
                        )
                        save_seconds = time.perf_counter() - started
                    if repeat >= 0:
                        measurements[mode]["preview_seconds"].append(preview_seconds)
                        measurements[mode]["save_seconds"].append(save_seconds)
                assert np.array_equal(previews["baseline"], previews["optimized"])
                assert np.array_equal(original_depth, depth)
                for name in ("color.png", "depth_raw.png", "depth_preview.png"):
                    assert (saved["baseline"] / name).read_bytes() == (saved["optimized"] / name).read_bytes()
                assert (saved["baseline"] / "frame.json").read_bytes() == (saved["optimized"] / "frame.json").read_bytes()
            for values in measurements.values():
                values["preview_median_seconds"] = statistics.median(values["preview_seconds"])
                values["save_median_seconds"] = statistics.median(values["save_seconds"])
            result = {
                "frame_dir": str(folder), "measurements": measurements,
                "preview_pixels_equal": True, "saved_files_equal": True, "input_unchanged": True,
                "input_sha256": {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                                 for name in ("color.png", "depth_raw.png", "frame.json")},
            }
            report["cases"].append(result)
            print(folder.parent.parent.name, {
                mode: {key: round(value * 1000, 3) for key, value in values.items() if "median" in key}
                for mode, values in measurements.items()
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
