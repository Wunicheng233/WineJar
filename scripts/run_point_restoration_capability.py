#!/usr/bin/env python3
"""Run pretrained P2P-Bridge on saved WineJar observations, without devices."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from neck_localization.point_restoration_baseline import (
    P2PBridgeRestorer, file_sha256, load_observation_frames, observation_cloud,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "experiments/neck_localization/external_baselines/p2p_bridge/source")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "experiments/neck_localization/external_baselines/p2p_bridge/latest.pth")
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/neck_localization/external_baselines/p2p_bridge/opt.yaml")
    parser.add_argument("--native-build", type=Path, default=ROOT / "experiments/neck_localization/external_baselines/p2p_bridge/build")
    parser.add_argument("--study", type=Path, default=ROOT / "experiments/neck_localization/runs/structural_stage_20261025_v1")
    parser.add_argument("--layout", choices=["close", "far", "development"], default="close")
    parser.add_argument("--development", type=Path, default=ROOT / "experiments/neck_localization/runs/active_stereo_development_20261024_v3")
    parser.add_argument("--conditions", nargs="+", choices=["stereo_raw", "ideal_rgb"], default=["stereo_raw", "ideal_rgb"])
    parser.add_argument("--camera-set", choices=["all", "cam_a"], default="all")
    parser.add_argument("--cases", nargs="+", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--max-points", type=int, default=8192)
    parser.add_argument("--upper-window-mm", type=float, default=70)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.layout == "development":
        render = args.development.resolve()
    else:
        mapping = args.study / args.layout / "input_source.json"
        name = json.loads(mapping.read_text())["render_directory"] if mapping.exists() else "render"
        render = args.study / args.layout / name
    folders = sorted((render / "geometry").iterdir())
    cases = [p.name for p in folders if p.is_dir() and (args.cases is None or p.name in args.cases)]
    if args.cases is not None and set(args.cases) != set(cases):
        raise ValueError("unknown cases requested")
    source_hashes = {str(p.relative_to(ROOT)): file_sha256(p) for p in [Path(__file__), ROOT / "neck_localization/point_restoration_baseline.py"]}
    upstream_files = list((args.source / "models").glob("*.py"))
    upstream_files += list((args.source / "third_party/openpoints/models/layers").glob("*.py"))
    upstream_files += [p for p in (args.source / "third_party/openpoints/cpp/pointnet2_batch/src").glob("*") if p.is_file()]
    provenance = {
        "method": "P2P-Bridge_official_PVDS_PUNet_XYZ_zeroshot",
        "upstream_url": "https://github.com/matvogel/P2P-Bridge",
        "upstream_commit": subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip(),
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_bytes": args.checkpoint.stat().st_size,
        "checkpoint_sha256": file_sha256(args.checkpoint), "config_sha256": file_sha256(args.config),
        "upstream_inference_source_sha256": {str(p.relative_to(args.source)): file_sha256(p) for p in sorted(upstream_files)},
        "native_extension_sha256": {p.name: file_sha256(p) for p in args.native_build.glob("pointnet2_batch_cuda*.so")},
        "adapter_sha256": source_hashes, "layout": args.layout, "cases": cases,
        "conditions": args.conditions, "steps": args.steps, "seed": args.seed,
        "camera_set": args.camera_set,
        "render_root": str(render.resolve()),
        "split": "development" if args.layout == "development" else "previously_viewed_capability_audit",
        "input_information": "saved RGB foreground + saved depth + saved camera parameters; no mesh/target/diagnostic mask",
        "status": "existing_cases_development_capability_audit_not_unseen_test",
        "adaptations": ["unchanged upstream backbone/sampler/native CUDA ops", "batch size 1 for 6 GB GPU",
                        "object-style patch normalization with room-style point-corresponding averaging",
                        "uncovered points preserved instead of replaced by random unrelated positions"],
        "no_training": True, "no_device_connection": True, "prepare_only": args.prepare_only,
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    model = None
    rows = []
    for case in cases:
        for condition in args.conditions:
            started = time.perf_counter()
            row = {"case": case, "layout": args.layout, "condition": condition,
                   "camera_set": args.camera_set}
            try:
                folder = render / condition / "scenes" / case / "observations"
                if not folder.exists() and condition == "stereo_raw":
                    folder = render / "raw" / "scenes" / case / "observations"
                frames = load_observation_frames(folder)
                if args.camera_set != "all":
                    frames = [frame for frame in frames if frame.camera.name == args.camera_set]
                input_files = [folder / "camera_true_parameters.json"]
                input_files += [folder / (f.camera.name + suffix) for f in frames for suffix in ["_rgb.png", "_depth_z_m.npz"]]
                points, camera_ids, uv, crop = observation_cloud(frames, upper_window_m=args.upper_window_mm / 1000,
                                                               max_points=args.max_points, seed=args.seed)
                path = args.output / f"{case}__{condition}.npz"
                if args.prepare_only:
                    restored, counts, restoration = points.copy(), np.zeros(len(points), int), {"not_inferred": True}
                else:
                    if model is None:
                        model = P2PBridgeRestorer(args.source, args.checkpoint, args.config, args.native_build,
                                                 device=args.device, steps=args.steps)
                        provenance["runtime"] = model.runtime
                        (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
                    restored, counts, restoration = model.restore(points, seed=args.seed)
                np.savez_compressed(path, points_world_m=restored, input_points_world_m=points,
                                    source_camera_ids=camera_ids, source_pixel_uv=uv, prediction_update_count=counts)
                row.update(status="prepared_not_inferred" if args.prepare_only else "inference_completed",
                           pointcloud=str(path.resolve()), pointcloud_sha256=file_sha256(path),
                           points=len(points), crop=crop, restoration=restoration,
                           input_files_sha256={str(p.resolve()): file_sha256(p) for p in input_files},
                           observations_directory=str(folder.resolve()),
                           camera_names=[f.camera.name for f in frames])
            except Exception as error:
                row.update(status="interface_or_execution_error", error_type=type(error).__name__, error=str(error),
                           traceback=traceback.format_exc())
            row["runtime_s"] = time.perf_counter() - started
            rows.append(row)
            (args.output / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in ["case", "condition", "status", "runtime_s"]}), flush=True)
    provenance["completed"] = True
    provenance["errors"] = sum(r["status"] == "interface_or_execution_error" for r in rows)
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if provenance["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
