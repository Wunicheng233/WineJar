#!/usr/bin/env python3
"""Offline official L-NDF functional-frame transfer and separate truth scoring.

Only the four development target files are read during inference. Audit target
files can be opened only by the separate ``evaluate`` command, after saved
predictions exist. No robot, device, or mesh sampling is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from neck_localization.lndf_baseline import (COMPATIBILITY, ObservedSimilarity,
    collar_query, file_sha256, function_pose, load_public_model,
    observation_cloud, pose_list, public_optimizer)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def render_root(source, layout):
    mapping = source / layout / "input_source.json"
    name = json.loads(mapping.read_text())["render_directory"] if mapping.exists() else "render"
    return source / layout / name


def build_manifest(source, development, layouts, conditions, camera):
    """Read configs and ordinary observations, never audit geometry labels."""
    manifest = []
    for layout in layouts:
        folder = development if layout == "development" else render_root(source, layout)
        config_path = folder / "effective_config.json"
        config = json.loads(config_path.read_text())
        for condition in conditions:
            for case in config["cases"]:
                name = case["name"]
                observation = folder / condition / "scenes" / name / "observations"
                manifest.append(dict(pair_id=f"{layout}/{condition}/{name}",
                    layout=layout, condition=condition, case=name, camera=camera,
                    observations=str(observation.resolve()),
                    config_sha256=file_sha256(config_path),
                    split="supervised_development_control" if layout == "development"
                          else "previously_viewed_capability_audit"))
    return manifest


def demo_inputs(development, camera, condition, upper_window_m=None):
    """Explicit supervised demonstrations; only development labels permitted."""
    config_path = development / "effective_config.json"
    config = json.loads(config_path.read_text())
    if len(config["cases"]) != 4:
        raise ValueError("The locked supervised demonstration set must have exactly four jars")
    items = []
    for case in config["cases"]:
        name = case["name"]
        cloud, provenance = observation_cloud(
            development / condition / "scenes" / name / "observations", camera, upper_window_m)
        transform = ObservedSimilarity.from_points(cloud)
        # Development supervision only; evaluation cases are never read here.
        target_path = development / "geometry" / name / "evaluation_only/target.json"
        target = json.loads(target_path.read_text())
        frame = function_pose(target["center_m"], target["normal"])
        items.append(dict(case=name, points=transform.points_to_model(cloud),
            frame=transform.pose_to_model(frame), radius=target["radius_m"]*transform.scale,
            half_height=.0015*transform.scale,
            provenance=dict(case=name, input=provenance,
                target_path=str(target_path.resolve()),
                target_sha256=file_sha256(target_path),
                label_use="supervised development demonstration, not audit truth",
                observed_center_m=transform.center_m.tolist(), observed_scale=transform.scale)))
    return items


def infer(args):
    import torch
    args.output.mkdir(parents=True, exist_ok=False)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    manifest = build_manifest(args.source, args.development, args.layouts,
                              args.conditions, args.camera)
    if args.limit:
        manifest = manifest[:args.limit]
    case_seeds = {name: args.seed+i+1 for i, name in enumerate(dict.fromkeys(r["case"] for r in manifest))}
    write_json(args.output / "manifest.json", manifest)
    upper_window_m = args.upper_window_mm / 1000 if args.upper_window_mm else None
    demos = demo_inputs(args.development, args.camera, args.demo_condition, upper_window_m)
    query = collar_query(float(np.median([d["radius"] for d in demos])),
                         float(np.median([d["half_height"] for d in demos])),
                         n_points=args.query_points, seed=args.seed)
    model = load_public_model(args.upstream, args.weights, args.device)
    optimizer, Demo = public_optimizer(args.upstream, model, query,
        device=args.device, iterations=args.iterations,
        initializations=args.initializations, workdir=args.output / "optimizer", cos_loss=args.cos_loss)
    for item in demos:
        optimizer.add_demo(Demo(item["points"], query, pose_list(np.eye(4)),
                                pose_list(item["frame"]), item["case"]))
    start = time.perf_counter()
    optimizer.process_demos()
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    demo_ms = (time.perf_counter()-start)*1000
    np.savez_compressed(args.output / "demonstrations.npz", query_points_model_m=query,
        target_descriptor=optimizer.target_act_hat.cpu().numpy())
    provenance = dict(method="official pretrained L-NDF functional-frame transfer",
        upstream_url="https://github.com/elchun/lndf_robot",
        upstream_commit=subprocess.check_output(["git", "-C", str(args.upstream),
            "rev-parse", "HEAD"], text=True).strip(),
        weights=str(args.weights.resolve()), weights_sha256=file_sha256(args.weights),
        upstream_network_sha256={str(p.relative_to(args.upstream)): file_sha256(p)
            for p in sorted((args.upstream / "src/ndf_robot/model/conv_occupancy_net").rglob("*.py"))},
        upstream_optimizer_sha256=file_sha256(args.upstream / "src/ndf_robot/opt/optimizer_lite.py"),
        adapter_sha256=file_sha256(ROOT / "neck_localization/lndf_baseline.py"),
        script_sha256=file_sha256(Path(__file__)), compatibility=COMPATIBILITY,
        demonstrations=[d["provenance"] for d in demos], demo_condition=args.demo_condition,
        demonstration_descriptor_ms=demo_ms, seed=args.seed,
        case_seeds=case_seeds,
        paired_random_initialization_rule="same case seed across raw/ideal and far/close",
        device=args.device, torch_version=torch.__version__, iterations=args.iterations,
        initializations=args.initializations, query_points=len(query),
        cos_loss=args.cos_loss, upper_window_mm=args.upper_window_mm,
        query_radius_model_m=float(np.median([d["radius"] for d in demos])),
        query_half_height_model_m=float(np.median([d["half_height"] for d in demos])),
        predicts_radius=False, inference_reads_audit_truth=False,
        no_hardware=True, normalized_using_observed_input_extent=True,
        conditions=list(args.conditions), layouts=list(args.layouts),
        source=str(args.source.resolve()), development=str(args.development.resolve()))
    write_json(args.output / "provenance.json", provenance)
    with (args.output / "predictions.jsonl").open("w") as handle:
        for index, item in enumerate(manifest):
            prediction = dict(**item, seed=case_seeds[item["case"]])
            start = time.perf_counter()
            try:
                # One reproducible official random restart sequence per case.
                np.random.seed(prediction["seed"])
                torch.manual_seed(prediction["seed"])
                cloud, cloud_provenance = observation_cloud(item["observations"], args.camera, upper_window_m)
                transform = ObservedSimilarity.from_points(cloud)
                if args.device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats()
                transforms, best_idx = optimizer.optimize_transform_implicit(
                    transform.points_to_model(cloud),
                    viz_path=str(args.output / "optimizer"), ee=True)
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                frame = transform.pose_to_world(transforms[best_idx])
                prediction.update(status="ok", center_m=frame[:3, 3].tolist(),
                    normal=frame[:3, 2].tolist(), frame_world=frame.tolist(),
                    best_index=int(best_idx), predicted_radius_m=None,
                    observed_center_m=transform.center_m.tolist(),
                    observed_scale=transform.scale, input=cloud_provenance,
                    peak_cuda_mib=(torch.cuda.max_memory_allocated()/2**20
                        if args.device.startswith("cuda") else None))
                save_path = args.output / "predictions" / (item["pair_id"] + ".npz")
                save_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(save_path, frame_world=frame,
                    all_frames_world=np.array([transform.pose_to_world(t) for t in transforms]),
                    input_points_world_m=cloud)
                prediction["prediction_sha256"] = file_sha256(save_path)
            except Exception as exc:
                prediction.update(status="error", error=repr(exc), traceback=traceback.format_exc())
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            prediction["elapsed_ms"] = (time.perf_counter()-start)*1000
            handle.write(json.dumps(prediction, allow_nan=False) + "\n")
            handle.flush()
            print(f"{index+1}/{len(manifest)} {item['pair_id']} {prediction['status']} "
                  f"{prediction['elapsed_ms']/1000:.2f}s", flush=True)
    write_json(args.output / "completed.json", dict(n=len(manifest),
        predictions_sha256=file_sha256(args.output / "predictions.jsonl"),
        inference_reads_audit_truth=False))


def evaluate(args):
    """Separate post-inference evaluation; no scoring changes predictions."""
    if not (args.predictions / "completed.json").exists():
        raise ValueError("Inference must finish and save all predictions before evaluation")
    completed = json.loads((args.predictions / "completed.json").read_text())
    if file_sha256(args.predictions / "predictions.jsonl") != completed["predictions_sha256"]:
        raise ValueError("Saved inference record changed after completion")
    provenance = json.loads((args.predictions / "provenance.json").read_text())
    source = Path(provenance["source"])
    rows = [json.loads(line) for line in (args.predictions / "predictions.jsonl").read_text().splitlines()]
    if len(rows) != completed["n"]:
        raise ValueError("Incomplete inference records")
    scored = []
    for row in rows:
        item = dict(row)
        render = Path(provenance["development"]) if row["layout"] == "development" else render_root(source, row["layout"])
        path = render / "geometry" / row["case"] / "evaluation_only/target.json"
        truth = json.loads(path.read_text())
        item["evaluation_truth_sha256"] = file_sha256(path)
        if row["status"] == "ok":
            offset = np.array(row["center_m"])-np.array(truth["center_m"])
            n = np.array(row["normal"]); n /= np.linalg.norm(n)
            true_n = np.array(truth["normal"])
            cos = np.clip(np.dot(n, true_n), -1, 1)
            item.update(center_mm=float(np.linalg.norm(offset)*1000),
                axis_deg=float(np.degrees(np.arccos(abs(cos)))),
                directed_normal_deg=float(np.degrees(np.arccos(cos))),
                axial_offset_mm=float(np.dot(offset, true_n)*1000),
                diameter_mm=None)
            item["center_axis_pass"] = bool(item["center_mm"] <= 4 and item["axis_deg"] <= 2)
            item["center_pass"] = bool(item["center_mm"] <= 4)
            item["axis_pass"] = bool(item["axis_deg"] <= 2)
            item["full_metrology_pass"] = None  # L-NDF does not estimate the neck diameter.
        else:
            item["center_axis_pass"] = False
        scored.append(item)
    summary = []
    for layout, condition in sorted({(r["layout"], r["condition"]) for r in rows}):
        selected = [r for r in scored if (r["layout"], r["condition"]) == (layout, condition)]
        item = dict(layout=layout, condition=condition, total=len(selected),
            solved=sum(r["status"] == "ok" for r in selected),
            errors=sum(r["status"] != "ok" for r in selected),
            center_axis_pass=sum(r["center_axis_pass"] for r in selected),
            center_pass=sum(r.get("center_pass", False) for r in selected),
            axis_pass=sum(r.get("axis_pass", False) for r in selected),
            wrong_partial_pose_outputs=sum(r["status"] == "ok" and not r["center_axis_pass"] for r in selected),
            full_metrology_pass=None, radius_not_predicted=True)
        for key in ["center_mm", "axis_deg", "directed_normal_deg", "elapsed_ms"]:
            values = [r[key] for r in selected if key in r]
            item[key+"_median"] = float(np.median(values)) if values else None
            item[key+"_max"] = float(max(values)) if values else None
        summary.append(item)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "records.json", scored)
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "scope.json", dict(no_hardware=True,
        independent_audit_jars=len({r["case"] for r in rows if r["layout"] != "development"}),
        development_control_jars=len({r["case"] for r in rows if r["layout"] == "development"}),
        prediction_records=len(rows), layouts=sorted({r["layout"] for r in rows}),
        layouts_are_paired=True, previously_viewed=True,
        supervised_demonstrations=4, predicted_radius=False,
        threshold_center_mm=4, threshold_axis_deg=2,
        success_is_partial_pose_only=True, no_new_method_trained=True,
        evaluation_script_sha256=file_sha256(Path(__file__)),
        predictions_sha256=completed["predictions_sha256"]))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("infer")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--development", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--upstream", type=Path, default=ROOT / "experiments/neck_localization/external_baselines/lndf/upstream")
    p.add_argument("--weights", type=Path, default=ROOT / "experiments/neck_localization/external_baselines/lndf/weights/lndf_weights.pth")
    p.add_argument("--layouts", nargs="+", default=["far", "close"])
    p.add_argument("--conditions", nargs="+", default=["stereo_raw", "ideal_rgb"])
    p.add_argument("--demo-condition", default="ideal_rgb", choices=["ideal_rgb", "stereo_raw"])
    p.add_argument("--camera", default="cam_a")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--initializations", type=int, default=20)
    p.add_argument("--query-points", type=int, default=1000)
    p.add_argument("--cos-loss", action="store_true", help="Official bottle evaluation configuration cosine objective")
    p.add_argument("--upper-window-mm", type=float, default=0, help="0 full foreground; 70 matches observed upper ROI used by restoration")
    p.add_argument("--seed", type=int, default=2026102707)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("evaluate")
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    infer(args) if args.command == "infer" else evaluate(args)
