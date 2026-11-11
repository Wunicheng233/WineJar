#!/usr/bin/env python3
"""Bounded CPU-only native-IR diagnostic on the FOUR opened development jars."""
import argparse
from dataclasses import asdict
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from neck_localization.active_stereo import quantize_depth
from neck_localization.band_geometry import evaluate_band, plan_band
from neck_localization.evaluation import score
from neck_localization.geometry import Camera
from neck_localization.native_stereo import estimate_native_stereo, map_rgb_onto_native
from neck_localization.stereo_robust import BlueScreenSettings
from neck_localization.task_validated_output import TaskValidationSettings

MODES = ("native_ideal", "native_ideal_z16", "native_raw", "native_spatial")


def sources():
    names = ["neck_localization/native_stereo.py", "scripts/evaluate_native_stereo.py",
             "neck_localization/stereo_robust.py", "neck_localization/active_stereo.py",
             "neck_localization/reliable_output.py", "neck_localization/task_validated_output.py",
             "neck_localization/joint_profile.py", "neck_localization/depth_profile.py",
             "neck_localization/support_feedback.py", "neck_localization/geometry.py"]
    return {name:sha256((ROOT/name).read_bytes()).hexdigest() for name in names}


def summarize(rows):
    result = []
    for mode in MODES:
        selected = [r for r in rows if r["condition_name"] == mode]
        item = dict(mode=mode, n=len(selected), candidates=sum(r.get("estimate") is not None for r in selected),
                    accepted=sum(r["accepted"] for r in selected),
                    correct_circle=sum(r["accepted"] and r["truth_pass"] for r in selected),
                    wrong_accepted=sum(r["accepted"] and not r["truth_pass"] for r in selected),
                    reasons={reason:sum(reason in r["reasons"] for r in selected)
                             for reason in sorted({reason for r in selected for reason in r["reasons"]})})
        for metric in ["center_mm", "normal_deg", "diameter_mm", "runtime_ms"]:
            values = [r[metric] for r in selected if r.get(metric) is not None]
            item[metric+"_median"] = float(np.median(values)) if values else None
            item[metric+"_max"] = float(np.max(values)) if values else None
        result.append(item)
    return result


def run(folder, output):
    config = json.loads((folder/"effective_config.json").read_text())
    if "development" not in config["dataset_split"] or len(config["cases"]) != 4:
        raise ValueError("Restricted to the four opened development jars")
    locked = sources()
    output.mkdir(parents=True, exist_ok=False)
    (output/"effective_config.json").write_text(json.dumps(config, indent=2)+"\n")
    provenance = dict(input=str(folder.resolve()), sources=locked, foreground=asdict(BlueScreenSettings()),
                      task_validation=asdict(TaskValidationSettings()), split="native_ir_development_diagnostic",
                      modes=MODES, geometry="frozen portfolio and thresholds; native IR K/T and Z",
                      truth_use="native_ideal/native_ideal_z16 are labelled diagnostic observations; target truth only after inference",
                      color_mapping="original RGB sampled onto measured native point locations; explicit mapped mask",
                      visibility="native measured-point RGB z-buffer only; disocclusion/matching errors remain",
                      limitation="not a sole-variable SDK-align ablation: sampling density and mapped-mask visibility also differ",
                      device_access=False, gpu_rendering=False)
    (output/"provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    rows = []
    with (output/"records.jsonl").open("w") as journal:
        for case in config["cases"]:
            name = case["name"]
            sensor = json.loads((folder/"geometry"/name/"sensor.json").read_text())
            rgb_folder = folder/"stereo_raw/scenes"/name/"observations"
            rgb_metadata = json.loads((rgb_folder/"camera_true_parameters.json").read_text())
            rgb_cameras = {item["name"]:Camera(item["name"], np.asarray(item["K"]),
                              np.asarray(item["T_world_from_camera"]), item["width"], item["height"])
                           for item in rgb_metadata["cameras"]}
            for mode in MODES:
                frames, masks, mappings = [], [], []
                destination = output/mode/"scenes"/name
                observation = destination/"observations"
                observation.mkdir(parents=True)
                mask_dir = destination/"foreground_masks"
                mask_dir.mkdir()
                camera_export = []
                for item in sensor["cameras"]:
                    camera_name = item["name"]
                    # Read original RGB and camera parameters only. No aligned
                    # ideal depth is opened while creating a native input.
                    original_rgb = cv2.cvtColor(cv2.imread(str(rgb_folder/(camera_name+"_rgb.png"))), cv2.COLOR_BGR2RGB)
                    rgb_frame = SimpleNamespace(camera=rgb_cameras[camera_name], rgb=original_rgb)
                    with np.load(folder/"sensor_diagnostics"/name/(camera_name+"_native.npz")) as native:
                        if mode.startswith("native_ideal"):
                            depth = native["ideal_left_z"].copy()
                            if mode.endswith("_z16"):
                                unit = sensor["settings"]["depth_unit_m"]
                                depth = quantize_depth(depth, unit).astype(np.float32)*unit
                        else:
                            depth = native["native_z16_m" if mode == "native_raw" else "native_filtered_m"].copy()
                    camera = Camera(camera_name, np.asarray(item["K_ir"]), np.asarray(item["T_world_from_left_ir"]),
                                    depth.shape[1], depth.shape[0])
                    frame, mask, mapping_valid, mapping = map_rgb_onto_native(rgb_frame, camera, depth)
                    frames.append(frame)
                    masks.append(mask)
                    mappings.append(mapping)
                    cv2.imwrite(str(observation/(camera_name+"_rgb.png")), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
                    (observation/(camera_name+"_source_rgb.png")).symlink_to((rgb_folder/(camera_name+"_rgb.png")).resolve())
                    np.savez_compressed(observation/(camera_name+"_depth_z_m.npz"), depth_z_m=depth)
                    cv2.imwrite(str(mask_dir/(camera_name+"_foreground.png")), mask.astype(np.uint8)*255)
                    cv2.imwrite(str(mask_dir/(camera_name+"_color_mapping_valid.png")), mapping_valid.astype(np.uint8)*255)
                    camera_export.append(dict(name=camera_name, K=camera.K.tolist(), T_world_from_camera=camera.T_world_from_camera.tolist(),
                                              width=camera.width, height=camera.height))
                (observation/"camera_true_parameters.json").write_text(json.dumps(dict(
                    kind="native_left_IR_true_synthetic_parameters", cameras=camera_export,
                    color="RGB reprojected to native observed points, original RGB saved separately"), indent=2)+"\n")
                (observation/"measurement_model.json").write_text(json.dumps(dict(mode=mode, mapping=mappings,
                    ideal_diagnostic=mode.startswith("native_ideal")), indent=2)+"\n")
                estimate = estimate_native_stereo(frames, masks, mappings)
                diagnostics = estimate.diagnostics
                band = (plan_band(dict(center_m=estimate.center, normal=estimate.normal, radius_m=estimate.radius), diagnostics)
                        if estimate.center is not None and "near_min_interval_from_axis_origin_m" in diagnostics else None)
                truth = json.loads((folder/"geometry"/name/"evaluation_only/target.json").read_text())
                row = dict(case=name, group=case["group"], condition_name=mode, factor=mode,
                           method="native_ir_rgb_mapped", split="native_ir_development_diagnostic",
                           evidence="original_RGB_mapped_mask", camera_set="all", **score(estimate, truth))
                row.setdefault("diagnostics", diagnostics)
                row["band"] = band
                row["band_evaluation"] = ({label:evaluate_band(band, truth, truth[field]) for label, field in
                    [("near_min_proxy", "near_min_interval_s_m"), ("full_annotated_neck", "search_interval_s_m")]} if band else None)
                rows.append(row)
                journal.write(json.dumps(row)+"\n")
                journal.flush()
                print(name, mode, row["accepted"], row.get("center_mm"), row.get("normal_deg"),
                      row.get("diameter_mm"), row["reasons"], flush=True)
    (output/"records.json.gz").write_bytes(gzip.compress(json.dumps(rows).encode(), mtime=0))
    (output/"summary.json").write_text(json.dumps(summarize(rows), indent=2)+"\n")
    for mode in MODES:
        selected = [r for r in rows if r["condition_name"] == mode]
        (output/mode/"records.json.gz").write_bytes(gzip.compress(json.dumps(selected).encode(), mtime=0))
    assert sources() == locked, "Sources changed during native diagnostic"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.run, args.output)
