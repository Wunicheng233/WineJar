#!/usr/bin/env python3
"""CPU-only paired foreground adaptation audit on saved DEVELOPMENT RGB-D."""
import argparse
from dataclasses import asdict
import gzip
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from neck_localization.band_geometry import evaluate_band, plan_band
from neck_localization.evaluation import score
from neck_localization.stereo_robust import BlueScreenSettings, estimate_stereo_robust, extract_stereo_masks
from neck_localization.task_validated_output import TaskValidationSettings
from scripts.run_active_stereo_stage1 import MODES, sources as sensor_sources, summarize
from scripts.run_depth_profile_baseline import load_frames


def run(folder, output):
    config = json.loads((folder/"effective_config.json").read_text())
    if "development" not in config["dataset_split"] or config["dataset_split"].endswith("_test"):
        raise ValueError("This diagnostic runner is restricted to development data")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(folder/"effective_config.json", output/"effective_config.json")
    old_rows = json.loads(gzip.decompress((folder/"records.json.gz").read_bytes()))
    source_files = [ROOT/"neck_localization/stereo_robust.py", Path(__file__)]
    locked = sensor_sources() | {str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in source_files}
    provenance = dict(input=str(folder.resolve()), split=config["dataset_split"], sources=locked,
                      foreground=asdict(BlueScreenSettings()), task_validation=asdict(TaskValidationSettings()),
                      changed="RGB foreground only; original depth and RGB preserved; frozen geometry/thresholds",
                      truth_use="scoring after inference only", gpu_rendering=False)
    (output/"provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    new_rows = []
    with (output/"records.jsonl").open("w") as journal:
        for case in config["cases"]:
            name = case["name"]
            for mode in MODES:
                source = folder/mode/"scenes"/name
                frames = load_frames(source/"observations", "rgb")
                # These are recomputed by inference from the original RGB;
                # assert hashes below so saved masks are exact, not illustrative.
                masks, detail = extract_stereo_masks(frames)
                destination = output/mode/"scenes"/name
                destination.mkdir(parents=True)
                for child in source.iterdir():
                    (destination/child.name).symlink_to(child.resolve(), target_is_directory=child.is_dir())
                mask_dir = destination/"foreground_masks"
                mask_dir.mkdir()
                for frame, mask in zip(frames, masks):
                    cv2.imwrite(str(mask_dir/(frame.camera.name+"_foreground.png")), mask.astype(np.uint8)*255)
                estimate = estimate_stereo_robust(frames)
                assert estimate.diagnostics["foreground"] == detail
                diagnostics = estimate.diagnostics
                band = (plan_band(dict(center_m=estimate.center, normal=estimate.normal, radius_m=estimate.radius), diagnostics)
                        if estimate.center is not None and "near_min_interval_from_axis_origin_m" in diagnostics else None)
                truth = json.loads((source/"evaluation_only/target.json").read_text())
                row = dict(case=name, group=case["group"], split=config["dataset_split"], geometry="photo_short_neck",
                           evidence="rgb", camera_set="all", factor=mode, condition_name=mode,
                           method="stereo_rgb_adapted", **score(estimate, truth))
                row.setdefault("diagnostics", diagnostics)
                row["band"] = band
                row["band_evaluation"] = ({name:evaluate_band(band, truth, truth[field]) for name, field in
                    [("near_min_proxy", "near_min_interval_s_m"), ("full_annotated_neck", "search_interval_s_m")]} if band else None)
                new_rows.append(row)
                journal.write(json.dumps(row)+"\n")
                journal.flush()
                print(name, mode, row["accepted"], row.get("center_mm"), row.get("normal_deg"),
                      row.get("diameter_mm"), row["reasons"], flush=True)
    (output/"records.json.gz").write_bytes(gzip.compress(json.dumps(new_rows).encode(), mtime=0))
    (output/"paired_records.json.gz").write_bytes(gzip.compress(json.dumps(old_rows+new_rows).encode(), mtime=0))
    (output/"summary.json").write_text(json.dumps(dict(frozen_rgb=summarize(old_rows), rgb_adapted=summarize(new_rows)), indent=2)+"\n")
    for mode in MODES:
        selected = [r for r in new_rows if r["condition_name"] == mode]
        (output/mode/"records.json.gz").write_bytes(gzip.compress(json.dumps(selected).encode(), mtime=0))
    assert locked == sensor_sources() | {str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in source_files}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.run, args.output)
