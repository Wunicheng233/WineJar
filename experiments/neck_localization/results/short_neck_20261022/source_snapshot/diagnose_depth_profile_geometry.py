#!/usr/bin/env python3
"""Explicit ORACLE interventions for error attribution, not estimator accuracy.

Reads generator dimensions/axis here by design. Never call this from the
ordinary estimator or count its results as observation-only localization.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_depth_profile_baseline import load_frames
from neck_localization.depth_profile import ProfileSettings, _point_cloud, _radius_profile, ProfileRejection
from neck_localization.jar_model import JarSpec, outer_radius, target_metadata, centerline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="+", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    ranges = []
    for group in args.scenes:
        for path in sorted(group.glob("*/model_manifest.json")):
            spec = JarSpec(**json.loads(path.read_text())["jar_spec"])
            truth = target_metadata(spec)
            midpoint = truth["s_m"]
            threshold = truth["radius_m"]+spec.near_min_delta_m
            function = lambda s: float(outer_radius(spec, s))-threshold
            low = brentq(function, midpoint-.06, midpoint)
            high = brentq(function, midpoint, spec.height_m)
            ranges.append({"group": group.name, "case": path.parent.name,
                           "generator_neck_interval_s_m": truth["search_interval_s_m"],
                           "generator_near_min_interval_s_m": truth["near_min_interval_s_m"],
                           "broader_threshold_interval_s_m": [low, high],
                           "signed_midpoint_bias_mm": ((low+high)/2-midpoint)*1000})
    rows = json.loads(args.records.read_text())
    candidates = [r for r in rows if r["evidence"] == "rgb" and r["method"] == "rim_guided_profile"
                  and r["camera_set"] == "cam_a" and r["center_mm"] is not None]
    worst = max(candidates, key=lambda r: r["center_mm"])
    folder = next(group / worst["case"] for group in args.scenes if (group / worst["case"]).exists())
    spec = JarSpec(**json.loads((folder / "model_manifest.json").read_text())["jar_spec"])
    truth = json.loads((folder / "evaluation_only/target.json").read_text())
    settings = ProfileSettings()
    cloud, _ = _point_cloud(load_frames(folder / "observations", "rgb", ["cam_a"]), "rgb", settings)
    estimated_mouth = np.array(worst["diagnostics"]["observed_mouth_center_m"])
    estimated_axis = np.array(worst["estimate"]["normal"])
    true_mouth = np.r_[spec.neck_offset_xy_m, spec.height_m]
    interventions = []
    for label, mouth, axis in [
            ("observations_only", estimated_mouth, estimated_axis),
            ("oracle_mouth_center_only", true_mouth, estimated_axis),
            ("oracle_axis_only", estimated_mouth, np.array(truth["normal"])),
            ("oracle_mouth_center_and_axis", true_mouth, np.array(truth["normal"]))]:
        record = {"condition": label, "oracle_input": label != "observations_only"}
        try:
            position, radius, _ = _radius_profile(cloud, mouth, axis, settings)
            record.update(center_error_mm=float(np.linalg.norm(mouth+position*axis-truth["center_m"])*1000),
                          diameter_error_mm=abs(radius-truth["radius_m"])*2000)
        except ProfileRejection as error:
            record["rejection"] = str(error)
        interventions.append(record)
    output = {"kind": "explicit_generator_oracle_diagnostic_not_normal_estimation",
              "range_definition": ranges, "worst_single_view_case": worst["case"],
              "mouth_center_error_mm": float(np.linalg.norm(estimated_mouth-true_mouth)*1000),
              "interventions": interventions}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps(output["interventions"], indent=2))


if __name__ == "__main__":
    main()
