#!/usr/bin/env python3
"""Compare frozen baseline, fixed-axis spline and joint-axis spline offline."""
from dataclasses import asdict
from hashlib import sha256
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neck_localization.depth_profile import estimate_depth_profile
from neck_localization.joint_profile import JointSettings, estimate_joint_profile
from neck_localization.evaluation import score, summarize
from neck_localization.scene_preparation import write_json
from scripts.run_depth_profile_baseline import load_frames

METHODS = ["rim_guided_profile", "spline_fixed_axis", "joint_axis_spline"]
CAMERA_SETS = ["cam_a", "cam_a,cam_b", "all"]


def source_hashes():
    names = ["neck_localization/joint_profile.py", "neck_localization/depth_profile.py",
             "neck_localization/geometry.py", "neck_localization/evaluation.py",
             "scripts/run_depth_profile_baseline.py", "scripts/run_joint_profile.py"]
    return {name: sha256((ROOT/name).read_bytes()).hexdigest() for name in names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["development", "test"], default="development")
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--cases", nargs="*")
    args = parser.parse_args()
    if args.split == "test":
        if args.freeze is None or len(args.scenes) != 1 or args.cases:
            raise ValueError("Test requires one complete scene set and a pre-evaluation source freeze")
        frozen = json.loads(args.freeze.read_text())
        assert frozen["sources"] == source_hashes(), "Estimator/runner changed after freeze"
        assert frozen["settings"] == asdict(JointSettings()), "Settings changed after freeze"
        actual = sha256((args.scenes[0]/"effective_config.json").read_bytes()).hexdigest()
        assert frozen["test_config_sha256"] == actual, "Test scene specifications changed"
    args.output.mkdir(parents=True, exist_ok=False)
    records, started = [], time.perf_counter()
    write_json(args.output/"settings.json", asdict(JointSettings()))
    for scene_root in args.scenes:
        for camera_file in sorted(scene_root.glob("*/observations/camera_true_parameters.json")):
            folder = camera_file.parent.parent
            if args.cases and folder.name not in args.cases:
                continue
            for camera_set in CAMERA_SETS:
                names = None if camera_set == "all" else camera_set.split(",")
                frames = load_frames(folder/"observations", "rgb", names)
                for method in METHODS:
                    estimate = (estimate_depth_profile(frames, evidence="rgb") if method == METHODS[0]
                                else estimate_joint_profile(frames, free_axis=method == METHODS[2]))
                    # No target/model manifest supplied to estimator. Truth only after return.
                    truth = json.loads((folder/"evaluation_only/target.json").read_text())
                    result = score(estimate, truth)
                    result.setdefault("diagnostics", estimate.diagnostics)
                    row = dict(case=folder.name, split=args.split, geometry="photo_short_neck",
                               evidence="rgb", camera_set=camera_set,
                               factor="ideal_true_cameras/"+camera_set, method=method, **result)
                    records.append(row)
                    write_json(args.output/"records.json", records)
                    print(json.dumps({k: row[k] for k in ["case", "camera_set", "method", "accepted",
                        "center_mm", "normal_deg", "diameter_mm", "runtime_ms", "reasons"]}), flush=True)
    if not records:
        raise ValueError("No cases")
    write_json(args.output/"summary.json", summarize(records))
    write_json(args.output/"provenance.json", dict(sources=source_hashes(), split=args.split,
        scenes=[str(p.resolve()) for p in args.scenes], settings=asdict(JointSettings()),
        elapsed_s=time.perf_counter()-started, methods=METHODS, camera_sets=CAMERA_SETS,
        test_freeze_sha256=sha256(args.freeze.read_bytes()).hexdigest() if args.freeze else None,
        scope="true cameras, ideal saved depth, ordinary RGB extraction; no layout search/no sensor noise/no learning"))


if __name__ == "__main__":
    main()
