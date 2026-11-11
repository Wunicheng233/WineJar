#!/usr/bin/env python3
"""Evaluate the limited rim-guided profile baseline on prepared static images."""
from dataclasses import asdict
from hashlib import sha256
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neck_localization.depth_profile import DepthFrame, ProfileSettings, estimate_depth_profile
from neck_localization.evaluation import score, summarize
from neck_localization.geometry import Camera
from neck_localization.scene_preparation import write_json


def load_frames(folder, evidence, camera_names=None):
    """Observation files only. In RGB mode even diagnostic masks stay unread."""
    metadata = json.loads((folder / "camera_true_parameters.json").read_text())
    frames = []
    for item in metadata["cameras"]:
        if camera_names is not None and item["name"] not in camera_names:
            continue
        camera = Camera(item["name"], np.array(item["K"]), np.array(item["T_world_from_camera"]),
                        item["width"], item["height"])
        name = item["name"]
        rgb = cv2.cvtColor(cv2.imread(str(folder / f"{name}_rgb.png")), cv2.COLOR_BGR2RGB)
        with np.load(folder / f"{name}_depth_z_m.npz") as data:
            depth = data["depth_z_m"]
        suffix = "exterior_mask_diagnostic" if evidence == "exterior_diagnostic" else "mask_diagnostic"
        mask = (cv2.imread(str(folder / f"{name}_{suffix}.png"), cv2.IMREAD_GRAYSCALE) > 0
                if evidence in ("mask_diagnostic", "exterior_diagnostic") else None)
        frames.append(DepthFrame(camera, rgb, depth, mask))
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["development", "heldout", "diagnostic"], default="development")
    parser.add_argument("--evidence", nargs="+", choices=["rgb", "mask_diagnostic", "exterior_diagnostic"],
                        default=["mask_diagnostic", "rgb"])
    parser.add_argument("--camera-sets", nargs="+", default=["all"],
                        help="e.g. cam_a cam_a,cam_b all; each is evaluated separately")
    args = parser.parse_args()
    estimator_sha = sha256((ROOT / "neck_localization/depth_profile.py").read_bytes()).hexdigest()
    if args.split == "heldout":
        config = json.loads((args.scenes / "effective_config.json").read_text())
        if config.get("frozen_estimator_sha256") != estimator_sha:
            raise ValueError("Held-out scenes require their pre-frozen estimator source hash to match")
    args.output.mkdir(parents=True, exist_ok=False)
    settings = ProfileSettings()
    write_json(args.output / "frozen_settings.json", asdict(settings))
    records = []
    started = time.perf_counter()
    for camera_file in sorted(args.scenes.glob("*/observations/camera_true_parameters.json")):
        folder = camera_file.parent.parent
        for camera_set in args.camera_sets:
            names = None if camera_set == "all" else camera_set.split(",")
            for evidence in args.evidence:
                frames = load_frames(folder / "observations", evidence, names)
                if not frames or (names is not None and len(frames) != len(names)):
                    raise ValueError("Unknown/duplicate camera in set: " + camera_set)
                for method, axis in [("vertical_profile", "vertical"), ("rim_guided_profile", "mouth_proxy")]:
                    estimator_evidence = "mask_diagnostic" if evidence == "exterior_diagnostic" else evidence
                    estimate = estimate_depth_profile(frames, evidence=estimator_evidence, axis_mode=axis,
                                                      settings=settings)
                    if evidence == "exterior_diagnostic":
                        estimate.diagnostics["mask_semantics"] = "GT visible exterior surface, excluding cavity; no target-ring labels"
                    # Truth is first opened only AFTER this estimate is computed.
                    truth = json.loads((folder / "evaluation_only/target.json").read_text())
                    result = score(estimate, truth)
                    records.append(dict(case=folder.name, split=args.split, geometry="photo_short_neck",
                                        evidence=evidence, camera_set=camera_set,
                                        factor="ideal_true_cameras/"+camera_set, method=method, **result))
                    print(json.dumps({"case": folder.name, "evidence": evidence, "camera_set": camera_set,
                                      "method": method, "accepted": result["accepted"],
                                      "center_mm": result["center_mm"], "normal_deg": result["normal_deg"],
                                      "diameter_mm": result["diameter_mm"], "reasons": result["reasons"]}), flush=True)
    if not records:
        raise ValueError("No prepared observation folders found")
    write_json(args.output / "records.json", records)
    write_json(args.output / "summary.json", summarize(records))
    sources = [ROOT / "neck_localization" / name for name in
               ("depth_profile.py", "geometry.py", "evaluation.py")]+[Path(__file__)]
    write_json(args.output / "provenance.json", {
        "scene_source": str(args.scenes.resolve()), "split": args.split,
        "scene_manifest_sha256": sha256((args.scenes / "manifest.json").read_bytes()).hexdigest(),
        "scene_config_sha256": sha256((args.scenes / "effective_config.json").read_bytes()).hexdigest(),
        "camera_sets": args.camera_sets,
        "evidence": args.evidence, "seed": 4101, "elapsed_s": time.perf_counter()-started,
        "source_sha256": {p.name: sha256(p.read_bytes()).hexdigest() for p in sources},
        "limitations": ["true camera parameters; no virtual calibration or sensor noise",
                        "mouth plane is an axis proxy; no independent local neck-axis recovery",
                        "planar crown and circular neck are synthetic model assumptions",
                        "RGB extraction assumes declared grey/clay jar and blue background",
                        "no uncertainty estimate; plane residual is not absolute accuracy",
                        "4 mm center is historical goal; 2 deg normal and 2 mm diameter are provisional analysis thresholds",
                        "small scene family, no real-world accuracy claim"]})


if __name__ == "__main__":
    main()
