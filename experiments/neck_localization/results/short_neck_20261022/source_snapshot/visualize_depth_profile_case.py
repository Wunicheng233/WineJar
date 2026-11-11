#!/usr/bin/env python3
"""Explain a saved success/rejection with observation stages and GT plot overlays.

Reconstructs intermediate observations with the frozen estimator. Ground truth
is opened afterwards, solely for labeled diagnostic overlays. Does not change
the saved prediction, RGB/depth inputs, or estimator parameters.
"""
import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_depth_profile_baseline import load_frames
from neck_localization.depth_profile import ProfileSettings, _point_cloud, _mouth_plane, _mouth_center, rgb_foreground
from neck_localization.geometry import basis, circle_points
from neck_localization.jar_model import JarSpec, centerline, neck_normal, outer_radius


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--camera-set", default="cam_a")
    parser.add_argument("--evidence", default="rgb", choices=["rgb", "mask_diagnostic", "exterior_diagnostic"])
    parser.add_argument("--method", default="rim_guided_profile", choices=["rim_guided_profile", "vertical_profile"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    provenance = json.loads((args.records.parent / "provenance.json").read_text())
    expected = provenance["source_sha256"]["depth_profile.py"]
    if sha256((ROOT / "neck_localization/depth_profile.py").read_bytes()).hexdigest() != expected:
        raise ValueError("Estimator source changed; use the frozen source before reconstructing old intermediate stages")
    data = args.records.read_bytes()
    records = json.loads(gzip.decompress(data) if args.records.suffix == ".gz" else data)
    row = next(r for r in records if r["case"] == args.case and r["camera_set"] == args.camera_set
               and r["evidence"] == args.evidence and r["method"] == args.method)
    folder = args.scenes / args.case
    names = None if args.camera_set == "all" else args.camera_set.split(",")
    frames = load_frames(folder / "observations", args.evidence, names)
    settings = ProfileSettings()
    evidence = "mask_diagnostic" if args.evidence == "exterior_diagnostic" else args.evidence
    cloud, counts = _point_cloud(frames, evidence, settings)
    plane_origin, plane_normal, plane_points = _mouth_plane(cloud, settings, np.random.default_rng(4101))
    mouth, mouth_radius = _mouth_center(plane_origin, plane_normal, plane_points)
    axis = plane_normal if args.method == "rim_guided_profile" else np.array([0., 0., 1.])
    if "diagnostics" in row:
        np.testing.assert_allclose(mouth, row["diagnostics"]["observed_mouth_center_m"], atol=1e-11)

    # Normal observation reconstruction above never opens these evaluation files.
    truth = json.loads((folder / "evaluation_only/target.json").read_text())
    spec = JarSpec(**json.loads((folder / "model_manifest.json").read_text())["jar_spec"])
    true_mouth = centerline(spec, spec.height_m)
    true_rim = circle_points(true_mouth, neck_normal(spec), float(outer_radius(spec, spec.height_m)), 240)
    args.output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    camera = frames[0].camera
    image_ax = axes[0, 0]
    image_ax.imshow(frames[0].rgb)
    mask = rgb_foreground(frames[0].rgb) if args.evidence == "rgb" else frames[0].diagnostic_mask
    if np.any(mask) and not np.all(mask):
        image_ax.contour(mask.astype(float), levels=[.5], colors=["#d5bc3c"], linewidths=.5)
    for point, label, color, marker in [
            (mouth, "Estimated mouth center", "#ff9933", "o"),
            (true_mouth, "GT mouth center", "#5ab6ff", "+"),
            (truth["center_m"], "GT target center", "#61d7ac", "x")]:
        pixel = camera.project(np.asarray(point)[None])[0]
        image_ax.plot(*pixel, marker=marker, color=color, markersize=8, label=label)
    if row.get("estimate") is not None:
        pixel = camera.project(np.array(row["estimate"]["center_m"])[None])[0]
        image_ax.plot(*pixel, marker="x", color="#ff554f", markersize=8, label="Estimated target center")
    pixel = camera.project(mouth[None])[0]
    image_ax.set(xlim=(max(0, pixel[0]-105), min(camera.width-1, pixel[0]+105)),
                 ylim=(min(camera.height-1, pixel[1]+105), max(0, pixel[1]-65)),
                 title="RGB crop: extracted foreground contour + center projections")
    image_ax.legend(fontsize=7, loc="lower left")

    u, v = basis(plane_normal)
    project = lambda p: np.column_stack(((np.asarray(p)-plane_origin) @ u, (np.asarray(p)-plane_origin) @ v))*1000
    xy = project(plane_points)
    hull = ConvexHull(xy)
    rim_ax = axes[0, 1]
    rim_ax.scatter(*xy.T, s=3, alpha=.25, color="grey", label="Observed plane inliers")
    polygon = xy[np.r_[hull.vertices, hull.vertices[0]]]
    rim_ax.plot(*polygon.T, color="#2a9d8f", linewidth=1, label="Convex hull used as outer-rim evidence")
    predicted_rim = circle_points(mouth, plane_normal, mouth_radius, 240)
    rim_ax.plot(*project(predicted_rim).T, color="#dc7a23", label="Fitted mouth circle")
    rim_ax.plot(*project(true_rim).T, "--", color="#1675ba", label="GT mouth rim projected into fitted plane")
    rim_ax.scatter(*project(mouth[None])[0], marker="x", color="#dc7a23", s=60)
    rim_ax.scatter(*project(true_mouth[None])[0], marker="+", color="#1675ba", s=60)
    rim_ax.set_aspect("equal", adjustable="datalim")
    rim_ax.set(xlabel="Plane coordinate u (mm)", ylabel="Plane coordinate v (mm)",
               title=f"Mouth-center error {np.linalg.norm(mouth-true_mouth)*1000:.2f} mm")
    rim_ax.legend(fontsize=6.5)
    rim_ax.grid(alpha=.15)

    delta = cloud-mouth
    axial = delta @ axis
    radius = np.linalg.norm(delta-axial[:, None]*axis, axis=1)
    edges = np.arange(-settings.profile_depth_m, -settings.exclude_crown_m+1e-9, settings.bin_width_m)
    centres = (edges[:-1]+edges[1:])/2
    bins = np.searchsorted(edges, axial, side="right")-1
    bin_counts = np.bincount(bins[(bins >= 0) & (bins < len(centres))], minlength=len(centres))
    radii = np.full(len(centres), np.nan)
    for i in np.flatnonzero(bin_counts >= settings.min_bin_points):
        radii[i] = np.quantile(radius[bins == i], settings.radial_quantile)
    profile_ax = axes[1, 0]
    profile_ax.plot(centres*1000, radii*1000, ".-", markersize=3, label="Observed 85th radial percentile; gaps left open")
    if "diagnostics" in row:
        diag = row["diagnostics"]
        profile_ax.plot(np.array(diag["axial_from_observed_mouth_m"])*1000,
                        np.array(diag["smoothed_radius_m"])*1000, label="Estimator-smoothed profile")
        low, high = np.array(diag["near_min_interval_from_mouth_m"])*1000
        profile_ax.axvspan(low, high, color="#1675ba", alpha=.15, label="Selected near-min interval")
    target_s = (np.array(truth["center_m"])-mouth) @ axis*1000
    profile_ax.axvline(target_s, color="#c15d33", linestyle="--", label="GT target center projected on estimated axis")
    profile_ax.axhline(truth["radius_m"]*1000, color="#c15d33", linestyle=":")
    profile_ax.set(xlabel="Axial position from estimated mouth (mm)", ylabel="Radius (mm)", title="Radius profile below mouth")
    profile_ax.legend(fontsize=6.5)
    profile_ax.grid(alpha=.15)

    count_ax = axes[1, 1]
    count_ax.bar(centres*1000, bin_counts, width=.42, color="#628caa")
    count_ax.axhline(settings.min_bin_points, color="#b55431", linestyle="--", label="Minimum 8 points per bin")
    supported = centres[bin_counts >= settings.min_bin_points]
    if len(supported) > 1:
        for a, b in zip(supported[:-1], supported[1:]):
            if b-a > settings.maximum_gap_m:
                count_ax.axvspan(a*1000, b*1000, color="#e65c49", alpha=.25)
    count_ax.set(xlabel="Axial position from estimated mouth (mm)", ylabel="Point count per 0.5 mm bin",
                 title="Support: red spans exceed the allowed 2.5 mm gap")
    count_ax.legend(fontsize=7)
    reason = "; ".join(row["reasons"]) or "no internal rejection"
    metrics = (f"center={row['center_mm']:.2f} mm, normal={row['normal_deg']:.2f} deg, diameter={row['diameter_mm']:.2f} mm"
               if row["center_mm"] is not None else "no solved target circle")
    fig.suptitle(f"{args.case} | {args.camera_set} | {args.evidence} | {args.method}\n"
                 f"accepted={row['accepted']}; evaluation_pass={row['truth_pass']} | {metrics}\n"
                 f"{reason} | GT overlays are diagnostic only", fontsize=10)
    fig.savefig(args.output / "steps.png", dpi=155)
    plt.close(fig)
    cv2.imwrite(str(args.output / "input_rgb.png"), cv2.cvtColor(frames[0].rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(args.output / "foreground_mask.png"), np.asarray(mask, np.uint8)*255)
    metadata = {key: row[key] for key in ("case", "camera_set", "evidence", "method", "accepted", "truth_pass",
                                         "center_mm", "normal_deg", "diameter_mm", "reasons")}
    metadata.update(observation_point_counts=counts, gt_used_only_in_plot=True,
                    estimator_sha256=expected, visualization_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output / "explanation.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(str((args.output / "steps.png").resolve()))


if __name__ == "__main__":
    main()
