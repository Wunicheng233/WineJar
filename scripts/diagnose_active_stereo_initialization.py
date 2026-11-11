#!/usr/bin/env python3
"""CPU-only forensic diagnostics of saved active-stereo initialization failures.

Does not change an estimator or run GPU rendering. Ideal RGB depth is used ONLY
to label diagnostic regions/errors, never to remove points from estimator input.
Quantile truncations are descriptive probes of the frozen RANSAC, not a selected
preprocessing policy or a new accuracy benchmark.
"""
import argparse
from dataclasses import asdict
from hashlib import sha256
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from neck_localization.depth_profile import ProfileRejection, ProfileSettings, _mouth_center, _mouth_plane, rgb_foreground
from neck_localization.stereo_robust import blue_screen_foreground
from scripts.run_depth_profile_baseline import load_frames


MODES = ("ideal_rgb", "ideal_ir_registered", "stereo_raw", "stereo_spatial")
QUANTILES = (95., 98., 99., 99.5, 99.9, 100.)


def stats(values):
    values = np.asarray(values)
    if len(values) == 0:
        return dict(n=0, median=None, p95_abs=None, maximum_abs=None)
    return dict(n=len(values), median=float(np.median(values)),
                p95_abs=float(np.percentile(np.abs(values), 95)), maximum_abs=float(np.max(np.abs(values))))


def station_keep(points, settings):
    return ((points[:, 2] >= settings.world_z_bounds_m[0]) &
            (points[:, 2] <= settings.world_z_bounds_m[1]) &
            (np.linalg.norm(points[:, :2], axis=1) < settings.world_xy_radius_m))


def ransac_probe(cloud, settings, upper_z=None):
    if upper_z is not None:
        cloud = cloud[cloud[:, 2] <= upper_z]
    top = cloud[cloud[:, 2] > cloud[:, 2].max()-settings.plane_top_window_m]
    upper = top[top[:, 2] > top[:, 2].max()-settings.plane_seed_top_m]
    rng = np.random.default_rng(4101)
    sample = upper[rng.choice(len(upper), min(6000, len(upper)), replace=False)]
    best_count, best_normal = 0, None
    # This is the frozen _mouth_plane's seed-consensus loop, without changing
    # its thresholds, and exposed even when the estimator rejects (<60 votes).
    for _ in range(settings.plane_trials):
        if len(sample) < 3:
            break
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(b-a, c-a)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal /= norm
        if normal[2] < 0:
            normal = -normal
        if normal[2] < np.cos(np.deg2rad(30)):
            continue
        count = int(np.count_nonzero(np.abs((sample-a) @ normal) < settings.plane_tolerance_m))
        if count > best_count:
            best_count, best_normal = count, normal
    return dict(cloud_points=len(cloud), z_max_mm=float(cloud[:, 2].max()*1000),
                top_35mm_points=len(top), seed_top_6mm_points=len(upper),
                sampled_seed_points=len(sample), best_seed_consensus=best_count,
                minimum_required_consensus=60,
                best_seed_normal=None if best_normal is None else best_normal.tolist())


def write_csv(path, rows):
    keys = sorted({key for row in rows for key in row})
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run(run_folder, output, foreground="frozen"):
    settings = ProfileSettings()
    foreground_function = rgb_foreground if foreground == "frozen" else blue_screen_foreground
    config = json.loads((run_folder/"effective_config.json").read_text())
    cases = [c["name"] for c in config["cases"]]
    records = [json.loads(line) for line in (run_folder/"records.jsonl").read_text().splitlines() if line]
    results, camera_rows, quantile_rows = [], [], []
    output.mkdir(parents=True, exist_ok=False)
    for case in cases:
        references = load_frames(run_folder/"ideal_rgb/scenes"/case/"observations", "rgb")
        reference_clouds = []
        for frame in references:
            yy, xx = np.nonzero(foreground_function(frame.rgb) & (frame.depth_z_m > 0))
            points = frame.camera.unproject(np.c_[xx, yy], frame.depth_z_m[yy, xx])
            reference_clouds.append(points[station_keep(points, settings)])
        reference_cloud = np.concatenate(reference_clouds)
        ref_origin, ref_normal, ref_inliers = _mouth_plane(reference_cloud, settings, np.random.default_rng(4101))
        ref_center, _ = _mouth_center(ref_origin, ref_normal, ref_inliers)
        ref_max_z = float(reference_cloud[:, 2].max())
        for mode in MODES:
            frames = load_frames(run_folder/mode/"scenes"/case/"observations", "rgb")
            clouds, sources, diagnostic_background = [], [], []
            mode_cameras = []
            for index, (frame, reference) in enumerate(zip(frames, references)):
                np.testing.assert_array_equal(frame.rgb, reference.rgb)
                mask = foreground_function(frame.rgb)
                valid = mask & np.isfinite(frame.depth_z_m) & (frame.depth_z_m > 0)
                yy, xx = np.nonzero(valid)
                points = frame.camera.unproject(np.c_[xx, yy], frame.depth_z_m[yy, xx])
                keep = station_keep(points, settings)
                cloud = points[keep]
                clouds.append(cloud)
                sources.append(np.full(len(cloud), index))
                background = reference.depth_z_m[yy[keep], xx[keep]] <= 0
                diagnostic_background.append(background)
                # Define the diagnostic crown footprint from ideal RGB depth.
                ry, rx = np.nonzero(mask & (reference.depth_z_m > 0))
                reference_points = frame.camera.unproject(np.c_[rx, ry], reference.depth_z_m[ry, rx])
                on_reference_crown = np.abs((reference_points-ref_origin) @ ref_normal) < settings.plane_tolerance_m
                cy, cx = ry[on_reference_crown], rx[on_reference_crown]
                crown_valid = np.isfinite(frame.depth_z_m[cy, cx]) & (frame.depth_z_m[cy, cx] > 0)
                actual_crown = frame.camera.unproject(np.c_[cx[crown_valid], cy[crown_valid]],
                                                     frame.depth_z_m[cy[crown_valid], cx[crown_valid]])
                plane_error_mm = ((actual_crown-ref_origin) @ ref_normal)*1000
                row = dict(case=case, mode=mode, camera=frame.camera.name,
                           image_pixels=int(mask.size), foreground_pixels=int(mask.sum()),
                           foreground_fraction=float(mask.mean()),
                           rgb_corner=frame.rgb[0, 0].tolist(), valid_mask_depth_pixels=int(valid.sum()),
                           station_points=len(cloud),
                           station_background_ghost_points=int(background.sum()),
                           above_reference_max_plus_20mm=int(np.count_nonzero(cloud[:, 2] > ref_max_z+.02)),
                           above_reference_max_plus_20mm_background=int(np.count_nonzero(background & (cloud[:, 2] > ref_max_z+.02))),
                           ideal_crown_pixels=len(cx), crown_valid_depth_pixels=int(crown_valid.sum()),
                           crown_depth_valid_fraction=float(crown_valid.mean()) if len(cx) else None,
                           crown_within_frozen_plane_tolerance=int(np.count_nonzero(np.abs(plane_error_mm) < settings.plane_tolerance_m*1000)))
                row.update({"crown_plane_error_mm_"+key:value for key, value in stats(plane_error_mm).items()})
                for q in QUANTILES:
                    row[f"station_z_p{q:g}_mm"] = float(np.percentile(cloud[:, 2], q)*1000)
                camera_rows.append(row)
                mode_cameras.append(row)
            cloud, ids, bg = np.concatenate(clouds), np.concatenate(sources), np.concatenate(diagnostic_background)
            probe = ransac_probe(cloud, settings)
            try:
                plane_origin, plane_normal, plane_inliers = _mouth_plane(cloud, settings, np.random.default_rng(4101))
                plane_center, plane_radius = _mouth_center(plane_origin, plane_normal, plane_inliers)
                refined = dict(center_m=plane_center.tolist(), normal=plane_normal.tolist(),
                               radius_m=float(plane_radius), support=len(plane_inliers),
                               center_difference_from_ideal_rgb_mm=float(np.linalg.norm(plane_center-ref_center)*1000),
                               normal_difference_from_ideal_rgb_deg=float(np.degrees(np.arccos(np.clip(plane_normal @ ref_normal, -1, 1)))))
            except (ProfileRejection, np.linalg.LinAlgError, ValueError) as error:
                refined = dict(rejected=str(error))
            upper_mask = cloud[:, 2] > cloud[:, 2].max()-settings.plane_seed_top_m
            top_seed_reference_plane_mm = ((cloud[upper_mask]-ref_origin) @ ref_normal)*1000
            saved = next(row for row in records if row["case"] == case and row["condition_name"] == mode)
            result = dict(case=case, mode=mode, reference_ideal_mouth_normal=ref_normal.tolist(),
                          reference_ideal_crown_support=len(ref_inliers), reference_ideal_z_max_mm=ref_max_z*1000,
                          observed=probe,
                          refined_mouth_initialization=refined,
                          seed_background_ghost_points=int(np.count_nonzero(bg & upper_mask)),
                          seed_per_camera={frames[i].camera.name:int(np.count_nonzero((ids == i) & upper_mask)) for i in range(len(frames))},
                          seed_reference_plane_error_mm=stats(top_seed_reference_plane_mm),
                          frozen_estimator_accepted=saved["accepted"], frozen_estimator_reasons=saved["reasons"],
                          frozen_candidate_metrics={k:saved.get(k) for k in ["center_mm", "normal_deg", "diameter_mm"]},
                          cameras=mode_cameras)
            if mode.startswith("stereo"):
                for q in QUANTILES:
                    z = float(np.percentile(cloud[:, 2], q))
                    q_probe = ransac_probe(cloud, settings, z)
                    q_probe.update(case=case, mode=mode, upper_z_percentile=q)
                    quantile_rows.append(q_probe)
            results.append(result)
            print(case, mode, "N", len(cloud), "seed", probe["seed_top_6mm_points"],
                  "consensus", probe["best_seed_consensus"], "background seed", result["seed_background_ghost_points"], flush=True)
    provenance = dict(input=str(run_folder.resolve()), settings=asdict(settings), cases=cases, foreground=foreground,
                      diagnostic_only=True, estimator_rerun=False, no_gt_mask_used_for_inference=True,
                      ideal_depth_use="diagnostic reference crown and background classification only",
                      quantile_use="descriptive frozen-RANSAC probes; no selected threshold or accepted-output result",
                      sources={str(path.relative_to(ROOT)):sha256(path.read_bytes()).hexdigest() for path in
                               [Path(__file__), ROOT/"neck_localization/depth_profile.py", ROOT/"neck_localization/joint_profile.py",
                                ROOT/"neck_localization/stereo_robust.py"]})
    (output/"diagnosis.json").write_text(json.dumps(dict(provenance=provenance, cases=results), indent=2)+"\n")
    write_csv(output/"cameras.csv", camera_rows)
    write_csv(output/"quantile_probes.csv", quantile_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--foreground", choices=["frozen", "blue"], default="frozen")
    args = parser.parse_args()
    run(args.run, args.output, args.foreground)
