#!/usr/bin/env python3
"""Evaluation-only audit of saved fit regions and paired depth postprocessing.

Reads synthetic target/ideal depth ONLY for diagnostics; never calls or changes
an estimator. Fractions along the true neck axis do not certify outer-surface
membership; an additional radial-profile test is reported separately.
"""
import argparse
import csv
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from neck_localization.geometry import Camera, circle_points
from neck_localization.viewer_navigation import active_stereo_datasets


def error_stats(values):
    a = np.asarray(values, float)
    return dict(n=len(a), signed_mean_mm=float(a.mean()) if len(a) else None,
                rmse_mm=float(np.sqrt(np.mean(a*a))) if len(a) else None,
                abs_p95_mm=float(np.percentile(abs(a), 95)) if len(a) else None)


def axis_values(points, truth):
    d = np.asarray(points)-truth['center_m']
    axial = d @ np.asarray(truth['normal'])
    return truth['s_m']+axial, np.linalg.norm(d-axial[..., None]*truth['normal'], axis=-1)


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    runs = ROOT/'experiments/neck_localization/runs'
    datasets = active_stereo_datasets(runs, 'true')
    records, folders = {}, {}
    for dataset in datasets:
        condition = dataset['condition']
        records[condition] = {r['case']:r for r in json.loads(gzip.decompress(Path(dataset['records']).read_bytes()))}
        folders[condition] = Path(dataset['scenes'])
    results, depth_rows, pooled, demo = [], [], {}, {}
    for case in records['stereo_raw']:
        truth = json.loads((folders['ideal_rgb']/case/'evaluation_only/target.json').read_text())
        with np.load(folders['ideal_rgb']/case/'evaluation_only/profile.npz') as profile:
            ss, rr, mouth = profile['s_m'], profile['radius_m'], profile['centerline_m'][-1]
        lo, hi = truth['search_interval_s_m']
        metadata = json.loads((folders['ideal_rgb']/case/'observations/camera_true_parameters.json').read_text())
        cameras = [Camera(c['name'], np.array(c['K']), np.array(c['T_world_from_camera']), c['width'], c['height']) for c in metadata['cameras']]
        cached = {}
        for camera in cameras:
            name = camera.name
            ideal = np.load(folders['ideal_rgb']/case/'observations'/f'{name}_depth_z_m.npz')['depth_z_m']
            rgb = cv2.cvtColor(cv2.imread(str(folders['ideal_rgb']/case/'observations'/f'{name}_rgb.png')), cv2.COLOR_BGR2RGB)
            image_mask = cv2.imread(str(folders['ideal_rgb']/case/'foreground_masks'/f'{name}_foreground.png'), cv2.IMREAD_GRAYSCALE) > 0
            yy, xx = np.indices(ideal.shape)
            q = camera.unproject(np.c_[xx.ravel(), yy.ravel()], ideal.ravel()).reshape(*ideal.shape, 3)
            s, rho = axis_values(q, truth)
            # Diagnostic outer-surface tolerance, declared rather than used by inference.
            outer_neck = image_mask & (ideal > 0) & (s >= lo) & (s <= hi) & (abs(rho-np.interp(s, ss, rr)) <= .001)
            depths = {condition:np.load(folders[condition]/case/'observations'/f'{name}_depth_z_m.npz')['depth_z_m'] for condition in records}
            comparison = ['stereo_raw', 'stereo_spatial', 'stereo_gp', 'stereo_gp_spatial']
            common = outer_neck.copy()
            for condition in comparison:
                common &= np.isfinite(depths[condition]) & (depths[condition] > 0)
            for condition, depth in depths.items():
                valid = outer_neck & np.isfinite(depth) & (depth > 0)
                errors = (depth[common]-ideal[common])*1000
                pooled.setdefault(condition, []).append(errors)
                depth_rows.append(dict(case=case, camera=name, condition=condition,
                    reference_outer_neck_pixels=int(outer_neck.sum()), valid_pixels=int(valid.sum()),
                    valid_fraction=float(valid.sum()/max(1, outer_neck.sum())),
                    common_four_method_pixels=int(common.sum()), **error_stats(errors)))
            cached[name] = camera, ideal, rgb
        for condition, rows in records.items():
            row = rows[case]
            d = row.get('diagnostics', {})
            if 'fit_point_positions_m' not in d:
                results.append(dict(case=case, condition=condition, fit_points=0, reasons=row['reasons']))
                continue
            points = np.array(d['fit_point_positions_m'])
            ids, names = np.array(d['fit_camera_ids']), d['fit_camera_names']
            weights = np.array(d['fit_point_weights'])
            reference = np.zeros_like(points)
            pixels = {}
            for name, (camera, ideal, rgb) in cached.items():
                selected = ids == names.index(name)
                uv = np.rint(camera.project(points[selected])).astype(int)
                reference[selected] = camera.unproject(uv, ideal[uv[:, 1], uv[:, 0]])
                pixels[name] = uv
            s, _ = axis_values(points, truth)
            ref_s, ref_rho = axis_values(reference, truth)
            ref_neck = (ref_s >= lo) & (ref_s <= hi)
            ref_outer = ref_neck & (abs(ref_rho-np.interp(ref_s, ss, rr)) <= .001)
            initial_normal = np.array(d['initial_mouth_normal'])
            result = dict(case=case, condition=condition, accepted=row['accepted'], fit_points=len(points),
                measured_neck_slab_fraction=float(np.mean((s >= lo) & (s <= hi))),
                reference_neck_slab_fraction=float(ref_neck.mean()),
                reference_outer_neck_fraction=float(ref_outer.mean()),
                reference_above_neck_fraction=float(np.mean(ref_s > hi)),
                weight_share_above_neck=float(weights[ref_s > hi].sum()/weights.sum()),
                measured_axial_midpoint_offset_mm=float((np.median(s)-truth['s_m'])*1000),
                initial_mouth_normal_error_deg=float(np.degrees(np.arccos(np.clip(initial_normal@truth['normal'], -1, 1)))),
                initial_mouth_center_error_mm=float(np.linalg.norm(np.array(d['initial_mouth_center_m'])-mouth)*1000),
                local_selection_interval_from_initial_axis_m=d.get('local_selection_interval_from_initial_axis_m'),
                reasons=row['reasons'])
            results.append(result)
            if case == 'test_nominal_000':
                demo[condition] = dict(truth=truth, s_mm=(s-truth['s_m'])*1000, ref_s_mm=(ref_s-truth['s_m'])*1000,
                    rgb=cached['cam_a'][2], camera=cached['cam_a'][0], pixels=pixels['cam_a'], summary=result)
    summary = {condition:error_stats(np.concatenate(values)) for condition, values in pooled.items()}
    (output/'fit_region_records.json').write_text(json.dumps(results, indent=2)+'\n')
    (output/'depth_comparison.json').write_text(json.dumps(dict(
        scope='diagnostic only: same RGB pixels vs ideal RGB axial Z; includes registration artifacts, not native sensor accuracy',
        region='true neck axial slab AND radius within 1mm of synthetic outer profile AND image-estimated foreground',
        pairing='intersection of valid depth across raw, spatial, GP and GP+spatial at each camera/case',
        pooled=summary, records=depth_rows), indent=2)+'\n')
    keys = sorted({k for r in results for k in r})
    with (output/'fit_region_records.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(results)
    conditions = ['stereo_raw', 'stereo_spatial', 'stereo_gp', 'stereo_gp_spatial']
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.3))
    for column, condition in enumerate(conditions):
        d = demo[condition]
        t = d['truth']; cam = d['camera']; rgb = d['rgb']; uv = d['pixels']
        ring = cam.project(circle_points(np.array(t['center_m']), np.array(t['normal']), t['radius_m'], 180))
        x0, x1 = max(0, int(ring[:, 0].min())-18), min(rgb.shape[1], int(ring[:, 0].max())+19)
        y0, y1 = max(0, int(ring[:, 1].min())-22), min(rgb.shape[0], int(ring[:, 1].max())+23)
        a = axes[0, column]; a.imshow(rgb, interpolation='nearest'); a.scatter(uv[:, 0], uv[:, 1], s=3, c='#ffb45d', alpha=.6)
        a.plot(ring[:, 0], ring[:, 1], c='#58e4f5', lw=.9); a.set_xlim(x0, x1); a.set_ylim(y1, y0)
        a.set_title(condition); a.set_xticks([]); a.set_yticks([])
        a = axes[1, column]
        low, high = (np.array(t['search_interval_s_m'])-t['s_m'])*1000
        a.axvspan(low, high, color='#b6dfc6', alpha=.5, label='True neck axial slab')
        bins = np.linspace(-25, 30, 56)
        a.hist(d['ref_s_mm'], bins=bins, histtype='step', color='#347fac', label='Ideal ray at fitted pixel')
        a.hist(d['s_mm'], bins=bins, histtype='step', color='#dd8031', label='Measured fitted point')
        a.set_xlabel('Axial offset from target (mm)'); a.set_ylabel('Fitting points')
        a.set_title(f"Neck pixels {100*d['summary']['reference_neck_slab_fraction']:.1f}%")
    axes[1, 0].legend(fontsize=7)
    fig.suptitle('Evaluation-only region audit: actual selected pixels (orange), target ring (cyan)\nSame jar and frozen geometry; truth is never fed back to inference', fontsize=12)
    fig.tight_layout(); fig.savefig(output/'default_case_fit_regions.png', dpi=160); plt.close(fig)
    print(json.dumps(dict(output=str(output), fit_records=len(results), paired_depth=summary), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
