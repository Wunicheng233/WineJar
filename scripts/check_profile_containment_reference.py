#!/usr/bin/env python3
"""Evaluation-only containment controls on FOUR development jars, offline.

Exact saved neck surfaces and target axes are deliberately used as references
here. These are engineering checks of a necessary image relation, never
estimator outputs, a new success rate, threshold fitting or unseen evaluation.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

from neck_localization.geometry import Camera, Estimate, circle_points
from neck_localization.stereo_robust import extract_stereo_masks
from neck_localization.structure_portfolio import PortfolioSettings, target_contact
from neck_localization.visible_profile_baseline import profile_projection_support


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def run(development, output, figure=None, foreground_cache=None):
    tracked, records, plot_data = {}, [], []

    def read_json(path):
        tracked[str(path.resolve())] = file_hash(path)
        return json.loads(path.read_text())

    cached = None
    if foreground_cache is not None:
        cached_manifest = read_json(foreground_cache/'manifest.json')
        cached = {(e['layout'], e['case'], e['camera']):e for e in cached_manifest['entries']}
    config = read_json(development/'effective_config.json')
    names = sorted(c['name'] for c in config['cases'])
    if len(names) != 4 or not all(n.startswith('development_') for n in names):
        raise ValueError('exactly_four_development_cases_required_no_audit_cases_allowed')
    threshold = PortfolioSettings().validation_contact_pixels
    for name in names:
        geometry = development/'geometry'/name
        target = read_json(geometry/'evaluation_only'/'target.json')
        profile_path = geometry/'evaluation_only'/'profile.npz'
        tracked[str(profile_path.resolve())] = file_hash(profile_path)
        with np.load(profile_path, allow_pickle=False) as data:
            s, r, centerline = (data[k].copy() for k in ('s_m', 'radius_m', 'centerline_m'))
        sensor = read_json(geometry/'sensor.json')
        metadata = read_json(development/'stereo_raw'/'scenes'/name/'observations'/'camera_true_parameters.json')
        camera_agreement = []
        for item in sensor['cameras']:
            saved = next(x for x in metadata['cameras'] if x['name'] == item['name'])
            camera_agreement.append(dict(camera=item['name'],
                K_max_abs=float(np.max(np.abs(np.asarray(item['K_rgb'])-np.asarray(saved['K'])))),
                T_max_abs=float(np.max(np.abs(np.asarray(item['T_world_from_rgb'])-np.asarray(saved['T_world_from_camera']))))))
        frames, ideal_masks = [], []
        for item in sorted(sensor['cameras'], key=lambda x:x['name']):
            rgb_path = development/'stereo_raw'/'scenes'/name/'observations'/(item['name']+'_rgb.png')
            tracked[str(rgb_path.resolve())] = file_hash(rgb_path)
            rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
            camera = Camera(item['name'], np.array(item['K_rgb']),
                np.array(item['T_world_from_rgb']), rgb.shape[1], rgb.shape[0])
            frames.append(SimpleNamespace(camera=camera, rgb=rgb))
            ideal_path = development/'ideal_rgb'/'scenes'/name/'observations'/(item['name']+'_depth_z_m.npz')
            tracked[str(ideal_path.resolve())] = file_hash(ideal_path)
            with np.load(ideal_path, allow_pickle=False) as data:
                ideal_masks.append(data['depth_z_m'] > 0)
        masks, foreground = extract_stereo_masks(frames)
        if cached is not None:
            masks = []
            for frame in frames:
                entry = cached[('development', name, frame.camera.name)]
                if entry['status'] != 'ok':
                    raise ValueError('foreground_cache_not_ok')
                mask_path = Path(entry['mask_path'])
                tracked[str(mask_path.resolve())] = file_hash(mask_path)
                if tracked[str(mask_path.resolve())] != entry['mask_sha256']:
                    raise ValueError('foreground_cache_file_hash_changed')
                rgb_path = development/'stereo_raw'/'scenes'/name/'observations'/(frame.camera.name+'_rgb.png')
                if file_hash(rgb_path) != entry['rgb_sha256']:
                    raise ValueError('foreground_cache_RGB_hash_mismatch')
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0
                if sha256(mask.astype(np.uint8).tobytes()).hexdigest() != entry['mask_pixels_sha256']:
                    raise ValueError('foreground_cache_pixels_hash_mismatch')
                masks.append(mask)
            foreground = dict(method='frozen_SAM2_foreground_cache', cache=str(foreground_cache.resolve()),
                              manifest_sha256=file_hash(foreground_cache/'manifest.json'))
        mesh_path = geometry/'assets'/'jar.obj'
        tracked[str(mesh_path.resolve())] = file_hash(mesh_path)
        vertices = np.array([[float(x) for x in line.split()[1:4]]
                             for line in mesh_path.read_text().splitlines() if line.startswith('v ')])
        mesh_checks = []
        for frame, mask, ideal in zip(frames, masks, ideal_masks):
            q = (vertices-frame.camera.T_world_from_camera[:3, 3]) @ frame.camera.T_world_from_camera[:3, :3]
            uv = frame.camera.project(vertices[q[:, 2] > 0])
            uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] <= frame.camera.width-1) &
                    (uv[:, 1] >= 0) & (uv[:, 1] <= frame.camera.height-1)]
            info = dict(camera=frame.camera.name, projected_vertices_inside_FOV=len(uv),
                geometric_foreground_pixels=int(ideal.sum()),
                geometric_foreground_pixels_removed_by_RGB_key=int((ideal & ~mask).sum()))
            for label, check_mask in [('RGB', mask), ('ideal_silhouette', ideal)]:
                field = np.maximum(distance_transform_edt(~check_mask)-.5, 0)
                error = map_coordinates(field, [uv[:, 1], uv[:, 0]], order=1)
                info[label] = dict(maximum_outside_pixels=float(error.max()),
                    rms_outside_pixels=float(np.sqrt(np.mean(error**2))),
                    vertices_more_than_two_pixels_outside=int((error > threshold).sum()))
            mesh_checks.append(info)
        lo, hi = target['search_interval_s_m']
        grid = np.unique(np.r_[np.arange(lo, hi, .0001), hi])
        center = np.asarray(target['center_m'])
        normal = np.asarray(target['normal'])
        # The actual saved mesh uses linear meridian segments. In the physical
        # neck its centreline is straight and its cross-section frame constant.
        # Do not extend this reference into the bent shoulder transition.
        predicted_centers = center+(grid-float(target['s_m']))[:, None]*normal
        saved_centers = np.column_stack([np.interp(grid, s, centerline[:, j]) for j in range(3)])
        axis_check = float(np.max(np.linalg.norm(saved_centers-predicted_centers, axis=1)))
        if axis_check > 1e-8:
            raise ValueError('reference_interval_is_not_the_saved_straight_neck')
        reference = Estimate(center=center, normal=normal, radius=target['radius_m'],
            diagnostics=dict(axis_origin_m=center.tolist(),
                profile_s_m=(grid-float(target['s_m'])).tolist(),
                profile_radius_m=np.interp(grid, s, r).tolist()))
        positive = profile_projection_support(reference, frames, masks)
        ideal_positive = profile_projection_support(reference, frames, ideal_masks)
        old_errors, _ = target_contact(frames, masks, center*1000, normal, target['radius_m']*1000)
        # Deliberately impossible translated surface is an engineering negative
        # control only. Its size does not choose or alter any model threshold.
        negative = Estimate(center=center+np.array([.25, 0, 0]), normal=normal,
            radius=reference.radius, diagnostics=dict(reference.diagnostics))
        negative.diagnostics['axis_origin_m'] = negative.center.tolist()
        shifted = profile_projection_support(negative, frames, masks)
        plot_data.append((name, frames, masks, ideal_masks, target, positive))
        records.append(dict(case=name, evaluation_only=True,
            exact_saved_neck_profile_reference=True, reference_neck_bounds_s_m=[lo, hi],
            reference_centerline_consistency_max_m=axis_check,
            saved_camera_metadata_agreement=camera_agreement,
            true_full_mesh_projection_checks=mesh_checks,
            positive=positive, positive_pass=positive['maximum_outside_pixels'] <= threshold,
            ideal_silhouette_positive=ideal_positive,
            ideal_silhouette_positive_pass=ideal_positive['maximum_outside_pixels'] <= threshold,
            old_true_circle_contact_rms_pixels=float(np.sqrt(np.mean(old_errors**2))),
            negative_translation_world_m=[.25, 0, 0], negative=shifted,
            negative_rejected=shifted['maximum_outside_pixels'] > threshold,
            foreground=foreground))
    sources = [Path(__file__), ROOT/'neck_localization'/'visible_profile_baseline.py',
               ROOT/'neck_localization'/'structure_portfolio.py',
               ROOT/'neck_localization'/'stereo_robust.py',
               ROOT/'neck_localization'/'geometry.py']
    result = dict(schema=1, role='evaluation_only_engineering_positive_and_negative_controls',
        no_hardware=True, no_depth_or_network_inference=True, no_parameter_changes=True,
        no_audit_12_cases_read=True, development=str(development.resolve()),
        foreground_cache=str(foreground_cache.resolve()) if foreground_cache else None,
        threshold_pixels=threshold, threshold_source='unchanged PortfolioSettings.validation_contact_pixels',
        reference='saved finite physical neck meridian with exact evaluation axis; '
                  'linear between saved mesh rings, 0.1 mm axial sampling, 360 angular samples',
        scope='tests necessary RGB containment geometry only; not an estimator capability result',
        positive_pass_count=sum(r['positive_pass'] for r in records),
        ideal_silhouette_positive_pass_count=sum(r['ideal_silhouette_positive_pass'] for r in records),
        negative_rejection_count=sum(r['negative_rejected'] for r in records),
        input_sha256=tracked, source_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in sources},
        inputs_unchanged=all(file_hash(p)==h for p,h in tracked.items()), records=records)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    if figure is not None:
        if figure.exists():
            raise FileExistsError(figure)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, 3, figsize=(15, 10))
        for row, (name, frames, masks, ideal_masks, target, positive) in enumerate(plot_data):
            for col, (frame, mask, ideal) in enumerate(zip(frames, masks, ideal_masks)):
                ax = axes[row, col]
                ring = frame.camera.project(circle_points(target['center_m'], target['normal'], target['radius_m'], 360))
                ax.imshow(frame.rgb)
                ax.contour(mask, levels=[.5], colors=['#ed6cf4'], linewidths=1.)
                ax.contour(ideal, levels=[.5], colors=['#4debd1'], linewidths=.8)
                ax.plot(ring[:, 0], ring[:, 1], '--', color='#ffe265', linewidth=.8)
                ax.set_xlim(ring[:, 0].min()-18, ring[:, 0].max()+18)
                ax.set_ylim(ring[:, 1].max()+25, ring[:, 1].min()-25)
                ax.set_title(name.replace('development_', '')+' / '+frame.camera.name+
                    '\ntrue-neck max outside RGB = %.2f px'%positive['views'][col]['maximum_outside_pixels'], fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
        fig.suptitle('Evaluation-only FOUR development controls; exact cameras and neck surface\n'
            'Magenta: supplied RGB segmentation; cyan: ideal geometric silhouette; yellow dashed: true section', fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, .94])
        fig.savefig(figure, dpi=160)
        plt.close(fig)
    print(json.dumps(dict(output=str(output), positive_pass=result['positive_pass_count'],
        ideal_silhouette_positive_pass=result['ideal_silhouette_positive_pass_count'],
        negative_rejected=result['negative_rejection_count'], cases=len(records),
        maximum_positive_outside_pixels=max(r['positive']['maximum_outside_pixels'] for r in records))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--development', type=Path,
        default=ROOT/'experiments/neck_localization/runs/active_stereo_development_20261024_v3')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--figure', type=Path)
    p.add_argument('--foreground-cache', type=Path)
    a = p.parse_args()
    run(a.development, a.output, a.figure, a.foreground_cache)
