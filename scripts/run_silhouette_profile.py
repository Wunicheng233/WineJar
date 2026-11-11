#!/usr/bin/env python3
"""Four DEVELOPMENT cases: RGB-only SOR baseline and separately labelled oracles."""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from neck_localization.evaluation import score
from neck_localization.geometry import Camera
from neck_localization.silhouette_profile import (
    SilhouetteImage, SilhouetteSettings, estimate_silhouette_profile,
)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def load_rgb_only(folder):
    metadata = json.loads((folder/'camera_true_parameters.json').read_text())
    images = []
    for item in metadata['cameras']:
        camera = Camera(item['name'], np.array(item['K']),
                        np.array(item['T_world_from_camera']), item['width'], item['height'])
        rgb = cv2.cvtColor(cv2.imread(str(folder/(camera.name+'_rgb.png'))), cv2.COLOR_BGR2RGB)
        images.append(SilhouetteImage(camera, rgb))
    return images


def visualize(images, estimates, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 4, figsize=(15, 10), constrained_layout=True)
    colors = ['tab:blue', 'tab:green', 'tab:orange']
    for row, (label, estimate) in enumerate(estimates):
        detail = estimate.diagnostics
        points = np.array(detail.get('fit_pixels', [])).reshape(-1, 2)
        ids = np.array(detail.get('fit_camera_ids', []))
        for index, image in enumerate(images):
            ax = axes[row, index]
            ax.imshow(image.rgb)
            q = points[ids == index]
            if len(q):
                ax.scatter(q[:, 0], q[:, 1], s=5, c='tab:red')
            # Observation-derived crop only, never a target-truth crop.
            foreground = detail.get('foreground', {}).get('cameras', [])
            if len(foreground) > index and foreground[index]['bbox_xyxy'] is not None:
                x0, y0, x1, y1 = foreground[index]['bbox_xyxy']
                middle, half_width = (x0+x1)/2, max(90, min(140, (x1-x0)*.4))
                ax.set_xlim(middle-half_width, middle+half_width)
                ax.set_ylim(min(image.camera.height, y0+85), y0-8)
            ax.set_title(f'{label}: {image.camera.name}', fontsize=9)
        ax = axes[row, 3]
        s = np.array(detail.get('axial_samples_mm', []))
        radius = np.array(detail.get('radial_samples_mm', []))
        if len(s):
            for index, image in enumerate(images):
                ax.scatter(s[ids == index], radius[ids == index], s=7, color=colors[index], label=image.camera.name)
            ax.plot(detail['profile_grid_axial_mm'], detail['profile_radius_mm'], c='k', lw=1)
            interval = np.array(detail['near_min_interval_from_axis_origin_m'])*1000
            ax.axvspan(*interval, color='gold', alpha=.25)
            ax.legend(fontsize=7)
        ax.set(xlabel='Axial coordinate from fitted origin (mm)', ylabel='Radius (mm)',
               title='Accepted' if estimate.accepted else 'REJECTED: '+','.join(estimate.reasons))
        ax.title.set_fontsize(8)
    fig.suptitle('RGB contour SOR probe: primary result and ORACLE DIAGNOSTICS (not algorithm scores)')
    fig.savefig(output, dpi=150)
    plt.close(fig)


def run(folder, output):
    config = json.loads((folder/'effective_config.json').read_text())
    if 'development' not in config['dataset_split'] or len(config['cases']) != 4:
        raise ValueError('This bounded runner requires exactly four development cases')
    output.mkdir(parents=True, exist_ok=False)
    settings = SilhouetteSettings()
    records = []
    for case in config['cases']:
        name = case['name']
        source = folder/'ideal_rgb/scenes'/name
        images = load_rgb_only(source/'observations')
        # Neither depth, diagnostic masks, model assets nor target metadata are
        # loaded until AFTER primary inference. RGB is common across depth modes.
        primary = estimate_silhouette_profile(images, settings)
        truth = json.loads((source/'evaluation_only/target.json').read_text())
        axis = (np.array(truth['center_m']), np.array(truth['normal']))
        oracle_axis = estimate_silhouette_profile(images, settings, diagnostic_axis=axis)
        axial = np.array(truth['search_interval_s_m'])-truth['s_m']
        oracle_axis_region = estimate_silhouette_profile(
            images, settings, diagnostic_axis=axis, diagnostic_axial_interval_from_anchor_m=axial)
        estimates = [('PRIMARY RGB only', primary), ('ORACLE axis', oracle_axis),
                     ('ORACLE axis + neck region', oracle_axis_region)]
        for label, estimate in estimates:
            row = dict(case=name, method=label, diagnostic_oracle=label.startswith('ORACLE'),
                       **score(estimate, truth))
            row['diagnostics'] = estimate.diagnostics
            records.append(row)
            print(name, label, row['accepted'], row.get('center_mm'), row.get('normal_deg'),
                  row.get('diameter_mm'), row['reasons'], flush=True)
        visualize(images, estimates, output/(name+'.png'))
        write_json(output/(name+'_input_contract.json'), dict(
            rgb_sha256={im.camera.name:sha256(im.rgb.tobytes()).hexdigest() for im in images},
            primary_input_files=[str(source/'observations/camera_true_parameters.json')]+
                               [str(source/'observations'/(im.camera.name+'_rgb.png')) for im in images],
            depth_files_read=False, gt_masks_read=False, target_truth_loaded_after_primary=True))
    write_json(output/'records.json', records)
    primary = [row for row in records if not row['diagnostic_oracle']]
    summary = dict(n=len(primary), accepted=sum(row['accepted'] for row in primary),
                   correct_outputs=sum(row['accepted'] and row['truth_pass'] for row in primary),
                   rejected=sum(not row['accepted'] for row in primary),
                   wrong_accepts=sum(row['accepted'] and not row['truth_pass'] for row in primary))
    for metric in ['center_mm', 'normal_deg', 'diameter_mm', 'runtime_ms']:
        values = [row[metric] for row in primary if row[metric] is not None]
        summary[metric] = {name:float(np.percentile(values, percentile)) for name, percentile in
                           [('median', 50), ('p95', 95), ('maximum', 100)]} if values else None
    write_json(output/'primary_summary.json', summary)
    sources = [ROOT/'neck_localization'/name for name in
               ['silhouette_profile.py', 'stereo_robust.py', 'geometry.py', 'evaluation.py']]+[Path(__file__)]
    snapshot = output/'source_snapshot'
    snapshot.mkdir()
    for path in sources:
        (snapshot/path.name).write_bytes(path.read_bytes())
    write_json(output/'provenance.json', dict(
        input=str(folder.resolve()), split='four_previously_viewed_development_cases', settings=asdict(settings),
        source_sha256={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in sources},
        scope='negative_or_positive_bounded_probe_not_a_frozen_final_test',
        oracles='GT axis and optional GT AXIAL neck interval; never counted as primary results',
        limitations=['true camera calibration', 'blue-key mask', 'pixel-derived contour tangent',
                     'occluding lip edges need not be smooth SOR generators', 'no calibrated uncertainty']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'experiments/neck_localization/runs/active_stereo_development_20261024_v3')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.run, args.output)
