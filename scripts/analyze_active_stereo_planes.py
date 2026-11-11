#!/usr/bin/env python3
"""Offline stereo-matcher diagnosis from saved plane IR; no render or hardware.

Ground-truth depth is used ONLY to diagnose correspondence and measurement error.
CPU costs are a mechanism probe, not a replacement for the production estimator.
Optional GPU replay uses a single upstream SimSense matcher at a time.
"""
import argparse
import gc
from hashlib import sha256
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / 'experiments/neck_localization/runs/active_stereo_planes_20261024_v2'
ROI = np.s_[160:320, 300:550]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def stats(values):
    values = np.asarray(values)
    return dict(mean=float(np.mean(values)), median=float(np.median(values)),
                rms=float(np.sqrt(np.mean(values**2))),
                p95_abs=float(np.percentile(abs(values), 95)),
                maximum_abs=float(np.max(abs(values))))


def load_case(folder, name):
    arrays = dict(np.load(folder / (name + '.npz')))
    ir = [cv2.imread(str(folder / (name + f'_ir_{i}.png')), cv2.IMREAD_GRAYSCALE)
          for i in range(2)]
    if any(im is None for im in ir):
        raise FileNotFoundError(name + ' IR pair missing')
    return arrays, ir


def census(im):
    """7x7 center-symmetric descriptor, matching upstream comparison and bit order."""
    h, w = im.shape
    padded = np.pad(im, 3)
    out = np.zeros_like(im, np.uint32)
    for i in range(4):
        for j in range(3 if i == 3 else 7):
            a = padded[i:i+h, j:j+w]
            b = padded[6-i:6-i+h, 6-j:6-j+w]
            out |= (a >= b).astype(np.uint32) << (i*7+j)
    return out


def cpu_block_disparity(left, right):
    """Integer CS-Census costs + block sum + parabola; NO SGM/noise/median/LR."""
    cl, cr = census(left), census(right)
    disparities = np.arange(10, 51)
    costs = []
    for disparity in disparities:
        # ROI is far enough from boundaries that roll/pad cannot contaminate it.
        cost = np.bitwise_count(cl ^ np.roll(cr, disparity, axis=1)).astype(np.float32)
        costs.append(cv2.boxFilter(cost, -1, (7, 7), normalize=False)[ROI])
    costs = np.stack(costs, axis=-1)
    index = np.argmin(costs, axis=-1)
    if np.any((index == 0) | (index == len(disparities)-1)):
        raise ValueError('Diagnostic disparity search reached its boundary')
    pick = lambda offset: np.take_along_axis(costs, (index+offset)[..., None], -1)[..., 0]
    a, b, c = pick(-1), pick(0), pick(1)
    denominator = a-2*b+c
    if np.any(denominator <= 0):
        raise ValueError('Invalid local parabola')
    return disparities[index] - .5*(c-a)/denominator


def photometric_curve(left, right, true_disparity):
    """Known correspondence warp + unknown constant disparity increment.

    Manual floating-point interpolation avoids OpenCV remap's 1/32-pixel table.
    This uses ground truth deliberately; it is not an inference method or score.
    """
    yy, xx = np.mgrid[160:320, 300:550]
    reference = left[ROI].astype(float)
    centered_reference = reference-reference.mean()
    deltas = np.linspace(-.75, .75, 301)
    ssd, ncc = [], []
    for delta in deltas:
        position = xx-true_disparity-delta
        low = np.floor(position).astype(int)
        fraction = position-low
        warped = (1-fraction)*right[yy, low] + fraction*right[yy, low+1]
        ssd.append(np.mean((reference-warped)**2))
        centered = warped-warped.mean()
        ncc.append(np.sum(centered_reference*centered) /
                   (np.linalg.norm(centered_reference)*np.linalg.norm(centered)))
    return deltas, np.array(ssd), np.array(ncc)


def native_plane_error(z, camera, distance, tilt_deg):
    yy, xx = np.mgrid[160:320, 300:550]
    rays = np.stack([xx, yy, np.ones_like(xx)], axis=-1) @ np.linalg.inv(camera['K_ir']).T
    transform = np.array(camera['T_world_from_left_ir'])
    points = (rays*z[..., None]) @ transform[:3, :3].T + transform[:3, 3]
    angle = np.deg2rad(tilt_deg)
    normal = np.array([np.cos(angle), np.sin(angle), 0])
    return (points-np.array([distance, 0, 0])) @ normal * 1000


def replay_matcher(folder, names, output, camera, seed):
    import inspect
    import sapien
    from sapien.sensor import StereoDepthSensorConfig
    from sapien.sensor.simsense_component import SimSenseComponent

    config = StereoDepthSensorConfig('D435')
    np.testing.assert_allclose(np.array(camera['K_ir'])[:2, 2] + .5,
                               config.ir_intrinsic[:2, 2], atol=1e-6)
    np.testing.assert_allclose(config.ir_intrinsic[:2, :2],
                               np.array(camera['K_ir'])[:2, :2], atol=1e-6)
    fb = camera['K_ir'][0][0]*camera['baseline_m']
    scene, entity = sapien.Scene(), sapien.Entity()
    scene.add_entity(entity)
    rows, saved = [], {}
    try:
        for median in [1, 3]:
            for noise in [0, 1]:
                keys = ['min_depth', 'max_depth', 'ir_noise_seed', 'ir_speckle_noise',
                        'ir_thermal_noise', 'rectified', 'census_width', 'census_height',
                        'max_disp', 'block_width', 'block_height', 'p1_penalty',
                        'p2_penalty', 'uniqueness_ratio', 'lr_max_diff', 'median_filter_size']
                kwargs = {key: getattr(config, key) for key in keys}
                kwargs.update(ir_noise_seed=seed, ir_speckle_noise=float(noise),
                              ir_thermal_noise=float(noise), median_filter_size=median,
                              depth_dilation=False)
                matcher = SimSenseComponent(
                    rgb_resolution=config.ir_resolution, ir_resolution=config.ir_resolution,
                    rgb_intrinsic=config.ir_intrinsic, ir_intrinsic=config.ir_intrinsic,
                    trans_pose_l=sapien.Pose(),
                    trans_pose_r=config.trans_pose_l.inv()*config.trans_pose_r, **kwargs)
                entity.add_component(matcher)
                try:
                    for name in names:
                        arrays, (left, right) = load_case(folder, name)
                        matcher.compute(left, right)
                        depth = matcher.get_ndarray().copy()
                        true_z, z = arrays['ideal_left_z'][ROI], depth[ROI]
                        good = (true_z > 0) & (z > 0)
                        row = dict(case=name, median=median, internal_ir_noise=bool(noise),
                                   valid_fraction=float(np.mean(good)),
                                   disparity_error_px=stats((fb/z-fb/true_z)[good]),
                                   optical_z_error_mm=stats((z-true_z)[good]*1000),
                                   max_abs_difference_from_saved_native_m=float(
                                       np.max(abs(depth-arrays['native_float_z']))))
                        rows.append(row)
                        saved[f'{name}_median{median}_noise{noise}'] = z
                        print(json.dumps(row), flush=True)
                finally:
                    entity.remove_component(matcher)
                    del matcher
                    gc.collect()
    finally:
        scene.remove_entity(entity)
    write_json(output/'matcher_replay.json', rows)
    np.savez_compressed(output/'matcher_replay_roi.npz', **saved)
    source = Path(inspect.getfile(SimSenseComponent))
    write_json(output/'matcher_replay_contract.json', dict(
        input='saved pre-noise uint8 PNG IR pairs; same first-channel truncation as CUDA float2uint8',
        noise='upstream gamma shape1333.33, scale1/1333.33; additive Gaussian sigma0.25 uint8 levels; off skips both',
        rng='fresh matcher per condition, original seed, original six-case order',
        scope='PNG replay, not a rerender; equality to saved original checked rather than assumed',
        max_disp=config.max_disp, source=str(source), source_sha256=sha256(source.read_bytes()).hexdigest()))


def run(folder, output, gpu_replay=False):
    output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((folder/'sensor.json').read_text())
    camera = metadata['cameras'][0]
    records = json.loads((folder/'records.json').read_text())
    cases = [r for r in records if r['mode'] == 'ideal_rgb']
    fb = camera['K_ir'][0][0]*camera['baseline_m']
    summaries, curves, maps = [], {}, {}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for case, ax in zip(cases, axes.flat):
        name = case['case']
        arrays, (left, right) = load_case(folder, name)
        true_z, z = arrays['ideal_left_z'][ROI], arrays['native_float_z'][ROI]
        if not np.all((true_z > 0) & (z > 0)):
            raise ValueError('Current diagnostic expects fully valid interior ROI')
        true_d, observed_d = fb/true_z, fb/z
        delta, ssd, ncc = photometric_curve(left, right, true_d)
        cpu_d = cpu_block_disparity(left, right)
        row = dict(case=name, true_disparity_px=stats(true_d),
                   observed_disparity_px=stats(observed_d),
                   native_disparity_error_px=stats(observed_d-true_d),
                   native_optical_z_error_mm=stats((z-true_z)*1000),
                   native_plane_error_mm=stats(native_plane_error(
                       z, camera, case['distance_m'], case['tilt_deg'])),
                   cpu_census_no_sgm_disparity_error_px=stats(cpu_d-true_d),
                   photometric_ssd_optimum_delta_px=float(delta[np.argmin(ssd)]),
                   photometric_ncc_optimum_delta_px=float(delta[np.argmax(ncc)]),
                   photometric_peak_ncc=float(np.max(ncc)))
        summaries.append(row)
        curves[name] = dict(delta_px=delta.tolist(), mean_ssd=ssd.tolist(), ncc=ncc.tolist())
        maps[name+'_true_disparity'] = true_d
        maps[name+'_native_disparity'] = observed_d
        maps[name+'_cpu_census_no_sgm'] = cpu_d
        ax.plot(delta, ssd, label='Clean IR photometric SSD')
        ax.axvline(0, color='k', ls=':', label='Known geometry')
        ax.axvline(np.median(observed_d-true_d), color='tab:red',
                   label='Saved matcher median error')
        ax.axvline(np.median(cpu_d-true_d), color='tab:green', ls='--',
                   label='CPU Census block, no SGM')
        ax.set(title=name, xlabel='Additional disparity (px)', ylabel='Mean SSD (uint8 levels squared)')
    axes.flat[0].legend(fontsize=7)
    fig.suptitle('Known-plane diagnostic: correct IR correspondence vs stereo-matcher bias')
    fig.savefig(output/'photometric_costs.png', dpi=160)
    plt.close(fig)
    write_json(output/'summary.json', summaries)
    write_json(output/'photometric_curves.json', curves)
    np.savez_compressed(output/'diagnostic_roi_maps.npz', **maps)
    write_json(output/'inputs.json', dict(
        input_folder=str(folder.resolve()), roi=[160, 320, 300, 550],
        file_sha256={f.name: sha256(f.read_bytes()).hexdigest()
                     for f in sorted(folder.iterdir()) if f.is_file()},
        script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Known-plane oracle diagnostic only; no estimator receives truth; no real D435i accuracy claim'))
    assert max(abs(r['photometric_ssd_optimum_delta_px']) for r in summaries) < .02
    assert max(abs(r['photometric_ncc_optimum_delta_px']) for r in summaries) < .02
    if gpu_replay:
        replay_matcher(folder, [r['case'] for r in cases], output, camera, metadata['settings']['seed'])
    return summaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay-matcher', action='store_true', help='Optional one-GPU-matcher replay, no rendering')
    args = parser.parse_args()
    run(args.input, args.output, args.replay_matcher)
