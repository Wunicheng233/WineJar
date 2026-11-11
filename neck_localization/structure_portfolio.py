"""Multiple observed rim planes + image support + robust local radius fitting.

Patent-inspired hypothesis/neighbor verification, NOT a faithful patent
implementation: point-density scoring is replaced by coverage/image checks.
Thresholds are declared development assumptions for simulated stereo noise.
"""
from dataclasses import asdict, dataclass, replace
from time import perf_counter

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.optimize import least_squares
from scipy.spatial import QhullError

from .depth_profile import ProfileRejection, ProfileSettings, _mouth_center
from .geometry import Estimate, basis
from .joint_profile import JointSettings
from .stereo_robust import extract_stereo_masks
from .structural_localization import image_rim_seed, projected_circle_conic, conic_pixel_residual, top_arc_pixels
from .support_feedback import WeightedAxisSplineProblem


@dataclass(frozen=True)
class PortfolioSettings:
    random_seed: int = 2026102503
    plane_trials: int = 400
    plane_tolerance_mm: float = 1.2
    silhouette_outlier_margin_pixels: float = 2.
    maximum_seed_count: int = 4
    knot_spacing_mm: float = 3.
    smoothing_mm3: float = .01
    robust_transition_mm: float = 1.5
    profile_half_span_mm: float = 10.
    max_fit_points: int = 1800
    feedback_iterations: int = 5
    max_nfev: int = 90
    validation_depth_rms_mm: float = 4.5
    validation_depth_mean_mm: float = 2.
    validation_contact_pixels: float = 2.
    maximum_seed_arc_pixels: float = 4.
    near_min_delta_mm: float = 1.
    pixel_sigma: float = .35
    optical_depth_sigma_mm_at_800mm: float = 4.
    radial_sigma_floor_mm: float = .25


def observed_cloud(frames, masks, settings=PortfolioSettings()):
    clouds, source = [], []
    # Fixed image uncertainty margin. This rejects gross visual-hull violations,
    # not all outward errors; hard zero-margin clipping would bias radii inward.
    distance = [distance_transform_edt(~m) for m in masks]
    for i, (f, m) in enumerate(zip(frames, masks)):
        yy, xx = np.nonzero(m & (f.depth_z_m > 0) & np.isfinite(f.depth_z_m))
        q = f.camera.unproject(np.c_[xx, yy], f.depth_z_m[yy, xx])
        valid = (q[:, 2] > .4) & (q[:, 2] < .76) & (np.linalg.norm(q[:, :2], axis=1) < .28)
        for g, field in zip(frames, distance):
            uv = g.camera.project(q)
            signed = map_coordinates(field, [uv[:, 1], uv[:, 0]], order=1, mode='constant', cval=1e3)
            valid &= signed <= settings.silhouette_outlier_margin_pixels
        clouds.append(q[valid]*1000); source.append(np.full(valid.sum(), i))
    return np.concatenate(clouds), np.concatenate(source)


def seed_hypotheses(frames, masks, cloud, settings=PortfolioSettings()):
    rng = np.random.default_rng(settings.random_seed)
    arcs = [top_arc_pixels(m, 4.) for m in masks]
    seeds = []
    try:
        center, normal, radius, detail = image_rim_seed(frames, masks)
        seeds.append(dict(origin=center*1000, normal=normal, radius=radius*1000,
                          label='rgb_circle', arc_rms=detail['rim_pixel_rms']))
    except (ValueError, np.linalg.LinAlgError):
        pass
    ztop = float(np.percentile(cloud[:, 2], 99.8))
    top = cloud[(cloud[:, 2] > ztop-35) & (cloud[:, 2] < ztop+3)]
    if len(top) > 4500:
        top = top[rng.choice(len(top), 4500, replace=False)]
    candidates = []
    for depth in [8., 16., 30.]:
        upper = top[top[:, 2] > ztop-depth]
        if len(upper) < 30: continue
        for _ in range(settings.plane_trials):
            a, b, c = upper[rng.choice(len(upper), 3, replace=False)]
            n = np.cross(b-a, c-a); length = np.linalg.norm(n)
            if length < 1e-8: continue
            n /= length
            if n[2] < 0: n = -n
            if n[2] < np.cos(np.deg2rad(30)): continue
            residual = abs((top-a)@n)
            keep = residual < settings.plane_tolerance_mm
            if keep.sum() < 30: continue
            candidates.append((int(keep.sum()), a, n))
    candidates.sort(key=lambda x:-x[0])
    for count, a, n in candidates:
        if len(seeds) >= 14: break
        if any(abs((a-s['origin'])@n) < 1.5 and n@s['normal'] > np.cos(np.deg2rad(3)) for s in seeds): continue
        for _ in range(3):
            points = top[abs((top-a)@n) < settings.plane_tolerance_mm]
            if len(points) < 30: break
            a = np.median(points, axis=0)
            centered = points-points.mean(axis=0)
            _, _, vt = np.linalg.svd(centered, full_matrices=False); n = vt[-1]
            if n[2] < 0: n = -n
        try:
            o, radius = _mouth_center(a/1000, n, points/1000)
        except (ValueError, QhullError, np.linalg.LinAlgError): continue
        o *= 1000; radius *= 1000
        conics = [projected_circle_conic(f.camera, o, n, radius) for f in frames]
        error = np.concatenate([conic_pixel_residual(p, c) for p, c in zip(arcs, conics)])
        arc = float(np.sqrt(np.mean(np.minimum(error**2, 100))))
        seeds.append(dict(origin=o, normal=n, radius=radius, label='depth_plane', arc_rms=arc,
                          plane_support=count))
    seeds.sort(key=lambda s:s['arc_rms'])
    return [s for s in seeds if s['arc_rms'] <= settings.maximum_seed_arc_pixels][:settings.maximum_seed_count]


def coordinates(points, origin, normal):
    d = points-origin; s = d@normal
    radial = d-s[:, None]*normal
    return s, np.linalg.norm(radial, axis=1), radial


def coarse_interval(s, radius, weights=None):
    grid = np.arange(float(np.percentile(s, 1))+1, float(np.percentile(s, 99))-1, .5)
    values = []
    for v in grid:
        use = abs(s-v) < 1.25
        if use.sum() < 10:
            values.append(np.nan)
        elif weights is None:
            values.append(np.median(radius[use]))
        else:
            r, w = radius[use], weights[use]
            order = np.argsort(r); r, w = r[order], w[order]
            values.append(r[min(np.searchsorted(np.cumsum(w), w.sum()/2), len(r)-1)])
    values = np.asarray(values); good = np.isfinite(values)
    if good.sum() < 10: raise ProfileRejection('insufficient_coarse_stereo_profile')
    values = np.interp(grid, grid[good], values[good])
    i = int(np.argmin(values)); left = right = i
    while left > 0 and values[left-1] <= values[i]+1: left -= 1
    while right+1 < len(grid) and values[right+1] <= values[i]+1: right += 1
    return float((grid[left]+grid[right])/2), [float(grid[left]), float(grid[right])]


def target_contact(frames, masks, center, normal, radius):
    """Two cylinder-generator contacts per view at a stationary neck section.

    This is an approximation when the chosen midpoint has nonzero r'(s).
    It uses RGB silhouette ONLY, not observed depth or target truth.
    """
    errors, pixels = [], []
    for f, m in zip(frames, masks):
        d = f.camera.T_world_from_camera[:3, 3]*1000-center
        d -= (d@normal)*normal; length = np.linalg.norm(d)
        if length <= radius: raise ProfileRejection('camera_inside_candidate_circle')
        u = d/length; v = np.cross(normal, u)
        e = radius/length*u
        e = np.array([e+np.sqrt(1-(radius/length)**2)*v, e-np.sqrt(1-(radius/length)**2)*v])
        uv = f.camera.project((center+radius*e)/1000)
        # Signed distance sampled on a display-independent binary RGB mask.
        field = distance_transform_edt(~m)-distance_transform_edt(m)
        values = map_coordinates(field, [uv[:, 1], uv[:, 0]], order=1, mode='constant', cval=100.)
        # Binary pixel centres have |distance|>=1. Boundary zero is half-way.
        values -= .5*np.sign(values)
        errors.extend(values); pixels.append(uv.tolist())
    return np.asarray(errors), pixels


def radial_variance(points, ids, frames, origin, normal, slopes, settings):
    """Propagate (u,v,Z) measurement scales to rho-r(s), in mm^2.

    Scales are declared simulation assumptions; independence between points is
    NOT implied and no covariance/confidence is claimed from this weighting.
    """
    s, rho, radial = coordinates(points, origin, normal)
    gradient = radial/np.maximum(rho[:, None], 1e-9)-slopes[:, None]*normal
    variance = np.full(len(points), settings.radial_sigma_floor_mm**2)
    for i, f in enumerate(frames):
        use = ids == i
        rotation = f.camera.T_world_from_camera[:3, :3]
        camera = f.camera.T_world_from_camera[:3, 3]*1000
        local = (points[use]-camera)@rotation
        z = local[:, 2]
        du = rotation[:, 0]*z[:, None]/f.camera.K[0, 0]
        dv = rotation[:, 1]*z[:, None]/f.camera.K[1, 1]
        dz = (points[use]-camera)/z[:, None]
        g = gradient[use]
        sigma_z = settings.optical_depth_sigma_mm_at_800mm*(z/800)**2
        variance[use] += (np.sum(g*du, axis=1)**2+np.sum(g*dv, axis=1)**2)*settings.pixel_sigma**2
        variance[use] += np.sum(g*dz, axis=1)**2*sigma_z**2
    return variance


def fit_seed(frames, masks, cloud, source, seed, settings, robust=True, feedback=True, anisotropic=False, fixed_axis=False):
    origin, normal = seed['origin'], seed['normal']
    s, rho, radial = coordinates(cloud, origin, normal)
    cp = np.array([f.camera.T_world_from_camera[:3, 3]*1000 for f in frames])
    view = cp[source]-cloud; view -= (view@normal)[:, None]*normal
    facing = np.sum(radial*view, axis=1)/np.maximum(rho*np.linalg.norm(view, axis=1), 1e-9)
    keep = (s > -45) & (s < -2) & (rho > 20) & (rho < 120) & (facing > (.03 if anisotropic else .35))
    initial_weights = 1/radial_variance(cloud, source, frames, origin, normal, np.zeros(len(cloud)), settings) if anisotropic else None
    middle, interval = coarse_interval(s[keep], rho[keep], initial_weights[keep] if anisotropic else None)
    half = max(settings.profile_half_span_mm, (interval[1]-interval[0])/2+3)
    keep &= abs(s-middle) < half
    indices = np.flatnonzero(keep)
    if len(indices) < 150: raise ProfileRejection('insufficient_observed_local_stereo_points')
    if len(indices) > settings.max_fit_points:
        indices = indices[np.linspace(0, len(indices)-1, settings.max_fit_points).astype(int)]
    # Hold out contiguous native/RGB image tiles, rather than interleaved points.
    block = []
    for i in indices:
        uv = frames[source[i]].camera.project(cloud[i:i+1]/1000)[0]
        block.append(int(np.floor(uv[0]/5)+3*np.floor(uv[1]/5)) % 4)
    held = np.asarray(block) == 0; training = ~held
    points, ids = cloud[indices], source[indices]
    joint = replace(JointSettings(), knot_spacing_mm=settings.knot_spacing_mm,
        smoothing_mm3=settings.smoothing_mm3, max_nfev=settings.max_nfev)
    problem = WeightedAxisSplineProblem(points[training], origin, normal, ids[training], joint)
    x = np.zeros(4); weights = np.ones(training.sum()); fit = None
    if anisotropic:
        weights = initial_weights[indices][training]; weights /= weights.mean()
    for iteration in range(settings.feedback_iterations if robust or feedback else 1):
        axis_used_weights = weights.copy()
        problem.set_weights(weights)
        bounds = np.array([15., 15., .22, .22])
        if not fixed_axis:
            fit = least_squares(problem.fun, x, jac=problem.jac, bounds=(-bounds, bounds),
                x_scale=[5, 5, .1, .1], max_nfev=settings.max_nfev, ftol=1e-7, xtol=1e-7, gtol=1e-6)
            x = fit.x
        state = problem.evaluate(x)
        weights = np.ones(len(problem.points))
        if anisotropic:
            variance = radial_variance(problem.points, ids[training], frames, state['origin'], state['normal'],
                problem.derivative(state['s'])@state['beta'], settings)
            weights = 1/variance; weights /= weights.mean()
        if robust:
            weights *= np.minimum(1, settings.robust_transition_mm/np.maximum(abs(state['radial_residual']), 1e-9))
        if feedback:
            mid, segment = coarse_interval(state['s'], np.linalg.norm(problem.points-state['origin']-state['s'][:, None]*state['normal'], axis=1), weights if anisotropic else None)
            distance = np.maximum(abs(state['s']-mid)-max(2., (segment[1]-segment[0])/2), 0)
            weights *= .02+.98*np.exp(-.5*(distance/2.5)**2)
    state = problem.evaluate(x); a, n = state['origin'], state['normal']
    # Profile readout uses robust full local context at the final axis. Axis
    # feedback weights must not erase the lips/transitions defining the interval.
    read = WeightedAxisSplineProblem(points[training], a, n, ids[training], joint)
    read_weights = 1/radial_variance(points[training], ids[training], frames, a, n, np.zeros(training.sum()), settings) if anisotropic else np.ones(training.sum())
    read_weights /= read_weights.mean()
    for _ in range(4 if robust else 1):
        read_used_weights = read_weights.copy()
        read.set_weights(read_weights); st = read.evaluate(np.zeros(4))
        read_weights = np.minimum(1., settings.robust_transition_mm/np.maximum(abs(st['radial_residual']), 1e-9))
        if anisotropic:
            variance = radial_variance(points[training], ids[training], frames, a, n, read.derivative(st['s'])@st['beta'], settings)
            base = 1/variance; base /= base.mean(); read_weights *= base
    axial = (points-a)@n
    grid = np.arange(float(np.percentile(axial, 2))+.5, float(np.percentile(axial, 98))-.5, .1)
    values = read.basis(grid)@st['beta']
    i = int(np.argmin(values)); left = right = i
    while left > 0 and values[left-1] <= values[i]+settings.near_min_delta_mm: left -= 1
    while right+1 < len(grid) and values[right+1] <= values[i]+settings.near_min_delta_mm: right += 1
    midpoint = float((grid[left]+grid[right])/2); radius = float(read.basis(midpoint)@st['beta'])
    center = a+midpoint*n
    residual = np.linalg.norm(points-a-axial[:, None]*n, axis=1)-read.basis(axial)@st['beta']
    relevant = held & (axial >= grid[left]-1.5) & (axial <= grid[right]+1.5)
    if relevant.sum() < 15: raise ProfileRejection('insufficient_spatial_block_validation_support')
    rms = float(np.sqrt(np.mean(residual[relevant]**2))); mean = float(np.mean(residual[relevant]))
    contacts, pixel_contacts = target_contact(frames, masks, center, n, radius)
    contact_rms = float(np.sqrt(np.mean(contacts**2)))
    result = Estimate(center=center/1000, normal=n, radius=radius/1000, residual=rms/1000)
    if fit is not None and not fit.success: result.reasons.append('robust_stereo_optimizer_not_converged')
    if left == 0 or right == len(grid)-1: result.reasons.append('near_minimum_touches_observed_boundary')
    if grid[right]-grid[left] < 3: result.reasons.append('insufficient_three_mm_band_span')
    if rms > settings.validation_depth_rms_mm or abs(mean) > settings.validation_depth_mean_mm:
        result.reasons.append('spatial_block_depth_mismatch')
    if contact_rms > settings.validation_contact_pixels: result.reasons.append('candidate_inconsistent_with_rgb_silhouette')
    if np.any(abs(x) > .98*np.array([15, 15, .22, .22])): result.reasons.append('axis_parameter_hits_bound')
    if not 20 < radius < seed['radius']-1: result.reasons.append('candidate_not_narrower_than_observed_lip')
    result.accepted = not result.reasons
    result.diagnostics.update(method='structure_portfolio', chosen_seed=seed['label'],
        seed_arc_rms_pixels=seed['arc_rms'], initial_mouth_center_m=(origin/1000).tolist(), initial_mouth_normal=normal.tolist(),
        axis_origin_m=(a/1000).tolist(), near_min_interval_from_axis_origin_m=[grid[left]/1000, grid[right]/1000],
        near_min_interval_length_m=float((grid[right]-grid[left])/1000),
        profile_s_m=(grid/1000).tolist(), profile_radius_m=(values/1000).tolist(),
        fit_point_positions_m=(points[training]/1000).tolist(), fit_camera_ids=ids[training].tolist(),
        fit_point_weights=axis_used_weights.tolist(), fit_camera_names=[f.camera.name for f in frames],
        heldout_points_m=(points[held]/1000).tolist(), heldout_rms_mm=rms, heldout_mean_mm=mean,
        target_contact_rms_pixels=contact_rms, target_contact_pixels=pixel_contacts,
        selection_score=rms/settings.validation_depth_rms_mm+contact_rms/settings.validation_contact_pixels,
        uncertainty='not_calibrated', camera_trust=[1.]*len(frames), robust=robust, feedback=feedback,
        weight_semantics='axis IRLS weights at final solve; readout has separate robust full-context weights',
        readout_weights=read_used_weights.tolist())
    result.diagnostics.update(anisotropic=anisotropic, fixed_axis=fixed_axis)
    return result


def estimate_portfolio(frames, *, settings=PortfolioSettings(), robust=True, feedback=True, multi=True, anisotropic=False, fixed_axis=False):
    started = perf_counter(); attempts = []; result = Estimate()
    masks, foreground = extract_stereo_masks(frames)
    try:
        cloud, source = observed_cloud(frames, masks, settings)
        if len(cloud) < 300: raise ProfileRejection('insufficient_multiview_consistent_observed_cloud')
        seeds = seed_hypotheses(frames, masks, cloud, settings)
        if not multi: seeds = seeds[:1]
        candidates = []
        for seed in seeds:
            try:
                candidate = fit_seed(frames, masks, cloud, source, seed, settings, robust, feedback, anisotropic, fixed_axis)
                candidates.append(candidate)
                attempts.append(dict(seed=seed['label'], arc_rms_pixels=seed['arc_rms'], accepted=candidate.accepted,
                    reasons=candidate.reasons, center_m=candidate.center.tolist(), normal=candidate.normal.tolist(),
                    radius_m=candidate.radius, score=candidate.diagnostics['selection_score']))
            except (ValueError, np.linalg.LinAlgError, QhullError) as error:
                attempts.append(dict(seed=seed['label'], accepted=False, reasons=[str(error)]))
        if candidates:
            result = min(candidates, key=lambda c:(not c.accepted, c.diagnostics['selection_score']))
        else:
            result.reasons = ['no_valid_structural_candidate']
        if not seeds: result.reasons = ['no_observed_rim_hypothesis']
        result.diagnostics['station_consistent_points'] = len(cloud)
    except (ValueError, np.linalg.LinAlgError, QhullError) as error:
        result.reasons = [str(error)]
    result.diagnostics.update(method='structure_portfolio', foreground=foreground, attempts=attempts,
        settings=asdict(settings), multi=multi, robust=robust, feedback=feedback,
        uncertainty='not_calibrated; noise-scale gates and agreement are not absolute error bounds')
    result.elapsed_ms = (perf_counter()-started)*1000
    return result
