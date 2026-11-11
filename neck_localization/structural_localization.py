"""Observed multi-view rim hypotheses and short-neck structure estimation.

RGB outer upper arcs constrain a FREE 3D rim circle (unknown radius), without
noisy depth normals. The rim is initialization only, never the tying target.
All inference inputs are RGB/depth plus supplied K/T; no mesh or target labels.
"""
from dataclasses import asdict, dataclass, replace
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import QhullError

from .depth_profile import ProfileRejection
from .geometry import Estimate, basis
from .joint_profile import JointSettings
from .reliable_output import ReliabilitySettings, _attempt, _initial
from .silhouette_profile import SilhouetteSettings, observed_hull_origin
from .stereo_robust import extract_stereo_masks
from .support_feedback import WeightedAxisSplineProblem
from .task_validated_output import TaskValidationSettings, _validate_task


@dataclass(frozen=True)
class StructuralSettings:
    top_arc_height_pixels: float = 4.
    maximum_rim_pixel_rms: float = 1.2
    rim_center_search_mm: float = 45.
    maximum_axis_slope: float = .5
    radius_bounds_mm: tuple = (25., 120.)
    seed_tangents: tuple = ((0., 0.), (.15, 0.), (-.15, 0.), (0., .15), (0., -.15))
    seed: int = 2026102501


def projected_circle_conic(camera, center_mm, normal, radius_mm):
    """Conic in integer-centre pixels; world parameters use mm in this helper."""
    u, v = basis(normal)
    rotation = camera.T_world_from_camera[:3, :3]
    p = center_mm/1000-camera.T_world_from_camera[:3, 3]
    H = camera.K @ np.column_stack((rotation.T @ u, rotation.T @ v, rotation.T @ p))
    inverse = np.linalg.inv(H)
    C = inverse.T @ np.diag([1., 1., -(radius_mm/1000)**2]) @ inverse
    return C/np.linalg.norm(C)


def conic_pixel_residual(pixels, conic):
    h = np.c_[pixels, np.ones(len(pixels))]
    gradient = h @ conic
    return np.sum(gradient*h, axis=1)/np.maximum(2*np.linalg.norm(gradient[:, :2], axis=1), 1e-12)


def top_arc_pixels(mask, height_pixels):
    valid = mask.any(axis=0)
    xx = np.flatnonzero(valid)
    yy = np.argmax(mask[:, valid], axis=0).astype(float)
    keep = yy <= yy.min()+height_pixels
    # The mask locates an integer foreground sample. Its boundary lies between
    # that sample and the preceding background sample; no true edge is used.
    return np.c_[xx[keep], yy[keep]-.5]


def image_rim_seed(frames, masks=None, settings=StructuralSettings()):
    if len(frames) < 2:
        raise ProfileRejection('rim_circle_requires_multiple_calibrated_views')
    if masks is None:
        masks, _ = extract_stereo_masks(frames)
    origin, hull_top = observed_hull_origin(frames, masks,
        replace(SilhouetteSettings(), origin_below_observed_hull_top_m=.004))
    arcs = [top_arc_pixels(m, settings.top_arc_height_pixels) for m in masks]
    if min(map(len, arcs)) < 10:
        raise ProfileRejection('insufficient_observed_upper_arc')
    def residual(x):
        center = origin*1000+x[:3]
        normal = np.r_[x[3:5], 1.]; normal /= np.linalg.norm(normal)
        return np.concatenate([conic_pixel_residual(p, projected_circle_conic(f.camera, center, normal, x[5]))
                               for f, p in zip(frames, arcs)])
    limit, slope = settings.rim_center_search_mm, settings.maximum_axis_slope
    bounds = ([-limit]*3+[-slope]*2+[settings.radius_bounds_mm[0]],
              [limit]*3+[slope]*2+[settings.radius_bounds_mm[1]])
    trials = []
    for tx, ty in settings.seed_tangents:
        fit = least_squares(residual, [0., 0., 0., tx, ty, 65.], bounds=bounds,
                            loss='soft_l1', f_scale=.5, x_scale=[10, 10, 10, .1, .1, 10],
                            max_nfev=120, ftol=1e-9, xtol=1e-9, gtol=1e-7)
        trials.append(fit)
    best = min(trials, key=lambda f:f.cost)
    normal = np.r_[best.x[3:5], 1.]; normal /= np.linalg.norm(normal)
    rms = float(np.sqrt(np.mean(residual(best.x)**2)))
    detail = dict(method='multi_view_observed_upper_arc_circle', observed_hull_top_m=hull_top,
        rim_fit_pixels=[p.tolist() for p in arcs], rim_camera_names=[f.camera.name for f in frames],
        rim_pixel_rms=rms, rim_optimizer_success=bool(best.success),
        rim_radius_m=float(best.x[5]/1000), rim_normal=normal.tolist(),
        rim_center_m=(origin+best.x[:3]/1000).tolist(),
        starts=[dict(cost=float(f.cost), success=bool(f.success)) for f in trials],
        assumptions='visible upper external outline is an arc of a circular planar rim; not true for arbitrary cap/lip',
        uncertainty='not_calibrated; pixel residual is not absolute accuracy')
    if not best.success or rms > settings.maximum_rim_pixel_rms:
        raise ProfileRejection('observed_rim_arcs_inconsistent')
    return origin+best.x[:3]/1000, normal, best.x[5]/1000, detail


def prepare_from_seed(frames, masks, depth, seed, *, support_mask=None, joint_settings=None):
    """Re-select from the FULL observed station pool at each explicit seed."""
    joint = replace(joint_settings or JointSettings(), fit_depth_mm=depth)
    origin, normal, lip = seed[:3]
    clouds, ids, counts = [], [], {}
    for i, (f, mask) in enumerate(zip(frames, masks)):
        valid = mask & np.isfinite(f.depth_z_m) & (f.depth_z_m > 0)
        if support_mask is not None:
            valid &= support_mask[i]
        yy, xx = np.nonzero(valid)
        q = f.camera.unproject(np.c_[xx, yy], f.depth_z_m[yy, xx])
        keep = (q[:, 2] > .4) & (q[:, 2] < .76) & (np.linalg.norm(q[:, :2], axis=1) < .28)
        clouds.append(q[keep]); ids.append(np.full(keep.sum(), i)); counts[f.camera.name] = int(keep.sum())
    cloud, source = np.concatenate(clouds), np.concatenate(ids)
    d = cloud-origin; s = d@normal
    radial = d-s[:, None]*normal; rho = np.linalg.norm(radial, axis=1)
    cameras = np.array([f.camera.T_world_from_camera[:3, 3] for f in frames])
    view = cameras[source]-cloud; view -= (view@normal)[:, None]*normal
    facing = np.sum(radial*view, axis=1)/np.maximum(rho*np.linalg.norm(view, axis=1), 1e-12)
    keep = (s > -depth/1000) & (s < -joint.exclude_top_mm/1000) & (facing > joint.facing_cosine_min)
    selected, selected_ids = cloud[keep], source[keep]
    if len(selected) < 200:
        raise ProfileRejection('insufficient_camera_facing_local_outer_candidates')
    ss, radius = s[keep]*1000, rho[keep]*1000
    bins = np.floor(ss/joint.support_bin_mm).astype(int)
    occupied, count = np.unique(bins, return_counts=True); occupied = occupied[count >= 8]
    if len(occupied) < 8:
        raise ProfileRejection('insufficient_coarse_local_profile_support')
    medians = np.array([np.median(radius[bins == b]) for b in occupied])
    middle = int(np.argmin(medians)); left = right = middle
    while left > 0 and medians[left-1] <= medians[middle]+joint.near_min_delta_mm:
        if (occupied[left]-occupied[left-1])*joint.support_bin_mm > joint.maximum_target_gap_mm: break
        left -= 1
    while right+1 < len(occupied) and medians[right+1] <= medians[middle]+joint.near_min_delta_mm:
        if (occupied[right+1]-occupied[right])*joint.support_bin_mm > joint.maximum_target_gap_mm: break
        right += 1
    low, high = occupied[left]*joint.support_bin_mm, (occupied[right]+1)*joint.support_bin_mm
    coarse = (low+high)/2
    half = max(joint.local_half_span_mm, (high-low)/2+joint.local_context_mm)
    local = abs(ss-coarse) < half
    selected, selected_ids = selected[local], selected_ids[local]
    if len(selected) < 200:
        raise ProfileRejection('insufficient_local_neck_support')
    indices = np.concatenate([idx[np.linspace(0, len(idx)-1, min(len(idx), joint.max_fit_points_per_camera)).astype(int)]
        for i in range(len(frames)) if len(idx := np.flatnonzero(selected_ids == i))])
    held, train = indices[::4], indices[np.arange(len(indices)) % 4 != 0]
    problem = WeightedAxisSplineProblem(selected[train]*1000, origin*1000, normal, selected_ids[train], joint)
    problem.set_weights(np.ones(len(train)))
    return dict(problem=problem, joint=joint, selected=selected, ids=selected_ids, train=train, held=held,
        origin=origin, normal=normal, lip=lip, counts=counts, selection=dict(
            coarse_minimum_from_initial_axis_m=coarse/1000, broad_outer_candidates=int(keep.sum()),
            coarse_near_minimum_interval_m=[low/1000, high/1000],
            local_selection_interval_from_initial_axis_m=[(coarse-half)/1000, (coarse+half)/1000]))


def estimate_structural(frames, *, settings=StructuralSettings(), masks=None, seed_override=None):
    started = perf_counter(); attempts = []; result = Estimate()
    if masks is None:
        masks, foreground = extract_stereo_masks(frames)
    else:
        foreground = dict(method='explicit_observation_only_mask')
    detail = {}
    try:
        seed = seed_override or image_rim_seed(frames, masks, settings)
        detail = seed[3] if len(seed) > 3 else {}
        for depth in ReliabilitySettings().candidate_depths_mm:
            try:
                prep = prepare_from_seed(frames, masks, depth, seed)
                candidate, _, _ = _attempt(prep, frames, ReliabilitySettings(), 'structural_rim_seed')
                attempts.append(dict(depth_mm=depth, path='continuous', reasons=candidate.reasons))
                result = candidate
                if result.accepted: break
                fit = _initial(prep['problem'], prep['joint'])
                for label, x, ok in [('joint', fit.x, bool(fit.success)), ('rim_axis', np.zeros(4), True)]:
                    try:
                        candidate = _validate_task(prep, x, frames, TaskValidationSettings(), label, ok)
                        result = candidate
                        attempts.append(dict(depth_mm=depth, path=label, reasons=candidate.reasons))
                        if result.accepted: break
                    except (ProfileRejection, ValueError, np.linalg.LinAlgError) as error:
                        attempts.append(dict(depth_mm=depth, path=label, reasons=[str(error)]))
                if result.accepted: break
            except (ProfileRejection, ValueError, np.linalg.LinAlgError, QhullError) as error:
                attempts.append(dict(depth_mm=depth, path='preparation_or_core', reasons=[str(error)]))
    except (ProfileRejection, ValueError, np.linalg.LinAlgError, QhullError) as error:
        result.reasons = [str(error)]
    if result.center is None and attempts:
        result.reasons = attempts[-1]['reasons']
    result.diagnostics.update(method='structural_rim_seed', foreground=foreground, attempts=attempts,
                              rim_seed=detail, settings=asdict(settings))
    result.elapsed_ms = (perf_counter()-started)*1000
    return result
