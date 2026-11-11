"""Observation-only correction of the portfolio's RGB validation geometry.

The fitted *finite surface* must project inside the whole object's foreground;
an interior neck section need not touch that foreground's outer boundary.
This is an engineering correctness baseline, not a new fitting algorithm or a
calibrated guarantee of pose accuracy. The frozen fitter is left unchanged.
"""
from copy import deepcopy
from dataclasses import asdict
from time import perf_counter

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.spatial import QhullError

from .depth_profile import ProfileRejection
from .geometry import Estimate, basis
from .stereo_robust import extract_stereo_masks
from .structure_portfolio import (
    PortfolioSettings, fit_seed, observed_cloud, seed_hypotheses,
)


_OLD_CONTACT_REASON = 'candidate_inconsistent_with_rgb_silhouette'


def profile_projection_support(candidate, frames, masks, *, angle_count=360):
    """Check one-sided containment of a sampled finite revolution surface.

    Use all saved axial samples (the frozen readout uses 0.1 mm spacing) and
    one-degree angular steps. Interior samples have zero error. Back-facing or
    occluded samples also must be contained in the whole silhouette, but are
    never required to coincide with a visible image edge. Max outside distance
    is the containment statistic; RMS is used only in candidate ranking.

    Front-facing/local-z-buffer counts are diagnostic only: a finite local
    model cannot decide whether the actual lip/shoulder occludes that surface.
    """
    d = candidate.diagnostics
    axial = np.asarray(d['profile_s_m'], float)
    radius = np.asarray(d['profile_radius_m'], float)
    origin = np.asarray(d['axis_origin_m'], float)
    normal = np.asarray(candidate.normal, float)
    if (axial.ndim != 1 or len(axial) < 2 or radius.shape != axial.shape or
            not np.isfinite(np.r_[axial, radius, origin, normal]).all() or
            np.linalg.norm(normal) < 1e-12 or
            np.any(radius <= 0) or np.any(np.diff(axial) <= 0)):
        raise ProfileRejection('invalid_fitted_profile_for_projection')
    normal = normal / np.linalg.norm(normal)
    u, v = basis(normal)
    theta = np.arange(angle_count) * (2*np.pi/angle_count)
    radial = np.cos(theta)[:, None]*u + np.sin(theta)[:, None]*v
    points = (origin[None, None, :] + axial[:, None, None]*normal +
              radius[:, None, None]*radial[None, :, :]).reshape(-1, 3)
    slopes = np.gradient(radius, axial)
    normals = (radial[None, :, :] - slopes[:, None, None]*normal).reshape(-1, 3)
    all_errors, views = [], []
    for frame, mask in zip(frames, masks):
        if mask.shape != (frame.camera.height, frame.camera.width):
            raise ProfileRejection('foreground_shape_does_not_match_camera')
        camera = frame.camera
        camera_position = camera.T_world_from_camera[:3, 3]
        local = (points-camera_position) @ camera.T_world_from_camera[:3, :3]
        if np.any(local[:, 2] <= 0):
            raise ProfileRejection('fitted_surface_behind_camera')
        uv = camera.project(points)
        # Distance to the foreground pixel squares; the binary boundary is
        # halfway between the nearest interior/exterior pixel centres.
        outside = np.maximum(distance_transform_edt(~mask)-.5, 0.)
        error = map_coordinates(outside, [uv[:, 1], uv[:, 0]], order=1,
                                mode='constant', cval=1e3)
        all_errors.append(error)

        front = np.einsum('ij,ij->i', normals, camera_position-points) > 0
        pixels = np.floor(uv+.5).astype(int)
        in_image = ((pixels[:, 0] >= 0) & (pixels[:, 0] < camera.width) &
                    (pixels[:, 1] >= 0) & (pixels[:, 1] < camera.height))
        # Only compare samples in the same discrete pixel. This is local
        # self-occlusion, not a reconstruction of the entire jar or new gate.
        ids = np.flatnonzero(in_image)
        flat = pixels[ids, 1]*camera.width + pixels[ids, 0]
        zbuffer = np.full(camera.width*camera.height, np.inf)
        np.minimum.at(zbuffer, flat, local[ids, 2])
        visible = np.zeros(len(points), bool)
        visible[ids] = front[ids] & (local[ids, 2] <= zbuffer[flat]+1e-9)
        views.append(dict(camera=camera.name, sample_count=len(points),
                          outside_rms_pixels=float(np.sqrt(np.mean(error**2))),
                          maximum_outside_pixels=float(error.max()),
                          front_facing_sample_count=int(front.sum()),
                          locally_visible_sample_count=int(visible.sum()),
                          outside_sample_count=int(np.count_nonzero(error > 0))))
    if not all_errors:
        raise ProfileRejection('no_camera_for_surface_projection')
    errors = np.concatenate(all_errors)
    return dict(outside_rms_pixels=float(np.sqrt(np.mean(errors**2))),
                maximum_outside_pixels=float(errors.max()), views=views,
                axial_sample_count=len(axial), angular_sample_count=angle_count,
                validation='one-sided containment of full finite fitted surface',
                visibility='front face + finite local z-buffer; diagnostic only; '
                           'unknown whole-jar occlusion is not a rejection',
                boundary_distance='outside foreground pixel squares, pixels',
                uses_ground_truth=False)


def revalidate_candidate(candidate, frames, masks, settings=PortfolioSettings()):
    """Create a freshly scored candidate; never select with the old score.

    All non-contact frozen gates, including 3 mm span and observed-boundary
    gates, remain part of main ``accepted``. The old decision is preserved only
    as evidence. Revalidation is done for *every seed before selection*.
    """
    result = deepcopy(candidate)
    diagnostics = result.diagnostics
    diagnostics['frozen_validation'] = dict(
        accepted=bool(candidate.accepted), reasons=list(candidate.reasons),
        selection_score=candidate.diagnostics.get('selection_score'),
        target_contact_rms_pixels=candidate.diagnostics.get('target_contact_rms_pixels'))
    result.reasons = [r for r in candidate.reasons if r != _OLD_CONTACT_REASON]
    support = profile_projection_support(result, frames, masks)
    if support['maximum_outside_pixels'] > settings.validation_contact_pixels:
        result.reasons.append('fitted_surface_projects_outside_rgb_foreground')
    result.accepted = not result.reasons
    span = float(diagnostics['near_min_interval_length_m'])
    diagnostics.update(
        method='visible_profile_baseline', surface_projection=support,
        selection_score=(float(diagnostics['heldout_rms_mm'])/settings.validation_depth_rms_mm +
                         support['outside_rms_pixels']/settings.validation_contact_pixels),
        selection_score_semantics='heldout radial RMS / frozen depth scale + '
                                  'one-sided surface outside RMS / frozen image scale',
        surface_containment_gate='maximum sampled outside distance <= frozen '
                                 'validation_contact_pixels; stricter than RMS containment',
        pose_gate_accepted=not [r for r in result.reasons if r != 'insufficient_three_mm_band_span'],
        band_usable=bool(span >= .003 and 'near_minimum_touches_observed_boundary' not in result.reasons),
        band_usable_semantics='observed near-minimum proxy spans 3 mm without touching support boundary; '
                              'not verified physical band containment',
        diameter_threshold_is_evaluation_only=True,
        engineering_correction=True,
        uncertainty='not_calibrated; containment is necessary, not sufficient for absolute accuracy')
    return result


def estimate_visible_profile(frames, *, settings=PortfolioSettings(), robust=True,
                             feedback=True, multi=True, anisotropic=False,
                             fixed_axis=False):
    """Frozen fitting + correct one-sided validation + per-seed reselection."""
    started = perf_counter()
    result, attempts = Estimate(), []
    masks, foreground = extract_stereo_masks(frames)
    try:
        cloud, source = observed_cloud(frames, masks, settings)
        if len(cloud) < 300:
            raise ProfileRejection('insufficient_multiview_consistent_observed_cloud')
        seeds = seed_hypotheses(frames, masks, cloud, settings)
        if not multi:
            seeds = seeds[:1]
        candidates = []
        for seed in seeds:
            try:
                frozen = fit_seed(frames, masks, cloud, source, seed, settings,
                                  robust, feedback, anisotropic, fixed_axis)
                candidate = revalidate_candidate(frozen, frames, masks, settings)
                candidates.append(candidate)
                attempts.append(dict(seed=seed['label'], arc_rms_pixels=seed['arc_rms'],
                    accepted=candidate.accepted, reasons=list(candidate.reasons),
                    center_m=candidate.center.tolist(), normal=candidate.normal.tolist(),
                    radius_m=candidate.radius, score=candidate.diagnostics['selection_score'],
                    surface_projection=candidate.diagnostics['surface_projection'],
                    frozen_validation=candidate.diagnostics['frozen_validation']))
            except (ValueError, np.linalg.LinAlgError, QhullError) as error:
                attempts.append(dict(seed=seed['label'], accepted=False, reasons=[str(error)]))
        if candidates:
            result = min(candidates, key=lambda c:(not c.accepted, c.diagnostics['selection_score']))
        else:
            result.reasons = ['no_valid_structural_candidate']
        if not seeds:
            result.reasons = ['no_observed_rim_hypothesis']
        result.diagnostics['station_consistent_points'] = len(cloud)
    except (ValueError, np.linalg.LinAlgError, QhullError) as error:
        result.reasons = [str(error)]
    result.diagnostics.update(method='visible_profile_baseline', foreground=foreground,
        attempts=attempts, settings=asdict(settings), multi=multi, robust=robust,
        feedback=feedback, anisotropic=anisotropic, fixed_axis=fixed_axis,
        uncertainty='not_calibrated; no absolute pose or physical band guarantee',
        engineering_correction=True)
    result.elapsed_ms = (perf_counter()-started)*1000
    return result
