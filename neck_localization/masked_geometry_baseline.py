"""Explicit foreground adapter for the two frozen geometry controls.

Only the caller-supplied masks change. The outer candidate loops mirror the
frozen modules; their fitters, settings, native validators and ranking are
called unchanged. No global replacement, mask extraction, CAD or truth reads.
"""
from copy import deepcopy
from dataclasses import asdict
from time import perf_counter

import numpy as np
from scipy.spatial import QhullError

from .depth_profile import ProfileRejection
from .geometry import Estimate
from .sor_baseline import SORSettings, _fit_candidate
from .structure_portfolio import PortfolioSettings, fit_seed, observed_cloud, seed_hypotheses
from .visible_profile_baseline import revalidate_candidate


def validate_explicit_masks(frames, masks, foreground):
    """Check the explicit API without inventing or repairing foreground."""
    if not isinstance(foreground, dict):
        raise ValueError('foreground_metadata_must_be_a_dict')
    if len(frames) != len(masks) or not len(frames):
        raise ValueError('one_explicit_mask_required_per_frame')
    checked = []
    for frame, mask in zip(frames, masks):
        mask = np.asarray(mask)
        expected = (frame.camera.height, frame.camera.width)
        if mask.ndim != 2 or mask.shape != expected or frame.rgb.shape[:2] != expected:
            raise ValueError('explicit_mask_shape_must_match_RGB_camera')
        if mask.dtype.kind not in 'bui' or not np.isin(mask, (0, 1)).all():
            raise ValueError('explicit_masks_must_be_boolean_or_binary_integer')
        checked.append(np.array(mask, dtype=bool, copy=True))
    return checked, deepcopy(foreground)


def estimate_masked_visible_profile(frames, masks, foreground, *, settings=PortfolioSettings(),
                                    robust=True, feedback=True, multi=True,
                                    anisotropic=False, fixed_axis=False):
    """Frozen visible-profile candidate loop, with explicit masks only."""
    started = perf_counter()
    result, attempts = Estimate(), []
    masks, foreground = validate_explicit_masks(frames, masks, foreground)
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
        engineering_correction=True, foreground_adapter='explicit_cached_mask_only')
    result.elapsed_ms = (perf_counter()-started)*1000
    return result


def estimate_masked_sor(frames, masks, foreground, *, settings=SORSettings()):
    """Frozen SOR candidate loop, with explicit masks only."""
    started = perf_counter(); result = Estimate(); attempts = []; candidates = []
    masks, foreground = validate_explicit_masks(frames, masks, foreground)
    try:
        cloud, source = observed_cloud(frames, masks)
        if len(cloud) < 300:
            raise ProfileRejection('sor_insufficient_station_cloud')
        seeds = seed_hypotheses(frames, masks, cloud)
        for seed in seeds:
            starts = [('rim_axis', None), ('vertical_axis', np.array([0., 0., 1.]))]
            for label, override in starts:
                try:
                    candidate = _fit_candidate(frames, cloud, source, seed, settings, override)
                    candidate.diagnostics['axis_start'] = label
                    candidates.append(candidate)
                    attempts.append(dict(seed=seed['label'], axis_start=label,
                        accepted=candidate.accepted, reasons=list(candidate.reasons),
                        center_m=candidate.center.tolist(), normal=candidate.normal.tolist(),
                        radius_m=candidate.radius, score=candidate.diagnostics['selection_score'],
                        nfev=candidate.diagnostics['optimizer_nfev']))
                except (ValueError, np.linalg.LinAlgError, QhullError) as error:
                    attempts.append(dict(seed=seed['label'], axis_start=label, accepted=False, reasons=[str(error)]))
        if candidates:
            result = min(candidates, key=lambda c:(not c.accepted, c.diagnostics['selection_score']))
        else:
            result.reasons = ['sor_no_fitted_candidate']
        result.diagnostics['station_consistent_points'] = len(cloud)
    except (ValueError, np.linalg.LinAlgError, QhullError) as error:
        result.reasons = [str(error)]
    result.diagnostics.update(method='standard_sor', attempts=attempts, settings=asdict(settings),
        foreground=foreground, uncertainty='not_calibrated; fitting residual is not absolute localization error',
        implementation='independent conventional robust axis+B-spline least squares; not full paper reproduction',
        shared_components='explicit cached masks, observed station cloud and observed rim initial hypotheses only',
        selection='accepted then held-out Huber error; no target silhouette tangency relation',
        foreground_adapter='explicit_cached_mask_only')
    result.elapsed_ms = (perf_counter()-started)*1000
    return result
