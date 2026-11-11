"""RGB blue-screen adaptation around the frozen local-geometry portfolio.

The sole estimation change is foreground extraction: chromatic background
rejection and largest external-contour filling preserve dark neck pixels. No
depth, dimensions, target labels or diagnostic masks enter this extraction.
Geometry selection below mirrors joint_profile.observed_initialization; the
solvers and all acceptance checks are imported unchanged from frozen modules.
This is a declared blue-background station method, not general segmentation.
"""
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from time import perf_counter

import cv2
import numpy as np
from scipy.spatial import QhullError

from .depth_profile import ProfileRejection, ProfileSettings, _mouth_center, _mouth_plane
from .geometry import Estimate
from .joint_profile import JointSettings
from .reliable_output import ReliabilitySettings, _attempt, _candidate, _initial
from .support_feedback import WeightedAxisSplineProblem
from .task_validated_output import TaskValidationSettings, _validate_task


@dataclass(frozen=True)
class BlueScreenSettings:
    # OpenCV hue is 0..179. Broad chromatic key for this blue station backdrop;
    # there is deliberately NO minimum value/brightness for foreground pixels.
    hue_low: int = 95
    hue_high: int = 125
    minimum_saturation: int = 65
    minimum_blue_over_red: int = 12
    minimum_blue_over_green: int = 4


def blue_screen_foreground(rgb, settings=BlueScreenSettings()):
    """Largest filled nonblue contour, without altering the original RGB.

    Filling admits the dark mouth cavity too. Frozen camera-facing geometry
    selection, not a fabricated RGB image, decides which depth points to fit.
    """
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected uint8 HxWx3 RGB")
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    signed = image.astype(np.int16)
    blue = ((hsv[..., 0] >= settings.hue_low) & (hsv[..., 0] <= settings.hue_high) &
            (hsv[..., 1] >= settings.minimum_saturation) &
            (signed[..., 2]-signed[..., 0] >= settings.minimum_blue_over_red) &
            (signed[..., 2]-signed[..., 1] >= settings.minimum_blue_over_green))
    contours, _ = cv2.findContours((~blue).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros(image.shape[:2], np.uint8)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(result, [largest], -1, 1, thickness=cv2.FILLED)
    return result.astype(bool)


def extract_stereo_masks(frames, settings=BlueScreenSettings()):
    """Return the exact RGB-only estimator masks and auditable metadata."""
    masks, details = [], []
    for frame in frames:
        mask = blue_screen_foreground(frame.rgb, settings)
        masks.append(mask)
        yy, xx = np.nonzero(mask)
        details.append(dict(camera=frame.camera.name, pixels=int(mask.sum()),
                            fraction=float(mask.mean()),
                            bbox_xyxy=None if not len(xx) else [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())],
                            mask_sha256=sha256(mask.astype(np.uint8).tobytes()).hexdigest(),
                            rgb_sha256=sha256(np.ascontiguousarray(frame.rgb).tobytes()).hexdigest()))
    return masks, dict(method="rgb_blue_key_largest_filled_external_contour", settings=asdict(settings),
                       inputs="original_RGB_only; no depth or target dimensions", cameras=details)


def _observed_initialization(frames, masks, settings, seed):
    """Frozen observed_initialization selection, with explicit RGB masks."""
    clouds, ids, counts = [], [], {}
    station = ProfileSettings()
    for index, (frame, mask) in enumerate(zip(frames, masks)):
        valid = mask & np.isfinite(frame.depth_z_m) & (frame.depth_z_m > 0)
        yy, xx = np.nonzero(valid)
        p = frame.camera.unproject(np.c_[xx, yy], frame.depth_z_m[yy, xx])
        keep = ((p[:, 2] >= station.world_z_bounds_m[0]) &
                (p[:, 2] <= station.world_z_bounds_m[1]) &
                (np.linalg.norm(p[:, :2], axis=1) < station.world_xy_radius_m))
        p = p[keep]
        clouds.append(p)
        ids.append(np.full(len(p), index))
        counts[frame.camera.name] = len(p)
    cloud, source = np.concatenate(clouds), np.concatenate(ids)
    if len(cloud) < 300:
        raise ProfileRejection("insufficient_surface_points_in_fixed_station_volume")
    plane_origin, normal, inliers = _mouth_plane(cloud, station, np.random.default_rng(seed))
    center, mouth_radius = _mouth_center(plane_origin, normal, inliers)
    d = cloud-center
    s = d @ normal
    radial = d-s[:, None]*normal
    rho = np.linalg.norm(radial, axis=1)
    camera_positions = np.array([f.camera.T_world_from_camera[:3, 3] for f in frames])
    view = camera_positions[source]-cloud
    view -= (view @ normal)[:, None]*normal
    cosine = np.sum(radial*view, axis=1)/np.maximum(rho*np.linalg.norm(view, axis=1), 1e-12)
    keep = ((s >= -settings.fit_depth_mm/1000) & (s <= -settings.exclude_top_mm/1000)
            & (cosine > settings.facing_cosine_min))
    selected, selected_ids = cloud[keep], source[keep]
    if len(selected) < 200:
        raise ProfileRejection("insufficient_camera_facing_local_outer_candidates")
    selected_s, selected_rho = s[keep]*1000, rho[keep]*1000
    bins = np.floor(selected_s/settings.support_bin_mm).astype(int)
    occupied, count = np.unique(bins, return_counts=True)
    occupied = occupied[count >= 8]
    if len(occupied) < 8:
        raise ProfileRejection("insufficient_coarse_local_profile_support")
    medians = np.array([np.median(selected_rho[bins == b]) for b in occupied])
    minimum = int(np.argmin(medians))
    left = right = minimum
    threshold = medians[minimum]+settings.near_min_delta_mm
    while (left > 0 and medians[left-1] <= threshold
           and (occupied[left]-occupied[left-1])*settings.support_bin_mm <= settings.maximum_target_gap_mm):
        left -= 1
    while (right+1 < len(occupied) and medians[right+1] <= threshold
           and (occupied[right+1]-occupied[right])*settings.support_bin_mm <= settings.maximum_target_gap_mm):
        right += 1
    low, high = occupied[left]*settings.support_bin_mm, (occupied[right]+1)*settings.support_bin_mm
    coarse_center = (low+high)/2
    half_span = max(settings.local_half_span_mm, (high-low)/2+settings.local_context_mm)
    local = np.abs(selected_s-coarse_center) <= half_span
    selection_info = dict(coarse_minimum_from_initial_axis_m=coarse_center/1000,
        coarse_near_minimum_interval_m=[low/1000, high/1000],
        local_selection_interval_from_initial_axis_m=[(coarse_center-half_span)/1000,
                                                      (coarse_center+half_span)/1000],
        broad_outer_candidates=len(selected),
        mouth_plane_support=len(inliers), foreground_station_counts=counts)
    selected, selected_ids = selected[local], selected_ids[local]
    if len(selected) < 200:
        raise ProfileRejection("insufficient_local_neck_support")
    fit_indices = []
    for index in range(len(frames)):
        indices = np.flatnonzero(selected_ids == index)
        if len(indices):
            fit_indices.extend(indices[np.linspace(0, len(indices)-1,
                min(len(indices), settings.max_fit_points_per_camera)).astype(int)])
    return selected, selected_ids, np.asarray(fit_indices), center, normal, mouth_radius, counts, selection_info


def _prepare(frames, masks, depth):
    joint = replace(JointSettings(), fit_depth_mm=depth)
    selected, ids, indices, origin, normal, lip, counts, selection = _observed_initialization(frames, masks, joint, 4101)
    held = indices[np.arange(len(indices)) % 4 == 0]
    train = indices[np.arange(len(indices)) % 4 != 0]
    problem = WeightedAxisSplineProblem(selected[train]*1000, origin*1000, normal, ids[train], joint)
    problem.set_weights(np.ones(len(train)))
    return dict(problem=problem, joint=joint, selected=selected, ids=ids, train=train, held=held,
                origin=origin, normal=normal, lip=lip, counts=counts, selection=selection)


def _reliable_output(frames, masks, settings=ReliabilitySettings()):
    """Frozen primary portfolio orchestration with explicit-mask preparation."""
    method = "adaptive_validated"
    attempts, last, last_prepared = [], None, None
    for depth in settings.candidate_depths_mm:
        prepared, fixed_x = None, None
        try:
            prepared = _prepare(frames, masks, depth)
            last_prepared = prepared
            result, fixed_x, fixed_success = _attempt(prepared, frames, settings, method)
            attempts.append(dict(depth_mm=depth, path="continuous_feedback", accepted=result.accepted, reasons=result.reasons))
            last = result
            if result.accepted:
                break
        except (ProfileRejection, np.linalg.LinAlgError, QhullError) as error:
            attempts.append(dict(depth_mm=depth, path="continuous_feedback", accepted=False, reasons=[str(error)]))
            if prepared is not None:
                fixed_x, fixed_success = prepared.get("fixed_x"), prepared.get("fixed_success", False)
        if fixed_x is not None:
            try:
                weights = np.ones(len(prepared["problem"].points))
                fallback = _candidate(prepared, fixed_x, weights, frames, method, settings, [], True, fixed_success)
                fallback.diagnostics["chosen_path"] = "validated_fixed_weight_alternative"
                attempts.append(dict(depth_mm=depth, path="fixed_alternative", accepted=fallback.accepted, reasons=fallback.reasons))
                last = fallback
                if fallback.accepted:
                    break
            except (ProfileRejection, np.linalg.LinAlgError, QhullError) as error:
                attempts.append(dict(depth_mm=depth, path="fixed_alternative", accepted=False, reasons=[str(error)]))
    if last is None:
        last = Estimate(reasons=attempts[-1]["reasons"] if attempts else ["no_candidate"])
    last.diagnostics.update(method=method, attempts=attempts, reliability_settings=asdict(settings),
                            camera_trust=[1.]*len(frames), uncertainty="not_estimated",
                            chosen_path=last.diagnostics.get("chosen_path", "continuous_feedback" if last.center is not None else "none"))
    if last.center is None and last_prepared is not None:
        _attach_rejected_preparation(last, last_prepared, frames)
    return last


def _attach_rejected_preparation(result, prep, frames):
    """Retain real solver-input points even when no circle could be read out."""
    problem = prep["problem"]
    result.diagnostics.update(prep["selection"])
    result.diagnostics.update(
        candidate_depth_mm=prep["joint"].fit_depth_mm,
        initial_mouth_center_m=prep["origin"].tolist(), initial_mouth_normal=prep["normal"].tolist(),
        fit_point_positions_m=(problem.points/1000).tolist(),
        fit_camera_ids=prep["ids"][prep["train"]].tolist(),
        fit_camera_names=[f.camera.name for f in frames],
        fit_point_weights=(problem.sqrt_weight**2).tolist(),
        fit_points=len(problem.points), heldout_points=len(prep["held"]),
        fit_visualization_semantics="last actually prepared training subset; solver/validation rejected; no circle fabricated")


def estimate_stereo_robust(frames, *, foreground_settings=BlueScreenSettings(), settings=TaskValidationSettings()):
    """Same frozen geometry/acceptance portfolio with RGB-only blue-screen masks."""
    started = perf_counter()
    masks, foreground = extract_stereo_masks(frames, foreground_settings)
    primary = _reliable_output(frames, masks)
    attempts = list(primary.diagnostics.get("attempts", []))
    best, last_alternative_preparation = None, None
    if not primary.accepted:
        for depth in ReliabilitySettings().candidate_depths_mm:
            try:
                prep = _prepare(frames, masks, depth)
                last_alternative_preparation = prep
                fit = _initial(prep["problem"], prep["joint"])
            except (ProfileRejection, np.linalg.LinAlgError, QhullError) as error:
                attempts.append(dict(depth_mm=depth, path="task_alternative_preparation", accepted=False, reasons=[str(error)]))
                continue
            candidates = []
            for label, x, ok in [("task_validated_joint_alternative", fit.x, bool(fit.success)),
                                 ("task_validated_observed_axis", np.zeros(4), True)]:
                try:
                    candidate = _validate_task(prep, x, frames, settings, label, ok)
                    attempts.append(dict(depth_mm=depth, path=label, accepted=candidate.accepted, reasons=candidate.reasons))
                    if candidate.accepted:
                        candidates.append(candidate)
                except (ProfileRejection, np.linalg.LinAlgError, QhullError) as error:
                    attempts.append(dict(depth_mm=depth, path=label, accepted=False, reasons=[str(error)]))
            if not candidates:
                continue
            if len(candidates) > 1:
                a, b = candidates[:2]
                gap_center = float(np.linalg.norm(a.center-b.center)*1000)
                gap_normal = float(np.degrees(np.arctan2(np.linalg.norm(np.cross(a.normal, b.normal)), a.normal @ b.normal)))
                if gap_center > settings.agreement_center_mm or gap_normal > settings.agreement_normal_deg:
                    attempts.append(dict(depth_mm=depth, path="task_alternative_agreement", accepted=False,
                                         reasons=["ambiguous_task_alternatives"], center_gap_mm=gap_center, normal_gap_deg=gap_normal))
                    continue
            best = min(candidates, key=lambda c:c.diagnostics["heldout_context_rms_mm"])
            break
    result = best if best is not None else primary
    if result.center is None and last_alternative_preparation is not None:
        _attach_rejected_preparation(result, last_alternative_preparation, frames)
    result.diagnostics.update(method="stereo_rgb_adapted", attempts=attempts,
                              primary_rejection=[] if primary.accepted else primary.reasons,
                              task_validation_settings=asdict(settings), foreground=foreground,
                              geometry_change="none: frozen mouth plane, axis/profile solvers and acceptance thresholds",
                              uncertainty="not_estimated")
    result.elapsed_ms = (perf_counter()-started)*1000
    return result
