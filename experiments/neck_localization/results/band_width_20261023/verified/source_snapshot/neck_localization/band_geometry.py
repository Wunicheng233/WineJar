"""Finite axial-width band audit; not contact mechanics or robot planning.

Metres, column vectors, outward/upward normal. Planner sees observations only.
The evaluation function alone accepts annotated usable intervals. A band is a
constant-radius cylindrical strip, with no radial thickness, strain or buckle.
"""
import numpy as np


def _unit(value):
    value = np.asarray(value, float)
    if value.shape != (3,) or not np.all(np.isfinite(value)) or np.linalg.norm(value) < 1e-12:
        raise ValueError("Expected finite nonzero 3-vector")
    return value / np.linalg.norm(value)


def plan_band(estimate, diagnostics, width_m=.005):
    """Keep the frozen target centre; erode its OBSERVED near-min interval.

    This is nominal compatibility, with no calibrated uncertainty allowance.
    An empty interval does not change/reject the original circle estimate.
    """
    if not np.isfinite(width_m) or width_m < 0:
        raise ValueError("Width must be finite and nonnegative")
    n = _unit(estimate["normal"])
    center = np.asarray(estimate["center_m"], float)
    origin = np.asarray(diagnostics["axis_origin_m"], float)
    bounds = np.asarray(diagnostics["near_min_interval_from_axis_origin_m"], float)
    radius = float(estimate["radius_m"])
    if (center.shape != (3,) or origin.shape != (3,) or bounds.shape != (2,)
            or not np.all(np.isfinite(np.r_[center, origin, bounds, radius]))
            or bounds[0] > bounds[1] or radius <= 0):
        raise ValueError("Invalid circle or observed interval")
    relative = bounds - (center-origin) @ n
    allowed = relative + np.array([width_m/2, -width_m/2])
    margin = float(min(-allowed[0], allowed[1]))
    compatible = bool(margin >= -1e-12)
    reasons = [] if compatible else ["observed_near_min_range_too_narrow_or_center_outside"]
    return dict(width_m=float(width_m), model="constant_radius_axial_strip_no_contact_physics",
                center_m=center.tolist(), normal=n.tolist(), radius_m=radius,
                observed_usable_interval_relative_center_m=relative.tolist(),
                allowed_center_shift_interval_m=allowed.tolist() if allowed[0] <= allowed[1]+1e-12 else None,
                nominal_margin_m=margin, nominal_compatible=compatible, reasons=reasons,
                uncertainty="not_estimated", criterion="observed_near_min_range_proxy_not_measured_groove")


def evaluate_band(band, truth, usable_interval_s_m):
    """Evaluation-only exact axial envelope of the WHOLE cylindrical strip.

    For q in [-w/2,w/2], p=c+q*n+R*(cos(phi)*u+sin(phi)*v).
    Its true-axis projection has half extent w/2*abs(n.nt)+R*|n cross nt|.
    No azimuth discretisation, centre error and tilt are not counted twice.
    This checks axial containment only; radial fit and shoulder collision are
    unmodelled. Truth intervals are separate assumptions, never planner inputs.
    """
    n, nt = _unit(band["normal"]), _unit(truth["normal"])
    bounds = np.asarray(usable_interval_s_m, float)-float(truth["s_m"])
    if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or bounds[0] > bounds[1]:
        raise ValueError("Invalid evaluation interval")
    width, radius = band["width_m"], band["radius_m"]
    offset = float((np.asarray(band["center_m"])-np.asarray(truth["center_m"])) @ nt)
    cosine = float(np.clip(n @ nt, -1, 1))
    tilt = float(radius*np.linalg.norm(np.cross(n, nt)))
    half = width/2*abs(cosine)+tilt
    envelope = np.array([offset-half, offset+half])
    margins = np.array([envelope[0]-bounds[0], bounds[1]-envelope[1]])
    contained = bool(np.min(margins) >= -1e-12)
    oracle = bool(bounds[1]-bounds[0] >= width-1e-12)
    predicted = band["nominal_compatible"]
    return dict(usable_interval_s_m=list(usable_interval_s_m),
                usable_length_mm=float(np.diff(bounds)[0]*1000),
                center_axial_offset_mm=offset*1000, tilt_swing_mm=tilt*1000,
                actual_strip_axial_envelope_relative_truth_center_mm=(envelope*1000).tolist(),
                boundary_margins_mm=(margins*1000).tolist(), minimum_margin_mm=float(min(margins)*1000),
                max_axial_overhang_mm=float(max(0, -min(margins))*1000),
                contained=contained, ideal_true_axis_center_can_fit=oracle,
                false_compatible=bool(predicted and not contained),
                false_incompatible=bool(not predicted and contained),
                scope="axial_geometry_only_no_contact_collision_or_material_model")
