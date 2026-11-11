"""Standard robust local surface-of-revolution (SOR) control, observation only.

This is an independently implemented conventional axis/meridian fit, NOT a
complete reproduction of Pottmann et al. (2002) or Liu et al. (CAD, 2006).
Those works establish SOR reconstruction and simultaneous pose/shape fitting
as mature geometry. Here a cubic B-spline meridian and four axis parameters
are optimized jointly by SciPy's bounded Huber least squares. The fixed point
support is never fed back from the estimated target, and no target-circle /
whole-object-silhouette tangency is assumed.

Only RGB masks, station-filtered observed depth and observed rim hypotheses
are shared with the portfolio. No weighted-feedback solver, truth, instance
CAD, known neck radius or per-instance target interval is consumed. Fit units
are mm; the public Estimate uses metres. No calibrated accuracy guarantee is
inferred from a small fitting residual.
"""
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import brentq, least_squares
from scipy.spatial import QhullError

from .depth_profile import ProfileRejection
from .geometry import Estimate, basis
from .stereo_robust import extract_stereo_masks
from .structure_portfolio import observed_cloud, seed_hypotheses


@dataclass(frozen=True)
class SORSettings:
    broad_depth_mm: float = 45.
    exclude_top_mm: float = 2.
    local_half_span_mm: float = 10.
    local_context_mm: float = 3.
    minimum_facing_cosine: float = .15
    maximum_points_per_camera: int = 650
    knot_spacing_mm: float = 3.
    smoothing: float = .01
    huber_transition_mm: float = 1.5
    axis_shift_bound_mm: float = 15.
    axis_tangent_bound: float = .25
    max_nfev: int = 180
    near_min_delta_mm: float = 1.
    validation_rms_mm: float = 4.5
    validation_mean_mm: float = 2.
    minimum_training_points: int = 120
    minimum_validation_points: int = 20


def _coords(points, origin, normal):
    delta = points-origin
    s = delta @ normal
    radial = delta-s[:, None]*normal
    return s, np.linalg.norm(radial, axis=1), radial


def _coarse_region(s, radius, settings):
    """Observed median profile only, used once before optimization."""
    grid = np.arange(np.percentile(s, 1)+1., np.percentile(s, 99)-1., .5)
    values = np.array([np.median(radius[abs(s-g) < 1.25])
                       if np.count_nonzero(abs(s-g) < 1.25) >= 10 else np.nan
                       for g in grid])
    good = np.isfinite(values)
    if good.sum() < 10:
        raise ProfileRejection('sor_insufficient_coarse_profile')
    values = np.interp(grid, grid[good], values[good])
    k = int(np.argmin(values)); lo = hi = k
    while lo > 0 and values[lo-1] <= values[k]+settings.near_min_delta_mm:
        lo -= 1
    while hi+1 < len(grid) and values[hi+1] <= values[k]+settings.near_min_delta_mm:
        hi += 1
    return float((grid[lo]+grid[hi])/2), [float(grid[lo]), float(grid[hi])]


class SorProblem:
    """Joint nonlinear axis + linear spline coefficients, with exact Jacobian.

    The two tangents parameterize an upward axis around the supplied initial
    normal. Axis origin Z is fixed to remove the along-axis gauge freedom.
    The objective is radial residual, not exact closest-surface distance.
    """
    def __init__(self, points_mm, origin_mm, normal, camera_ids, settings=SORSettings()):
        self.points = np.asarray(points_mm, float)
        self.origin = np.asarray(origin_mm, float)
        self.n0 = np.asarray(normal, float)/np.linalg.norm(normal)
        self.U = np.array(basis(self.n0)).T
        self.settings = settings
        s = (self.points-self.origin) @ self.n0
        h = settings.knot_spacing_mm
        # Fixed padded knot domain covers admissible axis changes. No target
        # coordinates are used and extrapolation is diagnosed at readout.
        lo = np.floor((s.min()-25.)/h)*h
        hi = np.ceil((s.max()+25.)/h)*h
        knots = np.r_[np.repeat(lo, 4), np.arange(lo+h, hi, h), np.repeat(hi, 4)]
        self.spline = BSpline(knots, np.eye(len(knots)-4), 3, extrapolate=True)
        self.derivative = self.spline.derivative()
        self.k = len(knots)-4
        ids = np.asarray(camera_ids, int)
        counts = np.bincount(ids)
        self.base_weights = 1./counts[ids]
        self.base_weights /= self.base_weights.mean()
        self.sqrt_weights = np.sqrt(self.base_weights)
        self.regularizer = np.diff(np.eye(self.k), n=2, axis=0)*np.sqrt(
            len(self.points)*settings.smoothing/h**3)

    def axis(self, x):
        a = self.origin+np.r_[x[:2], 0.]
        raw = self.n0+self.U @ x[2:4]
        length = np.linalg.norm(raw)
        n = raw/length
        dn = (np.eye(3)-np.outer(n, n)) @ self.U/length
        return a, n, dn

    def initial(self):
        s, rho, _ = _coords(self.points, self.origin, self.n0)
        b = self.spline(s)
        matrix = np.vstack([self.sqrt_weights[:, None]*b, self.regularizer])
        target = np.r_[self.sqrt_weights*rho, np.zeros(len(self.regularizer))]
        beta = np.linalg.lstsq(matrix, target, rcond=None)[0]
        return np.r_[np.zeros(4), beta]

    def evaluate(self, x):
        a, n, dn = self.axis(x)
        s, rho, radial = _coords(self.points, a, n)
        if np.min(rho) < 1e-6:
            raise ProfileRejection('sor_axis_intersects_selected_points')
        b = self.spline(s)
        beta = x[4:]
        dr = rho-b @ beta
        data = self.sqrt_weights*dr
        e = radial/rho[:, None]
        ds = np.c_[-np.broadcast_to(n[:2], (len(s), 2)), (self.points-a) @ dn]
        drho = np.c_[-e[:, :2], -s[:, None]*(e @ dn)]
        jaxis = drho-(self.derivative(s) @ beta)[:, None]*ds
        jac = np.vstack([self.sqrt_weights[:, None]*np.c_[jaxis, -b],
                         np.c_[np.zeros((len(self.regularizer), 4)), self.regularizer]])
        residual = np.r_[data, self.regularizer @ beta]
        return dict(residual=residual, jacobian=jac, radial_residual=dr,
                    origin=a, normal=n, s=s, beta=beta, weighted_data=data)

    def fun(self, x):
        return self.evaluate(x)['residual']

    def jac(self, x):
        return self.evaluate(x)['jacobian']

    def loss(self, z):
        """Huber on measured points, quadratic on spline roughness."""
        out = np.empty((3, len(z)))
        out[0], out[1], out[2] = z, 1., 0.
        large = (np.arange(len(z)) < len(self.points)) & (z > 1.)
        root = np.sqrt(z[large])
        out[0, large] = 2*root-1.
        out[1, large] = 1./root
        out[2, large] = -.5/(z[large]*root)
        return out

    def fit(self):
        bound = np.r_[np.repeat(self.settings.axis_shift_bound_mm, 2),
                      np.repeat(self.settings.axis_tangent_bound, 2), np.repeat(np.inf, self.k)]
        fit = least_squares(self.fun, self.initial(), jac=self.jac, bounds=(-bound, bound),
            x_scale=np.r_[5., 5., .1, .1, np.repeat(50., self.k)],
            loss=self.loss, f_scale=self.settings.huber_transition_mm,
            max_nfev=self.settings.max_nfev, ftol=1e-7, xtol=1e-7, gtol=1e-6)
        return fit, self.evaluate(fit.x)


def _readout(problem, state, points, settings):
    s = (points-state['origin']) @ state['normal']
    grid = np.arange(np.percentile(s, 2)+.5, np.percentile(s, 98)-.5, .1)
    if len(grid) < 20:
        raise ProfileRejection('sor_insufficient_readout_extent')
    values = problem.spline(grid) @ state['beta']
    index = int(np.argmin(values)); left = right = index
    threshold = values[index]+settings.near_min_delta_mm
    while left > 0 and values[left-1] <= threshold: left -= 1
    while right+1 < len(grid) and values[right+1] <= threshold: right += 1
    profile = lambda q: float(problem.spline(q) @ state['beta'])
    lower = float(grid[left]); upper = float(grid[right])
    if left > 0:
        lower = float(brentq(lambda q: profile(q)-threshold, grid[left-1], grid[left]))
    if right+1 < len(grid):
        upper = float(brentq(lambda q: profile(q)-threshold, grid[right], grid[right+1]))
    middle = (lower+upper)/2
    return middle, profile(middle), grid, values, [lower, upper], (left == 0 or right == len(grid)-1)


def _fit_candidate(frames, cloud, source, seed, settings, normal_override=None):
    origin = np.asarray(seed['origin'], float)
    normal = np.asarray(seed['normal'] if normal_override is None else normal_override, float)
    s, rho, radial = _coords(cloud, origin, normal)
    camera_positions = np.array([f.camera.T_world_from_camera[:3, 3]*1000 for f in frames])
    view = camera_positions[source]-cloud
    view -= (view @ normal)[:, None]*normal
    facing = np.sum(radial*view, axis=1)/np.maximum(rho*np.linalg.norm(view, axis=1), 1e-9)
    keep = ((s > -settings.broad_depth_mm) & (s < -settings.exclude_top_mm)
            & (rho > 20.) & (rho < 120.) & (facing > settings.minimum_facing_cosine))
    if keep.sum() < settings.minimum_training_points:
        raise ProfileRejection('sor_insufficient_local_outer_points')
    middle, interval = _coarse_region(s[keep], rho[keep], settings)
    selection_middle = middle
    half = max(settings.local_half_span_mm, (interval[1]-interval[0])/2+settings.local_context_mm)
    keep &= abs(s-middle) < half
    indices = []
    for i in range(len(frames)):
        available = np.flatnonzero(keep & (source == i))
        indices.extend(available[np.linspace(0, len(available)-1,
            min(len(available), settings.maximum_points_per_camera)).astype(int)])
    points, ids = cloud[indices], source[indices]
    tiles = np.zeros(len(points), int)
    for i, frame in enumerate(frames):
        use = ids == i
        uv = frame.camera.project(points[use]/1000)
        tiles[use] = (np.floor(uv[:, 0]/5)+3*np.floor(uv[:, 1]/5)).astype(int) % 4
    held = tiles == 0; training = ~held
    if training.sum() < settings.minimum_training_points or held.sum() < settings.minimum_validation_points:
        raise ProfileRejection('sor_insufficient_spatial_block_support')
    problem = SorProblem(points[training], origin, normal, ids[training], settings)
    fit, state = problem.fit()
    middle, radius, grid, values, segment, boundary = _readout(problem, state, points, settings)
    a, n = state['origin'], state['normal']
    axial, observed_radius, _ = _coords(points, a, n)
    residual = observed_radius-problem.spline(axial) @ state['beta']
    relevant = held & (axial >= segment[0]-1.5) & (axial <= segment[1]+1.5)
    validation = relevant if relevant.sum() >= settings.minimum_validation_points else held
    rms = float(np.sqrt(np.mean(residual[validation]**2)))
    mean = float(np.mean(residual[validation]))
    delta = settings.huber_transition_mm
    held_abs = abs(residual[held])
    score = float(np.mean(np.where(held_abs <= delta, .5*held_abs**2, delta*(held_abs-.5*delta))))
    result = Estimate(center=(a+middle*n)/1000, normal=n, radius=radius/1000, residual=rms/1000)
    if not fit.success: result.reasons.append('sor_optimizer_not_converged')
    if boundary: result.reasons.append('sor_near_minimum_touches_observed_boundary')
    if radius <= 0: result.reasons.append('sor_nonpositive_radius')
    if rms > settings.validation_rms_mm or abs(mean) > settings.validation_mean_mm:
        result.reasons.append('sor_spatial_block_depth_mismatch')
    if np.any(abs(fit.x[:4]) >= .98*np.array([settings.axis_shift_bound_mm]*2+[settings.axis_tangent_bound]*2)):
        result.reasons.append('sor_axis_parameter_hits_bound')
    result.accepted = not result.reasons
    robust_weights = np.minimum(1., delta/np.maximum(abs(state['weighted_data']), 1e-12))
    result.diagnostics.update(
        chosen_seed=seed['label'], seed_arc_rms_pixels=float(seed['arc_rms']),
        initial_mouth_center_m=(origin/1000).tolist(), initial_mouth_normal=normal.tolist(),
        axis_origin_m=(a/1000).tolist(), axis_parameters=fit.x[:4].tolist(),
        near_min_interval_from_axis_origin_m=(np.array(segment)/1000).tolist(),
        near_min_interval_length_m=float((segment[1]-segment[0])/1000),
        profile_s_m=(grid/1000).tolist(), profile_radius_m=(values/1000).tolist(),
        fit_point_positions_m=(points[training]/1000).tolist(), fit_camera_ids=ids[training].tolist(),
        fit_point_weights=(problem.base_weights*robust_weights).tolist(),
        fit_camera_names=[f.camera.name for f in frames],
        heldout_points_m=(points[held]/1000).tolist(), heldout_camera_ids=ids[held].tolist(),
        heldout_residual_mm=residual[held].tolist(), heldout_rms_mm=rms, heldout_mean_mm=mean,
        heldout_context_rms_mm=float(np.sqrt(np.mean(residual[held]**2))),
        selection_score=score, optimizer_nfev=int(fit.nfev), optimizer_cost=float(fit.cost),
        optimizer_message=str(fit.message), coarse_interval_mm=interval,
        fixed_selection_interval_mm=[selection_middle-half, selection_middle+half],
        fit_visualization_semantics='actual fixed training subset; Huber influence times equal-view base weight',
        weight_semantics='joint axis/curve Huber solve; no spatial feedback and no separate profile refit')
    return result


def estimate_sor(frames, *, settings=SORSettings()):
    """Mature-geometry control with observed multi-starts and fixed support.

    All candidate geometries are retained in ``attempts`` for later evaluation.
    Accepted candidates are preferred, then held-out Huber error selects among
    them. This observation-only policy is fixed before the capability cases.
    """
    started = perf_counter(); result = Estimate(); attempts = []; candidates = []
    masks, foreground = extract_stereo_masks(frames)
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
        shared_components='RGB masks, observed station cloud and observed rim initial hypotheses only',
        selection='accepted then held-out Huber error; no target silhouette tangency relation')
    result.elapsed_ms = (perf_counter()-started)*1000
    return result
