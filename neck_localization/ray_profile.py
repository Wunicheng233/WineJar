"""Image-space SOR envelope fitting, avoiding differentiated pixel contours.

Each outer-silhouette ray must touch the SAME free radius spline. We minimize
rho(t)-r(s(t)) along that ray, then optimize the axis/profile jointly. Optional
measured-depth residuals have a separate declared scale. No target dimensions,
model vertices or ideal depth enter the problem. This is a research prototype,
not a reproduction of a published complete algorithm.
"""
from dataclasses import asdict, dataclass
from time import perf_counter

import cv2
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import least_squares

from .depth_profile import ProfileRejection
from .geometry import Estimate, basis
from .stereo_robust import extract_stereo_masks
from .structural_localization import image_rim_seed, StructuralSettings


@dataclass(frozen=True)
class RayProfileSettings:
    below_rim_mm: float = 42.
    exclude_top_mm: float = 3.
    knot_spacing_mm: float = 3.
    second_difference_scale_mm: float = 1.5
    pixel_sigma: float = .6
    depth_sigma_mm: float = 4.
    depth_stride_pixels: int = 4
    maximum_depth_points: int = 900
    max_nfev: int = 130
    depth_weight: float = 1.
    maximum_validation_pixel_rms: float = 1.5
    near_min_delta_mm: float = 1.


def ray_axis_coordinates(origin, normal, cameras_mm, rays):
    delta = cameras_mm-origin
    vn = rays@normal
    cn = delta@normal
    u = rays-vn[:, None]*normal
    w = delta-cn[:, None]*normal
    aa = np.sum(u*u, axis=1)
    t0 = -np.sum(w*u, axis=1)/np.maximum(aa, 1e-12)
    closest = w+t0[:, None]*u
    rho2 = np.sum(closest*closest, axis=1)
    return aa, t0, rho2, vn, cn


def row_contour_rays(frames, masks):
    cameras, rays, ids, pixels, scales = [], [], [], [], []
    for i, (f, mask) in enumerate(zip(frames, masks)):
        occupied = mask.any(axis=1)
        yy = np.flatnonzero(occupied)
        left = np.argmax(mask[yy], axis=1).astype(float)-.5
        right = mask.shape[1]-.5-np.argmax(mask[yy, ::-1], axis=1)
        q = np.c_[np.r_[left, right], np.tile(yy, 2)]
        ray = np.c_[q, np.ones(len(q))]@np.linalg.inv(f.camera.K).T@f.camera.T_world_from_camera[:3, :3].T
        rays.append(ray); cameras.append(np.broadcast_to(f.camera.T_world_from_camera[:3, 3]*1000, ray.shape))
        ids.append(np.full(len(q), i)); pixels.append(q)
        scales.append(np.full(len(q), np.mean(np.diag(f.camera.K)[:2])))
    return tuple(np.concatenate(a) for a in [cameras, rays, ids, pixels, scales])


class RayProfileProblem:
    def __init__(self, frames, masks, seed, settings=RayProfileSettings(), depth_enabled=True):
        self.settings = settings
        self.origin = np.asarray(seed[0])*1000
        self.normal = np.asarray(seed[1]); self.U = np.array(basis(self.normal)).T
        all_samples = row_contour_rays(frames, masks)
        aa, t, rho2, vn, cn = ray_axis_coordinates(self.origin, self.normal, *all_samples[:2])
        s = cn+t*vn
        # Use sides away from the crown cap; no contour derivatives are needed.
        keep = (s > -settings.below_rim_mm) & (s < -settings.exclude_top_mm) & (rho2 > 20**2) & (rho2 < 120**2)
        self.samples = tuple(a[keep] for a in all_samples)
        self.depth_points, self.depth_ids, self.depth_pixels = [], [], []
        for i, (f, mask) in enumerate(zip(frames, masks)):
            yy, xx = np.indices(mask.shape)
            valid = mask & (f.depth_z_m > 0) & np.isfinite(f.depth_z_m)
            valid &= (xx % settings.depth_stride_pixels == 0) & (yy % settings.depth_stride_pixels == 0)
            y, x = np.nonzero(valid)
            q = f.camera.unproject(np.c_[x, y], f.depth_z_m[y, x])*1000
            d = q-self.origin; axial = d@self.normal
            rho = np.linalg.norm(d-axial[:, None]*self.normal, axis=1)
            view = f.camera.T_world_from_camera[:3, 3]*1000-q
            radial = d-axial[:, None]*self.normal
            view -= (view@self.normal)[:, None]*self.normal
            facing = np.sum(radial*view, axis=1)/np.maximum(rho*np.linalg.norm(view, axis=1), 1e-9)
            good = (axial > -settings.below_rim_mm) & (axial < -settings.exclude_top_mm) & (rho > 20) & (rho < 120) & (facing > .15)
            self.depth_points.append(q[good]); self.depth_ids.append(np.full(good.sum(), i)); self.depth_pixels.append(np.c_[x[good], y[good]])
        self.depth_points = np.concatenate(self.depth_points)
        self.depth_ids = np.concatenate(self.depth_ids); self.depth_pixels = np.concatenate(self.depth_pixels)
        if len(self.depth_points) > settings.maximum_depth_points:
            ids = np.linspace(0, len(self.depth_points)-1, settings.maximum_depth_points).astype(int)
            self.depth_points, self.depth_ids, self.depth_pixels = self.depth_points[ids], self.depth_ids[ids], self.depth_pixels[ids]
        self.depth_enabled = depth_enabled
        if len(self.samples[0]) < 30 or len(np.unique(self.samples[2])) < 2:
            raise ProfileRejection('insufficient_side_contour_rays')
        h = settings.knot_spacing_mm
        low, high = -settings.below_rim_mm-18, 18.
        knots = np.r_[[low]*4, np.arange(low+h, high, h), [high]*4]
        count = len(knots)-4
        self.basis = BSpline(knots, np.eye(count), 3, extrapolate=True)
        self.first = self.basis.derivative(); self.second = self.basis.derivative(2)
        self.penalty = np.diff(np.eye(count), n=2, axis=0)/settings.second_difference_scale_mm
        # Adjacent pixel rows are correlated. These are contiguous 3-row blocks,
        # held out from the objective; NOT claimed statistically independent.
        blocks = np.floor(self.samples[3][:, 1]/3).astype(int)
        self.held = blocks % 4 == 0
        self.train = ~self.held
        self.count = count

    def axis(self, x):
        a = self.origin+np.r_[x[:2], 0.]
        n = self.normal+self.U@x[2:4]; n /= np.linalg.norm(n)
        return a, n

    def silhouette(self, x):
        a, n = self.axis(x); beta = x[4:]
        c, ray = self.samples[:2]
        aa, t0, rho2, vn, cn = ray_axis_coordinates(a, n, c, ray)
        t = t0.copy()
        for _ in range(12):
            s = cn+t*vn
            dt = t-t0
            rho = np.sqrt(np.maximum(rho2+aa*dt**2, 1e-8))
            grad = aa*dt/rho-(self.first(s)@beta)*vn
            hess = aa/rho-aa**2*dt**2/rho**3-(self.second(s)@beta)*vn**2
            # Safeguarded local contact solve. A bound hit is reported below.
            step = np.clip(grad/np.maximum(hess, .003), -8, 8)
            t = np.clip(t-step, t0-50, t0+50)
        s = cn+t*vn
        rho = np.sqrt(np.maximum(rho2+aa*(t-t0)**2, 1e-8))
        g = rho-self.basis(s)@beta
        # Convert mm ray-envelope error to approximately image-normal pixels.
        slope = self.first(s)@beta
        z = np.maximum(t, 1.)
        pixels = g*self.samples[4]/z/np.sqrt(1+slope*slope)
        return pixels, s, c+t[:, None]*ray

    def residual(self, x):
        pixel, _, _ = self.silhouette(x)
        terms = [pixel[self.train]/self.settings.pixel_sigma]
        a, n = self.axis(x)
        if self.depth_enabled and len(self.depth_points):
            d = self.depth_points-a; s = d@n
            radial = np.linalg.norm(d-s[:, None]*n, axis=1)
            terms.append((radial-self.basis(s)@x[4:])/self.settings.depth_sigma_mm*self.settings.depth_weight)
        terms.append(self.penalty@x[4:])
        return np.concatenate(terms)

    def initial(self):
        aa, t, rho2, vn, cn = ray_axis_coordinates(self.origin, self.normal, *self.samples[:2])
        s = cn+t*vn
        design = self.basis(s)
        regularizer = self.penalty*.25
        beta = np.linalg.lstsq(np.r_[design, regularizer], np.r_[np.sqrt(rho2), np.zeros(len(regularizer))], rcond=None)[0]
        return np.r_[np.zeros(4), np.clip(beta, 22, 145)]


def estimate_ray_profile(frames, *, settings=RayProfileSettings(), depth_enabled=True):
    started = perf_counter(); result = Estimate(); detail = {}
    try:
        masks, foreground = extract_stereo_masks(frames)
        seed = image_rim_seed(frames, masks, StructuralSettings())
        problem = RayProfileProblem(frames, masks, seed, settings, depth_enabled)
        x0 = problem.initial()
        low = np.r_[[-25., -25., -.3, -.3], np.full(problem.count, 20.)]
        high = np.r_[[25., 25., .3, .3], np.full(problem.count, 150.)]
        fit = least_squares(problem.residual, x0, bounds=(low, high), loss='soft_l1', f_scale=1.,
            x_scale=np.r_[[5, 5, .1, .1], np.full(problem.count, 5.)], max_nfev=settings.max_nfev,
            ftol=1e-7, xtol=1e-7, gtol=1e-6)
        a, n = problem.axis(fit.x)
        pixel, s, contact = problem.silhouette(fit.x)
        grid = np.arange(np.percentile(s, 3)+.5, np.percentile(s, 97)-.5, .1)
        radii = problem.basis(grid)@fit.x[4:]
        minimum = int(np.argmin(radii)); left = right = minimum
        while left > 0 and radii[left-1] <= radii[minimum]+settings.near_min_delta_mm: left -= 1
        while right+1 < len(grid) and radii[right+1] <= radii[minimum]+settings.near_min_delta_mm: right += 1
        midpoint = float((grid[left]+grid[right])/2)
        radius = float(problem.basis(midpoint)@fit.x[4:])
        val = pixel[problem.held]
        val_rms = float(np.sqrt(np.mean(val**2))) if len(val) else float('inf')
        result.center = (a+midpoint*n)/1000; result.normal = n; result.radius = radius/1000
        result.residual = val_rms
        if not fit.success: result.reasons.append('ray_profile_optimizer_not_converged')
        if val_rms > settings.maximum_validation_pixel_rms: result.reasons.append('heldout_contour_mismatch')
        if left == 0 or right == len(grid)-1: result.reasons.append('near_minimum_touches_observed_boundary')
        if grid[right]-grid[left] < 3: result.reasons.append('insufficient_three_mm_band_span')
        if n[2] < np.cos(np.deg2rad(30)): result.reasons.append('normal_outside_working_range')
        if np.any(np.abs(fit.x[:4]) > .98*high[:4]): result.reasons.append('axis_parameter_hits_bound')
        result.accepted = not result.reasons
        detail.update(foreground=foreground, rim_seed=seed[3], evaluations=fit.nfev,
            objective=float(2*fit.cost), heldout_contour_rms_pixels=val_rms,
            axis_origin_m=(a/1000).tolist(), near_min_interval_from_axis_origin_m=[grid[left]/1000, grid[right]/1000],
            near_min_interval_length_m=float((grid[right]-grid[left])/1000),
            profile_s_m=(grid/1000).tolist(), profile_radius_m=(radii/1000).tolist(),
            actual_silhouette_pixels=problem.samples[3].tolist(), silhouette_camera_ids=problem.samples[2].tolist(),
            silhouette_heldout=problem.held.tolist(), silhouette_residual_pixels=pixel.tolist(),
            silhouette_contact_points_m=(contact/1000).tolist(),
            fit_point_positions_m=(problem.depth_points/1000).tolist() if depth_enabled else [],
            fit_camera_ids=problem.depth_ids.tolist() if depth_enabled else [],
            fit_point_weights=np.ones(len(problem.depth_points)).tolist() if depth_enabled else [],
            fit_camera_names=[f.camera.name for f in frames],
            fit_visualization_semantics='actual subsampled depth points; silhouette contacts are inferred, NOT observed point cloud',
            initial_mouth_center_m=np.asarray(seed[0]).tolist(), initial_mouth_normal=np.asarray(seed[1]).tolist())
    except (ValueError, np.linalg.LinAlgError) as error:
        result.reasons.append(str(error))
    result.diagnostics.update(detail, method='ray_profile_fused' if depth_enabled else 'ray_profile_rgb',
        settings=asdict(settings), uncertainty='not_calibrated', camera_trust=[1.]*len(frames),
        depth_enabled=depth_enabled, residual_semantics='approximate image-normal pixels; not metric absolute accuracy')
    result.elapsed_ms = (perf_counter()-started)*1000
    return result
