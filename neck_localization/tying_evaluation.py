"""Evaluation-only environment access. Never imported by the observation planner."""
import numpy as np
from scipy.spatial import cKDTree

from .station_tying import GunSpec, jaw_arcs, frame, move_circle
from .band_geometry import evaluate_band
from .jar_model import centerline, neck_normal
from .tying_cover import cover_mesh


def factory_filter(spec, low=.006, high=.010, tilt_limit_deg=15.):
    tilt = float(np.rad2deg(np.arccos(np.clip(neck_normal(spec)[2], -1, 1))))
    reasons = []
    if not low-1e-12 <= spec.neck_length_m <= high+1e-12:
        reasons.append('physical_neck_length_outside_6_to_10_mm')
    if tilt > tilt_limit_deg+1e-10:
        reasons.append('total_mouth_tilt_exceeds_15_deg')
    return dict(retained=not reasons, reasons=reasons, physical_neck_length_mm=spec.neck_length_m*1000,
                total_mouth_tilt_deg=tilt, criterion='model_physical_neck_not_observed_near_min_or_fit_pool')


def _obj_triangles(path):
    vertices, faces = [], []
    with open(path) as stream:
        for line in stream:
            if line.startswith('v '):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith('f '):
                faces.append([int(x.split('/')[0])-1 for x in line.split()[1:4]])
    return np.asarray(vertices)[np.asarray(faces)]


def _surface_cover(triangles, max_edge=.004):
    """Vertex cover with an explicit conservative surface-to-sample bound.

    Every point in a triangle is within max_edge/sqrt(3) of a vertex.
    Adaptive midpoint subdivision reduces the bound without editing the model.
    """
    finished = []
    for _ in range(12):
        edges = np.linalg.norm(triangles-np.roll(triangles, 1, axis=1), axis=2).max(axis=1)
        done = edges <= max_edge
        finished.append(triangles[done].reshape(-1, 3))
        triangles = triangles[~done]
        if not len(triangles):
            return np.concatenate(finished), max_edge/np.sqrt(3)
        a, b, c = triangles.transpose(1, 0, 2)
        ab, bc, ac = (a+b)/2, (b+c)/2, (a+c)/2
        triangles = np.concatenate([np.stack(t, axis=1) for t in
                                    [(a, ab, ac), (ab, b, bc), (ac, bc, c), (ab, bc, ac)]])
    raise RuntimeError('surface subdivision did not converge')


class HeadClearance:
    def __init__(self, obj_path, spec, config):
        self.config = config
        points, self.bound = _surface_cover(_obj_triangles(obj_path))
        cm = cover_mesh(spec, config)
        local, cover_bound = _surface_cover(cm['vertices'][cm['faces']])
        cover_points = local @ cm['rotation'].T+cm['origin']
        self.bound = max(self.bound, cover_bound)
        self.tree = cKDTree(np.concatenate([points, cover_points]))

    def evaluate(self, plan):
        if not plan.get('waypoints'):
            return dict(passed=False, reason='no_accepted_localization', minimum_lower_bound_mm=None)
        gun = GunSpec(**self.config['gun'])
        displacement = np.asarray(self.config['M_W_t2_t0'])[:3, 3]
        above, end = np.array(plan['W_T_above']), np.array(plan['W_T_G_t2'])
        samples = []
        # At most 1 mm translational steps and 0.5 deg total jaw steps.
        for u in np.linspace(0, 1, int(np.ceil(plan['descent_distance_m']/.001))+1):
            transform = above.copy()
            transform[:3, 3] = (1-u)*above[:3, 3]+u*end[:3, 3]
            samples.append((transform, gun.max_opening_deg))
        for opening in np.linspace(gun.max_opening_deg, 0, 121):
            samples.append((end, opening))
        minimum = float('inf')
        for transform, opening in samples:
            arcs = jaw_arcs(gun, opening)
            # Each chord <4 mm: endpoint+midpoint spacing bounds any missed point by 1 mm.
            pts = np.concatenate([np.concatenate([arc, (arc[:-1]+arc[1:])/2]) for arc in arcs])
            pts = pts @ transform[:3, :3].T+transform[:3, 3]-displacement
            distance = self.tree.query(pts)[0].min()
            minimum = min(minimum, distance-gun.head_tube_radius_m)
        # Includes between-path samples, jaw chord sampling and polygon sagitta.
        margin = minimum-self.bound-.001-.001-.0001
        return dict(passed=bool(margin > 0), minimum_lower_bound_mm=float(margin*1000),
                    status='clearance_certified_for_head_surrogate' if margin > 0 else 'clearance_unresolved_not_confirmed_collision',
                    minimum_sampled_distance_after_tube_mm=float(minimum*1000),
                    surface_cover_bound_mm=float(self.bound*1000),
                    evaluated_poses=len(samples),
                    scope='conservative_two_jaw_clearance_to_jar_and_assumed_cover_only',
                    excluded=['robot_self_collision', 'arm_environment_collision', 'gun_housing_collision',
                              'material_tension', 'locking', 'cutting', 'dynamic_tracking'])


def evaluate_transfer(plan, truth, physical_interval, row, config):
    if not plan.get('waypoints'):
        return dict(evaluated=False, fastening_success=None)
    moved = move_circle(truth, np.array(config['M_W_t2_t0']))
    estimate = plan['circle_t2']
    dc = np.array(estimate['center_m'])-moved['center_m']
    n, nt = np.array(estimate['normal']), np.array(moved['normal'])
    normal_error = float(np.rad2deg(np.arctan2(np.linalg.norm(np.cross(n, nt)), n @ nt)))
    band = dict(center_m=estimate['center_m'], normal=estimate['normal'], radius_m=estimate['radius_m'],
                width_m=config['gun']['band_width_m'], nominal_compatible=row.get('band', {}).get('nominal_compatible', False))
    moved['s_m'] = truth['s_m']
    physical = evaluate_band(band, moved, physical_interval)
    near_min = evaluate_band(band, moved, truth['near_min_interval_s_m'])
    original_dc = np.array(plan['circle_t0']['center_m'])-truth['center_m']
    return dict(evaluated=True, center_mm=float(np.linalg.norm(dc)*1000), xyz_mm=(dc*1000).tolist(),
                normal_component_mm=float(dc @ nt*1000),
                tangent_mm=float(np.linalg.norm(dc-(dc @ nt)*nt)*1000),
                normal_deg=normal_error, diameter_mm=float(abs(estimate['radius_m']-truth['radius_m'])*2000),
                center_error_invariance_m=float(abs(np.linalg.norm(dc)-np.linalg.norm(original_dc))),
                normal_error_invariance_deg=float(abs(normal_error-row['normal_deg'])),
                physical_band=physical, near_min_proxy_band=near_min,
                fastening_success=None, fastening_reason='flexible_band_contact_lock_and_cut_not_modelled')
