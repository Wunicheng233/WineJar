"""Offline observation-to-tying contract. Metres, radians, column vectors.

T_A_B maps B coordinates to A. W is the recognition world, B the robot
base, G the closed guide-circle centre (+Z toward the mouth), F the flange.
This module accepts observations/configuration only; no target truth or CAD.
"""
from dataclasses import dataclass, asdict

import numpy as np


def unit(x):
    x = np.asarray(x, float)
    if x.shape != (3,) or not np.isfinite(x).all() or np.linalg.norm(x) < 1e-12:
        raise ValueError("finite nonzero 3-vector required")
    return x / np.linalg.norm(x)


def pose(rotation=np.eye(3), translation=(0, 0, 0)):
    r, t = np.asarray(rotation, float), np.asarray(translation, float)
    if r.shape != (3, 3) or t.shape != (3,) or not np.isfinite(r).all() or not np.isfinite(t).all():
        raise ValueError("invalid rigid transform")
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-9) or not np.isclose(np.linalg.det(r), 1):
        raise ValueError("rotation must be right handed and orthonormal")
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = r, t
    return out


def frame(normal, preferred_x=(0, 1, 0)):
    z = unit(normal)
    x = np.asarray(preferred_x, float)
    x = x - (x @ z)*z
    if np.linalg.norm(x) < 1e-8:
        x = np.eye(3)[np.argmin(abs(z))]
        x -= (x @ z)*z
    x = unit(x)
    return np.column_stack((x, np.cross(z, x), z))


def move_circle(circle, T):
    c = np.asarray(circle['center_m'], float)
    # Validate the complete transform, including the homogeneous bottom row.
    T = np.asarray(T, float)
    if T.shape != (4, 4) or not np.allclose(T[3], [0, 0, 0, 1]):
        raise ValueError('invalid homogeneous transform')
    pose(T[:3, :3], T[:3, 3])
    return dict(center_m=(T[:3, :3] @ c + T[:3, 3]).tolist(),
                normal=unit(T[:3, :3] @ unit(circle['normal'])).tolist(),
                radius_m=float(circle['radius_m']))


@dataclass(frozen=True)
class GunSpec:
    inner_diameter_m: float = .150  # User confirmed available aperture, 2026-10-28.
    head_tube_radius_m: float = .0015  # Assumed, not measured.
    max_opening_deg: float = 60.
    band_width_m: float = .003
    approach_clearance_m: float = .045
    flange_behind_m: float = .190
    flange_above_m: float = .070
    arc_segments_per_half: int = 64

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in (
                self.inner_diameter_m, self.head_tube_radius_m, self.band_width_m,
                self.approach_clearance_m, self.flange_behind_m, self.flange_above_m)):
            raise ValueError('gun lengths must be positive and finite')
        if not 0 < self.max_opening_deg <= 90 or self.arc_segments_per_half < 16:
            raise ValueError('invalid jaw configuration')

    @property
    def arc_radius_m(self):
        return self.inner_diameter_m/2+self.head_tube_radius_m

    @property
    def T_F_G(self):
        # Flange +Z points down; guide +Z points up. This is a NEW assumed mount.
        return np.linalg.inv(pose(np.diag([1., -1., -1.]),
                                 [-self.flange_behind_m, 0, self.flange_above_m]))


def jaw_arcs(gun, opening_deg):
    """Two rear-hinged semicircles; opening is TOTAL angle between jaws."""
    if not 0 <= opening_deg <= gun.max_opening_deg+1e-10:
        raise ValueError('opening exceeds mechanical limit')
    r = gun.arc_radius_m
    hinge = np.array([-r, 0, 0])
    arcs = []
    for side in (1, -1):
        theta = np.linspace(0, side*np.pi, gun.arc_segments_per_half+1)
        points = np.c_[r*np.cos(theta), r*np.sin(theta), np.zeros(len(theta))]
        a = np.deg2rad(side*opening_deg/2)
        rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        arcs.append((points-hinge) @ rot.T+hinge)
    return arcs


def jaw_metrics(gun):
    arcs = jaw_arcs(gun, gun.max_opening_deg)
    return dict(closed_inner_diameter_m=gun.inner_diameter_m,
                open_tip_center_separation_m=float(np.linalg.norm(arcs[0][0]-arcs[1][0])),
                open_tip_clearance_m=float(np.linalg.norm(arcs[0][0]-arcs[1][0])-2*gun.head_tube_radius_m),
                polygon_sagitta_m=float(gun.arc_radius_m*(1-np.cos(np.pi/gun.arc_segments_per_half/2))),
                shared_rear_hinge_assumed=True)


def plan_tying(observation, config):
    """Freeze t0 observation; rigid transport; supplement yaw by station policy.

    Rejected estimates remain blocked. No gate is relaxed to produce a motion.
    A planned motion is not a collision/IK/fastening approval.
    """
    gun = GunSpec(**config['gun'])
    result = dict(jar_id=observation['jar_id'], observation_time_s=observation['time_s'],
                  calibration_id=observation['calibration_id'], input_frame='W_recognition',
                  accepted_localization=bool(observation['accepted']),
                  observation_reasons=observation.get('reasons', []),
                  fastening_success=None, execution_authorized=False,
                  gun=asdict(gun), scope='offline_kinematic_plan_no_material_or_controller')
    if not observation['accepted'] or not observation.get('estimate'):
        return result | dict(status='blocked_localization', waypoints=[])
    circle = observation['estimate']
    n = unit(circle['normal'])
    c = np.asarray(circle['center_m'], float)
    if c.shape != (3,) or not np.isfinite(c).all() or not np.isfinite(circle['radius_m']) or circle['radius_m'] <= 0:
        raise ValueError('invalid circle')
    if n[2] <= 0:
        raise ValueError('normal convention requires positive world Z')
    mouth = np.asarray(observation['mouth_seed_m'], float)
    if mouth.shape != (3,) or not np.isfinite(mouth).all():
        raise ValueError('observed mouth initializer is required')
    M = np.asarray(config['M_W_t2_t0'], float)
    B_T_W = np.asarray(config['B_T_W'], float)
    moved = move_circle(circle, M)
    # Validate B transform by the same routine before inversion/composition.
    move_circle(circle, B_T_W)
    rotation = frame(n, config['preferred_guide_x_W'])
    W_T_G0 = pose(rotation, c)
    W_T_G2 = M @ W_T_G0
    above = max(0., float((mouth-c) @ n))+gun.approach_clearance_m
    W_T_above = W_T_G2.copy()
    W_T_above[:3, 3] += above*W_T_G2[:3, 2]
    B_T_G = B_T_W @ W_T_G2
    B_T_F = B_T_G @ np.linalg.inv(gun.T_F_G)
    result.update(status='planned_pending_evaluation', circle_t0=circle, circle_t2=moved,
        W_T_G_t0=W_T_G0.tolist(), W_T_G_t2=W_T_G2.tolist(), B_T_G=B_T_G.tolist(), B_T_F=B_T_F.tolist(),
        W_T_above=W_T_above.tolist(), descent_distance_m=above,
        yaw_source='fixed_station_direction_projected_to_estimated_plane',
        mouth_plane='final_neck_axis_proxy_with_observed_mouth_initializer',
        post_cover_neck_observations=0,
        waypoints=[dict(phase='above', W_T_G=W_T_above.tolist(), opening_deg=gun.max_opening_deg),
                   dict(phase='descend', W_T_G=W_T_G2.tolist(), opening_deg=gun.max_opening_deg),
                   dict(phase='close', W_T_G=W_T_G2.tolist(), opening_deg=0),
                   dict(phase='feed_tighten_kinematic_only', W_T_G=W_T_G2.tolist(), opening_deg=0),
                   dict(phase='retract', W_T_G=W_T_above.tolist(), opening_deg=gun.max_opening_deg)])
    return result


def timeline(config, plan, t):
    """Deterministic sequence. Durations are demo assumptions, not cycle evidence."""
    durations = config['durations_s']
    phases = ['recognize', 'to_loading', 'cover', 'to_tying', 'above', 'descend', 'close', 'tighten', 'open', 'retract']
    end = np.cumsum([durations[p] for p in phases])
    index = min(int(np.searchsorted(end, t, side='right')), len(phases)-1)
    phase = phases[index]
    begin = 0 if index == 0 else end[index-1]
    u = np.clip((t-begin)/durations[phase], 0, 1)
    u = u*u*(3-2*u)
    load = np.asarray(config['loading_displacement_W_m'], float)
    final = np.asarray(config['M_W_t2_t0'], float)[:3, 3]
    displacement = (np.zeros(3) if index == 0 else load*u if index == 1 else load if index == 2
                    else load+(final-load)*u if index == 3 else final)
    cover = index >= 2
    if not plan.get('waypoints'):
        return dict(phase=phase, displacement_m=displacement, covered=cover, tool_visible=False, total_s=float(end[-1]))
    gun = GunSpec(**config['gun'])
    target, above = np.array(plan['W_T_G_t2']), np.array(plan['W_T_above'])
    transform = above.copy()
    if phase == 'descend':
        transform[:3, 3] = (1-u)*above[:3, 3]+u*target[:3, 3]
    elif phase in ('close', 'tighten', 'open'):
        transform = target.copy()
    elif phase == 'retract':
        transform[:3, 3] = (1-u)*target[:3, 3]+u*above[:3, 3]
    opening = (gun.max_opening_deg*(1-u) if phase == 'close' else gun.max_opening_deg*u if phase == 'open'
               else 0. if phase == 'tighten' else gun.max_opening_deg)
    fraction = u if phase == 'tighten' else 1. if index >= 8 else 0.
    radius = (1-fraction)*gun.inner_diameter_m/2+fraction*(plan['circle_t2']['radius_m']+config['cover_radial_allowance_m'])
    return dict(phase=phase, displacement_m=displacement, covered=cover, tool_visible=index >= 4,
                W_T_G=transform, opening_deg=opening, band_radius_m=radius,
                band_visible=index >= 7, total_s=float(end[-1]))
