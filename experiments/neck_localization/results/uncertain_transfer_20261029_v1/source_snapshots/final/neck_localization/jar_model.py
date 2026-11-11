"""Photo-informed, parameterized jar geometry for static offline scenes.

This is a synthetic circular-section family, not photogrammetry or a measured
model of the pictured jar.  The user supplied an approximate 0.60 m height and
0.125 m mouth outside diameter.  Body diameter, neck diameter, wall thickness,
and profile shape below remain explicit assumptions. The user later supplied
an approximate 10 mm exposed neck length, excluding the separately assumed lip.

The body and most of the shoulder remain unchanged. Only the collar tilts when a local
neck tilt or offset is requested.  ``s`` equals body Z below the transition and
is a nominal profile parameter in the transition.  In the straight neck it is
an actual axial coordinate in metres, with ``s=height_m`` at the mouth centre.
The mouth centre stays at Z=height_m; the highest vertex of a tilted lip can be
higher.  Every generated section is circular. Only in the body and straight neck is its
plane normal parallel to the centreline tangent; the collar joins the frames locally.

Target metadata is synthetic evaluation truth.  It must never be supplied to
an estimator as a measurement or initialization.  The near-minimum rule is a
declared design policy, not a claim about the correct physical binding plane.
"""
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class JarSpec:
    """Dimensions in metres; neck tilt is X/Y intrinsic XYZ Euler degrees.

    ``neck_offset_xy_m`` is the mouth centre's offset from the vertical body
    axis.  ``neck_profile='flat'`` provides a finite constant-radius interval;
    ``'shallow'`` provides a broad, shallow quadratic minimum.  Neither is a
    sharp manufactured V groove.  No random geometry is silently added.
    """

    height_m: float = 0.60
    body_radius_m: float = 0.18
    neck_radius_m: float = 0.0525
    lip_radius_m: float = 0.0625
    # Exposed neck BELOW the raised lip; lip height is a separate dimension.
    neck_length_m: float = 0.010
    lip_height_m: float = 0.006
    # (body height / shoulder-to-neck height, radius / maximum body radius).
    # The final neck junction is added separately, with a matching derivative.
    body_profile: tuple = ((0., .60), (.025, .64), (.12, .72), (.30, .84),
                           (.52, .95), (.72, 1.), (.81, .975),
                           (.89, .84), (.95, .60))
    neck_tilt_deg: tuple[float, float] = (0.0, 0.0)
    neck_offset_xy_m: tuple[float, float] = (0.0, 0.0)
    wall_thickness_m: float = 0.006
    bottom_thickness_m: float = 0.010
    neck_profile: str = "shallow"
    shallow_relief_m: float = 0.0018
    near_min_delta_m: float = 0.001

    def __post_init__(self):
        dimensions = (self.height_m, self.body_radius_m, self.neck_radius_m,
                      self.lip_radius_m, self.neck_length_m, self.lip_height_m,
                      self.wall_thickness_m, self.bottom_thickness_m)
        if not all(np.isfinite(x) and x > 0 for x in dimensions):
            raise ValueError("Jar dimensions must be finite and positive")
        if not self.wall_thickness_m < self.neck_radius_m < self.lip_radius_m < self.body_radius_m:
            raise ValueError("Require wall < neck radius < lip radius < body radius")
        if not self.bottom_thickness_m < self.height_m * 0.1:
            raise ValueError("Bottom thickness must be below one tenth of height")
        if not .002 <= self.neck_length_m < self.height_m * .2:
            raise ValueError("Exposed neck length must be at least 2 mm and below 20% of height")
        if not .001 <= self.lip_height_m < self.height_m * .05:
            raise ValueError("Lip height must be at least 1 mm and below 5% of height")
        profile = np.asarray(self.body_profile, dtype=float)
        if (profile.ndim != 2 or profile.shape[1] != 2 or len(profile) < 4
                or not np.all(np.isfinite(profile)) or profile[0, 0] != 0
                or np.any(np.diff(profile[:, 0]) <= 0)
                or profile[-1, 0] >= 1 or np.any(profile[:, 1] <= 0)
                or np.any(profile[:, 1] > 1) or np.max(profile[:, 1]) != 1):
            raise ValueError("Body profile needs increasing height fractions in [0,1), positive radii <=1, and a unit peak")
        if self.neck_profile not in ("shallow", "flat"):
            raise ValueError("neck_profile must be 'shallow' or 'flat'")
        if not 0 <= self.shallow_relief_m < self.lip_radius_m - self.neck_radius_m:
            raise ValueError("Shallow relief must be nonnegative and below lip relief")
        if not np.isfinite(self.near_min_delta_m) or self.near_min_delta_m < 0:
            raise ValueError("Near-minimum radius tolerance must be finite and nonnegative")
        if np.asarray(self.neck_tilt_deg).shape != (2,) or not np.all(np.isfinite(self.neck_tilt_deg)):
            raise ValueError("neck_tilt_deg requires two finite angles")
        if np.max(np.abs(self.neck_tilt_deg)) > 25:
            raise ValueError("This simplified geometry supports tilt components up to 25 degrees")
        if np.asarray(self.neck_offset_xy_m).shape != (2,) or not np.all(np.isfinite(self.neck_offset_xy_m)):
            raise ValueError("neck_offset_xy_m requires two finite offsets")
        if np.linalg.norm(self.neck_offset_xy_m) > 0.3 * self.body_radius_m:
            raise ValueError("Neck offset exceeds this simplified shoulder model's range")


def _neck_bounds(spec):
    stop = spec.height_m - spec.lip_height_m
    return stop - spec.neck_length_m, stop


def _relief(spec):
    return spec.shallow_relief_m if spec.neck_profile == "shallow" else 0.0


def _profile_parts(spec):
    start, stop = _neck_bounds(spec)
    amplitude = _relief(spec)
    profile = np.asarray(spec.body_profile, dtype=float)
    ss = start * np.r_[profile[:, 0], 1.]
    radii = np.r_[spec.body_radius_m * profile[:, 1], spec.neck_radius_m + amplitude]
    if radii[-2] <= radii[-1]:
        raise ValueError("Neck is too wide for the assumed body/shoulder profile")
    slopes = PchipInterpolator(ss, radii).derivative()(ss)
    slopes[-1] = -4 * amplitude / (stop - start)
    body = CubicHermiteSpline(ss, radii, slopes)
    peak = spec.height_m - min(.002, (spec.height_m - stop) * .3)
    lip_s = np.array([stop, peak, spec.height_m])
    lip_r = np.array([spec.neck_radius_m + amplitude, spec.lip_radius_m,
                      spec.lip_radius_m - min(.0005, .05 * (spec.lip_radius_m - spec.neck_radius_m))])
    lip = CubicHermiteSpline(lip_s, lip_r, [4 * amplitude / (stop - start), 0., 0.])
    return body, lip, ss, lip_s


def outer_radius(spec: JarSpec, s):
    """Analytic/spline outer radius; only the short neck uses the quadratic law."""
    values = np.asarray(s, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0) or np.any(values > spec.height_m):
        raise ValueError("Profile parameter must lie within [0, height_m]")
    start, stop = _neck_bounds(spec)
    body, lip, _, _ = _profile_parts(spec)
    half = (stop - start) / 2
    neck = spec.neck_radius_m + _relief(spec) * ((values - (start + stop) / 2) / half) ** 2
    return np.where(values < start, body(values), np.where(values <= stop, neck, lip(values)))


def neck_normal(spec: JarSpec):
    """Local neck normal; rotation R_y(ty) R_x(tx) applied to body +Z."""
    tx, ty = np.deg2rad(spec.neck_tilt_deg)
    return np.array([np.sin(ty) * np.cos(tx), -np.sin(tx), np.cos(ty) * np.cos(tx)])


def _rotation_vector(spec):
    return Rotation.from_euler("xy", spec.neck_tilt_deg, degrees=True).as_rotvec()


def _transition_values(spec, values, blend_start):
    """Rotate only the upper collar; do not bend the body centreline/frame."""
    start, _ = _neck_bounds(spec)
    t = np.clip((values - blend_start) / (start - blend_start), 0, 1)
    weight = t**3 * (10 - 15*t + 6*t*t)
    weight_prime = 30*t*t*(1-t)**2 / (start-blend_start)
    body = np.stack([np.zeros_like(values), np.zeros_like(values), values], axis=-1)
    normal = neck_normal(spec)
    top = np.r_[spec.neck_offset_xy_m, spec.height_m]
    neck = top + (values[..., None]-spec.height_m)*normal
    centres = body + weight[..., None]*(neck-body)
    derivative = (np.array([0., 0., 1.]) + weight_prime[..., None]*(neck-body)
                  + weight[..., None]*(normal-np.array([0., 0., 1.])))
    frames = Rotation.from_rotvec(weight.reshape(-1, 1)*_rotation_vector(spec)).as_matrix()
    frames = frames.reshape(values.shape+(3, 3))
    return centres, derivative, frames, weight_prime


def _neck_transition(spec):
    """Smallest local collar blend with positive swept-section Jacobian.

    Cross-section orientation changes independently of the centreline tangent
    in the transition.  This permits one side of a short neck to be higher
    while leaving the body and most of the shoulder exactly unchanged.
    With section normal n(s), the minimum axial Jacobian around a ring is
    n dot c' - r*||n'||.  Keep it positive; this is a local regularity check,
    not a proof against every possible global intersection.
    """
    start, _ = _neck_bounds(spec)
    rotvec = _rotation_vector(spec)
    angular_rate = np.linalg.norm(np.cross(rotvec, [0., 0., 1.]))
    for length in np.linspace(.015, min(.20*spec.height_m, .35*start), 61):
        blend_start = start-length
        ss = np.linspace(blend_start, start, 601)
        _, derivative, frames, weight_prime = _transition_values(spec, ss, blend_start)
        axial = np.einsum("ij,ij->i", frames[:, :, 2], derivative)
        radial = outer_radius(spec, ss)*angular_rate*weight_prime
        if np.min(axial-radial) >= .20:
            return blend_start, float(np.min(axial-radial)), float(np.max(radial/axial))
    raise ValueError("Requested local collar deformation has no regular short transition")


def centerline(spec: JarSpec, s, derivative=False):
    """Centres or derivatives; the normal matches the tangent in the neck only."""
    values = np.asarray(s, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0) or np.any(values > spec.height_m):
        raise ValueError("Profile parameter must lie within [0, height_m]")
    blend_start, _, _ = _neck_transition(spec)
    centres, tangent, _, _ = _transition_values(spec, values, blend_start)
    return tangent if derivative else centres


def target_metadata(spec: JarSpec):
    """Exact synthetic interval/midpoint, independent of mesh discretization.

    The selected interval is connected by construction in this model family.
    An eventual estimator must discover its own interval from observations.
    In particular a flat or shallow model is not proof of image observability.
    """
    start, stop = _neck_bounds(spec)
    midpoint = (start + stop) / 2
    amplitude = _relief(spec)
    fraction = 1. if amplitude == 0 else min(1., np.sqrt(spec.near_min_delta_m / amplitude))
    half_width = (stop - start) / 2 * fraction
    interval = [start, stop] if fraction == 1. else [midpoint - half_width, midpoint + half_width]
    minimum_interval = [start, stop] if amplitude == 0 else [midpoint, midpoint]
    normal = neck_normal(spec)
    return {
        "evaluation_only": True,
        "target_definition": "midpoint_of_connected_near_minimum_radius_interval_below_lip",
        "selection_policy": "Within search_s, r(s) <= min(r) + near_min_delta_m; select axial interval midpoint",
        "policy_is_synthetic_assumption": True,
        "search_interval_s_m": [start, stop],
        "minimum_interval_s_m": minimum_interval,
        "near_min_interval_s_m": interval,
        "near_min_interval_axial_length_m": interval[1] - interval[0],
        "near_min_delta_m": spec.near_min_delta_m,
        "s_m": midpoint,
        "center_m": centerline(spec, midpoint).tolist(),
        "normal": normal.tolist(),
        "normal_sign_convention": "toward_mouth; dot(body_up)>0",
        "radius_m": float(outer_radius(spec, midpoint)),
        "diameter_m": float(2 * outer_radius(spec, midpoint)),
        "profile_parameter_definition": "body_z_below_bridge; nominal_parameter_in_bridge; metric_axial_coordinate_in_straight_neck",
        "height_parameter_definition": "bottom_plane_to_mouth_center_z; tilted_lip_bbox_height_may_be_larger",
        "observability_verified": False,
    }


def _ring_points(spec, ss, radii, angular_segments):
    blend_start, _, _ = _neck_transition(spec)
    centres, _, frames, _ = _transition_values(spec, ss, blend_start)
    basis_u, basis_v = frames[:, :, 0], frames[:, :, 1]
    angles = 2 * np.pi * np.arange(angular_segments) / angular_segments
    return (centres[:, None, :] + radii[:, None, None] *
            (basis_u[:, None, :] * np.cos(angles)[None, :, None] +
             basis_v[:, None, :] * np.sin(angles)[None, :, None]))


def build_jar_mesh(spec: JarSpec, angular_segments=160, vertical_samples=180):
    """Build a closed, hollow vessel mesh with an open mouth and interior floor.

    Returned vertices/faces describe a watertight *solid wall* (the mouth is an
    actual hole into the cavity).  Winding faces outward, including interior
    faces toward the cavity.  UV indices have a separate azimuth seam.  This
    module does not import MuJoCo, render images, or estimate any geometry.
    """
    if not isinstance(angular_segments, int) or angular_segments < 16:
        raise ValueError("angular_segments must be an integer >= 16")
    if not isinstance(vertical_samples, int) or vertical_samples < 32:
        raise ValueError("vertical_samples must be an integer >= 32")
    start, stop = _neck_bounds(spec)
    _, _, body_knots, lip_knots = _profile_parts(spec)
    outer_s = np.unique(np.r_[np.linspace(0, spec.height_m, vertical_samples), body_knots,
                             np.linspace(start, stop, 13), lip_knots])
    inner_s = np.unique(np.r_[np.linspace(spec.bottom_thickness_m, spec.height_m, vertical_samples),
                             body_knots[body_knots > spec.bottom_thickness_m], start, stop])[::-1]
    outer_r = outer_radius(spec, outer_s)
    # A real cavity rather than a filled top disk.  Interior shape is assumed;
    # it carries no target feature or artificial high-contrast ring.
    inner_r = outer_radius(spec, inner_s) - spec.wall_thickness_m
    blend_start = start - min(.015, .25 * spec.neck_length_m)
    blend = np.clip((inner_s - blend_start) / (start - blend_start), 0, 1)
    blend = blend * blend * (3 - 2 * blend)
    inner_r = inner_r * (1 - blend) + (spec.neck_radius_m - spec.wall_thickness_m) * blend
    ring_s = np.r_[outer_s, inner_s]
    ring_r = np.r_[outer_r, inner_r]
    points = _ring_points(spec, ring_s, ring_r, angular_segments)
    vertices = points.reshape(-1, 3).tolist()
    texcoords = [[j / angular_segments, float(s / spec.height_m)]
                 for s in ring_s for j in range(angular_segments + 1)]
    faces, face_texcoords, face_regions = [], [], []
    for k in range(len(ring_s) - 1):
        if k < len(outer_s) - 1:
            region = 0  # external surface
        elif k == len(outer_s) - 1:
            region = 1  # mouth crown annulus, open in its centre
        else:
            region = 2  # interior surface
        for j in range(angular_segments):
            jnext = (j + 1) % angular_segments
            a, b = k * angular_segments + j, k * angular_segments + jnext
            c, d = (k + 1) * angular_segments + j, (k + 1) * angular_segments + jnext
            ta, tb = k * (angular_segments + 1) + j, k * (angular_segments + 1) + j + 1
            tc, td = (k + 1) * (angular_segments + 1) + j, (k + 1) * (angular_segments + 1) + j + 1
            faces.extend([(a, b, c), (b, d, c)])
            face_texcoords.extend([(ta, tb, tc), (tb, td, tc)])
            face_regions.extend([region, region])
    # Each bottom disk uses a centre vertex and triangle fan, with no collapsed
    # zero-radius ring or zero-area faces to upset mesh compilers.
    for ring, s, upward in [(0, 0., False), (len(ring_s) - 1, spec.bottom_thickness_m, True)]:
        centre_id = len(vertices)
        vertices.append(centerline(spec, s).tolist())
        centre_uv = len(texcoords)
        texcoords.append([.5, .5])
        rim_uv = len(texcoords)
        theta = 2 * np.pi * np.arange(angular_segments + 1) / angular_segments
        texcoords.extend(np.c_[.5 + .5 * np.cos(theta), .5 + .5 * np.sin(theta)].tolist())
        for j in range(angular_segments):
            a, b = ring * angular_segments + j, ring * angular_segments + (j + 1) % angular_segments
            if upward:
                faces.append((centre_id, a, b))
                face_texcoords.append((centre_uv, rim_uv + j, rim_uv + j + 1))
            else:
                faces.append((centre_id, b, a))
                face_texcoords.append((centre_uv, rim_uv + j + 1, rim_uv + j))
            face_regions.append(3 if not upward else 4)
    vertices = np.asarray(vertices, dtype=float)
    bridge_start, minimum_jacobian, orientation_risk = _neck_transition(spec)
    return {
        "vertices": vertices,
        "faces": np.asarray(faces, dtype=np.int32),
        "texcoords": np.asarray(texcoords, dtype=float),
        "face_texcoords": np.asarray(face_texcoords, dtype=np.int32),
        "face_regions": np.asarray(face_regions, dtype=np.int8),
        "face_region_names": {0: "outside", 1: "mouth_crown", 2: "inside", 3: "bottom_outside", 4: "floor_inside"},
        "profile_s_m": outer_s,
        "profile_radius_m": outer_r,
        "centerline_m": centerline(spec, outer_s),
        "outer_ring_vertex_indices": np.arange(len(outer_s) * angular_segments).reshape(-1, angular_segments),
        "bbox_min_m": vertices.min(axis=0),
        "bbox_max_m": vertices.max(axis=0),
        "geometry_diagnostics": {
            "local_collar_blend_start_s_m": bridge_start,
            "minimum_axial_jacobian": minimum_jacobian,
            "max_orientation_radius_over_axial_rate": orientation_risk,
            "minimum_axial_jacobian_limit": .20,
            "check_scope": "local_swept_surface_regularity; not_a_global_self_intersection_proof",
        },
        "target": target_metadata(spec),
    }
