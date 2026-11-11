"""Evaluation-only sufficient clearance certificate for rear-hinged jaws.

Sutherland-Hodgman slab clipping of exact mesh triangles avoids interpreting
a coarse point-sampling bound as a collision. No contact/tension is modelled.
"""
import numpy as np
from .station_tying import GunSpec
from .tying_cover import cover_mesh
from .tying_evaluation import _obj_triangles


def _clip_z(poly, level, lower):
    output = []
    if not len(poly):
        return output
    previous = poly[-1]
    prev_in = previous[2] >= level if lower else previous[2] <= level
    for current in poly:
        current_in = current[2] >= level if lower else current[2] <= level
        if current_in != prev_in:
            t = (level-previous[2])/(current[2]-previous[2])
            output.append(previous+t*(current-previous))
        if current_in:
            output.append(current)
        previous, prev_in = current, current_in
    return output


def slab_radial_envelope(triangles_local, low, high):
    """Maximum radial norm on a clipped triangle mesh, not vertex-only cropping.

    Norm is convex, so each clipped polygon attains its maximum at a vertex.
    New vertices at both slab planes must also be included.
    """
    tri = np.asarray(triangles_local, float)
    if low > high or tri.ndim != 3 or tri.shape[1:] != (3, 3) or not np.isfinite(tri).all():
        raise ValueError('invalid triangle mesh or slab')
    z = tri[:, :, 2]
    keep = (z.max(axis=1) >= low) & (z.min(axis=1) <= high)
    tri, z = tri[keep], z[keep]
    if not len(tri):
        return 0.
    inside = (z.min(axis=1) >= low) & (z.max(axis=1) <= high)
    maximum = float(np.linalg.norm(tri[inside, :, :2], axis=-1).max()) if inside.any() else 0.
    for item in tri[~inside]:
        clipped = _clip_z(_clip_z(item, low, True), high, False)
        if clipped:
            maximum = max(maximum, float(np.linalg.norm(np.array(clipped)[:, :2], axis=1).max()))
    return maximum


def head_bore_certificate(obj_path, spec, config, plan, *, jar_triangles=None, cover_triangles=None):
    """Sufficient, NOT necessary: whole relevant surface fits the inner bore.

    For each outward-opened semicircle, ideal centreline rho >= arc radius.
    A chord can intrude by at most the known polygon sagitta. Thus the bore
    is free throughout opening/closing; axial tube extent enlarges the slab.
    A failed certificate is unresolved, never declared an actual collision.
    """
    gun = GunSpec(**config['gun'])
    if not plan.get('waypoints'):
        return dict(certified=False, status='no_plan', minimum_margin_mm=None)
    T = np.array(plan['W_T_G_t0'])
    jar_tri = _obj_triangles(obj_path) if jar_triangles is None else np.asarray(jar_triangles)
    if cover_triangles is None:
        cover = cover_mesh(spec, config)
        cover_tri = cover['vertices'][cover['faces']] @ cover['rotation'].T+cover['origin']
    else:
        cover_tri = np.asarray(cover_triangles)
    combined = np.concatenate([jar_tri, cover_tri])
    local = (combined-T[:3, 3]) @ T[:3, :3]
    half = gun.head_tube_radius_m
    envelope = slab_radial_envelope(local, -half, plan['descent_distance_m']+half)
    sagitta = gun.arc_radius_m*(1-np.cos(np.pi/(2*gun.arc_segments_per_half)))
    margin = gun.inner_diameter_m/2-sagitta-envelope
    return dict(certified=bool(margin > 0), status='sufficient_head_bore_clearance' if margin > 0 else 'unresolved_not_collision',
                minimum_margin_mm=float(margin*1000), max_object_radial_envelope_mm=float(envelope*1000),
                polygon_sagitta_mm=float(sagitta*1000),
                scope='exact_triangle_mesh_plus_assumed_cover_and_two_jaw_tube_only',
                no_physical_reconstruction_or_material_validation=True)
