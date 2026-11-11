"""Instance body-template registration after the neck becomes invisible.

Standard nearest-neighbour point-to-plane ICP, Huber IRLS; no CAD/target truth.
Constrained axis/translation modes express a conveyor motion assumption,
not extra observations. Information spectra are observability diagnostics,
NOT calibrated covariance under correlated sensor errors.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from .station_tying import pose


def voxel_points(points, size=.003):
    p = np.asarray(points, float)
    _, ids = np.unique(np.floor(p/size).astype(np.int64), axis=0, return_index=True)
    return p[np.sort(ids)]


def surface_normals(points, neighbours=20):
    p = np.asarray(points, float)
    _, ids = cKDTree(p).query(p, k=min(neighbours, len(p)))
    q = p[ids]-p[ids].mean(axis=1, keepdims=True)
    _, vectors = np.linalg.eigh(np.einsum('nki,nkj->nij', q, q))
    return vectors[:, :, 0]


def register_body(source, target, initial, *, mode='axis', direction=(1, 0, 0), max_iterations=35,
                  max_distance_m=.012, robust=True, normals=None):
    source, target = np.asarray(source, float), np.asarray(target, float)
    if min(len(source), len(target)) < 50 or not np.isfinite(np.r_[source.ravel(), target.ravel()]).all():
        return dict(accepted=False, reason='insufficient_body_observations', transform=np.array(initial))
    if mode not in ('axis', 'translation', 'rigid'):
        raise ValueError('unknown motion model')
    n = surface_normals(target) if normals is None else np.asarray(normals, float)
    tree, T = cKDTree(target), np.array(initial, float).copy()
    axis = np.asarray(direction, float)
    axis /= np.linalg.norm(axis)
    pivot = target.mean(axis=0)
    span = max(.05, float(np.sqrt(np.mean(np.sum((target-pivot)**2, axis=1)))))
    history = []
    for iteration in range(max_iterations):
        moved = source @ T[:3, :3].T+T[:3, 3]
        distances, ids = tree.query(moved)
        keep = distances < max_distance_m
        if keep.sum() < 50:
            return dict(accepted=False, reason='insufficient_overlap', transform=T)
        p, q, normal = moved[keep], target[ids[keep]], n[ids[keep]]
        residual = np.sum((p-q)*normal, axis=1)
        if mode == 'axis':
            J = (normal @ axis)[:, None]
        elif mode == 'translation':
            J = normal
        else:
            J = np.c_[normal, np.cross(p-pivot, normal)/span]
        scale = max(.0002, 1.4826*float(np.median(abs(residual-np.median(residual)))))
        weights = np.minimum(1., 1.345*scale/np.maximum(abs(residual), 1e-12)) if robust else np.ones(len(residual))
        delta = np.linalg.lstsq(J*np.sqrt(weights[:, None]), -residual*np.sqrt(weights), rcond=1e-9)[0]
        if mode == 'axis':
            shift, rot = axis*delta[0], np.eye(3)
        elif mode == 'translation':
            shift, rot = delta, np.eye(3)
        else:
            shift, rot = delta[:3], Rotation.from_rotvec(delta[3:]/span).as_matrix()
        update = pose(rot, pivot+shift-rot @ pivot)
        T = update @ T
        history.append(float(np.sqrt(np.mean(residual**2))))
        if np.linalg.norm(delta) < 1e-7:
            break
    moved = source @ T[:3, :3].T+T[:3, 3]
    distances, ids = tree.query(moved)
    keep = distances < max_distance_m
    p, q, normal = moved[keep], target[ids[keep]], n[ids[keep]]
    residual = np.sum((p-q)*normal, axis=1)
    J6 = np.c_[normal, np.cross(p-pivot, normal)/span]
    eigen = np.linalg.eigvalsh(J6.T @ J6/len(J6))
    return dict(accepted=True, reason='registration_candidate_not_process_release', transform=T,
        mode=mode, point_plane_rms_m=float(np.sqrt(np.mean(residual**2))),
        overlap_fraction=float(keep.mean()), iterations=iteration+1,
        information_eigenvalues=eigen, rotation_scale_m=span,
        information_is_not_calibrated_covariance=True, history_rms_m=history)
