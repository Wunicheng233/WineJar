"""Observation-only engineering controls for pose and diameter trade-offs.

Both channels use the same IR exposure: no independence or calibrated
uncertainty is claimed. Preserve both frozen predictions and rejection gates.
"""
from copy import deepcopy
import numpy as np
from .band_geometry import plan_band


def pose_radius_control(pose_row, radius_row, *, refit=False, width_m=.003):
    for key in ('case', 'condition', 'camera_set'):
        if pose_row[key] != radius_row[key]:
            raise ValueError('channels must refer to the same jar and exposure: '+key)
    result = deepcopy(pose_row)
    result['method'] = 'pose_fs_radius_refit' if refit else 'pose_fs_radius'
    result['backend'] = result['method']
    result['accepted'] = bool(pose_row['accepted'] and radius_row['accepted'])
    result['reasons'] = sorted(set(pose_row['reasons']+radius_row['reasons']))
    if not pose_row.get('estimate') or not radius_row.get('estimate'):
        result['accepted'] = False
        result['reasons'].append('missing_channel_estimate')
        return result
    radius = radius_row['estimate']['radius_m']
    details = dict(pose_source='simsense_standard_sor', radius_source='foundation_stereo_visible_profile',
                   shared_IR_inputs=True, independence_assumed=False, refit=refit,
                   uses_target_truth=False, radius_fallback=False)
    if refit:
        cloud = np.array(radius_row['diagnostics']['fit_point_positions_m'], float)
        center = np.array(pose_row['estimate']['center_m'], float)
        normal = np.array(pose_row['estimate']['normal'], float)
        normal /= np.linalg.norm(normal)
        delta = cloud-center
        axial = delta @ normal
        keep = abs(axial) <= width_m/2
        rho = np.linalg.norm(delta[keep]-axial[keep, None]*normal, axis=1)
        details['slab_width_m'], details['slab_points'] = width_m, int(keep.sum())
        if len(rho) >= 30:
            # Huber location in one dimension; robust scale from this slab,
            # rather than inheriting residual weights from a different axis.
            value = float(np.median(rho))
            scale = max(.0002, 1.4826*float(np.median(abs(rho-value))))
            for _ in range(30):
                weight = np.minimum(1., 1.345*scale/np.maximum(abs(rho-value), 1e-12))
                updated = float(weight @ rho/weight.sum())
                if abs(updated-value) < 1e-10:
                    break
                value = updated
            radius = value
            details['robust_scale_m'] = scale
        else:
            details.update(radius_fallback=True, fallback_reason='fewer_than_30_FS_slab_points')
    result['estimate']['radius_m'] = float(radius)
    # The pose/interval channel is unchanged. A different radius can still
    # change the TRUE tilted-band envelope, which must be evaluated afresh.
    result['band'] = plan_band(result['estimate'], result['diagnostics'], width_m)
    result['diagnostics']['channel_composition'] = details
    return result
