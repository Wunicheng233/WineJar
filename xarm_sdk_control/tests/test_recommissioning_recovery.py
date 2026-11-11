"""Portable checks for the recovered workcell helpers; never connect hardware."""
import json

import cv2
import numpy as np
import pytest

from xarm_sdk_control.experiments.recommissioning_20261014 import full_loading_recovery as recovery


def test_rgbd_point_uses_frame_scale_and_candidate_transform(tmp_path, monkeypatch):
    depth = np.full((20, 20), 500, dtype=np.uint16)
    cv2.imwrite(str(tmp_path/'depth_raw.png'), depth)
    (tmp_path/'frame.json').write_text(json.dumps(dict(
        stream=dict(depth_scale_m_per_unit=.001),
        intrinsics=dict(camera_matrix=[[100, 0, 10], [0, 100, 10], [0, 0, 1]],
                        distortion_coefficients=[0, 0, 0, 0, 0]))))
    transform = np.eye(4)
    transform[:3, 3] = [.1, .2, .3]
    candidate = tmp_path/'candidate.json'
    candidate.write_text(json.dumps(dict(base_T_camera=transform.tolist())))
    monkeypatch.setattr(recovery, 'CAL', candidate)
    assert recovery.point(tmp_path, [12, 10]) == pytest.approx([110, 200, 800])
    depth[7:14, 9:16] = 0
    cv2.imwrite(str(tmp_path/'depth_raw.png'), depth)
    with pytest.raises(RuntimeError, match='Unstable depth'):
        recovery.point(tmp_path, [12, 10])


class QueryOnlyArm:
    """No motion or GPIO methods: preflight must only query the controller."""
    def __init__(self, tcp_limit=False, joint_limit=False, ik_code=0, jump=0):
        self.tcp_limit = tcp_limit
        self.joint_limit = joint_limit
        self.ik_code = ik_code
        self.jump = jump

    def get_servo_angle(self):
        return 0, [0]*6

    def get_inverse_kinematics(self, pose, **kwargs):
        return self.ik_code, [self.jump]*6

    def is_tcp_limit(self, pose, **kwargs):
        return 0, self.tcp_limit

    def is_joint_limit(self, angles, **kwargs):
        return 0, self.joint_limit


@pytest.mark.parametrize('kwargs', [dict(tcp_limit=True), dict(joint_limit=True),
                                   dict(ik_code=-1), dict(jump=21)])
def test_recovered_route_rejects_current_controller_failure(kwargs):
    with pytest.raises(RuntimeError):
        recovery.preflight(QueryOnlyArm(**kwargs), [0]*6,
                           [dict(label='test', pose=[10, 0, 0, 0, 0, 0])])


def test_recovered_route_checks_translation_and_rotation_samples():
    checks = recovery.preflight(QueryOnlyArm(), [0]*6,
                               [dict(label='test', pose=[25, 0, 0, 0, 0, 10])])
    assert len(checks) == 5
    assert checks[-1]['pose'] == [25, 0, 0, 0, 0, 10]
    assert all(not c['tcp_limit'] and not c['joint_limit'] for c in checks)
