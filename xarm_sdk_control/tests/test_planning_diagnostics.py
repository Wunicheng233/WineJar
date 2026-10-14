"""Check timing against a simulated clock, without connecting to a controller."""
import pytest

from conftest import motion_args
from xarm_sdk_control import cross_place_motion as motion


@pytest.mark.parametrize("flip_interior, expected_methods", [
    (False, ["dense"]), (True, ["dense", "dense", "dense", "endpoint"]),
])
def test_planning_records_ik_retries_and_preflight_separately(monkeypatch, flip_interior, expected_methods):
    now = [0.0]
    calls = []

    class Arm:
        joint_speed_limit = (0, 100.0)
        joint_acc_limit = (0, 200.0)

        def get_inverse_kinematics(self, pose, **kwargs):
            calls.append((pose, kwargs))
            now[0] += .003
            angle = pose[0] * .1
            if flip_interior and pose[0] < 20:
                angle += 90
            return 0, [angle, 0, 0, 0, 0, 0]

    arm = Arm()

    def preflight(actual_arm, requests, plans, radii):
        assert actual_arm is arm
        now[0] += .020

    monkeypatch.setattr(motion.time, "perf_counter", lambda: now[0])
    monkeypatch.setattr(motion, "preflight_cartesian_route", preflight)
    result = {}
    plans = motion.plan_route_motion_profile(
        arm, [0.0] * 6, [0.0] * 6,
        [("move", [20.0, 0, 0, 0, 0, 0], "empty_transit", None)],
        motion_args(), preflight_result=result,
    )
    diagnostics = result["planning_diagnostics"]
    attempts = diagnostics["estimation_attempts"]
    assert [a["method"] for a in attempts] == expected_methods
    assert [a["status"] for a in attempts] == ["failed"] * (len(attempts) - 1) + ["passed"]
    assert attempts[-1]["error"] is None
    for attempt in attempts:
        assert attempt["ik_calls"] > 0
        assert attempt["ik_seconds"] == pytest.approx(attempt["ik_calls"] * .003)
        assert attempt["seconds"] == pytest.approx(attempt["ik_seconds"])
    assert sum(a["ik_calls"] for a in attempts) == len(calls)
    assert all(kwargs["limited"] and not kwargs["input_is_radian"] for _, kwargs in calls)
    assert diagnostics["controller_preflight_seconds"] == pytest.approx(.020)
    assert result["status"] == "passed"
    assert plans[0].end_joint_angles_deg == pytest.approx([2, 0, 0, 0, 0, 0])


def test_orientation_only_route_cannot_bypass_rejection_via_fallback(monkeypatch):
    class Arm:
        joint_speed_limit = (0, 100.0)
        joint_acc_limit = (0, 200.0)
        def get_inverse_kinematics(self, *_args, **_kwargs):
            pytest.fail("Unsupported route must be rejected before IK")

    monkeypatch.setattr(motion, "preflight_cartesian_route",
                        lambda *_args: pytest.fail("Rejected route reached controller preflight"))
    result = {}
    with pytest.raises(motion.TrajectoryPlanningError, match="Pure orientation"):
        motion.plan_route_motion_profile(
            Arm(), [0.] * 6, [0.] * 6,
            [("rotate", [0, 0, 0, 0, 0, 30], "empty_transit", None)],
            motion_args(), preflight_result=result,
        )
    attempts = result["planning_diagnostics"]["estimation_attempts"]
    assert [attempt["method"] for attempt in attempts] == ["dense"] * 3 + ["endpoint"]
    assert all(attempt["status"] == "failed" and attempt["ik_calls"] == 0 for attempt in attempts)
