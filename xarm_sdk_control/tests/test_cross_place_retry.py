from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from conftest import RecordingArm, RecordingTimeline, RecordingVacuum, motion_args
from xarm_sdk_control import cross_place_retry as RETRY
from xarm_sdk_control.pick_verification import PickVerification


def verification(status: str) -> PickVerification:
    return PickVerification(status, "test", "bamboo", None, None, None)  # type: ignore[arg-type]


def retry_args(**overrides) -> SimpleNamespace:
    values = {
        **vars(motion_args()),
        "min_score": 0.45,
        "second_scan_min_score": 0.20,
        "max_pick_attempts": 3,
        "vision_env": "winejar-vision",
        "vision_socket": None,
        "require_vision_service": False,
        "allow_validated_clipped_leaf": False,
        "max_second_source_shift_mm": 80.0,
        "transport_clearance_mm": 100.0,
        "approach_clearance_mm": 25.0,
        "compression_mm": 13.0,
        "release_clearance_mm": 0.0,
        "precontact_clearance_mm": 5.0,
        "precision_lift_mm": 20.0,
        "simulate_first_pick_failure": False,
        "skip_second_controller_preflight": False,
        "min_x_mm": None,
        "max_x_mm": None,
        "min_y_mm": None,
        "max_y_mm": None,
        "min_z_mm": None,
        "max_z_mm": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


SOURCE = [0.287, 0.150, -0.044]
LIFTED = [0.287, 0.150, 0.149]
PLACEMENT = [0.400, -0.119, 0.049]
HOME = [215.194, -2.409, 300.0, 178.467, 0.581, -0.014]


def detection(point_base_m):
    return {
        "label": "bamboo leaf",
        "score": 0.9,
        "point_base_m": point_base_m,
        "touches_image_edge": False,
        "box_xyxy": [10.0, 10.0, 60.0, 200.0],
    }


class FakePlan:
    def __init__(self, label: str) -> None:
        self.label = label

    def to_dict(self) -> dict:
        return {"label": self.label}


def patch_retry_io(monkeypatch, tmp_path, verification_results, targeting_detections):
    """Monkeypatch run_scan / planner / executor / home for loop integration tests."""
    calls = {"home": 0, "verification_calls": 0, "targeting_calls": 0}

    def fake_run_scan(capture_root, base_t_camera, vision_env, min_score, **kwargs):
        if "verification" in str(capture_root):
            index = calls["verification_calls"]
            calls["verification_calls"] += 1
            return tmp_path / f"verify_{index}", [detection(verification_results[index])]
        calls["targeting_calls"] += 1
        return tmp_path / f"targeting_{calls['targeting_calls']}", [detection(point) for point in targeting_detections]

    def fake_plan(arm, start_pose, start_angles, route, args, run_controller_preflight=True):
        return [FakePlan(label) for label, _pose, _motion, _event in route]

    def fake_execute(arm, vacuum, route, args, holding_state, timeline, cycle_name,
                     segment_plans=None, **kwargs):
        callback = kwargs.get("event_callback")
        if callback is not None:
            callback("hold leaf above source for visual verification")

    def fake_home(arm, home_pose, speed, acceleration):
        calls["home"] += 1
        return list(map(float, HOME))

    monkeypatch.setattr(RETRY, "run_scan", fake_run_scan)
    monkeypatch.setattr(RETRY, "plan_route_motion_profile", fake_plan)
    monkeypatch.setattr(RETRY, "execute_route", fake_execute)
    monkeypatch.setattr(RETRY, "move_robot_home", fake_home)
    return calls


def test_decide_pick_retry_policy_mapping() -> None:
    assert RETRY.decide_pick_retry(verification("confirmed"), 1, 3).action == "proceed"
    assert RETRY.decide_pick_retry(verification("inconclusive"), 1, 3).action == "proceed"
    assert RETRY.decide_pick_retry(None, 1, 3).action == "proceed"
    assert RETRY.decide_pick_retry(verification("not_confirmed"), 2, 3).action == "recover_retry"
    assert RETRY.decide_pick_retry(verification("not_confirmed"), 3, 3).action == "abort"


def test_pick_not_confirmed_error_carries_verification_and_decision() -> None:
    check = verification("not_confirmed")
    decision = RETRY.decide_pick_retry(check, 1, 2)
    error = RETRY.PickNotConfirmedError(check, decision)
    assert error.verification is check
    assert error.decision.action == "recover_retry"
    assert "not confirmed" in str(error)


def test_verification_capture_exception_is_inconclusive(monkeypatch, tmp_path) -> None:
    def fail(capture_root, base_t_camera, vision_env, min_score, **kwargs):
        raise RuntimeError("camera gone")

    monkeypatch.setattr(RETRY, "run_scan", fail)
    frame, result = RETRY._capture_verification(
        capture_root=tmp_path / "v" / "captures",
        base_t_camera=np.eye(4),
        vision_env="winejar-vision",
        min_score=0.45,
        camera=None,
        vision_socket=None,
        require_service=False,
        position_bias_m=np.zeros(3),
        source_point_m=SOURCE,
        expected_object_point_m=LIFTED,
        allow_clipped_candidate=False,
        timeline=RecordingTimeline(),
        timeline_prefix="leaf_1.attempt_01.verification",
    )
    assert frame is None
    assert result.status == "inconclusive"
    assert "failed" in result.reason


def test_single_confirmed_attempt_succeeds_without_recovery(monkeypatch, tmp_path) -> None:
    calls = patch_retry_io(monkeypatch, tmp_path, [LIFTED], [SOURCE])
    outcome = RETRY.run_leaf_pick_with_retry(
        leaf_index=1,
        arm=RecordingArm(),
        vacuum=RecordingVacuum(),
        home_pose=HOME,
        placement_target_m=PLACEMENT,
        initial_target_m=SOURCE,
        source_stack_reference_m=SOURCE,
        place_yaw_offset_deg=45.0,
        layer_height_mm=0.0,
        return_home=False,
        args=retry_args(),
        holding_state={"value": False},
        timeline=RecordingTimeline(),
        run_dir=tmp_path,
        base_t_camera=np.eye(4),
        position_bias_m=np.zeros(3),
        camera=None,
    )
    assert outcome.success is True
    assert len(outcome.attempts) == 1
    assert outcome.final_verification.status == "confirmed"
    assert calls["home"] == 0


def test_at_source_first_then_lifted_succeeds_with_recovery(monkeypatch, tmp_path) -> None:
    calls = patch_retry_io(monkeypatch, tmp_path, [SOURCE, LIFTED], [SOURCE])
    outcome = RETRY.run_leaf_pick_with_retry(
        leaf_index=1,
        arm=RecordingArm(),
        vacuum=RecordingVacuum(),
        home_pose=HOME,
        placement_target_m=PLACEMENT,
        initial_target_m=SOURCE,
        source_stack_reference_m=SOURCE,
        place_yaw_offset_deg=45.0,
        layer_height_mm=0.0,
        return_home=False,
        args=retry_args(),
        holding_state={"value": False},
        timeline=RecordingTimeline(),
        run_dir=tmp_path,
        base_t_camera=np.eye(4),
        position_bias_m=np.zeros(3),
        camera=None,
    )
    assert outcome.success is True
    assert len(outcome.attempts) == 2
    assert outcome.attempts[0]["recovery_completed"] is True
    assert outcome.attempts[0]["retry_decision"]["action"] == "recover_retry"
    assert outcome.attempts[1]["retry_decision"]["action"] == "proceed"
    assert calls["home"] == 1
    assert calls["targeting_calls"] == 1


def test_always_at_source_exhausts_attempts(monkeypatch, tmp_path) -> None:
    calls = patch_retry_io(monkeypatch, tmp_path, [SOURCE, SOURCE], [SOURCE])
    outcome = RETRY.run_leaf_pick_with_retry(
        leaf_index=2,
        arm=RecordingArm(),
        vacuum=RecordingVacuum(),
        home_pose=HOME,
        placement_target_m=PLACEMENT,
        initial_target_m=SOURCE,
        source_stack_reference_m=SOURCE,
        place_yaw_offset_deg=-60.0,
        layer_height_mm=3.0,
        return_home=True,
        args=retry_args(max_pick_attempts=2),
        holding_state={"value": False},
        timeline=RecordingTimeline(),
        run_dir=tmp_path,
        base_t_camera=np.eye(4),
        position_bias_m=np.zeros(3),
        camera=None,
    )
    assert outcome.success is False
    assert len(outcome.attempts) == 2
    assert outcome.attempts[-1]["retry_decision"]["action"] == "abort"
    assert calls["home"] == 2
    assert calls["targeting_calls"] == 1


def test_outcome_cycle_record_preserves_baseline_keys() -> None:
    outcome = RETRY.LeafPickOutcome(
        success=True,
        leaf_index=1,
        source_point_m=SOURCE,
        attempts=[
            {
                "attempt": 1,
                "targeting_frame": "/tmp/frame",
                "route": [("lift", [0.0] * 6, "loaded_vertical", None)],
                "joint_speed_planning_seconds": 0.5,
                "joint_speed_plan": [{"label": "lift"}],
                "controller_preflight": {"checks_run": 1, "passed": True},
            }
        ],
        final_verification=verification("confirmed"),
    )
    record = outcome.cycle_record()
    for key in (
        "leaf",
        "frame_dir",
        "route",
        "joint_speed_planning_seconds",
        "joint_speed_plan",
        "controller_preflight",
    ):
        assert key in record
    assert record["completed"] is True
    assert record["pick_attempts_used"] == 1
    assert record["final_pick_verification_status"] == "confirmed"
