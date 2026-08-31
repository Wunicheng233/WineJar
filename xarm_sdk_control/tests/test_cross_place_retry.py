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

    def fake_plan(arm, start_pose, start_angles, route, args, run_controller_preflight=True,
                  preflight_result=None):
        if preflight_result is not None:
            preflight_result.update({"status": "passed", "error": None})
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


class SuctionErrorArm(RecordingArm):
    def __init__(self, fail_times: int) -> None:
        super().__init__()
        self.fail_times = fail_times
        self.reads = 0

    def get_cgpio_digital(self, channel):
        self.reads += 1
        value = 1 if self.reads <= self.fail_times else 0
        return 0, value


def test_decide_pick_retry_on_suction_failure_policy() -> None:
    assert RETRY.decide_pick_retry_on_suction_failure(2, 3).action == "recover_retry"
    assert RETRY.decide_pick_retry_on_suction_failure(3, 3).action == "abort"


def test_suction_failure_then_success_recovers_and_retries(monkeypatch, tmp_path) -> None:
    from xarm_sdk_control.suction_sensor import SuctionNotConfirmedError, SuctionSensorConfig, SuctionSensor, SuctionWaitOutcome

    calls = patch_retry_io(monkeypatch, tmp_path, [LIFTED], [SOURCE])
    arm = RecordingArm()
    sensor_arm = SuctionErrorArm(4)
    sensor = SuctionSensor(
        sensor_arm,
        SuctionSensorConfig(input_channel=0, success_value=0, debounce_seconds=0.05, timeout_seconds=2.0),
        clock=lambda: sensor_arm.reads * 0.02,
        sleep=lambda _s: None,
    )

    real_execute = RETRY.execute_route

    def fake_execute(*args, **kwargs):
        # First attempt: sensor never confirms (reads fail) -> real path raises.
        # Simulate via the real execute_route? Simpler: raise the error directly on call 1.
        if not fake_execute.calls:
            raise SuctionNotConfirmedError(0, SuctionWaitOutcome(False, 2.0, 2.0, 1))
        fake_execute.calls += 1

    fake_execute.calls = 0

    def wrapped_execute(arm, vacuum, route, args, holding_state, timeline, cycle_name,
                        segment_plans=None, **kwargs):
        if wrapped_execute.calls == 0:
            wrapped_execute.calls += 1
            raise SuctionNotConfirmedError(0, SuctionWaitOutcome(False, 2.0, 2.0, 1))
        wrapped_execute.calls += 1

    wrapped_execute.calls = 0
    monkeypatch.setattr(RETRY, "execute_route", wrapped_execute)

    outcome = RETRY.run_leaf_pick_with_retry(
        leaf_index=1,
        arm=arm,
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
        inspection_event=None,
    )

    assert outcome.success is True
    assert len(outcome.attempts) == 2
    assert outcome.attempts[0]["recovery_completed"] is True
    assert outcome.attempts[0]["failure_type"] == "suction_not_confirmed"
    assert "suction_failure" in outcome.attempts[0]
    assert calls["home"] == 1
    assert calls["targeting_calls"] == 1


def test_pressure_only_loop_skips_verification(monkeypatch, tmp_path) -> None:
    calls = patch_retry_io(monkeypatch, tmp_path, [LIFTED], [SOURCE])
    captured = {}

    def spy_execute(arm, vacuum, route, args, holding_state, timeline, cycle_name,
                    segment_plans=None, **kwargs):
        captured["event_callback"] = kwargs.get("event_callback")
        captured["suction_sensor"] = kwargs.get("suction_sensor")

    monkeypatch.setattr(RETRY, "execute_route", spy_execute)

    RETRY.run_leaf_pick_with_retry(
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
        inspection_event=None,
    )

    assert captured["event_callback"] is None
    assert captured["suction_sensor"] is None
    assert calls["verification_calls"] == 0


def test_retry_target_injection_uses_prompt_and_selector(monkeypatch, tmp_path) -> None:
    from xarm_sdk_control.suction_sensor import SuctionNotConfirmedError, SuctionWaitOutcome

    captured = {"prompts": [], "selector_called": 0}

    def fake_run_scan(capture_root, base_t_camera, vision_env, min_score, **kwargs):
        captured["prompts"].append(kwargs.get("text_prompt"))
        if "targeting" in str(capture_root):
            captured["selector_called"] += 1
            return tmp_path / "targeting", [detection(SOURCE)]
        return tmp_path / "verify", [detection(LIFTED)]

    def fake_plan(arm, start_pose, start_angles, route, args, run_controller_preflight=True,
                  preflight_result=None):
        if preflight_result is not None:
            preflight_result.update({"status": "passed", "error": None})
        return [FakePlan(label) for label, _p, _m, _e in route]

    def fake_execute(arm, vacuum, route, args, holding_state, timeline, cycle_name,
                     segment_plans=None, **kwargs):
        if fake_execute.calls == 0:
            fake_execute.calls += 1
            raise SuctionNotConfirmedError(0, SuctionWaitOutcome(False, 2.0, 2.0, 1))
        fake_execute.calls += 1

    fake_execute.calls = 0
    monkeypatch.setattr(RETRY, "run_scan", fake_run_scan)
    monkeypatch.setattr(RETRY, "plan_route_motion_profile", fake_plan)
    monkeypatch.setattr(RETRY, "execute_route", fake_execute)
    monkeypatch.setattr(RETRY, "move_robot_home", lambda *a, **k: HOME)

    def selector(dets, reference):
        return dets[0]

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
        inspection_event=None,
        retry_scan_prompt="lotus leaf.",
        retry_select=selector,
    )

    assert outcome.success is True
    assert captured["prompts"] == ["lotus leaf."]
    assert captured["selector_called"] == 1
