from __future__ import annotations

from conftest import load_script_module


MODULE = load_script_module("visual_pick_test", "xarm_sdk_control/scripts/20_visual_pick_test.py")


def test_parse_args_defaults(monkeypatch) -> None:
    monkeypatch.setattr(MODULE.sys, "argv", ["20_visual_pick_test.py"])
    args = MODULE.parse_args()
    assert args.run is False
    assert args.material == "ceramic"
    assert args.compression_mm is None
    assert args.test_lift_mm is None
    assert args.min_score is None
    assert args.approach_clearance_mm == 35.0
    assert args.release_clearance_mm == 10.0
    assert args.pick_offset_x_mm == 0.0
    assert args.pick_offset_y_mm == 0.0
    assert args.vacuum_settle_seconds == 0.9
    assert args.release_settle_seconds == 0.5
    assert args.inspection_hold_seconds == 2.0
    assert args.transit_speed == 80.0
    assert args.precision_speed == 15.0
    assert args.acc == 150.0


def test_per_material_defaults_resolve(monkeypatch) -> None:
    from xarm_sdk_control.pick_test_workflow import profile_for, resolve_pick_test_settings

    assert resolve_pick_test_settings(profile_for("ceramic")) == (0.45, 6.0, 40.0)
    assert resolve_pick_test_settings(profile_for("lotus-leaf")) == (0.35, 2.0, 40.0)
    assert resolve_pick_test_settings(profile_for("paper")) == (0.35, 1.0, 40.0)
    assert resolve_pick_test_settings(profile_for("ceramic"), compression_mm=8.0) == (0.45, 8.0, 40.0)


def test_material_choices_reject_unknown(monkeypatch) -> None:
    monkeypatch.setattr(MODULE.sys, "argv", ["20_visual_pick_test.py", "--material", "metal"])
    try:
        MODULE.parse_args()
    except SystemExit:
        pass
    else:
        raise AssertionError("Unknown materials must be rejected by argparse")


def test_run_requires_workspace_limits(monkeypatch) -> None:
    monkeypatch.setattr(MODULE.sys, "argv", ["20_visual_pick_test.py", "--run"])
    assert MODULE.main() == 3


def test_confirm_no_pressure_sensor_flag_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        MODULE.sys,
        "argv",
        ["20_visual_pick_test.py", "--confirm-no-pressure-sensor"],
    )
    try:
        MODULE.parse_args()
    except SystemExit:
        pass
    else:
        raise AssertionError("--confirm-no-pressure-sensor must be rejected")


def _reject_argv(monkeypatch, argv: list[str], expected: str) -> None:
    monkeypatch.setattr(MODULE.sys, "argv", ["20_visual_pick_test.py", *argv])
    try:
        MODULE.main()
    except ValueError as exc:
        assert expected in str(exc)
    else:
        raise AssertionError(f"{expected!r} must be rejected")


def test_negative_hold_time_is_rejected(monkeypatch) -> None:
    _reject_argv(monkeypatch, ["--inspection-hold-seconds", "-1"], "non-negative")


def test_compression_limit_is_enforced(monkeypatch) -> None:
    _reject_argv(monkeypatch, ["--compression-mm", "31"], "spring travel")


def test_negative_approach_clearance_is_rejected(monkeypatch) -> None:
    _reject_argv(monkeypatch, ["--approach-clearance-mm", "-5"], "non-negative")
