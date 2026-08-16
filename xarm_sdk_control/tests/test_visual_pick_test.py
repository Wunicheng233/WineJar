from __future__ import annotations

from conftest import load_script_module


MODULE = load_script_module("visual_pick_test", "xarm_sdk_control/scripts/20_visual_pick_test.py")


def test_parse_args_defaults(monkeypatch) -> None:
    monkeypatch.setattr(MODULE.sys, "argv", ["20_visual_pick_test.py"])
    args = MODULE.parse_args()
    assert args.run is False
    assert args.confirm_no_pressure_sensor is False
    assert args.compression_mm == 6.0
    assert args.approach_clearance_mm == 35.0
    assert args.release_clearance_mm == 10.0
    assert args.pick_offset_x_mm == 10.0
    assert args.vacuum_settle_seconds == 0.9
    assert args.release_settle_seconds == 0.5
    assert args.inspection_hold_seconds == 2.0
    assert args.transit_speed == 80.0
    assert args.precision_speed == 15.0
    assert args.acc == 150.0
    assert args.min_score == 0.45


def test_run_requires_workspace_limits(monkeypatch) -> None:
    monkeypatch.setattr(
        MODULE.sys,
        "argv",
        ["20_visual_pick_test.py", "--run", "--confirm-no-pressure-sensor"],
    )
    assert MODULE.main() == 3


def test_run_requires_pressure_sensor_confirmation(monkeypatch) -> None:
    monkeypatch.setattr(
        MODULE.sys,
        "argv",
        [
            "20_visual_pick_test.py",
            "--run",
            "--min-x-mm", "-1000", "--max-x-mm", "1000",
            "--min-y-mm", "-1000", "--max-y-mm", "1000",
            "--min-z-mm", "-1000", "--max-z-mm", "1000",
        ],
    )
    assert MODULE.main() == 2


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
