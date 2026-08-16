from __future__ import annotations

from types import SimpleNamespace

import pytest

from conftest import load_script_module
from xarm_sdk_control.pick_verification import PickVerification


MODULE = load_script_module("visual_pick_test", "xarm_sdk_control/scripts/20_visual_pick_test.py")


def args(**overrides):
    values = {
        "min_score": None,
        "compression_mm": None,
        "vacuum_settle_seconds": 0.9,
        "release_settle_seconds": 0.5,
        "inspection_hold_seconds": 2.0,
        "verify_pick": False,
        "max_attempts": 1,
        "simulate_first_pick_failure": False,
        "transit_speed": 80.0,
        "precision_speed": 15.0,
        "acc": 150.0,
        "run": False,
        "confirm_no_pressure_sensor": False,
        "min_x_mm": None,
        "max_x_mm": None,
        "min_y_mm": None,
        "max_y_mm": None,
        "min_z_mm": None,
        "max_z_mm": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_ceramic_default_compression_is_21_mm() -> None:
    score, compression = MODULE.validate_args(args(), MODULE.profile_for("ceramic"))

    assert score == 0.45
    assert compression == 21.0


def test_real_motion_requires_workspace_and_no_sensor_confirmation() -> None:
    with pytest.raises(ValueError, match="workspace"):
        MODULE.validate_args(
            args(run=True, confirm_no_pressure_sensor=True),
            MODULE.profile_for("ceramic"),
        )


def test_precision_steps_use_precision_speed() -> None:
    step = MODULE.PickTestStep("contact", [0, 0, 0, 0, 0, 0], "contact", "acquire")

    assert MODULE.speed_for(step, args()) == 15.0


def test_simulated_failure_requires_two_closed_loop_attempts() -> None:
    with pytest.raises(ValueError, match="requires --verify-pick"):
        MODULE.validate_args(
            args(simulate_first_pick_failure=True),
            MODULE.profile_for("ceramic"),
        )

    workspace = {
        "min_x_mm": 180.0,
        "max_x_mm": 520.0,
        "min_y_mm": -180.0,
        "max_y_mm": 180.0,
        "min_z_mm": -85.0,
        "max_z_mm": 500.0,
    }
    with pytest.raises(ValueError, match="max-attempts 2"):
        MODULE.validate_args(
            args(
                run=True,
                confirm_no_pressure_sensor=True,
                verify_pick=True,
                simulate_first_pick_failure=True,
                **workspace,
            ),
            MODULE.profile_for("ceramic"),
        )


def test_two_attempt_closed_loop_configuration_is_valid() -> None:
    score, compression = MODULE.validate_args(
        args(
            run=True,
            confirm_no_pressure_sensor=True,
            verify_pick=True,
            max_attempts=2,
            simulate_first_pick_failure=True,
            min_x_mm=180.0,
            max_x_mm=520.0,
            min_y_mm=-180.0,
            max_y_mm=180.0,
            min_z_mm=-85.0,
            max_z_mm=500.0,
        ),
        MODULE.profile_for("ceramic"),
    )

    assert (score, compression) == (0.45, 21.0)


def test_unconfirmed_closed_loop_result_has_nonzero_exit_code() -> None:
    failed = PickVerification("not_confirmed", "at source", "ceramic", None, 0.0, 0.0)
    confirmed = PickVerification("confirmed", "lifted", "ceramic", None, 25.0, 0.0)

    assert MODULE.result_exit_code(run_mode=True, verify_pick=True, final_verification=failed) == 2
    assert MODULE.result_exit_code(run_mode=True, verify_pick=True, final_verification=confirmed) == 0
    assert MODULE.result_exit_code(run_mode=True, verify_pick=False, final_verification=None) == 0
