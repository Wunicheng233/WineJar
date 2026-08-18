from __future__ import annotations

import numpy as np

from conftest import load_script_module
from xarm_sdk_control.cross_place_motion import accumulated_layer_heights_mm, build_cycle_route


MODULE = load_script_module("complete_loading", "xarm_sdk_control/scripts/21_complete_loading.py")

HOME = [215.0, -2.0, 300.0, 178.0, 0.0, 0.0]


def test_parse_args_defaults(monkeypatch) -> None:
    monkeypatch.setattr(MODULE.sys, "argv", ["21_complete_loading.py"])
    args = MODULE.parse_args()
    assert args.lotus_layer_mm == 4.0
    assert args.paper_layer_mm == 0.5
    assert args.ceramic_layer_mm == 5.0
    assert args.shell_layer_mm == 3.0
    assert args.second_layer_height_mm == 3.0
    assert args.lotus_compression_mm == 2.0
    assert args.paper_compression_mm == 1.0
    assert args.ceramic_compression_mm == 6.0
    assert args.shell_compression_mm == 13.0
    assert args.first_place_yaw_offset_deg == 45.0
    assert args.second_place_yaw_offset_deg == -60.0
    assert args.max_legs == 5
    assert args.intermediate_home is False
    assert args.min_score == 0.35
    assert args.second_scan_min_score == 0.20


def test_run_requires_sensor_confirmation(monkeypatch) -> None:
    monkeypatch.setattr(MODULE.sys, "argv", ["21_complete_loading.py", "--run"])
    assert MODULE.main() == 2


def test_run_requires_workspace_limits(monkeypatch) -> None:
    monkeypatch.setattr(
        MODULE.sys,
        "argv",
        ["21_complete_loading.py", "--run", "--confirm-no-pressure-sensor"],
    )
    assert MODULE.main() == 3


def _reject_argv(monkeypatch, argv: list[str], expected: str) -> None:
    monkeypatch.setattr(MODULE.sys, "argv", ["21_complete_loading.py", *argv])
    try:
        MODULE.main()
    except ValueError as exc:
        assert expected in str(exc)
    else:
        raise AssertionError(f"{expected!r} must be rejected")


def test_negative_layer_height_is_rejected(monkeypatch) -> None:
    _reject_argv(monkeypatch, ["--paper-layer-mm", "-1"], "non-negative")


def test_leg_count_is_bounded(monkeypatch) -> None:
    _reject_argv(monkeypatch, ["--max-legs", "6"], "1~5")


def test_jar_point_requires_all_three_overrides(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(MODULE.sys, "argv", ["21_complete_loading.py", "--jar-x-mm", "300"])
    try:
        MODULE.main()
    except ValueError as exc:
        assert "together" in str(exc)
    else:
        raise AssertionError("Partial jar-point overrides must be rejected")


def test_accumulated_layer_heights_for_five_legs() -> None:
    assert accumulated_layer_heights_mm([4.0, 0.5, 5.0, 3.0, 3.0]) == [0.0, 4.0, 4.5, 9.5, 12.5]


def test_lotus_leg_route_targets_jar_mouth() -> None:
    jar_mouth_m = [0.35, -0.05, 0.10]
    route = build_cycle_route(
        home_pose=HOME,
        leaf_point_m=[0.30, 0.10, -0.02],
        ceramic_point_m=jar_mouth_m,
        place_yaw_offset_deg=0.0,
        transport_clearance_mm=100.0,
        approach_clearance_mm=25.0,
        compression_mm=2.0,
        release_clearance_mm=0.0,
        layer_height_mm=0.0,
        return_home=True,
    )
    poses = {label: pose for label, pose, _motion, _event in route}
    assert poses["lower leaf to release height"][:2] == [350.0, -50.0]
    assert poses["lower leaf to release height"][2] == 100.0  # jar z + release clearance + layer 0
    assert poses["lower leaf to release height"][5] == 0.0


def test_stack_a_presence_raises_staging_guidance(monkeypatch) -> None:
    from types import SimpleNamespace

    def fail(*_args, **_kwargs):
        raise RuntimeError("shifted")

    monkeypatch.setattr(MODULE, "choose_second_source_leaf", fail)
    args = SimpleNamespace(allow_validated_clipped_leaf=False)
    try:
        MODULE.require_stack_a_present(
            [{"point_base_m": [0.5, 0.5, 0.0], "label": "bamboo leaf"}],
            [0.3, 0.1, 0.0],
            np.zeros(3),
            args,
        )
    except RuntimeError as exc:
        assert "re-stage" in str(exc)
    else:
        raise AssertionError("A shifted stack A must abort with staging guidance")
