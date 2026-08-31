from __future__ import annotations

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
    assert args.paper_compression_mm == 17.0
    assert args.ceramic_compression_mm == 7.0
    assert args.shell_compression_mm == 13.0
    assert args.ceramic_pick_offset_x_mm == 8.0
    assert args.first_place_yaw_offset_deg == 45.0
    assert args.second_place_yaw_offset_deg == -60.0
    assert args.max_legs == 5
    assert args.max_pick_attempts == 3
    assert args.skip_lotus is False
    assert args.start_material is None
    assert args.intermediate_home is False
    assert args.min_score == 0.35
    assert args.second_scan_min_score == 0.10
    assert args.shell_scan_min_score == 0.20
    assert args.shell_source_min_x_mm == 400.0
    assert args.shell_source_max_x_mm == 540.0
    assert args.shell_source_min_y_mm == 50.0
    assert args.shell_source_max_y_mm == 180.0
    assert args.shell_min_box_elongation == 2.2
    assert args.minimum_planned_speed_mm_s == 3.0
    assert args.transit_speed == 250.0
    assert args.loaded_transit_speed == 180.0
    assert args.vertical_speed == 170.0
    assert args.loaded_vertical_speed == 100.0
    assert args.precision_speed == 50.0
    assert args.contact_speed == 15.0
    assert args.acc == 400.0
    assert args.loaded_acc == 200.0
    assert args.precision_acc == 160.0
    assert args.joint_velocity_utilization == 0.35
    assert args.joint_acceleration_utilization == 0.35


def test_ceramic_position_bias_is_material_specific() -> None:
    base = MODULE.np.asarray([-0.015, 0.035, 0.0])

    ceramic = MODULE.material_position_bias_m(base, "ceramic", 8.0)
    paper = MODULE.material_position_bias_m(base, "paper", 8.0)

    assert MODULE.np.allclose(ceramic, [-0.007, 0.035, 0.0])
    assert MODULE.np.allclose(paper, base)
    assert MODULE.np.allclose(base, [-0.015, 0.035, 0.0])


def _shell_detection(
    point_mm: list[float],
    *,
    score: float,
    box: list[float],
    clipped: bool = False,
) -> dict:
    return {
        "label": "bamboo leaf",
        "score": score,
        "point_base_m": [value / 1000.0 for value in point_mm],
        "box_xyxy": box,
        "touches_image_edge": clipped,
    }


def test_shell_source_gate_rejects_high_confidence_lotus_misclassification() -> None:
    detections = [
        _shell_detection([228.0, -73.0, 33.0], score=0.80, box=[0, 0, 215, 239], clipped=True),
        _shell_detection([467.0, 134.0, -33.0], score=0.34, box=[74, 244, 635, 423]),
    ]

    selected = MODULE.select_shell_source_candidate(
        detections,
        min_x_mm=400.0,
        max_x_mm=540.0,
        min_y_mm=50.0,
        max_y_mm=180.0,
        min_box_elongation=2.2,
    )

    assert selected["point_base_m"] == [0.467, 0.134, -0.033]


def test_shell_source_gate_rejects_nonelongated_candidate_inside_region() -> None:
    detections = [
        _shell_detection([467.0, 134.0, -33.0], score=0.90, box=[100, 100, 300, 280]),
    ]

    try:
        MODULE.select_shell_source_candidate(
            detections,
            min_x_mm=400.0,
            max_x_mm=540.0,
            min_y_mm=50.0,
            max_y_mm=180.0,
            min_box_elongation=2.2,
        )
    except RuntimeError as exc:
        assert "source region" in str(exc)
    else:
        raise AssertionError("A broad non-shell mask must not pass the source gate")


def test_shell_source_gate_allows_minor_axial_clip_when_validated() -> None:
    detection = _shell_detection(
        [464.2, 140.9, -37.2],
        score=0.32,
        box=[72.4, 244.4, 638.3, 418.0],
        clipped=True,
    )
    detection["edge_sides"] = ["right"]

    selected = MODULE.select_shell_source_candidate(
        [detection],
        min_x_mm=400.0,
        max_x_mm=540.0,
        min_y_mm=50.0,
        max_y_mm=180.0,
        min_box_elongation=2.2,
        allow_axial_clipping=True,
    )

    assert MODULE.np.allclose(selected["point_base_m"], [0.4642, 0.1409, -0.0372])


def test_run_requires_workspace_limits(monkeypatch) -> None:
    monkeypatch.setattr(MODULE.sys, "argv", ["21_complete_loading.py", "--run"])
    assert MODULE.main() == 3


def test_confirm_no_pressure_sensor_flag_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        MODULE.sys,
        "argv",
        ["21_complete_loading.py", "--confirm-no-pressure-sensor"],
    )
    try:
        MODULE.parse_args()
    except SystemExit:
        pass
    else:
        raise AssertionError("--confirm-no-pressure-sensor must be rejected")


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


def test_skip_lotus_flag_is_available(monkeypatch) -> None:
    monkeypatch.setattr(MODULE.sys, "argv", ["21_complete_loading.py", "--skip-lotus"])
    assert MODULE.parse_args().skip_lotus is True


def test_skip_lotus_preserves_preplaced_layer_height() -> None:
    all_legs = [
        ("lotus", 0.0, 2.0, 4.0),
        ("paper", 0.0, 1.0, 0.5),
        ("ceramic", 0.0, 6.0, 5.0),
        ("shell_1", 45.0, 13.0, 3.0),
        ("shell_2", -60.0, 13.0, 3.0),
    ]
    heights = accumulated_layer_heights_mm([leg[3] for leg in all_legs])

    selected = MODULE.select_execution_legs(all_legs, heights, skip_lotus=True, max_legs=5)

    assert [leg[0] for leg in selected] == ["paper", "ceramic", "shell_1", "shell_2"]
    assert [leg[4] for leg in selected] == [4.0, 4.5, 9.5, 12.5]


def test_resume_at_shell_preserves_preplaced_layer_heights() -> None:
    all_legs = [
        ("lotus", 0.0, 2.0, 4.0),
        ("paper", 0.0, 20.0, 0.5),
        ("ceramic", 0.0, 10.0, 5.0),
        ("shell_1", 45.0, 16.0, 3.0),
        ("shell_2", -60.0, 16.0, 3.0),
    ]
    heights = accumulated_layer_heights_mm([leg[3] for leg in all_legs])

    selected = MODULE.select_execution_legs(
        all_legs,
        heights,
        skip_lotus=True,
        max_legs=5,
        start_material="shell_1",
    )

    assert [leg[0] for leg in selected] == ["shell_1", "shell_2"]
    assert [leg[4] for leg in selected] == [9.5, 12.5]


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
