from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from xarm_sdk_control import visual_targeting
from xarm_sdk_control.geometry import undirected_axis_delta_deg


@pytest.mark.parametrize("failed", [False, True])
def test_scan_records_capture_diagnostics_on_success_and_failure(monkeypatch, tmp_path, failed):
    from types import SimpleNamespace
    from xarm_sdk_control.task_timing import TaskTimeline

    diagnostics = {"status": "failed" if failed else "success", "restart_count": 1}
    error = RuntimeError("camera read timeout")
    def capture(_):
        if failed:
            raise error
        return tmp_path
    camera = SimpleNamespace(capture=capture, last_capture_diagnostics=diagnostics)
    timeline = TaskTimeline()
    monkeypatch.setattr(visual_targeting, "run_vision", lambda *_args, **_kwargs: {"detections": []})
    if failed:
        with pytest.raises(RuntimeError) as caught:
            visual_targeting.run_scan(tmp_path, np.eye(4), "test", .35, camera=camera, timeline=timeline)
        assert caught.value is error
    else:
        visual_targeting.run_scan(tmp_path, np.eye(4), "test", .35, camera=camera, timeline=timeline)
    event = timeline.events[1]
    assert event["name"] == ("scan.capture_failed" if failed else "scan.capture_complete")
    assert event["capture_diagnostics"] == diagnostics


def test_run_vision_passes_custom_text_prompt(tmp_path, monkeypatch) -> None:
    frame_dir = tmp_path / "frame"
    frame_dir.mkdir()
    (frame_dir / "vision_result.json").write_text(
        json.dumps({"detections": []}),
        encoding="utf-8",
    )
    observed: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        observed["cmd"] = cmd
        observed["kwargs"] = kwargs

    monkeypatch.setattr(visual_targeting.subprocess, "run", fake_run)

    visual_targeting.run_vision(
        frame_dir,
        "winejar-vision",
        0.35,
        text_prompt="bamboo leaf.",
        service_socket=None,
    )

    cmd = observed["cmd"]
    prompt_index = cmd.index("--text-prompt") + 1
    assert cmd[prompt_index] == "bamboo leaf."


def test_run_vision_uses_warm_service_without_subprocess(tmp_path, monkeypatch) -> None:
    frame_dir = tmp_path / "frame"
    frame_dir.mkdir()
    expected = {"detections": [{"label": "bamboo leaf"}]}
    (frame_dir / "vision_result.json").write_text(json.dumps(expected), encoding="utf-8")

    monkeypatch.setattr(
        visual_targeting,
        "request_vision_service",
        lambda **_kwargs: {"ok": True, "timing_seconds": {"total": 0.5}},
    )
    monkeypatch.setattr(
        visual_targeting.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("cold subprocess used")),
    )

    result = visual_targeting.run_vision(
        frame_dir,
        "winejar-vision",
        0.35,
        service_socket=tmp_path / "vision.sock",
        require_service=True,
    )
    assert result == expected


def test_mask_principal_axis_and_undirected_delta(tmp_path) -> None:
    mask = np.zeros((120, 160), dtype=np.uint8)
    cv2.rectangle(mask, (20, 50), (140, 70), 255, thickness=-1)
    cv2.imwrite(str(tmp_path / "mask.png"), mask)

    angle, elongation = visual_targeting.mask_principal_axis_deg(
        tmp_path,
        {"mask_path": "mask.png"},
    )

    assert min(abs(angle), abs(angle - 180.0)) < 1.0
    assert elongation > 10.0
    assert undirected_axis_delta_deg(170.0, 10.0) == 20.0


def test_mixed_grounding_label_uses_elongated_shape_as_bamboo() -> None:
    mixed_leaf = {
        "label": "bamboo leaf ceramic",
        "score": 0.64,
        "box_xyxy": [50.0, 260.0, 615.0, 450.0],
        "point_base_m": [0.48, 0.10, -0.03],
        "touches_image_edge": False,
    }
    ceramic = {
        "label": "ceramic disk",
        "score": 0.51,
        "box_xyxy": [20.0, 140.0, 140.0, 275.0],
        "point_base_m": [0.32, -0.15, 0.03],
        "touches_image_edge": False,
    }

    assert visual_targeting.select_ceramic([mixed_leaf, ceramic]) is ceramic
    leaves, _stacked = visual_targeting.select_leaf_waypoints(
        [mixed_leaf, ceramic],
        distinct_distance_m=0.05,
    )
    assert leaves[0] is mixed_leaf


def test_ambiguous_square_mixed_label_is_not_used_for_either_object() -> None:
    ambiguous = {
        "label": "bamboo leaf ceramic",
        "score": 0.9,
        "box_xyxy": [10.0, 10.0, 110.0, 110.0],
        "point_base_m": [0.4, 0.0, 0.0],
        "touches_image_edge": False,
    }
    assert visual_targeting.detection_object_kind(ambiguous) is None


def ceramic_detection(score: float, point_base_m=(0.3, 0.1, 0.02), clipped=False):
    return {
        "label": "ceramic disk",
        "score": score,
        "box_xyxy": [20.0, 140.0, 140.0, 275.0],
        "point_base_m": list(point_base_m),
        "touches_image_edge": clipped,
    }


def test_select_all_ceramic_returns_all_sorted_by_score() -> None:
    low = ceramic_detection(0.50, (0.20, 0.10, 0.02))
    high = ceramic_detection(0.80, (0.35, -0.05, 0.03))
    selected = visual_targeting.select_all_ceramic([low, high], min_score=0.45)
    assert [item["score"] for item in selected] == [0.80, 0.50]


def test_select_all_ceramic_applies_score_gate_and_rejects_clipped() -> None:
    weak = ceramic_detection(0.30, (0.20, 0.10, 0.02))
    clipped = ceramic_detection(0.70, (0.35, -0.05, 0.03), clipped=True)
    good = ceramic_detection(0.60, (0.40, 0.00, 0.02))
    selected = visual_targeting.select_all_ceramic([weak, clipped, good], min_score=0.45)
    assert [item["score"] for item in selected] == [0.60]


def test_select_all_ceramic_raises_on_empty_result() -> None:
    try:
        visual_targeting.select_all_ceramic([ceramic_detection(0.30)], min_score=0.45)
    except RuntimeError as exc:
        assert "confidence threshold" in str(exc)
    else:
        raise AssertionError("An empty ceramic selection must be rejected")


def test_select_all_ceramic_reports_clipped_disk() -> None:
    try:
        visual_targeting.select_all_ceramic([ceramic_detection(0.70, clipped=True)], min_score=0.45)
    except RuntimeError as exc:
        assert "image edge" in str(exc)
    else:
        raise AssertionError("Clipped ceramic disks must be rejected with guidance")


def test_lotus_paper_and_jar_labels_map_to_kinds() -> None:
    assert visual_targeting.detection_object_kind({"label": "lotus leaf"}) == "lotus"
    assert visual_targeting.detection_object_kind({"label": "white paper"}) == "paper"
    assert visual_targeting.detection_object_kind({"label": "wine jar"}) == "jar"


def test_ambiguous_new_kind_mixes_are_rejected() -> None:
    assert visual_targeting.detection_object_kind({"label": "bamboo leaf lotus leaf"}) is None
    assert visual_targeting.detection_object_kind({"label": "lotus leaf white paper"}) is None


def test_select_all_of_kind_matches_ceramic_wrapper() -> None:
    high = ceramic_detection(0.80)
    low = ceramic_detection(0.50, (0.20, 0.10, 0.02))
    selected = visual_targeting.select_all_of_kind("ceramic", [low, high], 0.45, display_name="ceramic disk")
    assert [item["score"] for item in selected] == [0.80, 0.50]
    assert [item["score"] for item in visual_targeting.select_all_ceramic([low, high], 0.45)] == [0.80, 0.50]
    try:
        visual_targeting.select_all_of_kind("ceramic", [ceramic_detection(0.30)], 0.45, display_name="ceramic disk")
    except RuntimeError as exc:
        assert "confidence threshold" in str(exc)
    else:
        raise AssertionError("An empty selection must be rejected")


def test_select_single_of_kind_picks_best_and_guides_on_clipped() -> None:
    low = ceramic_detection(0.50)
    high = ceramic_detection(0.80, (0.35, -0.05, 0.03))
    selected = visual_targeting.select_single_of_kind("ceramic", [low, high], display_name="ceramic disk")
    assert selected is high
    try:
        visual_targeting.select_single_of_kind("ceramic", [ceramic_detection(0.70, clipped=True)], display_name="ceramic disk")
    except RuntimeError as exc:
        assert "image edge" in str(exc)
    else:
        raise AssertionError("Clipped detections must be rejected with guidance")
