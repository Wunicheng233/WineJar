from __future__ import annotations

import json

import cv2
import numpy as np

from xarm_sdk_control import visual_targeting


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
    assert visual_targeting.undirected_axis_delta_deg(170.0, 10.0) == 20.0


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
