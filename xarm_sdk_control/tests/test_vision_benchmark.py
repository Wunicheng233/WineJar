"""Quality comparison must match objects, not zip potentially reordered detections."""
import numpy as np

from conftest import load_script_module

BENCHMARK = load_script_module("vision_benchmark", "vision/scripts/05_benchmark_inference.py")


def detection(label, x):
    return {"label": label, "score": .7, "box_xyxy": [x, 0, x + 10, 10]}


def test_reordered_detections_are_matched_before_measuring_point_shift():
    a, b = detection("bamboo leaf", 0), detection("bamboo leaf", 20)
    masks = [np.eye(3, dtype=np.uint8), np.ones((3, 3), dtype=np.uint8)]
    reference = ([a, b], masks, [[0, 0, 1], [.2, 0, 1]])
    candidate = ([b, a], masks[::-1], [[.2, 0, 1], [0, 0, 1]])
    report = BENCHMARK.compare_outputs(reference, candidate)
    assert report["unmatched_reference"] == report["unmatched_candidate"] == []
    assert [match["candidate_index"] for match in report["matches"]] == [1, 0]
    assert all(match["mask_iou"] == 1 and match["point_delta_mm"] == 0 for match in report["matches"])


def test_additional_detections_and_invalid_depth_remain_visible():
    a, b = detection("paper", 0), detection("paper", 20)
    mask = np.ones((3, 3), dtype=np.uint8)
    report = BENCHMARK.compare_outputs(([a], [mask], [[0, 0, 1]]), ([a, b], [mask, mask], [None, [0, 0, 1]]))
    assert report["unmatched_candidate"] == [1]
    assert report["matches"][0]["point_delta_mm"] is None
    assert report["matches"][0]["point_validity_equal"] is False


def test_different_labels_or_empty_outputs_are_not_false_matches():
    a, b = detection("paper", 0), detection("ceramic", 0)
    mask = np.ones((3, 3), dtype=np.uint8)
    reference = ([a], [mask], [[0, 0, 1]])
    for candidate in (([b], [mask], [[0, 0, 1]]), ([], [], [])):
        report = BENCHMARK.compare_outputs(reference, candidate)
        assert report["matches"] == []
        assert report["unmatched_reference"] == [0]
