from dataclasses import replace

import cv2
import numpy as np

from xarm_sdk_control.calibration_sampling import (
    StationaryBoardGate,
    check_view,
    duplicate_pose,
)
from xarm_sdk_control.multiview_calibration import CharucoDetection

MATRIX = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1.0]])


def detection(offset=0.0):
    points = np.zeros((25, 3), np.float32)
    points[:, :2] = np.mgrid[:5, :5].T.reshape(-1, 2) * 0.05
    pixels, _ = cv2.projectPoints(
        points,
        np.array([0.1, -0.1, 0.02]),
        np.array([-0.1, -0.1, 0.5]),
        MATRIX,
        np.zeros(5),
    )
    return CharucoDetection(pixels + offset, np.arange(25), points, 15)


def test_stationary_interval_and_missing_detection_reset():
    gate = StationaryBoardGate()
    pair = (detection(), detection())
    assert not gate.update(pair, 0.0)
    assert not gate.update(pair, 0.3)
    assert gate.update(pair, 0.61)
    assert not gate.update((None, pair[1]), 0.7)
    assert not gate.update(pair, 0.8)


def test_slow_drift_is_compared_to_anchor_not_only_previous_frame():
    gate = StationaryBoardGate()
    for i in range(10):
        assert not gate.update((detection(i * 0.6), detection()), i * 0.2)


def test_a_capture_gap_does_not_count_as_observed_stationarity():
    gate = StationaryBoardGate()
    pair = (detection(), detection())
    assert not gate.update(pair, 0.0)
    assert not gate.update(pair, 5.0)


def test_duplicate_pose_requires_change_from_all_saved_samples():
    first = (detection(), detection())
    assert duplicate_pose((detection(2), detection(2)), [first])
    assert not duplicate_pose((detection(20), detection(20)), [first])


def test_view_quality_accepts_well_spread_sharp_geometry():
    reason, metrics = check_view(detection(), (640, 480), MATRIX, np.zeros(5))
    assert reason is None
    assert metrics["pnp_rmse_px"] < 0.001


def test_compressed_corners_and_outlier_are_rejected():
    d = detection()
    pixels = d.corners_px.copy()
    pixels[:, :, 1] = 240
    reason, _ = check_view(
        replace(d, corners_px=pixels), (640, 480), MATRIX, np.zeros(5), minimum_area=0
    )
    assert reason == "corners too compressed"
    pixels = d.corners_px.copy()
    pixels[0] += 50
    reason, _ = check_view(
        replace(d, corners_px=pixels), (640, 480), MATRIX, np.zeros(5)
    )
    assert reason == "single-view reprojection error"


def test_stationary_diagnostics_distinguish_resets_and_ready():
    gate = StationaryBoardGate()
    pair = (detection(), detection())
    assert not gate.update(pair, 0.0)
    assert gate.status['reason'] == 'accumulating stationary time'
    assert not gate.update(pair, 0.3)
    assert gate.update(pair, 0.61)
    assert gate.status['reason'] is None
    assert gate.status['anchor_motion_px'] == [0.0, 0.0]
    assert not gate.update((detection(2), pair[1]), 0.7)
    assert gate.status['reason'] == 'board moved'
    assert not gate.update(pair, 2.0)
    assert gate.status['reason'] == 'processing interval too long'
    assert not gate.update((None, pair[1]), 2.1)
    assert gate.status['reason'] == 'board detection missing'


def test_changed_corner_ids_report_missing_overlap_and_serialize():
    import json

    gate = StationaryBoardGate()
    d = detection()
    gate.update((d, d), 0)
    changed = replace(d, corner_ids=d.corner_ids + 100)
    assert not gate.update((changed, d), 0.1)
    assert gate.status['reason'] == 'too few shared corners'
    assert gate.status['anchor_motion_px'] == [None, 0.0]
    json.dumps(gate.status, allow_nan=False)


def test_frame_rejection_explains_timing_separately_from_stationarity():
    from xarm_sdk_control.calibration_sampling import pair_frame_rejection

    first = {'frames': [{'frame_number': 10}, {'frame_number': 11}], 'skew_ms': 25}
    assert pair_frame_rejection(first, None) is None
    assert pair_frame_rejection(first, first) == 'repeated or reversed frame'
    assert pair_frame_rejection({**first, 'skew_ms': 101}, None) == 'frame skew too large'
    assert pair_frame_rejection({**first, 'skew_ms': float('nan')}, None) == 'invalid frame skew'
    assert pair_frame_rejection({**first, 'frames': []}, None) == 'invalid frame pair'
