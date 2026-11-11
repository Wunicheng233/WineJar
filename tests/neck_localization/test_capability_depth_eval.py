"""Offline audit checks: input integrity, metric depth and honest denominators."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from neck_localization.geometry import Estimate
from scripts import evaluate_capability_depth as evaluation


def rotation_y(degrees):
    c, s = np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


class AuditFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.predictions = self.root/"predictions_run"
        self.predictions.mkdir()
        self.scene = self.root/"scene"
        (self.scene/"observations").mkdir(parents=True)
        self.K = np.array([[80., 0., 31.5], [0., 80., 23.5], [0., 0., 1.]])
        self.rgb_transform = np.eye(4)
        self.rgb_transform[:3, :3] = rotation_y(3.)
        self.rgb_transform[:3, 3] = [.012, 0., .006]
        self.sensor = self.root/"sensor.json"
        self.pairs, self.index = [], {}
        self.depth = np.full((48, 64), .8004, np.float32)
        cameras = []
        for name in evaluation.ALL_CAMERAS:
            left, right = self.root/(name+"_left.png"), self.root/(name+"_right.png")
            cv2.imwrite(str(left), np.full((48, 64), 100, np.uint8))
            cv2.imwrite(str(right), np.full((48, 64), 101, np.uint8))
            cv2.imwrite(str(self.scene/"observations"/(name+"_rgb.png")),
                        np.full((48, 64, 3), 100, np.uint8))
            pair = dict(pair_id="development/case/"+name+"/clean", layout="development",
                        case="case", camera=name, condition="clean", split="development",
                        render_root=str(self.root), scene_dir=str(self.scene),
                        sensor_path=str(self.sensor), left_path=str(left), right_path=str(right),
                        input_sha256=[evaluation.file_hash(left), evaluation.file_hash(right)],
                        baseline_m=.05, K_ir=self.K.tolist(), T_world_from_left_ir=np.eye(4).tolist(),
                        width=64, height=48)
            cameras.append(dict(name=name, K_ir=self.K.tolist(), K_rgb=self.K.tolist(),
                                T_world_from_left_ir=np.eye(4).tolist(), baseline_m=.05,
                                T_world_from_rgb=self.rgb_transform.tolist()))
            path = self.predictions/"predictions"/(pair["pair_id"]+".npz")
            path.parent.mkdir(parents=True)
            np.savez_compressed(path, native_float_z=self.depth,
                                disparity_px=4./self.depth, valid_mask=np.ones(self.depth.shape, bool))
            self.index[pair["pair_id"]] = dict(pair_id=pair["pair_id"], status="ok", inference_ms=5.,
                input_sha256=pair["input_sha256"], prediction_sha256=evaluation.file_hash(path))
            self.pairs.append(pair)
        self.sensor.write_text(json.dumps(dict(cameras=cameras)))
        (self.predictions/"predictions.jsonl").write_text(
            "".join(json.dumps(row)+"\n" for row in self.index.values()))
        self.manifest = self.root/"manifest.json"
        self.manifest.write_text(json.dumps(dict(schema=1, pairs=self.pairs)))

    def test_read_prediction_rejects_focal_scale_and_status_integrity_errors(self):
        pair = self.pairs[0]
        depth, _ = evaluation.read_prediction(pair, self.predictions, self.index, {})
        np.testing.assert_array_equal(depth, self.depth)
        wrong_focal = copy.deepcopy(pair)
        wrong_focal["K_ir"][0][0] *= 1.4  # RGB fx accidentally used as IR fx.
        with self.assertRaisesRegex(ValueError, "IR_focal_baseline_mismatch"):
            evaluation.read_prediction(wrong_focal, self.predictions, self.index, {})
        wrong_index = copy.deepcopy(self.index)
        wrong_index[pair["pair_id"]]["input_sha256"] = ["0"*64]*2
        with self.assertRaisesRegex(ValueError, "prediction_input_sha256_mismatch"):
            evaluation.read_prediction(pair, self.predictions, wrong_index, {})
        wrong_index = copy.deepcopy(self.index)
        wrong_index[pair["pair_id"]]["prediction_sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError, "prediction_sha256_mismatch"):
            evaluation.read_prediction(pair, self.predictions, wrong_index, {})

    def test_registration_uses_quantized_raw_depth_then_rgb_optical_z(self):
        saved = self.depth.copy()
        frame, _, info = evaluation.prepare_frame(self.pairs[0], self.depth, {})
        yy, xx = np.indices(self.depth.shape)
        rays = np.stack(((xx-31.5)/80., (yy-23.5)/80., np.ones_like(xx)), axis=-1)
        normal = self.rgb_transform[:3, :3].T @ np.array([0., 0., 1.])
        expected = (.800-.006)/np.einsum("...i,i->...", rays, normal)
        valid = frame.depth_z_m > 0
        self.assertGreater(int(valid.sum()), int(valid.size*.85))
        np.testing.assert_allclose(frame.depth_z_m[valid], expected[valid], atol=2e-7)
        self.assertGreater(float(np.ptp(frame.depth_z_m[valid])), .02)
        self.assertEqual(info["native_quantized_valid"], self.depth.size)
        np.testing.assert_array_equal(self.depth, saved)

    def test_all_backend_calls_precede_truth_and_rejected_candidates_remain_visible(self):
        events = []
        def candidate(frames):
            events.append(("infer", len(frames)))
            return Estimate(center=np.array([0., 0., .6]), normal=np.array([0., 0., 1.]),
                            radius=.05, accepted=False, reasons=["unchanged_gate"])
        truth = dict(center_m=[0., 0., .6], normal=[0., 0., 1.], radius_m=.05)
        def read_truth(pair, tracked):
            self.assertEqual(events, [("infer", 3), ("infer", 3), ("infer", 1), ("infer", 1)])
            events.append(("truth", 0))
            return truth, {}
        with patch.dict(evaluation.BACKENDS, {"portfolio":candidate, "stereo_robust":candidate}, clear=True), \
             patch.object(evaluation, "_load_truth", side_effect=read_truth), \
             patch.object(evaluation, "measurement_metrics", return_value={}):
            records, _, summaries = evaluation.run(self.manifest, self.predictions, self.root/"evaluation")
        self.assertEqual(len(records), 4)
        for row in records:
            self.assertEqual(row["split"], "development")
            self.assertTrue(row["truth_pass"])
            self.assertFalse(row["accepted"])
        for summary in summaries:
            self.assertEqual(summary["candidate_geometry_pass"], 1)
            self.assertEqual(summary["correct_output"], 0)
            self.assertEqual(summary["rejected"], 1)
        self.assertTrue(json.loads((self.root/"evaluation"/"validation.json").read_text())["passed"])


class MeasurementDenominators(unittest.TestCase):
    def test_coverage_and_paired_common_pixels_do_not_hide_missing_predictions(self):
        ideal = np.ones((1, 4))
        first = np.array([[1.001, 0., 1.100, 1.002]])
        second = np.array([[1.003, 1.004, 0., 1.005]])
        region = np.ones(ideal.shape, bool)
        ordinary = evaluation.region_metrics(first, ideal, region)
        self.assertEqual(ordinary["reference_pixels"], 4)
        self.assertEqual(ordinary["missing_prediction_pixels"], 1)
        self.assertEqual(ordinary["valid_coverage"], .75)
        a = evaluation.common_region_metrics(first, ideal, region, [second], ["first", "second"])
        b = evaluation.common_region_metrics(second, ideal, region, [first], ["first", "second"])
        self.assertEqual(a["common_valid_pixels"], 2)
        self.assertEqual(a["common_coverage_of_fixed_reference"], .5)
        self.assertEqual(b["common_valid_pixels"], a["common_valid_pixels"])
        self.assertAlmostEqual(a["median_abs_mm"], 1.5)
        self.assertAlmostEqual(b["median_abs_mm"], 4.)
        self.assertGreater(ordinary["rmse_mm"], 50.)
        failed = evaluation.common_region_metrics(first, ideal, region, [], ["first", "second"],
                                                  {"second":"missing_prediction"})
        self.assertEqual(failed["status"], "comparison_input_error")
        self.assertNotIn("median_abs_mm", failed)

    def test_rejections_errors_and_wrong_accepts_all_remain_in_denominator(self):
        base = dict(estimate=None, accepted=False, truth_pass=False, status="input_error", reasons=["missing"])
        rows = [base, dict(base, estimate={}, truth_pass=True, status="ok", reasons=["gate"]),
                dict(base, estimate={}, accepted=True, status="ok", reasons=[]),
                dict(base, estimate={}, accepted=True, truth_pass=True, status="ok", reasons=[])]
        result = evaluation.aggregate(rows)
        self.assertEqual(result["n"], 4)
        self.assertEqual(result["candidate_geometry_pass"], 2)
        self.assertEqual(result["correct_output"], 1)
        self.assertEqual(result["wrong_accepted"], 1)
        self.assertEqual(result["input_or_backend_errors"], 1)
        self.assertEqual(result["correct_output_rate"], .25)


if __name__ == "__main__":
    unittest.main()
