"""Run with the vision environment: python -m unittest discover -s vision/tests."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
import torch

source = Path(__file__).resolve().parents[1] / "scripts/01_offline_grounded_sam2.py"
spec = importlib.util.spec_from_file_location("offline_precision_test", source)
vision = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = vision
spec.loader.exec_module(vision)


class Predictor:
    device = "cpu"

    def __init__(self):
        self.autocast_states = []

    def set_image(self, image):
        self.autocast_states.append(torch.is_autocast_enabled("cuda"))

    def predict(self, **kwargs):
        self.autocast_states.append(torch.is_autocast_enabled("cuda"))
        return np.ones((1, 1, 8, 8)), np.ones(1), None


class PrecisionTests(unittest.TestCase):
    def test_default_precision_preserves_masks_and_does_not_enable_autocast(self):
        predictor = Predictor()
        masks = vision.infer_sam2_masks(predictor, Image.new("RGB", (8, 8)), [{"box_xyxy": [0, 0, 7, 7]}])
        self.assertEqual(predictor.autocast_states, [False, False])
        np.testing.assert_array_equal(masks[0], np.ones((8, 8), np.uint8))

    def test_invalid_precision_and_cpu_bfloat16_are_rejected(self):
        for precision in ("float16", "bfloat16"):
            with self.assertRaises(ValueError):
                vision.infer_sam2_masks(Predictor(), Image.new("RGB", (8, 8)), [], precision)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "requires bfloat16 CUDA")
    def test_bfloat16_covers_encoder_and_decoder_and_restores_context(self):
        predictor = Predictor()
        predictor.device = "cuda"
        vision.infer_sam2_masks(predictor, Image.new("RGB", (8, 8)), [{"box_xyxy": [0, 0, 7, 7]}], "bfloat16")
        self.assertEqual(predictor.autocast_states, [True, True])
        self.assertFalse(torch.is_autocast_enabled("cuda"))

    def test_saved_result_reports_precision_with_default_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            cv2.imwrite(str(folder / "color.png"), np.zeros((8, 8, 3), np.uint8))
            cv2.imwrite(str(folder / "depth.png"), np.full((8, 8), 500, np.uint16))
            metadata = dict(files=dict(color_bgr="color.png", depth_raw="depth.png"),
                            stream=dict(depth_scale_m_per_unit=.001, coordinate_convention="test"),
                            intrinsics=dict(fx=100, fy=100, ppx=4, ppy=4))
            (folder / "frame.json").write_text(json.dumps(metadata))
            detection = dict(label="paper", score=.9, box_xyxy=[0, 0, 7, 7])
            with patch.object(vision, "infer_grounding_dino_loaded", return_value=[detection]):
                result = vision.process_saved_frame(folder, "paper.", None, None, Predictor(), torch.device("cpu"),
                                                    "test", "test", "test", None, .45, .15, .45)
            saved = json.loads((folder / "vision_result.json").read_text())
            self.assertEqual(saved["sam2"]["precision"], "float32")
            self.assertEqual(result["detections"][0]["point_camera_m"][2], .5)


if __name__ == "__main__":
    unittest.main()
