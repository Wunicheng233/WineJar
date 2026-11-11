import unittest
import numpy as np
from neck_localization.baseline_capability import disparity_to_depth, shared_ir_noise


class CapabilityConversions(unittest.TestCase):
    def test_native_metric_depth_and_invalid_support(self):
        d = np.full((4,100), 20., np.float32)
        depth, valid = disparity_to_depth(d, 400., .05)
        self.assertTrue(np.allclose(depth[:,20:], 1.))
        self.assertFalse(valid[:,:20].any())
        d[:,30:34] = [0., -1., np.nan, np.inf]
        depth, valid = disparity_to_depth(d, 400., .05)
        self.assertFalse(valid[:,30:34].any())
        self.assertTrue((depth[:,30:34] == 0).all())

    def test_shared_noise_is_reproducible_without_changing_source(self):
        image = np.full((200,200), 128, np.uint8)
        first = shared_ir_noise(image, 17)
        self.assertTrue(np.array_equal(first, shared_ir_noise(image, 17)))
        self.assertFalse(np.array_equal(first, shared_ir_noise(image, 18)))
        self.assertTrue((image == 128).all())
        self.assertLess(abs(first.mean()-128), .2)
        self.assertTrue(np.array_equal(shared_ir_noise(image, 4, 0, 0), image))

    def test_noise_rejects_unscaled_float_input(self):
        with self.assertRaises(ValueError):
            shared_ir_noise(np.ones((3,3)), 1)


if __name__ == '__main__':
    unittest.main()
