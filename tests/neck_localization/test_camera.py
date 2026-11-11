"""Camera geometry and virtual-calibration checks; never access real devices."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import unittest
import mujoco
import numpy as np

from neck_localization.calibration import virtual_calibrate
from neck_localization.geometry import look_at


def camera_rig():
    return [look_at("left", [0.45, -0.45, 0.75], [0, 0, 0.2]),
            look_at("right", [0.5, 0.4, 0.7], [0, 0, 0.2]),
            look_at("top", [-0.3, 0.1, 0.95], [0, 0, 0.2])]


class CameraGeometryTests(unittest.TestCase):
    def test_project_unproject_off_axis_world_points(self):
        camera = camera_rig()[0]
        points = np.array([[0.01, 0.04, 0.22], [-0.02, -0.03, 0.18], [0.05, 0.01, 0.2]])
        depth = ((points - camera.T_world_from_camera[:3, 3]) @ camera.T_world_from_camera[:3, :3])[:, 2]
        np.testing.assert_allclose(camera.unproject(camera.project(points), depth), points, atol=1e-12)
        pixels = camera.project(points)
        self.assertTrue(np.all(pixels > 0))
        self.assertTrue(np.all(pixels[:, 0] < camera.width))
        self.assertTrue(np.all(pixels[:, 1] < camera.height))

    def test_mujoco_axial_depth_and_integer_pixel_centers(self):
        width, height, fovy = 320, 240, 60
        xml = '''<mujoco><visual><global offwidth="320" offheight="240"/>
          <quality offsamples="0"/></visual><worldbody>
          <camera name="camera" pos="0 0 1" fovy="60"/>
          <geom type="plane" size="2 2 .1" quat="0.984807753 0 0.173648178 0"/>
          </worldbody></mujoco>'''
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            renderer.enable_depth_rendering()
            renderer.update_scene(data, camera="camera")
            depth = renderer.render().copy()
        focal = height / (2 * np.tan(np.deg2rad(fovy) / 2))
        theta = np.deg2rad(20)
        u = np.arange(width)
        expected = np.cos(theta) / (np.cos(theta) - np.sin(theta) * (u - (width - 1) / 2) / focal)
        self.assertLess(float(np.max(np.abs(depth - expected[None, :]))), 1e-5)
        # Off-axis range differs substantially, whereas rendered depth matches Z.
        ray_length = np.sqrt(1 + ((u - (width - 1) / 2) / focal) ** 2)
        self.assertGreater(abs(float(depth[height // 2, 10] - expected[10] * ray_length[10])), 0.1)


class VirtualCalibrationTests(unittest.TestCase):
    def test_ideal_intrinsics_and_relative_extrinsics_recovery(self):
        cameras = camera_rig()
        estimated, diagnostics = virtual_calibrate(cameras, seed=61, corner_sigma_px=0)
        self.assertEqual(diagnostics["kind"], "synthetic_projected_board_corners")
        for truth, recovered in zip(cameras, estimated):
            np.testing.assert_allclose(recovered.K, truth.K, atol=0.003)
            np.testing.assert_allclose(recovered.T_world_from_camera, truth.T_world_from_camera, atol=1e-5)
        self.assertLess(max(diagnostics["heldout_rig_reference_pnp_reprojection_rms_px"]), 0.003)

    def test_board_scale_error_biases_baseline_despite_small_reprojection(self):
        cameras = camera_rig()
        scale_error = 0.02
        _, diagnostics = virtual_calibrate(cameras, seed=61, corner_sigma_px=0, board_scale_error=scale_error)
        for item in diagnostics["evaluation_truth_only"][1:]:
            ratio = item["estimated_baseline_mm"] / item["true_baseline_mm"]
            self.assertAlmostEqual(ratio, 1 / (1 + scale_error), delta=1e-4)
            self.assertGreater(item["relative_translation_error_mm"], 5)
        self.assertLess(max(diagnostics["heldout_rig_reference_pnp_reprojection_rms_px"]), 0.003)


if __name__ == "__main__":
    unittest.main()
