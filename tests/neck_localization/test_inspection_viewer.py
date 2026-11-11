"""Check result-overlay projection independently of window/display availability."""
import os
import unittest

import glfw
import mujoco
import numpy as np
from OpenGL import GL

from neck_localization.geometry import Camera
from neck_localization.inspection_viewer import present_image, project_scene, draw_cloud


class InspectionProjectionTests(unittest.TestCase):
    def test_mono_projection_matches_calibrated_camera_at_multiple_aspects(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
            <camera name="view" pos="0 0 1" fovy="60"/>
            <geom type="sphere" size=".1"/>
            </worldbody></mujoco>''')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        scene = mujoco.MjvScene(model, maxgeom=100)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid = 0
        mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, camera,
                              mujoco.mjtCatBit.mjCAT_ALL, scene)
        scene.stereo = mujoco.mjtStereo.mjSTEREO_NONE
        pose = np.diag([1., -1., -1., 1.])
        pose[:3, 3] = [0, 0, 1]
        points = np.random.default_rng(134).uniform([-.2, -.15, -.1], [.2, .15, .3], (100, 3))
        for width, height in [(640, 480), (1077, 960), (420, 285)]:
            with self.subTest(width=width, height=height):
                focal = height/(2*np.tan(np.deg2rad(60)/2))
                K = np.array([[focal, 0, (width-1)/2], [0, focal, (height-1)/2], [0, 0, 1]])
                expected = Camera("view", K, pose, width, height).project(points)
                np.testing.assert_allclose(project_scene(scene, points, width, height), expected, atol=4e-5)


@unittest.skipUnless(os.environ.get("DISPLAY") and os.environ.get("MUJOCO_GL") not in ("egl", "osmesa"),
                     "Desktop GLFW presentation test; run with env -u MUJOCO_GL")
class WindowPresentationTests(unittest.TestCase):
    def test_cloud_points_align_with_camera_and_use_mujoco_depth_order(self):
        self.assertTrue(glfw.init())
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        window = glfw.create_window(320, 240, "point cloud projection check", None, None)
        context = None
        try:
            self.assertIsNotNone(window)
            glfw.make_context_current(window)
            model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
                <camera name="view" pos="0 0 1" fovy="60"/>
                <geom type="sphere" size=".1"/></worldbody></mujoco>''')
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)
            scene = mujoco.MjvScene(model, maxgeom=100)
            camera = mujoco.MjvCamera()
            camera.type, camera.fixedcamid = mujoco.mjtCamera.mjCAMERA_FIXED, 0
            mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, camera,
                                  mujoco.mjtCatBit.mjCAT_ALL, scene)
            scene.stereo = mujoco.mjtStereo.mjSTEREO_NONE
            scene.ngeom = 0
            w, h = glfw.get_framebuffer_size(window)
            rect = mujoco.MjrRect(0, 0, w, h)
            mujoco.mjr_render(rect, scene, context)
            # Near red point must hide the later-drawn far green point.
            points = np.array([[0, 0, .2], [0, 0, 0], [.12, .03, .15]], np.float32)
            colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], np.uint8)
            draw_cloud(scene, points, colors, w, h)
            actual = np.empty((h, w, 3), np.uint8)
            mujoco.mjr_readPixels(actual, None, rect, context)
            actual = actual[::-1]
            for index in (0, 2):
                x, y = np.rint(project_scene(scene, points[index:index+1], w, h)[0]).astype(int)
                patch = actual[y-2:y+3, x-2:x+3]
                self.assertTrue(np.any(np.all(patch == colors[index], axis=2)))
            self.assertFalse(np.any(np.all(actual == colors[1], axis=2)))
        finally:
            if context is not None:
                context.free()
            if window:
                glfw.destroy_window(window)
            glfw.terminate()

    def test_composite_survives_depth_test_and_preserves_gl_state(self):
        self.assertTrue(glfw.init())
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        window = glfw.create_window(160, 120, "presentation check", None, None)
        context = None
        try:
            self.assertIsNotNone(window)
            glfw.make_context_current(window)
            model = mujoco.MjModel.from_xml_string("<mujoco/>")
            context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)
            width, height = glfw.get_framebuffer_size(window)
            # An old 3D depth buffer must not reject any of the 2D composite.
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glDepthFunc(GL.GL_NEVER)
            image = np.random.default_rng(918).integers(0, 256, (height, width, 3), dtype=np.uint8)
            present_image(image, context)
            actual = np.empty_like(image)
            mujoco.mjr_readPixels(actual, None, mujoco.MjrRect(0, 0, width, height), context)
            np.testing.assert_array_equal(actual[::-1], image)
            self.assertTrue(GL.glIsEnabled(GL.GL_DEPTH_TEST))
            self.assertEqual(GL.glGetIntegerv(GL.GL_DEPTH_FUNC), GL.GL_NEVER)
        finally:
            if context is not None:
                context.free()
            if window:
                glfw.destroy_window(window)
            glfw.terminate()


if __name__ == "__main__":
    unittest.main()
