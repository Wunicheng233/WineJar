import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from neck_localization.inspection_viewer import InspectionViewer
from neck_localization.viewer_navigation import (
    NavButton, Rect, active_stereo_datasets, framebuffer_point, hit_button, viewer_layout,
)


class NavigationTests(unittest.TestCase):
    def test_layout_keeps_controls_inputs_and_both_3d_views_separate(self):
        for width, height in [(1000, 680), (1480, 960), (1920, 1080)]:
            for large in (False, True):
                layout = viewer_layout(width, height, large)
                self.assertEqual(layout.sidebar.width, layout.main.x)
                self.assertEqual(layout.main.x+layout.main.width, layout.inputs.x)
                self.assertEqual(layout.inputs.x+layout.inputs.width, width)
                self.assertTrue(layout.main.contains(layout.neck.x, layout.neck.y))
                self.assertTrue(layout.main.contains(layout.neck.x+layout.neck.width-1,
                                                    layout.neck.y+layout.neck.height-1))

    def test_high_dpi_conversion_and_minimized_window(self):
        self.assertEqual(framebuffer_point(100, 50, (740, 480), (1480, 960)), (200, 100))
        self.assertIsNone(framebuffer_point(1, 1, (0, 0), (0, 0)))

    def test_disabled_and_scrolled_buttons_cannot_receive_hidden_clicks(self):
        buttons = [NavButton('visible', 'one', Rect(10, 80, 100, 30)),
                   NavButton('disabled', 'two', Rect(10, 120, 100, 30), enabled=False),
                   NavButton('under_header', 'three', Rect(10, 45, 100, 30))]
        self.assertEqual(hit_button(buttons, 50, 90, 64, 700), 'visible')
        self.assertIsNone(hit_button(buttons, 50, 130, 64, 700))
        self.assertIsNone(hit_button(buttons, 50, 50, 64, 700))

    def test_sidebar_and_camera_inputs_do_not_rotate_scene(self):
        viewer = InspectionViewer.__new__(InspectionViewer)
        viewer.window = object(); viewer.large_neck = True; viewer.show_neck = True
        viewer.camera, viewer.scene = object(), object()
        viewer.neck_camera, viewer.neck_scene = object(), object()
        layout = viewer_layout(1480, 960, True)
        with patch('glfw.get_window_size', return_value=(740, 480)), patch('glfw.get_framebuffer_size', return_value=(1480, 960)):
            self.assertIsNone(viewer._pointer_view(30, 100))
            self.assertIsNone(viewer._pointer_view((layout.inputs.x+30)/2, 100))
            self.assertIs(viewer._pointer_view((layout.neck.x+10)/2, (layout.neck.y+10)/2)[0], viewer.neck_camera)
            self.assertIs(viewer._pointer_view((layout.main.x+20)/2, (layout.main.y+20)/2)[0], viewer.camera)

    def test_mouse_and_keyboard_actions_have_consistent_cloud_modes(self):
        viewer = InspectionViewer.__new__(InspectionViewer)
        viewer.point_cloud = viewer.cloud_local = viewer.cloud_weights = viewer.cloud_source_colors = False
        viewer.depth_mode = False
        viewer._activate('color:weights')
        self.assertTrue(viewer.point_cloud and viewer.cloud_local and viewer.cloud_weights)
        viewer._activate('view:cloud')
        self.assertTrue(viewer.point_cloud)
        self.assertFalse(viewer.cloud_local or viewer.cloud_weights)
        viewer._activate('input:depth')
        self.assertTrue(viewer.depth_mode)
        viewer._activate('view:model')
        self.assertFalse(viewer.point_cloud)

    def test_comparison_buttons_only_reference_available_saved_experiments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root/'active_stereo_adapted_test_20261025_v1/stereo_rgb_adapted/stereo_raw'
            (folder/'scenes').mkdir(parents=True)
            (folder/'records.json.gz').write_bytes(b'not opened by discovery')
            choices = active_stereo_datasets(root)
            self.assertEqual([d['condition'] for d in choices], ['stereo_raw'])
            self.assertEqual(active_stereo_datasets(root, 'native'), [])


if __name__ == '__main__':
    unittest.main()
