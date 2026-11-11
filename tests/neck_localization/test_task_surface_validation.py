from copy import deepcopy
from types import SimpleNamespace
import unittest
import numpy as np

from neck_localization.geometry import Estimate,look_at
from neck_localization.visible_profile_baseline import revalidate_candidate
from neck_localization.task_surface_validation import task_surface_support,revalidate_task_candidate


class TaskSurfaceTests(unittest.TestCase):
    def fixture(self):
        camera=look_at('a',[0,-.4,.6],[0,0,.6],width=200,height=160)
        frame=SimpleNamespace(camera=camera)
        grid=np.linspace(-.012,.012,49)
        result=Estimate(center=np.array([0,0,.6]),normal=np.array([0.,0.,1.]),radius=.05,accepted=True,
            diagnostics=dict(profile_s_m=grid.tolist(),profile_radius_m=np.full(len(grid),.05).tolist(),
                axis_origin_m=[0,0,.6],near_min_interval_length_m=.006,
                heldout_rms_mm=.1,target_contact_rms_pixels=0.,selection_score=.1))
        return result,[frame]

    def test_context_mismatch_does_not_invalidate_supported_operation(self):
        candidate,frames=self.fixture();mask=np.zeros((160,200),bool);mask[77:83,:]=True
        context=revalidate_candidate(candidate,frames,[mask])
        self.assertFalse(context.accepted)
        self.assertGreater(context.diagnostics['surface_projection']['maximum_outside_pixels'],2)
        task=revalidate_task_candidate(context,frames,[mask])
        self.assertTrue(task.accepted)
        self.assertEqual(task.diagnostics['task_surface_projection']['maximum_outside_pixels'],0.)
        self.assertEqual(task.diagnostics['selection_score'],context.diagnostics['selection_score'])
        np.testing.assert_array_equal(task.center,context.center)

    def test_missing_target_foreground_still_rejected(self):
        candidate,frames=self.fixture();mask=np.zeros((160,200),bool)
        context=revalidate_candidate(candidate,frames,[mask])
        task=revalidate_task_candidate(context,frames,[mask])
        self.assertFalse(task.accepted)
        self.assertIn('operation_surface_projects_outside_rgb_foreground',task.reasons)

    def test_nonRGB_rejection_is_preserved(self):
        candidate,frames=self.fixture();candidate.reasons=['spatial_block_depth_mismatch']
        context=revalidate_candidate(candidate,frames,[np.ones((160,200),bool)])
        task=revalidate_task_candidate(context,frames,[np.ones((160,200),bool)])
        self.assertFalse(task.accepted);self.assertIn('spatial_block_depth_mismatch',task.reasons)

    def test_no_extrapolation_into_unobserved_band(self):
        candidate,frames=self.fixture();candidate.center[2]+=.0115
        context=revalidate_candidate(candidate,frames,[np.ones((160,200),bool)])
        task=revalidate_task_candidate(context,frames,[np.ones((160,200),bool)])
        self.assertFalse(task.accepted);self.assertIn('operation_surface_outside_fitted_support',task.reasons)

    def test_validation_does_not_mutate_saved_geometry(self):
        candidate,frames=self.fixture();before=deepcopy(candidate.diagnostics)
        support=task_surface_support(candidate,frames,[np.ones((160,200),bool)])
        self.assertEqual(candidate.diagnostics,before)
        self.assertAlmostEqual(np.diff(support['axis_interval_m'])[0],.003)
        self.assertFalse(support['uses_ground_truth'])

    def test_invalid_width_fails(self):
        candidate,frames=self.fixture()
        for width in [0.,-.003,np.nan]:
            with self.assertRaises(ValueError):task_surface_support(candidate,frames,[np.ones((160,200),bool)],width)


if __name__=='__main__':unittest.main()
