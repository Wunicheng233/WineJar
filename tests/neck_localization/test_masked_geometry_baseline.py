"""Explicit foreground input and frozen candidate-policy adapter contracts."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import cv2

from neck_localization.geometry import Estimate
from neck_localization import masked_geometry_baseline as adapter
from scripts import run_masked_geometry_capability_stage as masked_stage


class ExplicitMaskAdapterTests(unittest.TestCase):
    def setUp(self):
        self.frames = [SimpleNamespace(camera=SimpleNamespace(name='cam_a',width=6,height=4),
                                      rgb=np.zeros((4,6,3),np.uint8))]
        self.masks = [np.ones((4,6),bool)]
        self.foreground = {'method':'synthetic_explicit_masks','cameras':[{'camera':'cam_a'}]}

    def candidate(self,accepted,score):
        return Estimate(center=np.array([0.,0.,.6]),normal=np.array([0.,0.,1.]),radius=.05,
            accepted=accepted,reasons=[] if accepted else ['native_gate'],
            diagnostics={'selection_score':score,'surface_projection':{},'frozen_validation':{},'optimizer_nfev':1})

    def test_explicit_masks_validate_shape_binary_and_preserve_caller_inputs(self):
        checked,metadata=adapter.validate_explicit_masks(self.frames,self.masks,self.foreground)
        checked[0][:]=False;metadata['cameras'][0]['camera']='changed'
        self.assertTrue(self.masks[0].all())
        self.assertEqual(self.foreground['cameras'][0]['camera'],'cam_a')
        for invalid in [[],[np.ones((5,6),bool)],[np.full((4,6),2,np.uint8)],[np.ones((4,6),float)]]:
            with self.subTest(invalid=str([x.shape for x in invalid])):
                with self.assertRaises(ValueError):
                    adapter.validate_explicit_masks(self.frames,invalid,self.foreground)

    def test_visible_wrapper_keeps_native_gate_priority_and_original_score_ranking(self):
        # A rejected perfect-score candidate must not beat an accepted one.
        seeds=[{'label':'rejected','arc_rms':0.},{'label':'accepted','arc_rms':1.}]
        candidates=[self.candidate(False,0.),self.candidate(True,5.)]
        with patch('neck_localization.visible_profile_baseline.extract_stereo_masks',side_effect=AssertionError('must not extract masks')), \
             patch.object(adapter,'observed_cloud',return_value=(np.zeros((400,3)),np.zeros(400,int))) as cloud, \
             patch.object(adapter,'seed_hypotheses',return_value=seeds), \
             patch.object(adapter,'fit_seed',side_effect=candidates) as fitter, \
             patch.object(adapter,'revalidate_candidate',side_effect=lambda c,*args:c) as validator:
            result=adapter.estimate_masked_visible_profile(self.frames,self.masks,self.foreground)
        self.assertTrue(result.accepted)
        self.assertEqual(result.diagnostics['selection_score'],5.)
        self.assertEqual(len(result.diagnostics['attempts']),2)
        self.assertEqual(validator.call_count,2)
        self.assertEqual(fitter.call_count,2)
        np.testing.assert_array_equal(cloud.call_args.args[1][0],self.masks[0])
        self.assertEqual(result.diagnostics['foreground']['method'],'synthetic_explicit_masks')

    def test_sor_wrapper_keeps_two_axis_starts_and_acceptance_then_score(self):
        seeds=[{'label':'first','arc_rms':0.},{'label':'second','arc_rms':1.}]
        candidates=[self.candidate(False,0.),self.candidate(True,5.),
                    self.candidate(True,2.),self.candidate(True,3.)]
        with patch('neck_localization.sor_baseline.extract_stereo_masks',side_effect=AssertionError('must not extract masks')), \
             patch.object(adapter,'observed_cloud',return_value=(np.zeros((400,3)),np.zeros(400,int))), \
             patch.object(adapter,'seed_hypotheses',return_value=seeds), \
             patch.object(adapter,'_fit_candidate',side_effect=candidates) as fitter:
            result=adapter.estimate_masked_sor(self.frames,self.masks,self.foreground)
        self.assertEqual(fitter.call_count,4)
        self.assertIsNone(fitter.call_args_list[0].args[-1])
        np.testing.assert_array_equal(fitter.call_args_list[1].args[-1],[0.,0.,1.])
        self.assertTrue(result.accepted)
        self.assertEqual(result.diagnostics['selection_score'],2.)
        self.assertEqual(result.diagnostics['axis_start'],'rim_axis')
        self.assertEqual(len(result.diagnostics['attempts']),4)


class MaskedStageContracts(unittest.TestCase):
    def test_cached_mask_integrity_is_checked_before_explicit_use(self):
        with tempfile.TemporaryDirectory(prefix='winejar-mask-contract-') as directory:
            root=Path(directory)
            (root/'observations').mkdir()
            rgb_path=root/'observations/cam_a_rgb.png'
            rgb=np.full((4,6,3),100,np.uint8)
            cv2.imwrite(str(rgb_path),rgb)
            mask_path=root/'mask.png'
            cv2.imwrite(str(mask_path),np.full((4,6),255,np.uint8))
            cache=root/'cache.json'
            cache.write_text(json.dumps(dict(schema=1,inputs_and_adapter_unchanged=True,
                no_ground_truth_read=True,no_depth_or_CAD_read=True,
                entries=[dict(layout='development',case='synthetic',camera='cam_a',status='ok',
                    rgb_path=str(rgb_path),rgb_sha256=masked_stage.shared.file_hash(rgb_path),
                    mask_path=str(mask_path),mask_sha256=masked_stage.shared.file_hash(mask_path),
                    mask_pixels_sha256=masked_stage.hashlib.sha256(np.ones((4,6),np.uint8).tobytes()).hexdigest(),
                    prompt_box_xyxy=[0,0,5,3],source_coarse_mask_bbox_xyxy=[0,0,5,3],
                    predicted_scores=[.9],selected_index=0)],
                protocol=dict(name='synthetic_fixed_protocol'),model={})))
            pairs=[dict(layout='development',case='synthetic',camera='cam_a',scene_dir=str(root))]
            frames=[SimpleNamespace(camera=SimpleNamespace(name='cam_a'),rgb=rgb)]
            masks,foreground=masked_stage.load_foreground(cache,pairs,frames,{})
            self.assertTrue(masks[0].all())
            self.assertEqual(foreground['cameras'][0]['mask_sha256'],
                             masked_stage.hashlib.sha256(np.ones((4,6),np.uint8).tobytes()).hexdigest())
            cv2.imwrite(str(mask_path),np.zeros((4,6),np.uint8))
            with self.assertRaisesRegex(ValueError,'mask_file_sha256_mismatch'):
                masked_stage.load_foreground(cache,pairs,frames,{})

    def test_all_explicit_mask_backends_complete_before_truth_even_after_exception(self):
        events=[]
        pairs=[dict(camera=name,pair_id='synthetic/'+name,layout='development',case='synthetic',
                    condition='clean',split='synthetic_test') for name in masked_stage.shared.ALL_CAMERAS]
        masks=[np.ones((2,2),bool) for _ in pairs]
        foreground={'method':'synthetic_cache'}
        def prepared(pair,depth,tracked):
            return SimpleNamespace(camera=SimpleNamespace(name=pair['camera'])),None,{}
        def backend(method):
            def infer(frames,explicit_masks,explicit_foreground,*,settings):
                self.assertIs(explicit_masks,masks)
                self.assertIs(explicit_foreground,foreground)
                events.append(method)
                if method=='visible_profile':raise RuntimeError('synthetic_failure')
                return Estimate(center=np.array([0.,0.,.6]),normal=np.array([0.,0.,1.]),radius=.05,accepted=True)
            return infer,None
        def truth(pair,tracked):
            self.assertEqual(events,['visible_profile','standard_sor'])
            events.append('truth')
            return dict(center_m=[0.,0.,.6],normal=[0.,0.,1.],radius_m=.05),{}
        with patch.object(masked_stage.shared,'load_prediction_index',return_value={}), \
             patch.object(masked_stage.shared,'read_prediction',return_value=(np.ones((2,2)),{})), \
             patch.object(masked_stage.shared,'prepare_frame',side_effect=prepared), \
             patch.object(masked_stage,'load_foreground',return_value=(masks,foreground)), \
             patch.object(masked_stage,'backend',side_effect=backend), \
             patch.object(masked_stage.shared,'_load_truth',side_effect=truth):
            records,_,_=masked_stage.worker((pairs,'/synthetic/predictions',
                ['visible_profile','standard_sor'],'/synthetic/cache'))
        self.assertEqual(events[-1],'truth')
        summary=masked_stage.aggregate(records)
        self.assertEqual(summary['n'],2)
        self.assertEqual(summary['input_or_backend_errors'],1)
        self.assertEqual(summary['full_circle_correct_output_rate'],.5)


if __name__=='__main__':
    unittest.main()
