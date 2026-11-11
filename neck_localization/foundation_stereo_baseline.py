"""Thin headless adapter for pinned official FoundationStereo inference.

No retraining, ROI, target dimensions or ground-truth geometry. The official
network and checkpoint remain unchanged. Only unused visualization imports and
redundant pretrained-backbone downloads are avoided at construction time.
"""
import ast
from pathlib import Path
import sys
import time
import types

import numpy as np


class FoundationStereoRunner:
    def __init__(self, repo, checkpoint, *, low_memory=True, iterations=32):
        import torch
        import timm
        from omegaconf import OmegaConf
        self.torch, self.iterations, self.low_memory = torch, iterations, low_memory
        repo, checkpoint = Path(repo).resolve(), Path(checkpoint).resolve()
        sys.path.insert(0, str(repo))
        # Compile the exact two pure inference helpers from upstream Utils.
        # This avoids importing Open3D/pandas just for a headless forward pass.
        source = (repo / 'Utils.py').read_text()
        tree = ast.parse(source)
        helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name in ('freeze_model', 'get_resize_keep_aspect_ratio')]
        if len(helpers) != 2:
            raise ValueError('Upstream Utils interface changed')
        module = types.ModuleType('Utils')
        module.np = np
        exec(compile(ast.Module(body=helpers, type_ignores=[]), str(repo / 'Utils.py'), 'exec'), module.__dict__)
        sys.modules['Utils'] = module
        from core.foundation_stereo import FoundationStereo
        from core.utils.utils import InputPadder
        self.padder_class = InputPadder
        cfg = OmegaConf.load(checkpoint.parent / 'cfg.yaml')
        if 'vit_size' not in cfg:
            cfg.vit_size = 'vitl'
        cfg.low_memory = bool(low_memory)
        hub_load, create_model = torch.hub.load, timm.create_model

        def local_hub(repository, model, *args, **kwargs):
            if repository == 'facebookresearch/dinov2':
                if kwargs.get('pretrained', False):
                    raise ValueError('Unexpected separate DINO weight request')
                kwargs['source'] = 'local'
                return hub_load(str(repo / 'dinov2'), model, *args, **kwargs)
            return hub_load(repository, model, *args, **kwargs)

        def construction_only(name, *args, **kwargs):
            # All tensor state is subsequently loaded strictly from the full
            # official checkpoint, including the feature backbone.
            kwargs['pretrained'] = False
            return create_model(name, *args, **kwargs)

        torch.hub.load, timm.create_model = local_hub, construction_only
        try:
            model = FoundationStereo(cfg)
        finally:
            torch.hub.load, timm.create_model = hub_load, create_model
        checkpoint_data = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint_data['model'], strict=True)
        self.checkpoint_epoch = checkpoint_data.get('epoch')
        self.checkpoint_step = checkpoint_data.get('global_step')
        del checkpoint_data
        self.config = OmegaConf.to_container(cfg, resolve=True)
        self.model = model.eval().cuda()
        torch.set_grad_enabled(False)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def predict(self, left, right):
        torch = self.torch
        if left.shape != right.shape or left.ndim != 2 or left.dtype != np.uint8:
            raise ValueError('Expected equally sized uint8 IR pairs')
        h, w = left.shape
        tensors = [torch.as_tensor(np.repeat(im[..., None], 3, axis=2)).cuda().float()[None].permute(0,3,1,2)
                   for im in (left, right)]
        padder = self.padder_class(tensors[0].shape, divis_by=32, force_square=False)
        a, b = padder.pad(*tensors)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            disparity = self.model(a, b, iters=self.iterations, test_mode=True, low_memory=self.low_memory)
        disparity = padder.unpad(disparity.float()).cpu().numpy().reshape(h, w)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) * 1000
        return disparity, dict(inference_ms=elapsed, gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())
