"""Observation-only adapter for the official P2P-Bridge XYZ checkpoint.

The upstream model, diffusion schedule, and CUDA operators are used unchanged.
Only unused training/evaluation imports are isolated. No robot or camera APIs
are imported. This is a pretrained, conditional point-restoration baseline,
not a new estimator and not an independent test of the existing scene family.
"""
from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import time
import types

import numpy as np
from scipy.spatial import cKDTree


def load_observation_frames(folder):
    """Read the saved RGB, depth, and camera metadata without renderer imports."""
    import cv2
    from .depth_profile import DepthFrame
    from .geometry import Camera

    folder = Path(folder)
    metadata = json.loads((folder / "camera_true_parameters.json").read_text())
    frames = []
    for item in metadata["cameras"]:
        name = item["name"]
        camera = Camera(name, np.array(item["K"]), np.array(item["T_world_from_camera"]),
                        item["width"], item["height"])
        rgb = cv2.imread(str(folder / f"{name}_rgb.png"))
        if rgb is None:
            raise FileNotFoundError(folder / f"{name}_rgb.png")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        with np.load(folder / f"{name}_depth_z_m.npz") as data:
            depth = data["depth_z_m"]
        frames.append(DepthFrame(camera, rgb, depth))
    return frames


def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def observation_cloud(frames, *, upper_window_m=0.070, max_points=8192, seed=42):
    """Use RGB foreground, the existing fixed workcell, and observed upper Z.

    The 99.5th percentile is a measured envelope used only to set the lower
    crop boundary. It is neither the true neck boundary nor a target label.
    The chosen camera parameters are supplied by the saved observation.
    """
    from dataclasses import asdict
    from .depth_profile import ProfileSettings
    from .stereo_robust import extract_stereo_masks

    settings = ProfileSettings()
    masks, mask_metadata = extract_stereo_masks(frames)
    points, camera_ids, pixel_uv = [], [], []
    for camera_id, (frame, mask) in enumerate(zip(frames, masks)):
        valid = mask & np.isfinite(frame.depth_z_m) & (frame.depth_z_m > 0)
        yy, xx = np.nonzero(valid)
        cloud = frame.camera.unproject(np.c_[xx, yy], frame.depth_z_m[yy, xx])
        keep = ((cloud[:, 2] >= settings.world_z_bounds_m[0])
                & (cloud[:, 2] <= settings.world_z_bounds_m[1])
                & (np.linalg.norm(cloud[:, :2], axis=1) < settings.world_xy_radius_m))
        points.append(cloud[keep])
        camera_ids.append(np.full(int(keep.sum()), camera_id, dtype=np.int16))
        pixel_uv.append(np.c_[xx, yy][keep].astype(np.int32))
    if not points or sum(len(p) for p in points) < 16:
        raise ValueError("insufficient_observed_foreground")
    points = np.concatenate(points)
    camera_ids = np.concatenate(camera_ids)
    pixel_uv = np.concatenate(pixel_uv)
    envelope = float(np.percentile(points[:, 2], 99.5))
    keep = points[:, 2] >= envelope - upper_window_m
    points, camera_ids, pixel_uv = points[keep], camera_ids[keep], pixel_uv[keep]
    full_count = len(points)
    if full_count < 16:
        raise ValueError("insufficient_observed_upper_surface")
    if max_points and len(points) > max_points:
        ids = np.sort(np.random.default_rng(seed).choice(len(points), max_points, replace=False))
        points, camera_ids, pixel_uv = points[ids], camera_ids[ids], pixel_uv[ids]
    return points, camera_ids, pixel_uv, {
        "rule": "existing_rgb_foreground_and_fixed_station_then_z_ge_observed_z_p995_minus_window",
        "observed_z_p995_m": envelope,
        "upper_window_m": float(upper_window_m),
        "full_upper_point_count": full_count,
        "selected_point_count": len(points),
        "maximum_points": max_points,
        "sampling_seed": seed,
        "foreground": mask_metadata,
        "foreground_settings_sha256": sha256(json.dumps(mask_metadata["settings"], sort_keys=True).encode()).hexdigest(),
        "foreground_module_sha256": file_sha256(Path(__file__).with_name("stereo_robust.py")),
        "station_settings": asdict(settings),
        "station_module_sha256": file_sha256(Path(__file__).with_name("depth_profile.py")),
        "camera_counts": {frame.camera.name: int((camera_ids == i).sum()) for i, frame in enumerate(frames)},
        "evaluation_labels_read": False,
    }


def normalize_unit_sphere(points):
    """Official object inference normalization: bbox midpoint and max radius."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be finite N x 3")
    center = (points.max(axis=0) + points.min(axis=0)) / 2
    scale = float(np.linalg.norm(points - center, axis=1).max())
    if scale <= 1e-12:
        raise ValueError("degenerate point cloud")
    return (points - center) / scale, center, scale


def deterministic_fps(points, count):
    """CPU FPS beginning at index 0, matching the deterministic convention."""
    points = np.asarray(points)
    count = min(int(count), len(points))
    chosen = np.empty(count, dtype=np.int64)
    distance = np.full(len(points), np.inf)
    current = 0
    for i in range(count):
        chosen[i] = current
        distance = np.minimum(distance, ((points - points[current]) ** 2).sum(axis=1))
        current = int(distance.argmax())
    return chosen


def patch_indices(points, *, patch_size=2048, oversampling=3):
    """Official object-style FPS seeds + nearest-neighbour patches.

    If an observation has fewer points than the checkpoint's 2048, existing
    points are repeated (never perturbed or synthesized). These repeats are
    explicitly tracked when aggregating predictions.
    """
    points = np.asarray(points)
    n = len(points)
    count = max(1, int(oversampling * n / patch_size))
    seeds = deterministic_fps(points, count)
    k = min(n, patch_size)
    _, indices = cKDTree(points).query(points[seeds], k=k)
    indices = np.asarray(indices, dtype=np.int64).reshape(len(seeds), k)
    if k < patch_size:
        indices = np.tile(indices, (1, int(np.ceil(patch_size / k))))[:, :patch_size]
    # A few peripheral points may not be covered by the official seed count.
    # Preserve them unchanged; do not invent positions or silently replace them.
    return indices


def average_patch_predictions(input_points, indices, predictions):
    """Point-corresponding averaging, as in official room inference.

    This adapter uses object normalization/patches and room-style averaging
    instead of the official object's final FPS resampling. It preserves source
    camera/pixel identities for the downstream re-projection audit.
    """
    input_points = np.asarray(input_points)
    sums = np.zeros_like(input_points, dtype=np.float64)
    counts = np.zeros(len(input_points), dtype=np.int64)
    for index, prediction in zip(indices, predictions):
        np.add.at(sums, index, prediction)
        np.add.at(counts, index, 1)
    output = input_points.copy()
    valid = counts > 0
    output[valid] = sums[valid] / counts[valid, None]
    return output, counts


def _namespace(name, path=None):
    module = types.ModuleType(name)
    module.__path__ = [str(path)] if path is not None else []
    sys.modules[name] = module
    return module


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def isolate_upstream_inference_imports(source, native_build):
    """Load only unchanged upstream operators needed by PVCNN inference."""
    import torch

    source = Path(source).resolve()
    sys.path.insert(0, str(native_build))
    import pointnet2_batch_cuda

    sys.path.insert(0, str(source))
    _namespace("third_party", source / "third_party")
    _namespace("third_party.openpoints", source / "third_party/openpoints")
    _namespace("third_party.openpoints.models", source / "third_party/openpoints/models")
    cpp = _namespace("third_party.openpoints.cpp")
    cpp.pointnet2_cuda = pointnet2_batch_cuda
    folder = source / "third_party/openpoints/models/layers"
    layers = _namespace("third_party.openpoints.models.layers", folder)
    exported = {
        "ball_query": ["ball_query"],
        "group": ["pvcnn_grouping"],
        "sampling": ["furthest_point_sample_pvcnn"],
        "interpolatation": ["nearest_neighbor_interpolate"],
        "voxelization": ["avg_voxelize"],
        "devoxelization": ["trilinear_devoxelize"],
    }
    for filename, names in exported.items():
        module = _load_file("third_party.openpoints.models.layers." + filename, folder / (filename + ".py"))
        for name in names:
            setattr(layers, name, getattr(module, name))
    # The model's loss.py eagerly imports a training-only EMD extension even
    # when the checkpoint uses MSE. Fail loudly if EMD is ever actually called.
    _namespace("metrics", source / "metrics")
    _namespace("metrics.emd_assignment")
    emd = _namespace("metrics.emd_assignment.emd_module")

    class UnavailableEMD:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("training-only EMD was not installed for this inference adapter")

    emd.emdModule = UnavailableEMD
    return {"backend": "unchanged_upstream_pointnet2_batch_cuda", "torch": torch.__version__,
            "training_only_emd_disabled": True}


class P2PBridgeRestorer:
    """Strictly load pretrained backbone and EMA; never accept partial weights."""
    def __init__(self, source, checkpoint, config, native_build, *, device="cuda:0", steps=5):
        import torch
        from omegaconf import OmegaConf

        self.runtime = isolate_upstream_inference_imports(source, native_build)
        from models.p2pb import P2PB
        from models.unet_pvc import PVCNN2Unet

        cfg = OmegaConf.load(config)
        if cfg.diffusion.loss_type != "mse":
            raise ValueError("this adapter is restricted to the official XYZ/MSE checkpoint")
        cfg.gpu = device
        self.cfg, self.device, self.steps = cfg, device, steps
        self.model = P2PB(cfg, PVCNN2Unet(cfg)).to(device)
        weights = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = weights["model_state"]
        model_state = {k[len("model.module."):]: v for k, v in state.items() if k.startswith("model.module.")}
        ema_state = {k[len("ema."):]: v for k, v in state.items() if k.startswith("ema.")}
        self.model.model.load_state_dict(model_state, strict=True)
        self.model.ema.load_state_dict(ema_state, strict=True)
        self.model.eval()
        self.model.ema.eval()
        self.runtime.update(checkpoint_step=int(weights["step"]), strict_model_tensors=len(model_state),
                            strict_ema_tensors=len(ema_state), use_ema=True, steps=steps,
                            device=str(device), cuda_version=torch.version.cuda,
                            gpu_name=torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else None)
        del weights, state, model_state, ema_state

    def restore(self, points, *, seed=42, oversampling=3):
        import torch

        using_cuda = torch.device(self.device).type == "cuda"
        if using_cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        torch.manual_seed(seed)
        normalized, center, scale = normalize_unit_sphere(points)
        indices = patch_indices(normalized, patch_size=int(self.cfg.data.npoints), oversampling=oversampling)
        patches = normalized[indices]
        patch_centers = patches.mean(axis=1, keepdims=True)
        centered = patches - patch_centers
        # Match official object inference: one shared maximum across patches.
        patch_scale = float(np.linalg.norm(centered, axis=-1).max())
        inference_started = time.perf_counter()
        predictions = []
        for patch, patch_center in zip(centered, patch_centers):
            tensor = torch.as_tensor(patch / patch_scale, dtype=torch.float32, device=self.device).T[None]
            # The original sampler sets its backbone back to train() after a
            # sample. Explicit eval here keeps every patch in inference mode.
            self.model.eval()
            self.model.ema.eval()
            with torch.inference_mode():
                result = self.model.sample(x_start=tensor, use_ema=True, steps=self.steps,
                                           log_count=self.steps, verbose=False)
            prediction = result["x_pred"][0].T.cpu().numpy() * patch_scale + patch_center
            predictions.append(prediction)
        inference_ms = (time.perf_counter() - inference_started) * 1000
        restored, counts = average_patch_predictions(normalized, indices, predictions)
        restored = restored * scale + center
        if not np.isfinite(restored).all():
            raise RuntimeError("pretrained model produced nonfinite coordinates")
        return restored, counts, {
            "normalization_center_m": center.tolist(), "normalization_scale_m": scale,
            "patch_count": len(indices), "patch_size": int(self.cfg.data.npoints),
            "oversampling": oversampling, "patch_scale_normalized": patch_scale,
            "aggregation": "official_room_style_pointwise_average_of_object_style_knn_patches",
            "uncovered_points_preserved": int((counts == 0).sum()),
            "minimum_updates": int(counts.min()), "maximum_updates": int(counts.max()),
            "model_patch_loop_ms": inference_ms,
            "restoration_total_ms": (time.perf_counter() - started) * 1000,
            "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)) if using_cuda else None,
            "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)) if using_cuda else None,
        }
