"""Offline L-NDF capability adapter; no evaluation truth enters inference.

The public pretrained ConvONet and public OccNetOptimizer methods are loaded
unchanged. Compatibility glue replaces ROS/plotting imports and torch-scatter
forward reductions, not the learned field or descriptor objective. A reversible
observed-extent similarity transform places large jars inside the pretrained
network's 0.11 m input cube (the network itself multiplies coordinates by 10).
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
from scipy.spatial.transform import Rotation


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class ObservedSimilarity:
    center_m: np.ndarray
    scale: float

    @classmethod
    def from_points(cls, points_m, target_extent_m=.08):
        points = np.asarray(points_m, float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 30:
            raise ValueError("Need at least 30 measured 3D points")
        if not np.isfinite(points).all():
            raise ValueError("Nonfinite measured points")
        center = points.mean(axis=0)
        extent = float(2 * np.max(np.abs(points-center)))
        if extent <= 1e-6:
            raise ValueError("Degenerate observed extent")
        return cls(center, float(target_extent_m / extent))

    def points_to_model(self, points_m):
        return (np.asarray(points_m) - self.center_m) * self.scale

    def points_to_world(self, points):
        return np.asarray(points) / self.scale + self.center_m

    def pose_to_model(self, pose):
        result = np.asarray(pose, float).copy()
        result[:3, 3] = self.points_to_model(result[:3, 3])
        return result

    def pose_to_world(self, pose):
        result = np.asarray(pose, float).copy()
        result[:3, 3] = self.points_to_world(result[:3, 3])
        return result


def pose_list(pose):
    pose = np.asarray(pose)
    return np.r_[pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_quat()].tolist()


def apply_pose_numpy(points, pose):
    """Numerically equivalent to upstream ROS/trimesh 7D-pose conversion."""
    pose = np.asarray(pose)
    return np.asarray(points) @ Rotation.from_quat(pose[3:]).as_matrix().T + pose[:3]


def function_pose(center_m, normal):
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    # Fixed world-X tangent convention; yaw is not scored for a circular neck.
    u = np.array([1., 0., 0.]) - n[0] * n
    if np.linalg.norm(u) < 1e-6:
        u = np.array([0., 1., 0.]) - n[1] * n
    u /= np.linalg.norm(u)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([u, np.cross(n, u), n])
    pose[:3, 3] = center_m
    return pose


def collar_query(radius_model_m, half_height_model_m, n_points=1000, seed=711):
    """One fixed probing collar derived only from demonstration dimensions.

    L-NDF accepts arbitrary user query geometry. This collar is a task adapter,
    not a new learned model and not an estimate of an evaluation jar's radius.
    """
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0, 2 * np.pi, n_points)
    radius = radius_model_m * rng.uniform(.8, 1.2, n_points)
    z = rng.uniform(-half_height_model_m, half_height_model_m, n_points)
    return np.column_stack([radius * np.cos(angle), radius * np.sin(angle), z])


def observation_cloud(folder, camera_name="cam_a", upper_window_m=None):
    """Saved ordinary RGB/depth/camera files only; no masks or target labels."""
    import json
    import cv2
    from .geometry import Camera
    from .stereo_robust import blue_screen_foreground
    folder = Path(folder)
    metadata_path = folder / "camera_true_parameters.json"
    metadata = json.loads(metadata_path.read_text())
    item = next(c for c in metadata["cameras"] if c["name"] == camera_name)
    camera = Camera(item["name"], np.array(item["K"]),
                    np.array(item["T_world_from_camera"]), item["width"], item["height"])
    rgb_path = folder / f"{camera_name}_rgb.png"
    depth_path = folder / f"{camera_name}_depth_z_m.npz"
    rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
    with np.load(depth_path) as data:
        depth = data["depth_z_m"]
    mask = blue_screen_foreground(rgb)
    valid = mask & np.isfinite(depth) & (depth > 0)
    yy, xx = np.nonzero(valid)
    cloud = camera.unproject(np.column_stack([xx, yy]), depth[yy, xx])
    crop = None
    if upper_window_m is not None:
        # Same declared station and observed upper-envelope rule as the
        # restoration adapter. This is an observation prior, not a neck label.
        from .depth_profile import ProfileSettings
        settings = ProfileSettings()
        keep = ((cloud[:, 2] >= settings.world_z_bounds_m[0]) &
                (cloud[:, 2] <= settings.world_z_bounds_m[1]) &
                (np.linalg.norm(cloud[:, :2], axis=1) < settings.world_xy_radius_m))
        cloud = cloud[keep]
        if len(cloud) < 30:
            raise ValueError("Insufficient station foreground")
        envelope = float(np.percentile(cloud[:, 2], 99.5))
        cloud = cloud[cloud[:, 2] >= envelope-upper_window_m]
        crop = dict(rule="fixed_station_and_z_ge_observed_z_p995_minus_window",
            observed_z_p995_m=envelope, upper_window_m=float(upper_window_m),
            world_z_bounds_m=list(settings.world_z_bounds_m),
            world_xy_radius_m=settings.world_xy_radius_m, evaluation_labels_read=False)
    provenance = dict(camera=camera_name, points=len(cloud),
        input_sha256={p.name: file_sha256(p) for p in [metadata_path, rgb_path, depth_path]},
        mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
        crop=crop,
        evidence="RGB-only blue-background mask plus valid saved depth; optional observed upper station crop")
    return cloud, provenance


def _torch_scatter_compat():
    import torch
    module = types.ModuleType("torch_scatter")

    def scatter_mean(src, index, dim=-1, out=None, dim_size=None):
        dim %= src.ndim
        if dim != src.ndim - 1:
            raise NotImplementedError("L-NDF adapter supports last-axis reductions only")
        index = index.expand_as(src)
        size = out.shape[-1] if out is not None else dim_size
        result = src.new_zeros((*src.shape[:-1], size))
        result.scatter_add_(dim, index, src)
        count = src.new_zeros(result.shape)
        count.scatter_add_(dim, index, torch.ones_like(src))
        result = result / count.clamp_min(1)
        if out is not None:
            out.copy_(result)
            return out
        return result

    def scatter_max(src, index, dim=-1, out=None, dim_size=None):
        dim %= src.ndim
        if dim != src.ndim - 1:
            raise NotImplementedError("L-NDF adapter supports last-axis reductions only")
        size = out.shape[-1] if out is not None else dim_size
        result = src.new_zeros((*src.shape[:-1], size))
        result.scatter_reduce_(dim, index.expand_as(src), src, reduce="amax", include_self=False)
        if out is not None:
            out.copy_(result)
            result = out
        # Upstream encoder never uses argmax; explicitly do not invent indices.
        return result, None

    module.scatter_mean, module.scatter_max = scatter_mean, scatter_max
    return module


def _source_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_public_model(upstream, weights, device="cpu"):
    """Load identical public network and strict-match all checkpoint tensors."""
    import torch
    from scipy.spatial import cKDTree
    upstream = Path(upstream)
    sys.path.insert(0, str(upstream / "src"))
    if "torch_scatter" not in sys.modules and importlib.util.find_spec("torch_scatter") is None:
        sys.modules["torch_scatter"] = _torch_scatter_compat()
    kd = types.ModuleType("ndf_robot.utils.libkdtree")
    kd.KDTree = cKDTree  # Imported by common.py, not called by the network.
    sys.modules["ndf_robot.utils.libkdtree"] = kd
    from ndf_robot.model.conv_occupancy_net.conv_occupancy_net import ConvolutionalOccupancyNetwork
    model = ConvolutionalOccupancyNetwork(latent_dim=128, return_features=True,
        sigmoid=False, acts="last")
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).to(device)
    return model


def public_optimizer(upstream, model, query_points, *, device="cpu", iterations=500,
                     initializations=20, workdir=None, cos_loss=False):
    """Execute unmodified public process_demos/optimize_transform_implicit.

    Import-only robotics/plotting dependencies are bypassed with an AST class
    loader. Visualization is disabled. Original helper rotation math, Healpix
    initialization grid, loss, optimizer and learning rate are retained.
    """
    import torch
    import torch.nn.functional as F
    upstream = Path(upstream)
    utils_dir = upstream / "src/ndf_robot/utils"
    torch_util = _source_module(utils_dir / "torch_util.py", "lndf_official_torch_util")
    torch3d = _source_module(utils_dir / "torch3d_util.py", "lndf_official_torch3d_util")
    # generate_healpix_grid imports just this airobot transform helper.
    airobot = types.ModuleType("airobot")
    airobot_utils = types.ModuleType("airobot.utils")
    common = types.ModuleType("airobot.utils.common")
    common.euler2rot = lambda angles: Rotation.from_euler("xyz", angles).as_matrix()
    airobot_utils.common = common
    airobot.utils = airobot_utils
    sys.modules.update({"airobot": airobot, "airobot.utils": airobot_utils,
                        "airobot.utils.common": common})
    source = ast.parse((utils_dir / "util.py").read_text())
    nodes = [n for n in source.body if isinstance(n, ast.FunctionDef)
             and n.name in {"generate_healpix_grid", "transform_pcd"}]
    globals_util = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(utils_dir / "util.py"), "exec"), globals_util)
    grid_cache = {}
    def grid(**kwargs):
        key = tuple(sorted(kwargs.items()))
        if key not in grid_cache:
            grid_cache[key] = globals_util["generate_healpix_grid"](**kwargs)
        return grid_cache[key]
    util = types.SimpleNamespace(apply_pose_numpy=apply_pose_numpy,
        transform_pcd=globals_util["transform_pcd"], generate_healpix_grid=grid,
        safe_makedirs=lambda path: Path(path).mkdir(parents=True, exist_ok=True))
    optimizer_path = upstream / "src/ndf_robot/opt/optimizer_lite.py"
    source = ast.parse(optimizer_path.read_text())
    nodes = [n for n in source.body if isinstance(n, ast.ClassDef)
             and n.name in {"Demo", "OccNetOptimizer"}]
    namespace = dict(np=np, torch=torch, nn=torch.nn, F=F, os=__import__("os"),
        osp=__import__("os").path, util=util, torch_util=torch_util, torch3d_util=torch3d,
        multiplot=lambda *a, **k: None, log_debug=lambda *a, **k: None)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(optimizer_path), "exec"), namespace)
    optimizer_class = namespace["OccNetOptimizer"]
    # Explicit device selection prevents the upstream automatic CUDA selection
    # from allocating GPU during our CPU preparation / resource queue stage.
    optimizer = optimizer_class.__new__(optimizer_class)
    optimizer.__dict__.update(model=model, model_type=model.model_type,
        query_pts=query_points, query_pts_override=True, demos=[], target_act_hat=None,
        n_obj_points=2000, n_query_points=1500, cos_loss=cos_loss,
        loss_fn=(lambda output, target: -F.cosine_similarity(output, target, dim=1).mean())
            if cos_loss else torch.nn.L1Loss(), dev=torch.device(device), opt_iterations=iterations,
        noise_scale=0., noise_decay=.5, rand_translate=True, use_tsne=False,
        M_override=initializations, opt_fname_prefix="functional_collar", save_all_opt=False,
        viz_files=[], debug_viz_path=str(workdir or "."), viz_path=str(workdir or "."))
    optimizer.rot_grid = grid(size=1e6)
    optimizer._visualize_reconstruction = lambda *a, **k: None
    optimizer._visualize_pose = lambda *a, **k: None
    return optimizer, namespace["Demo"]


COMPATIBILITY = [
    "public ConvONet and strict checkpoint loading, latent_dim128/acts_last/scaling10/grid32",
    "public optimizer_lite Demo/process_demos/optimize_transform_implicit bodies unchanged",
    "torch_scatter forward mean/max via PyTorch scatter_add/scatter_reduce; unused KDTree via scipy",
    "ROS/trimesh 7D transform via scipy; exact upstream torch rotation math and Healpix grid",
    "plotting disabled; model parameter gradients disabled while preserving query gradients",
    "observed input extent maps to .08 m; points and demo/query poses transformed together",
    "optional upper station crop uses observed z995 minus fixed70mm, same restoration ROI prior",
    "L1 default or exact official bottle cos_loss; all optimizer iteration bodies unchanged",
    "task collar query dimensions from four supervised development jars only; no test labels in inference",
]
