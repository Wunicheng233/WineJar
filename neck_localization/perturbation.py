"""Observation-space stress factors. Assumed magnitudes are not sensor calibration.

All corrupted camera models are applied to raw measured Z again; world points
are never retained from the true calibration when a camera model changes.
"""
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import Camera, Observation


def perturb(observations, factor, seed, calibrated_cameras=None):
    rng=np.random.default_rng(seed)
    output=[]
    by_name={c.name:c for c in (calibrated_cameras or [])}
    for obs in observations:
        camera=deepcopy(by_name.get(obs.camera.name,obs.camera))
        uv=obs.pixels.copy()
        original=obs.camera
        pcam=(obs.points_world-original.T_world_from_camera[:3,3])@original.T_world_from_camera[:3,:3]
        depth_pixels=original.project(obs.points_world)
        z=pcam[:,2].copy()
        if factor.get("views") and camera.name not in factor["views"]:
            continue
        if factor.get("focal_fraction"):
            camera.K[0,0]*=1+factor["focal_fraction"]
            camera.K[1,1]*=1+factor["focal_fraction"]
        camera.K[:2,2]+=factor.get("principal_shift_px",0)
        camera.T_world_from_camera[:3,3]+=factor.get("common_translation_m",0)
        if factor.get("camera_translation_m") and camera.name=="cam1":
            camera.T_world_from_camera[:3,3]+=np.array([factor["camera_translation_m"],0,0])
        if factor.get("camera_rotation_deg") and camera.name=="cam1":
            camera.T_world_from_camera[:3,:3]=Rotation.from_euler("z",factor["camera_rotation_deg"],degrees=True).as_matrix()@camera.T_world_from_camera[:3,:3]
        if factor.get("radial_k1"):
            # Observed image has radial distortion, estimator deliberately assumes D=0.
            xy=np.c_[uv,np.ones(len(uv))]@np.linalg.inv(original.K).T
            xy[:,:2]*=1+factor["radial_k1"]*np.sum(xy[:,:2]**2,axis=1,keepdims=True)
            uv=(xy@original.K.T)[:,:2]
        uv+=rng.normal(0,factor.get("edge_sigma_px",0),uv.shape)
        if factor.get("edge_radial_bias_px"):
            direction=uv-np.median(uv,axis=0)
            direction/=np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-9)
            uv+=factor["edge_radial_bias_px"]*direction
        z=z*(1+factor.get("depth_scale_fraction",0))+factor.get("depth_offset_m",0)
        if camera.name=="cam1":
            z+=factor.get("one_camera_depth_offset_m",0)
        if factor.get("disparity_sigma_px"):
            fb=original.K[0,0]*.05  # 50mm is an assumed D435-like stereo baseline.
            z=fb/(fb/z+rng.normal(0,factor["disparity_sigma_px"],len(z)))
        if factor.get("outlier_fraction"):
            count=int(len(uv)*factor["outlier_fraction"])
            idx=rng.choice(len(uv),count,replace=False)
            uv[idx]+=rng.uniform(-20,20,(count,2))
            depth_pixels[idx]+=rng.uniform(-20,20,(count,2))
            z[idx]+=rng.uniform(-.025,.025,count)
        keep=rng.uniform(size=len(uv))>=factor.get("hole_fraction",0)
        if factor.get("hole_sector_fraction"):
            relative=uv-np.mean(uv,axis=0)
            angle=np.mod(np.arctan2(relative[:,1],relative[:,0]),2*np.pi)
            keep &= angle >= 2*np.pi*factor["hole_sector_fraction"]
        # Per-view motion is a deterministic asynchronous rigid translation.
        points=camera.unproject(depth_pixels,z)
        if factor.get("motion_m"):
            index=int(camera.name[-1])
            points+=np.array([index*factor["motion_m"],0,0])
            uv=camera.project(points)
        output.append(Observation(camera,uv[keep],points[keep]))
    return output
