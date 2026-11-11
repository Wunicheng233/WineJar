"""Prepare reviewable static MuJoCo assets and RGB-D inputs, without estimation.

Photo-inspired geometry and procedural materials are synthetic assets, not a
metric reconstruction or a RealSense rendering/sensor model.
"""
from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "egl")

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import platform
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import Camera, look_at
from .jar_model import JarSpec, build_jar_mesh


def _numbers(values):
    return " ".join(f"{x:.10g}" for x in np.asarray(values).ravel())


def write_json(path, data):
    def encode(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=encode, allow_nan=False)+"\n")


def write_obj(path, mesh, regions=None):
    vertices, faces = mesh["vertices"], mesh["faces"]
    uv, face_uv = mesh["texcoords"], mesh["face_texcoords"]
    if regions is not None:
        keep=np.isin(mesh["face_regions"],regions)
        faces,face_uv=faces[keep],face_uv[keep]
    with path.open("w") as stream:
        stream.write("# Synthetic circular-section jar, metres. Not a measured reconstruction.\n")
        for point in vertices:
            stream.write("v "+_numbers(point)+"\n")
        for point in uv:
            stream.write("vt "+_numbers(point)+"\n")
        for triangle, texture_triangle in zip(faces, face_uv):
            stream.write("f "+" ".join(f"{int(v)+1}/{int(t)+1}" for v,t in zip(triangle,texture_triangle))+"\n")


def ceramic_texture(seed, width=512, height=512):
    """Independent mottling; there is no geometry-target-dependent stripe or ring."""
    rng=np.random.default_rng(seed)
    field=np.zeros((height,width),np.float32)
    for size,weight in [(7,.45),(23,.30),(83,.15)]:
        noise=rng.uniform(size=(size,size)).astype(np.float32)
        # A wrapped azimuth avoids a sharp artificial UV seam.
        noise=np.c_[noise,noise[:,0]]
        field+=weight*cv2.resize(noise,(width,height),interpolation=cv2.INTER_CUBIC)
    field+=rng.normal(0,.025,(height,width)).astype(np.float32)
    low,high=np.percentile(field,[3,97])
    mottling=np.clip((field-low)/(high-low),0,1)
    # Verified with a coloured UV diagnostic: image row 0 maps to the mouth,
    # image row H-1 to the base for the OBJ/MuJoCo texture path used here.
    height_fraction=1-np.linspace(0,1,height)[:,None]
    # Grey-white weathering covers a dark ceramic substrate.  Mix layers rather
    # than painting a black cap: the reference has mid-grey flecks, pale scuffs
    # and a broken shoulder boundary.  These are image-inspired colours, not
    # calibrated physical albedo measurements.
    transition=np.clip((height_fraction-.64+.16*(mottling-.5))/.26,0,1)
    dark_coverage=transition*transition*(3-2*transition)
    fine_noise=rng.uniform(size=(height,width)).astype(np.float32)
    fine=cv2.GaussianBlur(fine_noise,(0,0),.65)
    fine=(fine-fine.mean())/(fine.std()+1e-6)
    substrate=.27+.16*mottling+.018*fine
    patina=.77+.10*mottling+.014*fine
    value=patina*(1-dark_coverage)+substrate*dark_coverage
    # Sparse pale worn patches remain even on the dark upper shoulder/lip.
    chips=np.clip((mottling+.12*fine-.66)/.22,0,1)
    pale_remnants=.52*chips*dark_coverage
    value=value*(1-pale_remnants)+patina*pale_remnants
    # Interrupted vertical wear marks are strongest near the base, as in the
    # photo.  The colour field never reads the target range or neck geometry.
    columns=cv2.resize(rng.uniform(size=(24,180)).astype(np.float32),
                       (width,height),interpolation=cv2.INTER_CUBIC)
    scratches=np.clip((columns-.65)/.28,0,1)*np.clip((mottling-.25)/.55,0,1)
    base_wear=np.clip((.19-height_fraction)/.19,0,1)
    value-=scratches*(.08+.27*base_wear)*(1-.65*dark_coverage)
    value=np.clip(value,.20,.90)
    color=np.stack((value*.995,value,value*1.005),axis=-1)
    return (np.clip(color,0,1)*255).astype(np.uint8)


def scene_cameras(scene_config):
    target=np.asarray(scene_config["nominal_target_world_m"],float)
    width,height=scene_config.get("resolution",[640,480])
    cameras=[]
    for item in scene_config["cameras"]:
        az,el=np.deg2rad([item["azimuth_deg"],item["elevation_deg"]])
        distance=item["distance_m"]
        position=target+distance*np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
        cameras.append(look_at(item["name"],position,target,width,height,scene_config.get("fovy_deg",50)))
    return cameras


def _add_camera(world,camera,fovy):
    gl_rotation=camera.T_world_from_camera[:3,:3]@np.diag([1,-1,-1])
    ET.SubElement(world,"camera",name=camera.name,pos=_numbers(camera.T_world_from_camera[:3,3]),
                  xyaxes=_numbers(gl_rotation[:,:2].T),fovy=str(fovy))


def scene_xml(cameras,scene_config,material):
    root=ET.Element("mujoco",model="winejar_scene_preparation")
    ET.SubElement(root,"compiler",angle="degree",meshdir="assets",texturedir="assets")
    ET.SubElement(root,"option",gravity="0 0 0")
    ET.SubElement(root,"statistic",extent="1",center="0 0 .30")
    visual=ET.SubElement(root,"visual")
    ET.SubElement(visual,"global",offwidth=str(cameras[0].width),offheight=str(cameras[0].height))
    ET.SubElement(visual,"quality",offsamples="0",shadowsize="1024")
    ET.SubElement(visual,"map",znear=".01",zfar="5")
    ET.SubElement(visual,"headlight",ambient=".35 .35 .35",diffuse=".45 .45 .45",specular=".05 .05 .05")
    asset=ET.SubElement(root,"asset")
    ET.SubElement(asset,"texture",name="sky",type="skybox",builtin="gradient",rgb1=".08 .12 .18",rgb2=".14 .19 .26",width="256",height="1536")
    ET.SubElement(asset,"mesh",name="jar_mesh",file="jar.obj",smoothnormal="true")
    ET.SubElement(asset,"mesh",name="jar_inside_mesh",file="jar_inside.obj",smoothnormal="true")
    ET.SubElement(asset,"texture",name="ceramic_texture",type="2d",file="ceramic.png")
    if material=="mottled_gray":
        ET.SubElement(asset,"material",name="jar_material",texture="ceramic_texture",texrepeat="1 1",texuniform="false",rgba="1 1 1 1",specular=".12",shininess=".15")
    elif material=="plain_clay":
        ET.SubElement(asset,"material",name="jar_material",rgba=".54 .48 .40 1",specular=".07",shininess=".1")
    else:
        raise ValueError(f"Unknown material {material}")
    ET.SubElement(asset,"material",name="interior_material",rgba=".055 .050 .043 1",specular=".02",shininess=".05")
    world=ET.SubElement(root,"worldbody")
    for name,pos,diffuse in [("key","1 -1 2",".65 .65 .65"),("fill","-1 1 1.5",".35 .35 .35")]:
        ET.SubElement(world,"light",name=name,pos=pos,dir="0 0 -1",diffuse=diffuse,ambient=".10 .10 .10",castshadow="false")
    ET.SubElement(world,"geom",name="floor",type="plane",pos="0 0 -.06",size="2 2 .01",rgba=".08 .12 .18 1",contype="0",conaffinity="0")
    ET.SubElement(world,"geom",name="static_conveyor",type="box",pos="0 0 -.025",size=".65 .42 .025",rgba=".06 .10 .15 1",contype="0",conaffinity="0")
    body=ET.SubElement(world,"body",name="jar",pos="0 0 0")
    ET.SubElement(body,"geom",name="jar_surface",type="mesh",mesh="jar_mesh",material="jar_material",contype="0",conaffinity="0")
    ET.SubElement(body,"geom",name="jar_interior",type="mesh",mesh="jar_inside_mesh",material="interior_material",contype="0",conaffinity="0")
    for cam in cameras:
        _add_camera(world,cam,scene_config.get("fovy_deg",50))
    # Fixed review views are for human inspection, never estimator inputs.
    for name,position,target,fovy in [
            ("review_overview",[.9,-.9,.75],[0,0,.30],35),
            ("review_front",[0,-1.45,.30],[0,0,.30],28),
            ("review_neck",[.25,-.25,.625],[0,0,.584],32)]:
        camera=look_at(name,position,target,cameras[0].width,cameras[0].height,fovy)
        _add_camera(world,camera,fovy)
    return ET.tostring(root,encoding="unicode")


def render_scene_file(xml_path,cameras,output_dir):
    model=mujoco.MjModel.from_xml_path(str(xml_path))
    data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    jar_body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"jar")
    gids=np.flatnonzero(model.geom_bodyid==jar_body)
    observations=output_dir/"observations"
    observations.mkdir()
    review=output_dir/"review"
    review.mkdir()
    quality={}
    with mujoco.Renderer(model,cameras[0].height,cameras[0].width) as renderer:
        for camera in cameras:
            renderer.update_scene(data,camera=camera.name)
            rgb=renderer.render().copy()
            cv2.imwrite(str(observations/(camera.name+"_rgb.png")),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            renderer.enable_depth_rendering()
            depth=renderer.render().copy()
            renderer.disable_depth_rendering()
            renderer.enable_segmentation_rendering()
            seg=renderer.render()
            mask=(np.isin(seg[:,:,0],gids)&(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM)))
            renderer.disable_segmentation_rendering()
            np.savez_compressed(observations/(camera.name+"_depth_z_m.npz"),depth_z_m=depth)
            cv2.imwrite(str(observations/(camera.name+"_mask_diagnostic.png")),mask.astype(np.uint8)*255)
            y,x=np.nonzero(mask)
            quality[camera.name]={"jar_pixels":len(y),"jar_bbox_xyxy":([int(x.min()),int(y.min()),int(x.max()),int(y.max())] if len(y) else None),
                                  "depth_on_jar_m":([float(depth[mask].min()),float(depth[mask].max())] if len(y) else None)}
        for name in ("overview","front","neck"):
            renderer.update_scene(data,camera="review_"+name)
            rgb=renderer.render().copy()
            cv2.imwrite(str(review/(name+".png")),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    return quality


def prepare_scene_bundle(config,output_dir):
    """Export static assets, same fixed camera rig across cases, separate target truth."""
    if config.get("schema") != 2:
        raise ValueError("Use preparation schema 2: neck_length_m excludes the separate lip_height_m; legacy schema 1 had different semantics")
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=False)
    write_json(output_dir/"effective_config.json",config)
    cameras=scene_cameras(config["scene"])
    records=[]
    for index,item in enumerate(config["cases"]):
        name=item["name"]
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in name):
            raise ValueError("Case names must be simple lowercase identifiers")
        folder=output_dir/name
        assets=folder/"assets"
        assets.mkdir(parents=True)
        spec=JarSpec(**(config["jar"]|item.get("jar",{})))
        mesh=build_jar_mesh(spec,angular_segments=config.get("angular_segments",160),vertical_samples=config.get("vertical_samples",180))
        write_obj(assets/"jar.obj",mesh,regions=[0,1,3])
        write_obj(assets/"jar_inside.obj",mesh,regions=[2,4])
        cv2.imwrite(str(assets/"ceramic.png"),cv2.cvtColor(ceramic_texture(config.get("seed",20261021)+index),cv2.COLOR_RGB2BGR))
        xml=scene_xml(cameras,config["scene"],item.get("material","mottled_gray"))
        (folder/"scene.xml").write_text(xml)
        quality=render_scene_file(folder/"scene.xml",cameras,folder)
        camera_data={"kind":"true_simulation_camera_parameters_not_virtual_calibration","depth":"float32 optical-axis Z in metres; co-located ideal RGB/depth",
                     "cameras":[{"name":c.name,"width":c.width,"height":c.height,"K":c.K,"T_world_from_camera":c.T_world_from_camera} for c in cameras]}
        write_json(folder/"observations/camera_true_parameters.json",camera_data)
        evaluation=folder/"evaluation_only"
        evaluation.mkdir()
        write_json(evaluation/"target.json",mesh["target"])
        np.savez_compressed(evaluation/"profile.npz",s_m=mesh["profile_s_m"],radius_m=mesh["profile_radius_m"],centerline_m=mesh["centerline_m"])
        vertex=mesh["vertices"]
        record={"name":name,"relative_dir":name,"jar_spec":asdict(spec),"material":item.get("material","mottled_gray"),
                "mesh_vertices":len(vertex),"mesh_triangles":len(mesh["faces"]),"bbox_min_m":vertex.min(axis=0),"bbox_max_m":vertex.max(axis=0),
                "geometry_diagnostics":mesh.get("geometry_diagnostics",{}),
                "geometry_source":"photo-inspired synthetic parameterization; no metric single-view reconstruction",
                "quality":quality}
        write_json(folder/"model_manifest.json",record)
        records.append(record)
    module_dir=Path(__file__).parent
    files=[module_dir/"jar_model.py",module_dir/"geometry.py",Path(__file__)]
    manifest={"stage":"scene_preparation_only_no_localization_or_noise_benchmark", "schema":2,"cases":records,
              "neck_length_definition":"exposed neck below lip, excluding lip_height_m",
              "review_views":"overview/front/neck images are for human inspection only; not measurement-camera observations",
              "world":"metres; fixed jar body base at origin; +Z up; static support top Z=0", "target_truth_location":"each_case/evaluation_only/target.json",
              "limitations":["Approximate user height/mouth diameter; unmeasured body/neck dimensions assumed","Single photograph cannot determine unseen shape or true 3D neck orientation",
                              "Circular sections are an initial hypothesis","Near-min interval uses an explicit synthetic tolerance, not an identified real contact band",
                              "Ideal co-located RGB/depth is not a RealSense active stereo/registration/noise simulation","Geometry and target truth not supplied to any estimator; no estimator invoked"],
              "versions":{"python":platform.python_version(),"mujoco":mujoco.__version__,"numpy":np.__version__,"opencv":cv2.__version__},
              "source_sha256":{p.name:sha256(p.read_bytes()).hexdigest() for p in files}}
    write_json(output_dir/"manifest.json",manifest)
    return manifest
