"""Parameterized, static MuJoCo metrology scenes, with separate evaluation truth.

Rendering is geometric, not a RealSense sensor simulator. The jar groove has an
artificial contrast band: evidence extraction here does not validate bare clay.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
from dataclasses import dataclass
from xml.etree.ElementTree import Element, SubElement, tostring
import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import Camera, Observation, circle_points, look_at


def _s(values):
    return " ".join(f"{v:.10g}" for v in np.asarray(values).ravel())


def _surface_mesh(asset, name, profile, segments, ellipticity=0):
    """Closed surface of revolution; profile traverses its cross-section boundary."""
    vertices, faces = [], []
    for radius, z in profile:
        for j in range(segments):
            t = 2*np.pi*j/segments
            rr = radius * (1 + ellipticity*np.cos(2*t))
            vertices.append((rr*np.cos(t), rr*np.sin(t), z))
    for k in range(len(profile)):
        kk = (k+1) % len(profile)
        for j in range(segments):
            a, b = k*segments+j, k*segments+(j+1) % segments
            c, d = kk*segments+j, kk*segments+(j+1) % segments
            faces.extend([(a,b,c),(b,d,c)])
    SubElement(asset, "mesh", name=name, vertex=_s(vertices), face=_s(np.array(faces, int)).replace(".0", ""))


@dataclass
class Rendered:
    cameras: list[Camera]
    rgb: list[np.ndarray]
    depth: list[np.ndarray]
    masks: list[np.ndarray]
    truth: dict
    xml: str


def cameras_for(spec):
    w, h = spec.get("resolution", [640,480])
    layout = spec.get("layout", "wide")
    angles = [(0,65),(120,20),(240,20)]
    if layout == "narrow":
        angles = [(0,40),(8,35),(16,30)]
    elif layout == "balanced":
        angles = [(0,35),(120,35),(240,35)]
    result = []
    # Fixed rig looks at a nominal workspace, never retargets to exact scene center.
    target = np.array([0.,0.,.40])
    for i,(az,el) in enumerate(angles):
        az, el = np.deg2rad([az,el])
        p = target + spec.get("distance_m", .72) * np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
        result.append(look_at(f"cam{i}",p,target,w,h,spec.get("fovy_deg",50)))
    return result


def render(spec):
    cameras = cameras_for(spec)
    center = np.array(spec["center_m"],float)
    rot = Rotation.from_euler("xyz", spec["rotation_deg"],degrees=True)
    r = spec["radius_m"]
    segments = spec.get("segments",256)
    style = spec.get("geometry", "coupon")
    root = Element("mujoco", model="offline_neck_metrology")
    SubElement(root,"compiler", angle="radian")
    vis = SubElement(root,"visual")
    SubElement(vis,"global", offwidth=str(cameras[0].width),offheight=str(cameras[0].height))
    SubElement(vis,"quality", offsamples=str(spec.get("offsamples",0)), shadowsize="1024")
    SubElement(vis,"map", znear="0.01",zfar="5")
    SubElement(root,"statistic",extent="1")
    asset=SubElement(root,"asset")
    world=SubElement(root,"worldbody")
    SubElement(world,"light",pos="0 -1 2",dir="0 0 -1",diffuse=_s([spec.get("light",.8)]*3),ambient=".25 .25 .25",castshadow="false")
    SubElement(world,"geom",type="plane",size="2 2 .01",rgba=".07 .08 .09 1",contype="0",conaffinity="0")
    for cam in cameras:
        # OpenCV camera -> MuJoCo/OpenGL: diag(1,-1,-1).
        glrot=cam.T_world_from_camera[:3,:3]@np.diag([1,-1,-1])
        SubElement(world,"camera",name=cam.name,pos=_s(cam.T_world_from_camera[:3,3]),xyaxes=_s(glrot[:,:2].T),fovy=str(spec.get("fovy_deg",50)))
    xyzw=rot.as_quat()
    body=SubElement(world,"body",name="object",pos=_s(center),quat=_s(np.r_[xyzw[3],xyzw[:3]]))
    eps=spec.get("ellipticity",0.)
    if style == "coupon":
        # True target is the outer upper boundary at z=0. Finite rim thickness is explicit.
        profile=[(r,0),(r*.80,0),(r*.80,-.0002),(r,-.0002)]
        _surface_mesh(asset,"target_mesh",profile,segments,eps)
    elif style == "groove":
        # Unique minimum-radius cross section z=0; the thin light band is artificial.
        band_half=.0007
        band_dr=.004*(band_half/.007)**2
        profile=[(r+band_dr,-band_half),(r,0),(r+band_dr,band_half),(r-.0003,band_half),(r-.0003,-band_half)]
        _surface_mesh(asset,"target_mesh",profile,segments,eps)
        lower=[(r+.075,-.18),(r+.075,-.10),(r+.035,-.026),(r+.004,-.007),(r+band_dr,-band_half),(r-.001,-band_half),(r-.001,-.18)]
        upper=[(r+band_dr,band_half),(r+.004,.007),(r+.013,.014),(r+.013,.025),(r-.006,.025),(r-.006,band_half)]
        for name,prof in [("jar_lower",lower),("jar_lip",upper)]:
            _surface_mesh(asset,name,prof,segments,eps)
            SubElement(body,"geom",name=name,type="mesh",mesh=name,rgba=".30 .11 .035 1",contype="0",conaffinity="0")
    else:
        raise ValueError(f"Unknown geometry {style}")
    band_color=".85 .85 .85 1" if spec.get("contrast_band",True) else ".30 .11 .035 1"
    SubElement(body,"geom",name="target_band",type="mesh",mesh="target_mesh",rgba=band_color,contype="0",conaffinity="0")
    if spec.get("occlusion",False):
        SubElement(body,"geom",name="occluder",type="box",pos=_s([r*.65,0,.012]),size=_s([r*.40,r*1.3,.035]),rgba=".12 .16 .20 1",contype="0",conaffinity="0")
    xml=tostring(root,encoding="unicode")
    model=mujoco.MjModel.from_xml_string(xml)
    data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    rgb,depth,masks=[],[],[]
    gid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,"target_band")
    with mujoco.Renderer(model,cameras[0].height,cameras[0].width) as renderer:
        for cam in cameras:
            renderer.update_scene(data,camera=cam.name)
            rgb.append(renderer.render().copy())
            renderer.enable_depth_rendering()
            depth.append(renderer.render().copy())
            renderer.disable_depth_rendering()
            renderer.enable_segmentation_rendering()
            seg=renderer.render()
            masks.append(((seg[:,:,0]==gid)&(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))).astype(np.uint8)*255)
            renderer.disable_segmentation_rendering()
    truth={"center_m":center.tolist(),"normal":rot.apply([0,0,1]).tolist(),"radius_m":r,
           "target_definition":"outer_upper_coupon_boundary" if style=="coupon" else "minimum_radius_groove_cross_section",
           "geometry":style,"ellipticity":eps,"contrast_band_artificial":style=="groove"}
    return Rendered(cameras,rgb,depth,masks,truth,xml)


def _skeleton(mask):
    img=(mask>0).astype(np.uint8)*255
    skeleton=np.zeros_like(img)
    kernel=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    while np.any(img):
        eroded=cv2.erode(img,kernel)
        skeleton|=img-cv2.dilate(eroded,kernel)
        img=eroded
    return skeleton


def extract_observations(rendered, style, evidence="rgb"):
    """Extract only from arrays and a declared appearance/shape prior; no target pose.

    rgb uses a neutral bright appearance prior. mask_diagnostic uses GT geom IDs.
    Projected exact edges are generated separately as an explicit diagnostic.
    """
    if evidence not in ("rgb","mask_diagnostic"):
        raise ValueError("Unknown image evidence")
    obs=[]
    for camera,rgb,depth,gtmask in zip(rendered.cameras,rendered.rgb,rendered.depth,rendered.masks):
        if evidence=="mask_diagnostic":
            mask=gtmask.copy()
        else:
            hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
            # Frozen on development scenes: hue is undefined for neutral grey;
            # use low saturation and modest intensity, with a dark background.
            mask=cv2.inRange(hsv,np.array([0,0,35],np.uint8),np.array([179,55,255],np.uint8))
        n,labels,stats,_=cv2.connectedComponentsWithStats(mask)
        if n<2:
            continue
        # Keep sufficiently sized components; partial arc patches are legitimate.
        keep=[]
        for label in range(1,n):
            x,y,w,h,area=stats[label]
            # The declared object is fully inside the image; exclude neutral
            # background components connected to the frame, without a GT ROI.
            if area>=8 and x>0 and y>0 and x+w<camera.width and y+h<camera.height:
                keep.append(label)
        mask=np.isin(labels,keep).astype(np.uint8)*255
        if style=="coupon":
            contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue
            pixels=max(contours,key=cv2.contourArea).reshape(-1,2)
        else:
            y,x=np.nonzero(_skeleton(mask))
            pixels=np.c_[x,y]
        if len(pixels)<12:
            continue
        pixels=pixels[::max(1,len(pixels)//240)]
        z=depth[pixels[:,1],pixels[:,0]]
        original_pixels=pixels.copy()
        if style=="coupon":
            # Integer contours mark the final foreground pixel centre. Approximate
            # the continuous boundary halfway to the background (not subpixel truth).
            smooth=cv2.GaussianBlur((mask>0).astype(np.float32),(5,5),1)
            dx=cv2.Sobel(smooth,cv2.CV_32F,1,0,ksize=3)
            dy=cv2.Sobel(smooth,cv2.CV_32F,0,1,ksize=3)
            gradient=np.c_[dx[pixels[:,1],pixels[:,0]],dy[pixels[:,1],pixels[:,0]]]
            gradient/=np.maximum(np.linalg.norm(gradient,axis=1,keepdims=True),1e-9)
            pixels=pixels.astype(float)-.5*gradient
        valid=np.isfinite(z)&(z>0)&(z<4)
        pixels,z=pixels[valid].astype(float),z[valid]
        if len(pixels)>=12:
            # Raw depth belongs to the original pixel centre, not a fabricated
            # depth value at the corrected boundary. The plane-ray variant can
            # explicitly intersect the refined RGB ray with the observed plane.
            obs.append(Observation(camera,pixels,camera.unproject(original_pixels[valid],z)))
    return obs


def exact_edge_diagnostic(rendered, count=120):
    """Full exact projected ring, including invisible parts: solver diagnostic ONLY."""
    gt=rendered.truth
    points=circle_points(gt["center_m"],gt["normal"],gt["radius_m"],count)
    return [Observation(cam,cam.project(points),points.copy()) for cam in rendered.cameras]
