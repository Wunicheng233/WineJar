#!/usr/bin/env python3
"""Export existing native diagnostic results for the MuJoCo viewer; no inference."""
import argparse
import copy
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np

from neck_localization.geometry import Camera


def numbers(values):
    return " ".join(f"{x:.17g}" for x in np.asarray(values).ravel())


def validate_scene(path, cameras):
    """Check native pose/resolution and effective MuJoCo projection on CPU."""
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    checks = []
    for camera in cameras:
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera.name)
        np.testing.assert_allclose(data.cam_xpos[index], camera.T_world_from_camera[:3, 3], atol=1e-8)
        np.testing.assert_allclose(data.cam_xmat[index].reshape(3, 3) @ np.diag([1., -1., -1.]),
                                   camera.T_world_from_camera[:3, :3], atol=1e-8)
        np.testing.assert_array_equal(model.cam_resolution[index], [camera.width, camera.height])
        scene = mujoco.MjvScene(model, maxgeom=20)
        view = mujoco.MjvCamera()
        view.type, view.fixedcamid = mujoco.mjtCamera.mjCAMERA_FIXED, index
        mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, view, mujoco.mjtCatBit.mjCAT_ALL, scene)
        gl = mujoco.mjv_averageCamera(scene.camera[0], scene.camera[1])
        forward = np.asarray(gl.forward, float)
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, gl.up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        pixels = np.array([[20., 20.], [camera.width/2, camera.height/2], [camera.width-20., camera.height-20.]])
        points = camera.unproject(pixels, [1., 1.2, .8])
        delta = points-gl.pos
        z, x, y = delta @ forward, delta @ right, delta @ up
        sy = float(gl.frustum_top-gl.frustum_bottom)
        sx = sy*camera.width/camera.height
        left = gl.frustum_center-sx/2
        projected = np.column_stack((camera.width*(gl.frustum_near*x/z-left)/sx-.5,
                                    camera.height*(1-(gl.frustum_near*y/z-gl.frustum_bottom)/sy)-.5))
        error = float(np.max(np.abs(projected-pixels)))
        np.testing.assert_allclose(projected, pixels, atol=.00015)
        checks.append(dict(camera=camera.name, pose_assertions_passed=True,
                           resolution=[camera.width, camera.height], projection_max_abs_px=error,
                           computed_fovy_deg=float(model.cam_fovy[index])))
        del scene
    return checks


def export(source, output):
    provenance = json.loads((source/"provenance.json").read_text())
    geometry_source = Path(provenance["input"])/"geometry"
    original_rows = json.loads(gzip.decompress((source/"records.json.gz").read_bytes()))
    output.mkdir(parents=True, exist_ok=False)
    checks, rows = [], []
    for original in original_rows:
        row = copy.deepcopy(original)
        row["source_evidence"] = row["evidence"]
        row["evidence"] = "rgb"  # viewer category: observed RGB supplied mask
        row["viewer_input_semantics"] = "RGB→IR mapped color, not original camera RGB; native IR Z"
        row["diagnostics"]["foreground"] = dict(method="saved_original_RGB_to_native_mapped_mask",
                                                  source_evidence=original["evidence"], not_ground_truth=True)
        mode, case = row["condition_name"], row["case"]
        folder = output/mode/"scenes"/case
        folder.mkdir(parents=True)
        for child in ["observations", "foreground_masks"]:
            (folder/child).symlink_to((source/mode/"scenes"/case/child).resolve(), target_is_directory=True)
        geometry = geometry_source/case
        for child in ["assets", "evaluation_only", "model_manifest.json"]:
            (folder/child).symlink_to((geometry/child).resolve(), target_is_directory=(geometry/child).is_dir())
        metadata = json.loads((folder/"observations/camera_true_parameters.json").read_text())
        cameras = [Camera(item["name"], np.asarray(item["K"]), np.asarray(item["T_world_from_camera"]),
                          item["width"], item["height"]) for item in metadata["cameras"]]
        tree = ET.parse(geometry/"scene.xml")
        tree.getroot().find("visual/global").set("offwidth", str(cameras[0].width))
        tree.getroot().find("visual/global").set("offheight", str(cameras[0].height))
        for camera in cameras:
            node = tree.getroot().find(f"worldbody/camera[@name='{camera.name}']")
            rotation = camera.T_world_from_camera[:3, :3] @ np.diag([1., -1., -1.])
            node.set("pos", numbers(camera.T_world_from_camera[:3, 3]))
            node.set("xyaxes", numbers(rotation[:, :2].T))
            # Installed MuJoCo 3.8 rejects simultaneous fovy and sensorsize;
            # exact pixel intrinsics below derive the matching fovy instead.
            node.attrib.pop("fovy", None)
            node.set("resolution", f"{camera.width} {camera.height}")
            node.set("sensorsize", f"{camera.width} {camera.height}")
            node.set("focalpixel", numbers([camera.K[0, 0], camera.K[1, 1]]))
            # MuJoCo physical principal offset has the opposite sign to the
            # desired integer-centre image-coordinate displacement. Check the
            # effective frustum against independent Camera.unproject below.
            node.set("principalpixel", numbers([(camera.width-1)/2-camera.K[0, 2],
                                                 (camera.height-1)/2-camera.K[1, 2]]))
        tree.write(folder/"scene.xml", encoding="unicode")
        validation = validate_scene(folder/"scene.xml", cameras)
        checks.append(dict(case=case, mode=mode, cameras=validation))
        rows.append(row)
    for mode in sorted({r["condition_name"] for r in rows}):
        selected = [r for r in rows if r["condition_name"] == mode]
        (output/mode/"records.json.gz").write_bytes(gzip.compress(json.dumps(selected).encode(), mtime=0))
        (output/mode/"settings.json").write_text(json.dumps(dict(preprocessing=dict(kind="native_ir_color_mapping", condition=mode),
            camera_parameter_mode="true native IR K/T", color_semantics="RGB→IR取色（非相机原始RGB）", depth_semantics="native IR optical-axis Z"), indent=2)+"\n")
    (output/"camera_validation.json").write_text(json.dumps(checks, indent=2)+"\n")
    (output/"provenance.json").write_text(json.dumps(dict(source=str(source.resolve()),
        source_records_sha256=sha256((source/"records.json.gz").read_bytes()).hexdigest(),
        export_only_no_inference=True, original_records_unchanged=True,
        source_evidence_preserved=True, native_intrinsics_in_xml=True,
        sources={str(Path(__file__).relative_to(ROOT)):sha256(Path(__file__).read_bytes()).hexdigest()}), indent=2)+"\n")
    print("exported", len(rows), "records and", sum(len(x["cameras"]) for x in checks), "camera checks", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.output)
