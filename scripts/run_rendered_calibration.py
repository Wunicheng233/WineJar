#!/usr/bin/env python3
"""Render/detect/calibrate three RGB cameras without feeding true K to solver."""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from neck_localization.geometry import Camera
from neck_localization.scene_preparation import scene_cameras
from neck_localization.rendered_calibration import (BoardSettings, RasterCheckerboard,
    board_pose_plan, T_camera_from_board_plan, detect_checkerboard, calibrate_intrinsics,
    estimate_camera_pose, pose_error, transform_from_rt)


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def run(output, scene_config, seed=2026102551):
    import sapien
    from sapien.sensor import StereoDepthSensorConfig
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    board = BoardSettings()
    plan = board_pose_plan(seed)
    write_json(output/'plan.json', dict(seed=seed, board=asdict(board), frames=plan,
        intrinsics='No true K or board poses enter calibrateCamera',
        world_anchor='Three fixed checkerboard faces on an ideally surveyed common-world fixture; one face per camera',
        scope='RGB K and world RGB pose only; IR K, depth, RGB-IR transform remain preset truth'))
    preset = StereoDepthSensorConfig('D435')
    renderer_K = preset.rgb_intrinsic.copy()
    K_integer = renderer_K.copy(); K_integer[:2, 2] -= .5
    cameras = [Camera(c.name, K_integer.copy(), c.T_world_from_camera.copy(), *preset.rgb_resolution)
               for c in scene_cameras(scene_config)]
    truth = dict(schema_version='winejar-camera-parameters-v1', role='evaluation_only', cameras=[])
    estimated = dict(schema_version='winejar-camera-parameters-v1',
        calibration_source='image-detected checkerboard; known ideal world fixture; zero distortion model', cameras=[])
    report = dict(cameras={}, detections=[], limitations=[
        'Perfect planar board, exact square size and fixture poses; no board manufacturing/metrology uncertainty',
        'Ideal pinhole raster RGB, no extra distortion/noise; camera errors arise from rendered image sampling and corner extraction',
        'RGB calibration only; no IR calibration, RGB-IR alignment calibration or robot/world calibration',
        'Reprojection residual is not absolute localization accuracy'])
    for camera in cameras:
        start = time.perf_counter()
        renderer = RasterCheckerboard(camera, renderer_K, board)
        detections, board_poses, observations = {}, {}, []
        for item in plan:
            folder = output/camera.name/item['frame_id']; folder.mkdir(parents=True)
            Twb = camera.T_world_from_camera@T_camera_from_board_plan(item, board)
            image = renderer.capture(Twb)
            cv2.imwrite(str(folder/'rgb.png'), image[..., ::-1])
            row = dict(camera=camera.name, **item)
            # True board poses/corner projections are written separately and
            # used below ONLY after actual image detection has returned.
            truth_pixels = camera.project(board.object_points()@Twb[:3, :3].T+Twb[:3, 3])
            write_json(folder/'evaluation_only.json', dict(T_world_from_board=Twb.tolist(), ideal_corner_pixels=truth_pixels.tolist()))
            board_poses[item['frame_id']] = Twb
            try:
                corners, diagnostic = detect_checkerboard(image, board)
                detections[item['frame_id']] = corners
                np.save(folder/'detected_corners.npy', corners)
                overlay = image.copy()
                cv2.drawChessboardCorners(overlay, (board.columns, board.rows), corners, True)
                cv2.imwrite(str(folder/'detected_overlay.png'), overlay[..., ::-1])
                difference = corners.reshape(-1, 2)-truth_pixels
                row.update(found=True, corner_error_rms_px=float(np.sqrt(np.mean(np.sum(difference*difference, axis=1)))),
                           corner_signed_mean_xy_px=difference.mean(axis=0).tolist(), detection=diagnostic)
                if item['split'] == 'train': observations.append(corners)
            except ValueError as error:
                row.update(found=False, reason=str(error))
            write_json(folder/'detection.json', row)
            report['detections'].append(row)
            print(camera.name, item['frame_id'], 'detected', row['found'],
                  'corner_rms', row.get('corner_error_rms_px'), flush=True)
        if len(observations) < 15 or sum(r['found'] and r['split']=='validation' for r in report['detections'] if r['camera']==camera.name) < 5:
            raise RuntimeError('Not all predeclared 15 train + 5 validation boards detected; retain failure evidence and diagnose')
        solution = calibrate_intrinsics(observations, (camera.width, camera.height), board)
        anchor_id = 'fixed_world_fixture'
        Test, anchor_diagnostic = estimate_camera_pose(detections[anchor_id], solution['K'], solution['distortion'], board_poses[anchor_id], board)
        record = dict(name=camera.name, K=solution['K'].tolist(), distortion=solution['distortion'].ravel().tolist(),
                      T_world_from_camera=Test.tolist(), width=camera.width, height=camera.height,
                      extrinsic_reference='known world pose of fixed surveyed checkerboard face')
        estimated['cameras'].append(record)
        truth['cameras'].append(dict(name=camera.name, K=camera.K.tolist(), distortion=[0.]*5,
            T_world_from_camera=camera.T_world_from_camera.tolist(), width=camera.width, height=camera.height))
        metrics = dict(train_reprojection_rms_px=solution['rms_px'], intrinsics_delta_px=(solution['K']-camera.K).tolist(),
                       world_anchor_pose_error=pose_error(Test, camera.T_world_from_camera), world_anchor=anchor_diagnostic,
                       flags=solution['flags'], distortion_model=solution['distortion_model'], validation=[])
        for item in plan:
            if item['split'] != 'validation': continue
            identifier = item['frame_id']; detected = detections[identifier]
            Tval, info = estimate_camera_pose(detected, solution['K'], solution['distortion'], board_poses[identifier], board)
            # Independent held-out board projection under the FIXED final world
            # camera pose; not another fitted per-view residual.
            Tcb = np.linalg.inv(Test)@board_poses[identifier]
            rv = cv2.Rodrigues(Tcb[:3, :3])[0]
            predicted = cv2.projectPoints(board.object_points(), rv, Tcb[:3, 3], solution['K'], solution['distortion'])[0]
            difference = predicted.reshape(-1, 2)-detected.reshape(-1, 2)
            metrics['validation'].append(dict(frame_id=identifier,
                fixed_world_reprojection_rms_px=float(np.sqrt(np.mean(np.sum(difference*difference, axis=1)))),
                per_view_pnp_pose_error=pose_error(Tval, camera.T_world_from_camera), **info))
        metrics['elapsed_seconds'] = time.perf_counter()-start
        report['cameras'][camera.name] = metrics
        write_json(output/'estimated_camera_parameters.json', estimated)
        write_json(output/'evaluation_only/camera_true_parameters.json', truth)
        write_json(output/'calibration_report.json', report)
        del renderer
        print(camera.name, 'K', solution['K'], 'anchor_error', metrics['world_anchor_pose_error'], flush=True)
    write_json(output/'provenance.json', dict(opencv_version=cv2.__version__, sapien_version=sapien.__version__,
        sources={name:sha256((ROOT/name).read_bytes()).hexdigest() for name in
                 ['neck_localization/rendered_calibration.py', 'scripts/run_rendered_calibration.py']},
        input_scene_sha256=sha256(json.dumps(scene_config, sort_keys=True).encode()).hexdigest(),
        pixel_convention='SAPIEN corner-origin K; OpenCV integer centres use cx,cy minus .5',
        corner_order='Only observed red/green printed markers, known print geometry and detected corners; no truth sorting',
        no_true_intrinsic_initialization=True, fixture_world_poses_are_ideal_known_reference=True))
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scene-config', type=Path, default=ROOT/'experiments/neck_localization/configs/preparation_v2.json')
    p.add_argument('--seed', type=int, default=2026102551)
    args=p.parse_args()
    config=json.loads(args.scene_config.read_text())
    run(args.output, config['scene'], args.seed)
