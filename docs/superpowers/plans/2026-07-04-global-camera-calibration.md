# Global Camera Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal eye-to-hand calibration path that estimates `base_T_camera` for a fixed RealSense camera and xArm base.

**Architecture:** Keep hardware sampling and numeric solving separate. Sampling records one `camera_point_m <-> base_point_m` pair per run; solving loads all samples and estimates a rigid transform with SVD/Kabsch.

**Tech Stack:** Python 3.11, NumPy, OpenCV, pyrealsense2, xArm Python SDK, unittest.

---

### Task 1: Rigid Transform Solver

**Files:**
- Create: `xarm_sdk_control/calibration_utils.py`
- Create: `xarm_sdk_control/tests/test_global_camera_calibration.py`

- [x] Add a tested `solve_rigid_transform(source_points, target_points)` function mapping camera points to base points.
- [x] Validate at least 3 non-collinear points and reject degenerate input.

### Task 2: One-Sample Collection Script

**Files:**
- Create: `xarm_sdk_control/scripts/11_collect_global_camera_sample.py`

- [x] Capture synchronized RealSense color/depth.
- [x] Accept `--pixel u,v` to deproject one selected image point into `camera_point_m`.
- [x] Read current xArm TCP pose and use TCP xyz as `base_point_m`.
- [x] Save raw images, annotated image, and `sample.json`.

### Task 3: Calibration Solve Script

**Files:**
- Create: `xarm_sdk_control/scripts/12_solve_global_camera_calibration.py`

- [x] Load timestamped global-camera samples.
- [x] Solve `base_T_camera`, report mean/max residual error, and save `global_camera_<timestamp>.json`.

### Task 4: Verification

**Files:**
- Modify: none beyond files above.

- [x] Run unit tests.
- [x] Run Python syntax checks for new scripts.
