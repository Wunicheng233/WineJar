# Handeye Sample Collection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a safe sample collection script for eye-in-hand calibration using the xArm TCP pose and RealSense chessboard observations.

**Architecture:** The script performs one-shot sampling only. It connects to xArm in read-only style to query TCP pose, starts RealSense RGB-D streams, detects an 8x11 inner-corner chessboard, estimates target pose with OpenCV `solvePnP`, and saves images plus JSON metadata. It never enables motion or sends movement commands.

**Tech Stack:** Python 3.11, xArm Python SDK, pyrealsense2, OpenCV, NumPy, YAML.

---

### Task 1: Sample Collection Script

**Files:**
- Create: `xarm_sdk_control/scripts/06_collect_handeye_sample.py`

- [x] **Step 1: Create a one-shot handeye sample script**

The script must:
- Load robot IP from `xarm_sdk_control/config/robot.yaml`.
- Connect to xArm and call only pose/status getters.
- Capture one aligned RealSense color/depth frame.
- Detect chessboard inner corners with pattern `(8, 11)`.
- Use `solvePnP` with square size `0.020 m`.
- Save `color.png`, `depth_raw.png`, `annotated.png`, and `sample.json`.
- Print explicit confirmation that no robot motion is commanded.

- [x] **Step 2: Run script once in the lab**

Run:

```bash
conda run -n winejar-xarm-sdk python xarm_sdk_control/scripts/06_collect_handeye_sample.py
```

Expected:
- RealSense camera is detected.
- xArm TCP pose is read.
- Chessboard corners are detected.
- Sample directory is created under `xarm_sdk_control/data/handeye_samples/`.

- [x] **Step 3: Stop for user review**

Report:
- Whether detection succeeded.
- The detected chessboard pattern.
- The current TCP pose.
- The saved sample path.
