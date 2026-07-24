# Winejar Vacuum Simulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a focused MuJoCo scene for the vacuum-suction leaf placement task.

**Architecture:** Create a vacuum-specific xArm include and a new scene that keeps only the wine jar, ceramic disc, two stacked bamboo leaves, and one overhead global camera. Add a validation script that loads the XML and checks required bodies/sites/camera.

**Tech Stack:** MuJoCo MJCF XML, Python MuJoCo loader checks.

---

### Task 1: Vacuum xArm Include

**Files:**
- Create: `mujoco_xarm6/scene/xarm6_vacuum.xml`

- [x] Keep the six xArm joints and velocity actuators.
- [x] Replace the gripper subtree with a compact vacuum cup and `vacuum_suction_site`.

### Task 2: Focused Vacuum Scene

**Files:**
- Create: `mujoco_xarm6/scene/scene_winejar_vacuum.xml`

- [x] Include `xarm6_vacuum.xml`.
- [x] Place ceramic disc on the jar mouth.
- [x] Place two bamboo leaves stacked on the material table.
- [x] Add target sites for crossed placement at 0 and 90 degrees.
- [x] Add one overhead global camera.

### Task 3: Validation

**Files:**
- Create: `mujoco_xarm6/test/inspect_winejar_vacuum_scene.py`

- [x] Load the scene with MuJoCo.
- [x] Verify required bodies, sites, and camera names.
- [x] Print key world positions for user inspection.
