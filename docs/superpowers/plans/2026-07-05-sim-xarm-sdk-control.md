# Sim xArm SDK Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drive the MuJoCo xArm scene through a Python SDK-like adapter instead of direct object teleportation.

**Architecture:** Add a `SimXArmAPI` layer exposing familiar `get_position`, `set_position`, and `set_cgpio_digital` calls. The scene makes leaves free bodies and uses inactive weld constraints that are activated only when the simulated vacuum cup is close enough.

**Tech Stack:** MuJoCo Python, NumPy, unittest, MJCF equality weld constraints.

---

### Task 1: Make Leaves Movable and Attachable

**Files:**
- Modify: `mujoco_xarm6/scene/scene_winejar_vacuum.xml`

- [x] Add `freejoint` to both bamboo leaves.
- [x] Add inactive weld constraints between the vacuum end effector and each leaf.

### Task 2: SDK-Like Simulation Adapter

**Files:**
- Create: `mujoco_xarm6/sim/xarm_sim_api.py`
- Create: `mujoco_xarm6/test/test_xarm_sim_api.py`

- [x] Implement site pose reading.
- [x] Implement Jacobian-based IK to solve TCP target poses.
- [x] Implement velocity-actuator stepping toward IK joint targets.
- [x] Implement DO9 vacuum on/off with distance-gated weld attachment.

### Task 3: Leaf Placement Demo Script

**Files:**
- Create: `mujoco_xarm6/scripts/01_vacuum_leaf_place_sequence.py`

- [x] Load the vacuum scene.
- [x] Move through approach, pick, transfer, place, release waypoints using `SimXArmAPI`.
- [x] Save final diagnostics for leaf/target distances.
