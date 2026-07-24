# Tie Gun End Effector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a simplified vertical automatic tie-gun end effector to the right-side xArm6 in the MuJoCo production demo scene.

**Architecture:** Extend the existing prefixed xArm generator so it can produce three right-arm variants: bare flange, tie-gun flange, and the existing left vacuum arm. The tie-gun geometry is built from MuJoCo primitives directly inside the right end-effector body, with sites for alignment and later animation. The production scene will include the tie-gun right arm and the inspection script will verify the model is loaded, attached, oriented downward, and free of obvious static-layout errors.

**Tech Stack:** Python ElementTree, MuJoCo MJCF XML, MuJoCo Python loader, NumPy validation.

---

### Task 1: Extend The Prefixed xArm Generator With A Tie-Gun End Effector

**Files:**
- Modify: `mujoco_xarm6/tools/generate_prefixed_xarm_models.py`
- Generate: `mujoco_xarm6/scene/xarm6_right_tie_gun.xml`

- [ ] **Step 1: Replace the boolean `bare` flag with an `end_effector` mode**

  Change `prefix_model` to accept `end_effector: str`, with accepted values:

  ```python
  def prefix_model(
      source_root: ET.Element,
      prefix: str,
      end_effector: str,
      base_pos: str,
      base_euler: str,
  ) -> ET.Element:
      ...
      if end_effector == "bare":
          make_bare_end_effector(root, prefix)
      elif end_effector == "tie_gun":
          make_tie_gun_end_effector(root, prefix)
      elif end_effector == "vacuum":
          pass
      else:
          raise ValueError(f"Unknown end effector mode: {end_effector}")
  ```

- [ ] **Step 2: Add helpers for capsule arc segments**

  Add these functions below `make_bare_end_effector`:

  ```python
  def arc_points(radius: float, z: float, start_deg: float, end_deg: float, segments: int):
      points = []
      for index in range(segments + 1):
          t = index / segments
          angle = math.radians(start_deg + (end_deg - start_deg) * t)
          points.append((radius * math.cos(angle), radius * math.sin(angle), z))
      return points


  def add_capsule_arc(parent: ET.Element, prefix: str, name: str, radius: float, z: float, start_deg: float, end_deg: float, segments: int, tube_radius: float, material: str):
      points = arc_points(radius, z, start_deg, end_deg, segments)
      for index, (a, b) in enumerate(zip(points, points[1:])):
          ET.SubElement(parent, "geom", {
              "name": f"{prefix}{name}_{index}",
              "type": "capsule",
              "fromto": f"{a[0]:.5f} {a[1]:.5f} {a[2]:.5f} {b[0]:.5f} {b[1]:.5f} {b[2]:.5f}",
              "size": f"{tube_radius:.5f}",
              "material": material,
              "mass": "0.01",
              "friction": "0.8 0.8 0.8",
          })
  ```

  Also add `import math` at the top of the file.

- [ ] **Step 3: Implement `make_tie_gun_end_effector`**

  Replace the prefixed `vacuum_end_effector` body with a vertical tie-gun assembly:

  ```python
  def make_tie_gun_end_effector(root: ET.Element, prefix: str):
      target_name = prefix + "vacuum_end_effector"
      for body in root.iter("body"):
          if body.get("name") != target_name:
              continue
          body.set("name", prefix + "tie_gun_end_effector")
          body.attrib["pos"] = "0 0 0.015"
          for child in list(body):
              body.remove(child)

          ET.SubElement(body, "inertial", {
              "pos": "0 0 0.055",
              "mass": "0.45",
              "diaginertia": "0.00035 0.00035 0.00018",
          })
          ET.SubElement(body, "geom", {
              "name": prefix + "tie_gun_connector",
              "type": "cylinder",
              "pos": "0 0 0.012",
              "size": "0.030 0.015",
              "material": prefix + "gray",
              "mass": "0.12",
              "friction": "0.8 0.8 0.8",
          })
          ET.SubElement(body, "geom", {
              "name": prefix + "tie_gun_body",
              "type": "box",
              "pos": "0 0 0.060",
              "size": "0.050 0.035 0.030",
              "material": prefix + "black",
              "mass": "0.20",
              "friction": "0.8 0.8 0.8",
          })
          ET.SubElement(body, "geom", {
              "name": prefix + "tie_gun_air_port",
              "type": "cylinder",
              "pos": "0.055 0 0.072",
              "euler": "0 1.5708 0",
              "size": "0.008 0.028",
              "material": prefix + "gray",
              "mass": "0.03",
              "friction": "0.8 0.8 0.8",
          })

          jaw_z = -0.080
          ring_radius = 0.105
          add_capsule_arc(body, prefix, "tie_gun_left_open_jaw", ring_radius, jaw_z, 20, 160, 7, 0.006, prefix + "black")
          add_capsule_arc(body, prefix, "tie_gun_right_open_jaw", ring_radius, jaw_z, 200, 340, 7, 0.006, prefix + "black")
          add_capsule_arc(body, prefix, "tie_gun_outer_band_preview", ring_radius * 0.92, jaw_z - 0.004, 0, 360, 16, 0.003, prefix + "gray")
          add_capsule_arc(body, prefix, "tie_gun_final_band_preview", 0.060, jaw_z - 0.010, 0, 360, 16, 0.0025, prefix + "black")

          ET.SubElement(body, "site", {
              "name": prefix + "tie_gun_center_site",
              "pos": f"0 0 {jaw_z:.5f}",
              "type": "sphere",
              "size": "0.010",
              "rgba": "0 0.7 1 0.8",
          })
          ET.SubElement(body, "site", {
              "name": prefix + "tie_gun_approach_site",
              "pos": f"0 0 {jaw_z + 0.160:.5f}",
              "type": "sphere",
              "size": "0.008",
              "rgba": "0 1 0.3 0.7",
          })
          ET.SubElement(body, "site", {
              "name": prefix + "tie_gun_neck_target_site",
              "pos": f"0 0 {jaw_z - 0.010:.5f}",
              "type": "sphere",
              "size": "0.008",
              "rgba": "1 0 0 0.7",
          })
          ET.SubElement(body, "site", {
              "name": prefix + "link_tcp",
              "pos": f"0 0 {jaw_z:.5f}",
              "type": "sphere",
              "size": "0.009",
              "rgba": "1 0 0 0.5",
              "group": "1",
          })
          return
      raise RuntimeError(f"Could not find body {target_name}")
  ```

- [ ] **Step 4: Generate all three prefixed models**

  Update `main()` to generate:

  ```python
  write_xml(
      SCENE_DIR / "xarm6_left_vacuum.xml",
      prefix_model(source_root, "left_", end_effector="vacuum", base_pos="-0.12 0.62 0.12", base_euler="0 0 -1.5708"),
  )
  write_xml(
      SCENE_DIR / "xarm6_right_bare.xml",
      prefix_model(source_root, "right_", end_effector="bare", base_pos="-0.12 -0.62 0.12", base_euler="0 0 1.5708"),
  )
  write_xml(
      SCENE_DIR / "xarm6_right_tie_gun.xml",
      prefix_model(source_root, "right_", end_effector="tie_gun", base_pos="-0.12 -0.62 0.12", base_euler="0 0 1.5708"),
  )
  ```

- [ ] **Step 5: Run the generator and verify the new model loads**

  Run:

  ```bash
  conda run -n winejar-mujoco python mujoco_xarm6/tools/generate_prefixed_xarm_models.py
  conda run -n winejar-mujoco python -c "import mujoco; m=mujoco.MjModel.from_xml_path('mujoco_xarm6/scene/xarm6_right_tie_gun.xml'); print(m.nbody, m.ngeom, m.nsite, m.nu)"
  ```

  Expected:

  - `xarm6_right_tie_gun.xml` is written.
  - The MuJoCo loader exits successfully.
  - The printed model has `nu=6`.

### Task 2: Attach The Tie Gun To The Production Demo Scene

**Files:**
- Modify: `mujoco_xarm6/scene/scene_winejar_production_demo.xml`

- [ ] **Step 1: Replace the right-arm include**

  Change:

  ```xml
  <include file="xarm6_right_bare.xml"/>
  ```

  to:

  ```xml
  <include file="xarm6_right_tie_gun.xml"/>
  ```

- [ ] **Step 2: Update the production home keyframe**

  Keep the same `qpos` values, because the tie-gun model does not add joints. The keyframe should still contain:

  ```xml
  <key name="production_home"
       qpos="0 -0.610865 -1.221730 0 1.832596 0 0 -0.610865 -1.221730 0 1.832596 0 -0.75 0.52 0.4815 0.707107 0 0 0.707107 -0.75 0.52 0.4845 0.707107 0 0 0.707107 -0.48 0.52 0.498 1 0 0 0"
       ctrl="0 0 0 0 0 0 0 0 0 0 0 0"/>
  ```

- [ ] **Step 3: Load the production scene**

  Run:

  ```bash
  conda run -n winejar-mujoco python -c "import mujoco; m=mujoco.MjModel.from_xml_path('mujoco_xarm6/scene/scene_winejar_production_demo.xml'); print(m.nbody, m.ngeom, m.nsite, m.ncam, m.nu)"
  ```

  Expected:

  - The scene loads without XML errors.
  - `nu` remains `12`, because both xArm6 models still have six actuators.

### Task 3: Update Production Scene Inspection

**Files:**
- Modify: `mujoco_xarm6/test/inspect_winejar_production_demo_scene.py`

- [ ] **Step 1: Replace the right bare TCP site with tie-gun sites**

  In `SITES`, replace:

  ```python
  "right_link_tcp",
  ```

  with:

  ```python
  "right_tie_gun_center_site",
  "right_tie_gun_approach_site",
  "right_tie_gun_neck_target_site",
  "right_link_tcp",
  ```

- [ ] **Step 2: Add tie-gun body names to `BODIES`**

  Add:

  ```python
  "right_tie_gun_end_effector",
  ```

  The primitive geoms are not bodies, so do not add `right_tie_gun_connector` or `right_tie_gun_body` to `BODIES`.

- [ ] **Step 3: Add validation for tie-gun attachment and downward orientation**

  Add this check near the end of `main()`:

  ```python
  right_tcp = data.site_xpos[model.site("right_link_tcp").id]
  tie_center = data.site_xpos[model.site("right_tie_gun_center_site").id]
  neck_target = data.site_xpos[model.site("right_tie_gun_neck_target_site").id]
  if np.linalg.norm(right_tcp - tie_center) > 1e-6:
      raise AssertionError("right_link_tcp should coincide with right_tie_gun_center_site")
  if neck_target[2] >= tie_center[2]:
      raise AssertionError("Tie-gun neck target should be below the ring center for vertical insertion")
  print("Tie-gun TCP and vertical target orientation OK")
  ```

- [ ] **Step 4: Add validation for ring radius**

  Add this check after the orientation check:

  ```python
  neck_radius = 0.060
  open_radius = 0.105
  if open_radius <= neck_radius:
      raise AssertionError("Tie-gun open ring radius must exceed final neck radius")
  print(f"Tie-gun ring radius OK: open={open_radius:.3f} m, final={neck_radius:.3f} m")
  ```

- [ ] **Step 5: Run the updated inspection**

  Run:

  ```bash
  conda run -n winejar-mujoco python mujoco_xarm6/test/inspect_winejar_production_demo_scene.py
  ```

  Expected:

  - The scene loads.
  - Existing conveyor, material support, freejoint, and workspace checks still pass.
  - New tie-gun checks pass.

### Task 4: Viewer And Syntax Verification

**Files:**
- Check: `mujoco_xarm6/test/view_winejar_scene.py`
- Check: `mujoco_xarm6/tools/generate_prefixed_xarm_models.py`
- Check: `mujoco_xarm6/test/inspect_winejar_production_demo_scene.py`

- [ ] **Step 1: Verify the production home keyframe still works**

  Run:

  ```bash
  conda run -n winejar-mujoco python -c "import mujoco; m=mujoco.MjModel.from_xml_path('mujoco_xarm6/scene/scene_winejar_production_demo.xml'); d=mujoco.MjData(m); kid=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, 'production_home'); assert kid >= 0; mujoco.mj_resetDataKeyframe(m,d,kid); mujoco.mj_forward(m,d); print('right_tie_gun_center_site=', [round(float(x),3) for x in d.site_xpos[m.site('right_tie_gun_center_site').id]])"
  ```

  Expected:

  - The command exits successfully.
  - It prints the right tie-gun center site position.

- [ ] **Step 2: Verify Python files compile**

  Run:

  ```bash
  conda run -n winejar-mujoco python -m py_compile \
    mujoco_xarm6/tools/generate_prefixed_xarm_models.py \
    mujoco_xarm6/test/inspect_winejar_production_demo_scene.py \
    mujoco_xarm6/test/view_winejar_scene.py
  ```

  Expected:

  - The command exits with code `0`.

- [ ] **Step 3: Manual viewer command for the user**

  Provide this command:

  ```bash
  conda activate winejar-mujoco
  python mujoco_xarm6/test/view_winejar_scene.py \
    --scene mujoco_xarm6/scene/scene_winejar_production_demo.xml \
    --key production_home
  ```

  Expected:

  - The right xArm6 has a visible vertical tie-gun head.
  - The ring is below the flange.
  - The jaws are visibly open.
  - The tie-band preview is visible and centered on the ring.
