# Compliant Bamboo Leaves Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the two rigid staged bamboo leaves in the production demo with segmented compliant approximations that can bend while preserving existing pick-site and suction-weld names.

**Architecture:** Generate a dedicated MJCF include for the staged bamboo leaves instead of hand-writing all segments in the main scene. Each leaf keeps a stable root body name and uses nested segment bodies with hinge joints, tapered overlapping box geoms, and a middle pick site. The production scene includes this generated file, while inspection scripts verify segment count, support height, named sites, and basic non-rigid behavior.

**Tech Stack:** Python ElementTree, MuJoCo MJCF XML, MuJoCo Python loader, NumPy validation.

---

### Task 1: Create A Generator For Segmented Bamboo Leaves

**Files:**
- Create: `mujoco_xarm6/tools/generate_compliant_material_models.py`
- Generate: `mujoco_xarm6/scene/compliant_bamboo_leaf_pair.xml`

- [ ] **Step 1: Create the generator file with XML helpers**

  Create `mujoco_xarm6/tools/generate_compliant_material_models.py` with:

  ```python
  #!/usr/bin/env python3
  from __future__ import annotations

  import math
  import xml.etree.ElementTree as ET
  from pathlib import Path


  ROOT = Path(__file__).resolve().parents[1]
  SCENE_DIR = ROOT / "scene"
  OUT_PATH = SCENE_DIR / "compliant_bamboo_leaf_pair.xml"

  SEGMENT_COUNT = 11
  LEAF_LENGTH = 0.42
  LEAF_MAX_WIDTH = 0.125
  LEAF_THICKNESS = 0.003
  LEAF_MASS = 0.010


  def indent(elem: ET.Element, level: int = 0):
      space = "\n" + level * "  "
      if len(elem):
          if not elem.text or not elem.text.strip():
              elem.text = space + "  "
          for child in elem:
              indent(child, level + 1)
          if not child.tail or not child.tail.strip():
              child.tail = space
      if level and (not elem.tail or not elem.tail.strip()):
          elem.tail = space


  def add_geom(body: ET.Element, attrs: dict[str, str]):
      ET.SubElement(body, "geom", attrs)
  ```

- [ ] **Step 2: Add width profile and leaf generation helpers**

  Append:

  ```python
  def segment_width(index: int) -> float:
      if SEGMENT_COUNT == 1:
          return LEAF_MAX_WIDTH
      center = (SEGMENT_COUNT - 1) / 2.0
      normalized = abs(index - center) / center
      return LEAF_MAX_WIDTH * (0.24 + 0.76 * (1.0 - normalized ** 1.7))


  def add_segment_body(parent: ET.Element, leaf_name: str, index: int, material: str, segment_length: float):
      x = -LEAF_LENGTH / 2.0 + segment_length * (index + 0.5)
      width = segment_width(index)
      body = ET.SubElement(parent, "body", {
          "name": f"{leaf_name}_seg_{index:02d}",
          "pos": f"{x:.5f} 0 0",
      })
      if index > 0:
          ET.SubElement(body, "joint", {
              "name": f"{leaf_name}_bend_{index:02d}",
              "type": "hinge",
              "axis": "0 1 0",
              "pos": f"{-segment_length / 2.0:.5f} 0 0",
              "range": "-0.70 0.70",
              "damping": "0.020",
              "stiffness": "0.080",
              "armature": "0.0005",
              "limited": "true",
          })
      add_geom(body, {
          "name": f"{leaf_name}_seg_{index:02d}_geom",
          "type": "box",
          "size": f"{segment_length * 0.56:.5f} {width / 2.0:.5f} {LEAF_THICKNESS / 2.0:.5f}",
          "material": material,
          "mass": f"{LEAF_MASS / SEGMENT_COUNT:.6f}",
          "friction": "1 1 1",
      })
      if index == SEGMENT_COUNT // 2:
          ET.SubElement(body, "site", {
              "name": f"{leaf_name}_pick_site",
              "pos": f"0 0 {LEAF_THICKNESS / 2.0 + 0.003:.5f}",
              "type": "sphere",
              "size": "0.007",
              "rgba": "0 1 0 0.7",
          })
          ET.SubElement(body, "site", {
              "name": f"{leaf_name}_center_site",
              "pos": f"0 0 {LEAF_THICKNESS / 2.0 + 0.002:.5f}",
              "type": "sphere",
              "size": "0.006",
              "rgba": "1 0 0 0.6",
          })
      return body


  def add_leaf(worldbody: ET.Element, leaf_name: str, pos: str, material: str):
      segment_length = LEAF_LENGTH / SEGMENT_COUNT
      root = ET.SubElement(worldbody, "body", {
          "name": leaf_name,
          "pos": pos,
          "euler": "0 0 1.5708",
      })
      ET.SubElement(root, "freejoint", {"name": f"{leaf_name}_freejoint"})
      for index in range(SEGMENT_COUNT):
          add_segment_body(root, leaf_name, index, material, segment_length)
      return root
  ```

- [ ] **Step 3: Add `main()` to write the include file**

  Append:

  ```python
  def main():
      root = ET.Element("mujoco", {"model": "compliant bamboo leaf pair"})
      asset = ET.SubElement(root, "asset")
      ET.SubElement(asset, "material", {
          "name": "compliant_bamboo_mat_bottom",
          "rgba": "0.26 0.48 0.18 1",
          "specular": "0.08",
          "shininess": "0.12",
      })
      ET.SubElement(asset, "material", {
          "name": "compliant_bamboo_mat_top",
          "rgba": "0.34 0.58 0.22 1",
          "specular": "0.08",
          "shininess": "0.12",
      })
      worldbody = ET.SubElement(root, "worldbody")
      add_leaf(worldbody, "staged_bamboo_leaf_bottom", "-0.75 0.52 0.4815", "compliant_bamboo_mat_bottom")
      add_leaf(worldbody, "staged_bamboo_leaf_top", "-0.75 0.52 0.4845", "compliant_bamboo_mat_top")
      indent(root)
      OUT_PATH.write_text(
          "<?xml version='1.0' encoding='utf-8'?>\n" + ET.tostring(root, encoding="unicode"),
          encoding="utf-8",
      )
      print(f"Wrote {OUT_PATH}")


  if __name__ == "__main__":
      main()
  ```

- [ ] **Step 4: Generate the include and load it alone**

  Run:

  ```bash
  conda run -n winejar-mujoco python mujoco_xarm6/tools/generate_compliant_material_models.py
  conda run -n winejar-mujoco python -c "import mujoco; m=mujoco.MjModel.from_xml_path('mujoco_xarm6/scene/compliant_bamboo_leaf_pair.xml'); print(m.nbody, m.njnt, m.ngeom, m.nsite)"
  ```

  Expected:

  - `compliant_bamboo_leaf_pair.xml` is written.
  - The model loads.
  - `nbody` is at least `23` because there are two roots plus 22 segment bodies.
  - `nsite` is at least `4` for pick and center sites.

### Task 2: Include Segmented Leaves In The Production Scene

**Files:**
- Modify: `mujoco_xarm6/scene/scene_winejar_production_demo.xml`

- [ ] **Step 1: Add the compliant leaf include**

  Add this include after the xArm includes:

  ```xml
  <include file="compliant_bamboo_leaf_pair.xml"/>
  ```

- [ ] **Step 2: Remove the old rigid staged bamboo leaf bodies**

  Delete the two existing rigid bodies from the main scene:

  ```xml
  <body name="staged_bamboo_leaf_bottom" ...>
    ...
  </body>
  <body name="staged_bamboo_leaf_top" ...>
    ...
  </body>
  ```

  Leave `staged_metal_weight` unchanged.

- [ ] **Step 3: Keep suction weld names unchanged**

  Confirm these welds still reference the same root body names:

  ```xml
  <weld name="left_suction_weld_leaf_top" body1="left_vacuum_end_effector" body2="staged_bamboo_leaf_top"
        active="false" solref="0.004 1" solimp="0.95 0.99 0.001"/>
  <weld name="left_suction_weld_leaf_bottom" body1="left_vacuum_end_effector" body2="staged_bamboo_leaf_bottom"
        active="false" solref="0.004 1" solimp="0.95 0.99 0.001"/>
  ```

- [ ] **Step 4: Update the `production_home` keyframe**

  Remove the two old 7-value freejoint qpos blocks for the rigid bamboo leaves. Keep the arm qpos and the metal weight qpos:

  ```xml
  <key name="production_home"
       qpos="0 -0.610865 -1.221730 0 1.832596 0 0 -0.610865 -1.221730 0 1.832596 0 -0.48 0.52 0.498 1 0 0 0"
       ctrl="0 0 0 0 0 0 0 0 0 0 0 0"/>
  ```

  This lets MuJoCo use the include-defined default pose for the segmented leaves and explicitly positions only the metal weight.

- [ ] **Step 5: Load the production scene**

  Run:

  ```bash
  conda run -n winejar-mujoco python -c "import mujoco; m=mujoco.MjModel.from_xml_path('mujoco_xarm6/scene/scene_winejar_production_demo.xml'); print(m.nbody, m.njnt, m.ngeom, m.nsite, m.nu, m.nq)"
  ```

  Expected:

  - The scene loads.
  - `nu` remains `12`.
  - `nbody` and `njnt` increase compared with the rigid leaf scene.

### Task 3: Add A Dedicated Compliant Material Inspector

**Files:**
- Create: `mujoco_xarm6/test/inspect_compliant_material_models.py`

- [ ] **Step 1: Create the inspector skeleton**

  Create:

  ```python
  #!/usr/bin/env python3
  from __future__ import annotations

  from pathlib import Path

  import mujoco
  import numpy as np


  ROOT = Path(__file__).resolve().parents[1]
  SCENE_PATH = ROOT / "scene" / "scene_winejar_production_demo.xml"
  SEGMENT_COUNT = 11


  def body_exists(model, name: str) -> bool:
      return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0


  def site_exists(model, name: str) -> bool:
      return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) >= 0
  ```

- [ ] **Step 2: Add static structure checks**

  Append:

  ```python
  def check_leaf_structure(model, leaf_name: str):
      if not body_exists(model, leaf_name):
          raise AssertionError(f"Missing leaf root body: {leaf_name}")
      missing = [
          f"{leaf_name}_seg_{index:02d}"
          for index in range(SEGMENT_COUNT)
          if not body_exists(model, f"{leaf_name}_seg_{index:02d}")
      ]
      if missing:
          raise AssertionError(f"Missing segment bodies for {leaf_name}: {missing}")
      for site_name in [f"{leaf_name}_pick_site", f"{leaf_name}_center_site"]:
          if not site_exists(model, site_name):
              raise AssertionError(f"Missing site: {site_name}")


  def check_table_support(model, data, leaf_name: str, expected_bottom_z: float):
      for index in range(SEGMENT_COUNT):
          body_id = model.body(f"{leaf_name}_seg_{index:02d}").id
          bottom_z = float(data.xpos[body_id][2] - 0.003 / 2.0)
          gap = bottom_z - expected_bottom_z
          print(f"  {leaf_name}_seg_{index:02d} support gap: {gap * 1000.0:.2f} mm")
          if abs(gap) > 0.002:
              raise AssertionError(f"{leaf_name}_seg_{index:02d} is not supported correctly, gap={gap:.4f} m")
  ```

- [ ] **Step 3: Add a dynamic non-rigid response check**

  Append:

  ```python
  def check_nonrigid_response(model, data, leaf_name: str):
      middle = model.body(f"{leaf_name}_seg_{SEGMENT_COUNT // 2:02d}").id
      end = model.body(f"{leaf_name}_seg_00").id
      mujoco.mj_forward(model, data)
      middle_before = data.xpos[middle].copy()
      end_before = data.xpos[end].copy()

      dof_bodyids = np.array([model.dof_bodyid[dof_id] for dof_id in range(model.nv)])
      middle_dofs = np.where(dof_bodyids == middle)[0]
      if len(middle_dofs) == 0:
          raise AssertionError(f"{leaf_name} middle segment has no movable dofs")
      for _ in range(120):
          data.qfrc_applied[middle_dofs] = -0.002
          mujoco.mj_step(model, data)
      data.qfrc_applied[:] = 0
      mujoco.mj_forward(model, data)

      middle_delta = float(np.linalg.norm(data.xpos[middle] - middle_before))
      end_delta = float(np.linalg.norm(data.xpos[end] - end_before))
      print(f"  {leaf_name} response: middle_delta={middle_delta:.6f} m, end_delta={end_delta:.6f} m")
      if middle_delta <= end_delta + 0.0001:
          raise AssertionError(f"{leaf_name} did not show a stronger middle response than end response")
  ```

- [ ] **Step 4: Add `main()`**

  Append:

  ```python
  def main():
      model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
      data = mujoco.MjData(model)
      mujoco.mj_forward(model, data)
      print(f"Loaded scene: {SCENE_PATH}")
      print(f"nbody={model.nbody}, njnt={model.njnt}, ngeom={model.ngeom}, nsite={model.nsite}, nq={model.nq}, nv={model.nv}")

      table_top_z = data.xpos[model.body("left_material_table").id][2] + 0.24
      for leaf_name, expected_bottom_z in [
          ("staged_bamboo_leaf_bottom", table_top_z),
          ("staged_bamboo_leaf_top", table_top_z + 0.003),
      ]:
          print(f"\nChecking {leaf_name}")
          check_leaf_structure(model, leaf_name)
          check_table_support(model, data, leaf_name, expected_bottom_z)
          check_nonrigid_response(model, data, leaf_name)
      print("\nCompliant bamboo leaf checks OK")


  if __name__ == "__main__":
      main()
  ```

- [ ] **Step 5: Run the new inspector**

  Run:

  ```bash
  conda run -n winejar-mujoco python mujoco_xarm6/test/inspect_compliant_material_models.py
  ```

  Expected:

  - Scene loads.
  - Each leaf has 11 segment bodies.
  - Pick and center sites exist.
  - Support gaps are within 2 mm.
  - Dynamic response check passes.

### Task 4: Update Production Scene Inspector For Segmented Leaves

**Files:**
- Modify: `mujoco_xarm6/test/inspect_winejar_production_demo_scene.py`

- [ ] **Step 1: Add helper functions for segmented leaves**

  Add:

  ```python
  def segment_body_names(prefix: str, count: int = 11):
      return [f"{prefix}_seg_{index:02d}" for index in range(count)]


  def assert_segmented_leaf(model, prefix: str, count: int = 11):
      missing = [
          name for name in segment_body_names(prefix, count)
          if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) < 0
      ]
      if missing:
          raise AssertionError(f"Missing segmented leaf bodies for {prefix}: {missing}")
  ```

- [ ] **Step 2: Replace rigid support checks**

  Replace the existing support checks for:

  ```python
  ("staged_bamboo_leaf_bottom", 0.003 / 2.0, table_top_z, 0.0015),
  ("staged_bamboo_leaf_top", 0.003 / 2.0, table_top_z + 0.003, 0.0015),
  ```

  with segment checks:

  ```python
  for leaf_name, expected_bottom_z in [
      ("staged_bamboo_leaf_bottom", table_top_z),
      ("staged_bamboo_leaf_top", table_top_z + 0.003),
  ]:
      assert_segmented_leaf(model, leaf_name)
      for segment_name in segment_body_names(leaf_name):
          bottom_z = float(data.xpos[model.body(segment_name).id][2] - 0.003 / 2.0)
          gap = bottom_z - expected_bottom_z
          if abs(gap) > 0.002:
              raise AssertionError(f"{segment_name} is not supported correctly, gap={gap:.4f} m")
  ```

- [ ] **Step 3: Keep metal weight support check**

  Keep the existing metal weight support/freejoint check:

  ```python
  body_name = "staged_metal_weight"
  bottom_z = float(data.xpos[model.body(body_name).id][2] - 0.018)
  gap = bottom_z - table_top_z
  if abs(gap) > 0.0015:
      raise AssertionError(f"{body_name} is not supported correctly, gap={gap:.4f} m")
  if body_joint_count(model, body_name) == 0:
      raise AssertionError(f"{body_name} has no joint, so it is fixed and cannot be picked")
  ```

- [ ] **Step 4: Run production inspection**

  Run:

  ```bash
  conda run -n winejar-mujoco python mujoco_xarm6/test/inspect_winejar_production_demo_scene.py
  ```

  Expected:

  - Existing robot, tie-gun, conveyor, and material checks pass.
  - New segmented leaf checks pass.

### Task 5: Final Verification

**Files:**
- Check: `mujoco_xarm6/tools/generate_compliant_material_models.py`
- Check: `mujoco_xarm6/test/inspect_compliant_material_models.py`
- Check: `mujoco_xarm6/test/inspect_winejar_production_demo_scene.py`
- Check: `mujoco_xarm6/test/view_winejar_scene.py`

- [ ] **Step 1: Run all generators**

  Run:

  ```bash
  conda run -n winejar-mujoco python mujoco_xarm6/tools/generate_prefixed_xarm_models.py
  conda run -n winejar-mujoco python mujoco_xarm6/tools/generate_compliant_material_models.py
  ```

  Expected:

  - xArm include files are regenerated.
  - `compliant_bamboo_leaf_pair.xml` is regenerated.

- [ ] **Step 2: Run inspections**

  Run:

  ```bash
  conda run -n winejar-mujoco python mujoco_xarm6/test/inspect_winejar_production_demo_scene.py
  conda run -n winejar-mujoco python mujoco_xarm6/test/inspect_compliant_material_models.py
  ```

  Expected:

  - Both commands exit with code `0`.

- [ ] **Step 3: Verify keyframe and Python syntax**

  Run:

  ```bash
  conda run -n winejar-mujoco python -c "import mujoco; m=mujoco.MjModel.from_xml_path('mujoco_xarm6/scene/scene_winejar_production_demo.xml'); d=mujoco.MjData(m); kid=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, 'production_home'); assert kid >= 0; mujoco.mj_resetDataKeyframe(m,d,kid); mujoco.mj_forward(m,d); print('production_home nq=', m.nq, 'nv=', m.nv)"
  conda run -n winejar-mujoco python -m py_compile \
    mujoco_xarm6/tools/generate_compliant_material_models.py \
    mujoco_xarm6/test/inspect_compliant_material_models.py \
    mujoco_xarm6/test/inspect_winejar_production_demo_scene.py \
    mujoco_xarm6/test/view_winejar_scene.py
  ```

  Expected:

  - Keyframe loads.
  - Python files compile.

- [ ] **Step 4: Manual viewer command for the user**

  Provide:

  ```bash
  conda activate winejar-mujoco
  python mujoco_xarm6/test/view_winejar_scene.py \
    --scene mujoco_xarm6/scene/scene_winejar_production_demo.xml \
    --key production_home
  ```

  Expected:

  - The two staged bamboo leaves still appear as two stacked leaves.
  - They should no longer be a single rigid mesh each.
  - No segment should visibly float above the material table.
