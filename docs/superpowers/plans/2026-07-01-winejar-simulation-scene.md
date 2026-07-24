# Wine Jar Simulation Scene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a first static MuJoCo scene for the wine jar loading task with xArm6, gripper, raised base, realistic jar mesh, material table, ordered rigid materials, and inspection tooling.

**Architecture:** Create an isolated conda environment before running MuJoCo tools. Keep the existing xArm6 robot model unchanged and create a separate `scene_winejar.xml` that includes it. Generate simple deterministic STL assets for the jar and irregular materials, then validate the scene through a lightweight MuJoCo loading script that prints key site positions.

**Tech Stack:** Conda, Python 3.11, MuJoCo XML, Python 3 standard library, binary STL generation, existing xArm6 MuJoCo assets.

---

## File Structure

- Create: `mujoco_xarm6/environment.yml`
  - Defines the isolated conda environment for scene generation, inspection, and viewer launch.
- Create: `mujoco_xarm6/scene/assets/materials/`
  - Stores generated STL meshes for the jar, lotus leaf, and bamboo leaf.
- Create: `mujoco_xarm6/tools/generate_scene_assets.py`
  - Generates deterministic binary STL meshes from parameterized geometry.
- Create: `mujoco_xarm6/scene/scene_winejar.xml`
  - Defines the new static scene and includes `xarm6.xml`.
- Create: `mujoco_xarm6/test/inspect_winejar_scene.py`
  - Loads the scene and prints key body/site coordinates.
- Create: `mujoco_xarm6/test/check_material_asset_dimensions.py`
  - Verifies generated material STL dimensions that affect layout and grasp references.
- Create: `mujoco_xarm6/test/check_material_layout.py`
  - Verifies material top-down placement does not overlap and grasp sites are exposed near the robot-side table edge.
- Modify: none of the existing scene files.

## Task 0: Create Isolated Conda Environment

**Files:**
- Create: `mujoco_xarm6/environment.yml`

- [ ] **Step 1: Create the environment definition**

Write `mujoco_xarm6/environment.yml`:

```yaml
name: winejar-mujoco
channels:
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - numpy
  - glfw
  - pip:
      - mujoco
      - opencv-python
      - gymnasium
```

- [ ] **Step 2: Create or update the conda environment**

Run:

```bash
conda env create -f mujoco_xarm6/environment.yml
```

Expected when creating for the first time: conda creates an environment named `winejar-mujoco`.

If the environment already exists, run:

```bash
conda env update -f mujoco_xarm6/environment.yml --prune
```

Expected when updating: conda updates `winejar-mujoco` without installing packages into the system Python environment.

- [ ] **Step 3: Verify core packages inside the environment**

Run:

```bash
conda run -n winejar-mujoco python - <<'PY'
import mujoco
import numpy
import glfw

print("mujoco", mujoco.__version__)
print("numpy", numpy.__version__)
print("glfw", glfw.__version__)
PY
```

Expected: prints package versions without import errors.

## Task 1: Generate Scene Mesh Assets

**Files:**
- Create: `mujoco_xarm6/tools/generate_scene_assets.py`
- Create by running script: `mujoco_xarm6/scene/assets/materials/wine_jar.stl`
- Create by running script: `mujoco_xarm6/scene/assets/materials/lotus_leaf.stl`
- Create by running script: `mujoco_xarm6/scene/assets/materials/bamboo_leaf.stl`

- [ ] **Step 1: Create the asset generator script**

Write `mujoco_xarm6/tools/generate_scene_assets.py` with deterministic binary STL mesh generation:

```python
#!/usr/bin/env python3
from __future__ import annotations

import math
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "scene" / "assets" / "materials"


def normal(a, b, c):
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length == 0:
        return (0.0, 0.0, 1.0)
    return (nx / length, ny / length, nz / length)


def write_binary_stl(path: Path, name: str, triangles):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"{name} generated for wine jar MuJoCo scene".encode("ascii")[:80]
    header = header + b" " * (80 - len(header))
    with path.open("wb") as f:
        f.write(header)
        f.write(struct.pack("<I", len(triangles)))
        for a, b, c in triangles:
            nx, ny, nz = normal(a, b, c)
            values = (
                nx,
                ny,
                nz,
                a[0],
                a[1],
                a[2],
                b[0],
                b[1],
                b[2],
                c[0],
                c[1],
                c[2],
            )
            f.write(struct.pack("<12fH", *values, 0))


def generate_wine_jar(path: Path):
    profile = [
        (0.00, 0.105),
        (0.06, 0.125),
        (0.20, 0.175),
        (0.36, 0.165),
        (0.46, 0.090),
        (0.53, 0.060),
        (0.55, 0.060),
    ]
    segments = 96
    triangles = []

    rings = []
    for z, r in profile:
        ring = []
        for i in range(segments):
            theta = 2.0 * math.pi * i / segments
            ring.append((r * math.cos(theta), r * math.sin(theta), z))
        rings.append(ring)

    for j in range(len(rings) - 1):
        lower = rings[j]
        upper = rings[j + 1]
        for i in range(segments):
            ni = (i + 1) % segments
            triangles.append((lower[i], lower[ni], upper[ni]))
            triangles.append((lower[i], upper[ni], upper[i]))

    bottom_center = (0.0, 0.0, profile[0][0])
    top_center = (0.0, 0.0, profile[-1][0])
    for i in range(segments):
        ni = (i + 1) % segments
        triangles.append((bottom_center, rings[0][i], rings[0][ni]))
        triangles.append((top_center, rings[-1][ni], rings[-1][i]))

    write_binary_stl(path, "wine_jar", triangles)


def generate_lotus_leaf(path: Path):
    radius = 0.22
    thickness = 0.004
    angle0 = -math.pi / 4.0
    angle1 = math.pi / 4.0
    segments = 32
    triangles = []

    top_center = (0.0, 0.0, thickness / 2.0)
    bottom_center = (0.0, 0.0, -thickness / 2.0)
    top_arc = []
    bottom_arc = []
    for i in range(segments + 1):
        t = angle0 + (angle1 - angle0) * i / segments
        top_arc.append((radius * math.cos(t), radius * math.sin(t), thickness / 2.0))
        bottom_arc.append((radius * math.cos(t), radius * math.sin(t), -thickness / 2.0))

    for i in range(segments):
        triangles.append((top_center, top_arc[i], top_arc[i + 1]))
        triangles.append((bottom_center, bottom_arc[i + 1], bottom_arc[i]))
        triangles.append((top_arc[i], bottom_arc[i], bottom_arc[i + 1]))
        triangles.append((top_arc[i], bottom_arc[i + 1], top_arc[i + 1]))

    triangles.append((top_center, bottom_center, bottom_arc[0]))
    triangles.append((top_center, bottom_arc[0], top_arc[0]))
    triangles.append((top_center, top_arc[-1], bottom_arc[-1]))
    triangles.append((top_center, bottom_arc[-1], bottom_center))

    write_binary_stl(path, "lotus_leaf", triangles)


def generate_bamboo_leaf(path: Path):
    length = 0.42
    half_length = length / 2.0
    half_width = 0.0625
    thickness = 0.003
    segments = 48
    triangles = []

    top_center = (0.0, 0.0, thickness / 2.0)
    bottom_center = (0.0, 0.0, -thickness / 2.0)
    top = []
    bottom = []
    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        x = half_length * math.cos(theta)
        y = half_width * math.sin(theta)
        top.append((x, y, thickness / 2.0))
        bottom.append((x, y, -thickness / 2.0))

    for i in range(segments):
        ni = (i + 1) % segments
        triangles.append((top_center, top[i], top[ni]))
        triangles.append((bottom_center, bottom[ni], bottom[i]))
        triangles.append((top[i], bottom[i], bottom[ni]))
        triangles.append((top[i], bottom[ni], top[ni]))

    write_binary_stl(path, "bamboo_leaf", triangles)


def main():
    generate_wine_jar(OUT_DIR / "wine_jar.stl")
    generate_lotus_leaf(OUT_DIR / "lotus_leaf.stl")
    generate_bamboo_leaf(OUT_DIR / "bamboo_leaf.stl")
    print(f"Generated scene assets in {OUT_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the asset generator**

Run:

```bash
conda run -n winejar-mujoco python mujoco_xarm6/tools/generate_scene_assets.py
```

Expected output:

```text
Generated scene assets in /home/wunicheng/Git/WineJar/mujoco_xarm6/scene/assets/materials
```

- [ ] **Step 3: Verify assets exist**

Run:

```bash
ls -lh mujoco_xarm6/scene/assets/materials
```

Expected: the directory contains `wine_jar.stl`, `lotus_leaf.stl`, and `bamboo_leaf.stl`.

## Task 2: Create The Wine Jar MuJoCo Scene

**Files:**
- Create: `mujoco_xarm6/scene/scene_winejar.xml`

- [ ] **Step 1: Create `scene_winejar.xml`**

Create a new scene that follows the existing `scene_peghole.xml` structure and includes the existing xArm6 robot:

```xml
<mujoco model="winejar loading scene">
  <include file="xarm6.xml"/>

  <option integrator="implicitfast" gravity="0 0 -9.81" timestep="0.001"/>
  <statistic center="0.30 0.25 0.40" extent="1.1"/>

  <visual>
    <quality shadowsize="2048"/>
    <headlight diffuse="0.28 0.28 0.27" ambient="0.15 0.16 0.18" specular="0.16 0.16 0.16"/>
    <rgba haze="0.72 0.76 0.80 1"/>
    <global azimuth="135" elevation="-25"/>
  </visual>

  <asset>
    <texture type="2d" name="workbench_texture" file="textures/workbench_laminate.png"/>
    <material name="table_laminate" texture="workbench_texture" texuniform="true" texrepeat="2 2"
              rgba="0.66 0.68 0.70 1" specular="0.14" shininess="0.18" reflectance="0.02"/>
    <texture type="skybox" builtin="gradient" rgb1="0.82 0.86 0.90" rgb2="0.26 0.30 0.34"
             width="512" height="3072"/>
    <texture type="2d" name="ground_texture" builtin="checker"
             rgb1="0.18 0.19 0.20" rgb2="0.21 0.22 0.23" width="512" height="512"/>
    <material name="ground_material" texture="ground_texture" texuniform="true" texrepeat="18 18"
              specular="0.08" shininess="0.12" reflectance="0.02"/>

    <material name="base_mat" rgba="0.25 0.27 0.28 1" specular="0.15" shininess="0.20"/>
    <material name="jar_mat" rgba="0.48 0.23 0.11 1" specular="0.18" shininess="0.32"/>
    <material name="jar_collision_mat" rgba="0.48 0.23 0.11 0.12"/>
    <material name="lotus_mat" rgba="0.18 0.45 0.20 1" specular="0.08" shininess="0.15"/>
    <material name="paper_mat" rgba="0.94 0.93 0.88 1" specular="0.05" shininess="0.10"/>
    <material name="ceramic_mat" rgba="0.86 0.84 0.78 1" specular="0.22" shininess="0.35"/>
    <material name="bamboo_mat" rgba="0.34 0.55 0.22 1" specular="0.08" shininess="0.12"/>
    <material name="label_mat" rgba="0.93 0.86 0.58 1" specular="0.05" shininess="0.10"/>

    <mesh name="wine_jar_mesh" file="materials/wine_jar.stl"/>
    <mesh name="lotus_leaf_mesh" file="materials/lotus_leaf.stl"/>
    <mesh name="bamboo_leaf_mesh" file="materials/bamboo_leaf.stl"/>
  </asset>

  <worldbody>
    <light name="key_light" pos="-0.35 -0.55 1.4" dir="0.25 0.35 -1" directional="true"
           diffuse="0.52 0.50 0.46" specular="0.22 0.21 0.19" castshadow="true"/>
    <light name="fill_light" pos="0.75 0.45 1.05" dir="-0.5 -0.25 -1" directional="true"
           diffuse="0.18 0.21 0.26" specular="0.06 0.07 0.09" castshadow="false"/>
    <geom name="floor" pos="0 0 -0.02" size="0 0 0.05" type="plane" material="ground_material"/>

    <body name="robot_base_pedestal" pos="0 0 0.15">
      <geom name="robot_base_pedestal_geom" type="box" size="0.22 0.22 0.15" material="base_mat"
            mass="2000" friction="1 1 1"/>
    </body>

    <body name="wine_jar" pos="0.60 0 0">
      <geom name="wine_jar_visual" type="mesh" mesh="wine_jar_mesh" material="jar_mat"
            contype="0" conaffinity="0"/>
      <geom name="wine_jar_belly_collision" type="cylinder" pos="0 0 0.23" size="0.175 0.23"
            material="jar_collision_mat" mass="8" friction="1 1 1"/>
      <geom name="wine_jar_shoulder_collision" type="cylinder" pos="0 0 0.43" size="0.10 0.07"
            material="jar_collision_mat" mass="2" friction="1 1 1"/>
      <geom name="wine_jar_neck_collision" type="cylinder" pos="0 0 0.52" size="0.06 0.04"
            material="jar_collision_mat" mass="1" friction="1 1 1"/>
      <site name="jar_mouth_center" pos="0 0 0.55" type="sphere" size="0.012" rgba="1 0 0 0.8"/>
      <site name="jar_label_target" pos="0.061 0 0.49" type="sphere" size="0.01" rgba="0 0.25 1 0.8"/>
      <site name="jar_centerline" pos="0 0 0.275" type="cylinder" size="0.004 0.275" rgba="1 1 1 0.25"/>
    </body>

    <body name="material_table" pos="0.30 0.84 0.30">
      <geom name="material_table_geom" type="box" size="0.65 0.34 0.30" material="table_laminate"
            mass="2000" friction="1 1 1"/>
    </body>

    <body name="lotus_leaf" pos="0.02 0.55 0.612" euler="0 0 0">
      <geom name="lotus_leaf_geom" type="mesh" mesh="lotus_leaf_mesh" material="lotus_mat"
            mass="0.03" friction="1 1 1"/>
      <site name="lotus_leaf_center_site" pos="0 0 0.004" type="sphere" size="0.006" rgba="1 0 0 0.7"/>
      <site name="lotus_leaf_grasp_site" pos="0 -0.11 0.004" type="sphere" size="0.006" rgba="0 1 0 0.7"/>
    </body>

    <body name="white_paper" pos="0.16 0.55 0.615">
      <geom name="white_paper_geom" type="box" size="0.1485 0.105 0.002" material="paper_mat"
            mass="0.01" friction="1 1 1"/>
      <site name="white_paper_center_site" pos="0 0 0.003" type="sphere" size="0.006" rgba="1 0 0 0.7"/>
      <site name="white_paper_grasp_site" pos="0 -0.105 0.003" type="sphere" size="0.006" rgba="0 1 0 0.7"/>
    </body>

    <body name="ceramic_disc" pos="0.30 0.55 0.618">
      <geom name="ceramic_disc_geom" type="cylinder" size="0.05 0.005" material="ceramic_mat"
            mass="0.20" friction="1 1 1"/>
      <site name="ceramic_disc_center_site" pos="0 0 0.006" type="sphere" size="0.006" rgba="1 0 0 0.7"/>
      <site name="ceramic_disc_grasp_site" pos="0 -0.05 0.006" type="sphere" size="0.006" rgba="0 1 0 0.7"/>
    </body>

    <body name="bamboo_leaf_1" pos="0.43 0.50 0.615" euler="0 0 0">
      <geom name="bamboo_leaf_1_geom" type="mesh" mesh="bamboo_leaf_mesh" material="bamboo_mat"
            mass="0.01" friction="1 1 1"/>
      <site name="bamboo_leaf_1_center_site" pos="0 0 0.003" type="sphere" size="0.006" rgba="1 0 0 0.7"/>
      <site name="bamboo_leaf_1_grasp_site" pos="0 -0.0625 0.003" type="sphere" size="0.006" rgba="0 1 0 0.7"/>
    </body>

    <body name="bamboo_leaf_2" pos="0.43 0.60 0.615" euler="0 0 1.5708">
      <geom name="bamboo_leaf_2_geom" type="mesh" mesh="bamboo_leaf_mesh" material="bamboo_mat"
            mass="0.01" friction="1 1 1"/>
      <site name="bamboo_leaf_2_center_site" pos="0 0 0.003" type="sphere" size="0.006" rgba="1 0 0 0.7"/>
      <site name="bamboo_leaf_2_grasp_site" pos="-0.0625 0 0.003" type="sphere" size="0.006" rgba="0 1 0 0.7"/>
    </body>

    <body name="label_paper" pos="0.56 0.55 0.615">
      <geom name="label_paper_geom" type="box" size="0.10 0.04 0.002" material="label_mat"
            mass="0.01" friction="1 1 1"/>
      <site name="label_paper_center_site" pos="0 0 0.003" type="sphere" size="0.006" rgba="1 0 0 0.7"/>
      <site name="label_paper_grasp_site" pos="0 -0.04 0.003" type="sphere" size="0.006" rgba="0 1 0 0.7"/>
    </body>

    <camera name="winejar_overview_cam" pos="1.15 -0.85 1.10" euler="0.85 0 0.90" fovy="45" resolution="1920 1080"/>
  </worldbody>
</mujoco>
```

- [ ] **Step 2: Verify the scene references only existing/generated assets**

Run:

```bash
rg -n "file=\"|mesh name=" mujoco_xarm6/scene/scene_winejar.xml
```

Expected: file references are `xarm6.xml`, `textures/workbench_laminate.png`, `materials/wine_jar.stl`, `materials/lotus_leaf.stl`, and `materials/bamboo_leaf.stl`.

## Task 3: Add Scene Inspection Script

**Files:**
- Create: `mujoco_xarm6/test/inspect_winejar_scene.py`

- [ ] **Step 1: Create the inspection script**

Write:

```python
#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import mujoco


ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = ROOT / "scene" / "scene_winejar.xml"

SITES = [
    "jar_mouth_center",
    "jar_label_target",
    "lotus_leaf_center_site",
    "lotus_leaf_grasp_site",
    "white_paper_center_site",
    "white_paper_grasp_site",
    "ceramic_disc_center_site",
    "ceramic_disc_grasp_site",
    "bamboo_leaf_1_center_site",
    "bamboo_leaf_1_grasp_site",
    "bamboo_leaf_2_center_site",
    "bamboo_leaf_2_grasp_site",
    "label_paper_center_site",
    "label_paper_grasp_site",
]

BODIES = [
    "robot_base_pedestal",
    "wine_jar",
    "material_table",
    "lotus_leaf",
    "white_paper",
    "ceramic_disc",
    "bamboo_leaf_1",
    "bamboo_leaf_2",
    "label_paper",
]


def fmt(vec):
    return f"({vec[0]: .3f}, {vec[1]: .3f}, {vec[2]: .3f})"


def main():
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print(f"Loaded scene: {SCENE_PATH}")
    print(f"nbody={model.nbody}, ngeom={model.ngeom}, nsite={model.nsite}, ncam={model.ncam}")

    print("\nBodies:")
    for name in BODIES:
        body_id = model.body(name).id
        print(f"  {name}: {fmt(data.xpos[body_id])}")

    print("\nSites:")
    for name in SITES:
        site_id = model.site(name).id
        print(f"  {name}: {fmt(data.site_xpos[site_id])}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the inspection script**

Run:

```bash
conda run -n winejar-mujoco python mujoco_xarm6/test/inspect_winejar_scene.py
```

Expected: the script loads the scene and prints the key body/site positions. The jar mouth should print near `(0.600, 0.000, 0.550)`, and material center sites should print near their planned table positions.

## Task 4: Validate Scene Loading And Existing Scene Isolation

**Files:**
- Read only: `mujoco_xarm6/scene/scene_peghole.xml`
- Read only: `mujoco_xarm6/scene/xarm6.xml`
- Verify: `mujoco_xarm6/scene/scene_winejar.xml`

- [ ] **Step 1: Confirm existing files were not modified**

Run:

```bash
find mujoco_xarm6/scene -maxdepth 1 -type f -printf "%f\n" | sort
```

Expected: existing scene files remain present and `scene_winejar.xml` is added.

- [ ] **Step 2: Check XML load failure surface if MuJoCo reports an error**

Run:

```bash
conda run -n winejar-mujoco python mujoco_xarm6/test/inspect_winejar_scene.py
```

Expected: PASS. If it fails, inspect the MuJoCo error message and fix only the new scene XML or generated assets.

- [ ] **Step 3: Optional manual viewer check**

Run:

```bash
conda run -n winejar-mujoco python - <<'PY'
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("mujoco_xarm6/scene/scene_winejar.xml")
data = mujoco.MjData(model)
mujoco.viewer.launch(model, data)
PY
```

Expected: a viewer opens with xArm6 on a raised base, the jar in front, and the material table on the left side. Close the viewer after inspection.

## Task 5: Document The Implementation Result

**Files:**
- Create or modify: `docs/superpowers/specs/2026-07-01-winejar-simulation-scene-design.md`
- Create: final response summary to user

- [ ] **Step 1: Add an implementation notes section if needed**

If implementation required deviations from the design, append a short section to the design doc:

```markdown
## Implementation Notes After Build

- The first generated scene keeps the existing `xarm6.xml` unchanged.
- STL assets are binary STL files for MuJoCo compatibility.
- The jar uses a visual STL and simple cylinder collision geoms.
- Materials are rigid bodies with center/grasp sites for later planning.
- Bamboo leaves use length `0.42 m` and max width `0.125 m`, with long axes placed into the table and grasp sites at the robot-side front tips.
- The material table was enlarged to center `(0.30, 0.84, 0.25)` and size `1.30 m x 0.68 m x 0.50 m`; material positions were re-laid out so footprints do not overlap.
```

- [ ] **Step 2: Run final verification**

Run:

```bash
conda run -n winejar-mujoco python mujoco_xarm6/tools/generate_scene_assets.py
conda run -n winejar-mujoco python mujoco_xarm6/test/check_material_asset_dimensions.py
conda run -n winejar-mujoco python mujoco_xarm6/test/check_material_layout.py
conda run -n winejar-mujoco python mujoco_xarm6/test/inspect_winejar_scene.py
```

Expected: all commands complete successfully.

- [ ] **Step 3: Report completion**

Tell the user:

- Which files were created.
- What the inspection script printed for key positions.
- Whether manual viewer verification was run.
- That no existing peghole scene files were modified.
- That git commit was skipped because this workspace is not a git repository.

## Self-Review

- Spec coverage: The plan covers conda isolation, asset generation, new scene XML, layout, object/site names, validation, and isolation from existing scenes.
- Placeholder scan: No task contains `TBD`, `TODO`, or an unspecified implementation step.
- Type/name consistency: File names, mesh names, body names, and site names match across the scene XML and inspection script.
