# Wine Jar Simulation Scene Design

## Goal

Build the first MuJoCo simulation scene for the wine jar loading project. This stage only creates a clear, inspectable static scene with the xArm6 gripper, a realistic wine jar, a material table, and ordered rigid materials. It does not implement grasp planning, motion control, vision, flexible materials, or tying/locking.

## Current Project Context

The workspace contains:

- `项目策划.docx`: project proposal for the wine jar loading and tying workflow.
- `xarm6/`: xArm manuals, SDKs, ROS package, UFactory Studio, and vision examples.
- `mujoco_xarm6/scene/xarm6.xml`: existing xArm6 MuJoCo model with gripper, D435 model, wrist camera, force/torque sensor, and velocity actuators.
- `mujoco_xarm6/scene/scene_peghole.xml`: existing simple scene with xArm6, table, lights, a red object, and camera.
- `mujoco_xarm6/test/test4joint.py`: existing controller and inspection code for MuJoCo loading, viewer launch, camera rendering, joint velocity control, TCP velocity control, and force sensing.

The new scene should reference the existing style and structure where useful, especially the `include file="xarm6.xml"` pattern, lighting/material organization, worldbody layout, and use of MuJoCo sites for future task references.

## Scope

In scope:

- Create an isolated Python environment before running MuJoCo tooling.
- Create a new wine jar task scene.
- Place the xArm6 with its existing gripper on a 0.30 m high base.
- Place the wine jar directly in front of the robot.
- Place the material table on the robot's left side.
- Arrange rigid material objects in process order.
- Add center and grasp reference sites for materials.
- Add jar task reference sites.
- Add a script to load and inspect the scene.

Out of scope:

- Real grasp execution.
- Motion planning.
- ROS2 integration.
- UFactory hardware control.
- Vision detection.
- Flexible material simulation.
- Tying/locking with rope or strap.
- Physical calibration against a real robot cell.

## Coordinate System And Layout

Use the following world convention:

- `+X`: robot forward direction.
- `+Y`: robot left side.
- `+Z`: vertical upward direction.

Primary layout:

- Robot base pedestal center: `(0.0, 0.0, 0.10)`.
- Wine jar centerline: `x = 0.60`, `y = 0.0`.
- Wine jar bottom: `z = 0.0`.
- Wine jar mouth center: approximately `(0.60, 0.0, 0.46)`.
- Material table center: `(0.00, 0.84, 0.25)`.
- Material table top height: `z = 0.50`.

The robot has a low pedestal in the scene, while the material table top is set slightly above the 0.46 m jar mouth. The jar stays in the robot's forward direction, while the material table is placed on the robot's left side.

## Wine Jar Model

The wine jar should not be modeled as stacked cylinders. That would distort the side surface and become misleading for later label placement and tying work.

Use a parameterized surface-of-revolution STL generated from a radius-height profile. The first profile should approximate a traditional jar shape:

| Height z (m) | Radius (m) | Meaning |
| --- | ---: | --- |
| 0.00 | 0.125 | bottom, diameter 0.25 m |
| 0.08 | 0.140 | lower expansion |
| 0.20 | 0.155 | mid body |
| 0.35 | 0.160 | widest belly, diameter 0.32 m |
| 0.41 | 0.110 | shoulder contraction |
| 0.445 | 0.060 | mouth outer radius, diameter 0.12 m |
| 0.46 | 0.060 | top of 1.5 cm hollow mouth cylinder |

The mouth cylinder is hollow in the visual mesh. Its outer diameter is `0.12 m`, and its inner diameter uses the project proposal mouth inner diameter `0.105 m`.

Generate:

- Visual mesh: `mujoco_xarm6/scene/assets/materials/wine_jar.stl`.
- Collision approximation: a small set of cylinders or capsule-like primitives matching bottom, belly, shoulder, and neck. Use stable simple collisions first instead of relying on detailed mesh collision.

Jar sites:

- `jar_mouth_center`: located at the center of the mouth opening.
- `jar_label_target`: located on the front side of the neck/upper shoulder area for later label placement.
- `jar_centerline`: optional invisible reference site around the jar central axis.

## Material Models

All materials are rigid in the first version. Flexible material simulation is intentionally deferred.

Material dimensions:

- Lotus leaf: 90-degree sector with radius `0.22 m`.
- White paper: A4 rectangle, `0.297 m x 0.210 m`.
- Ceramic disc: circular disc, diameter `0.10 m`.
- Bamboo leaves: eye/ellipse-like thin pieces, length `0.42 m`, max width `0.125 m`.
- Label paper: rectangle, `0.20 m x 0.08 m`.

Recommended asset strategy:

- Lotus leaf: generated thin STL mesh sector.
- Bamboo leaf: generated thin STL mesh, shared by both leaf bodies.
- White paper: MuJoCo box geom.
- Ceramic disc: MuJoCo cylinder geom.
- Label paper: MuJoCo box geom.

Generated mesh paths:

- `mujoco_xarm6/scene/assets/materials/lotus_leaf.stl`
- `mujoco_xarm6/scene/assets/materials/bamboo_leaf.stl`

Each material body should include:

- One center site named `<material>_center_site`.
- One grasp site named `<material>_grasp_site`.

For the two bamboo leaves, use distinct names:

- `bamboo_leaf_1_center_site`
- `bamboo_leaf_1_grasp_site`
- `bamboo_leaf_2_center_site`
- `bamboo_leaf_2_grasp_site`

## Material Table And Object Placement

Material table:

- Center: `(0.00, 0.84, 0.25)`.
- Size: approximately `1.30 m x 0.68 m x 0.50 m`.
- Top height: `0.50 m`.
- Long axis along `X`.
- Short axis along `Y`.

Material center positions:

| Material | Position (m) | Notes |
| --- | --- | --- |
| Lotus leaf | `(-0.45, 0.46, 0.512)` | first process material, sector opens into the table |
| White paper | `(-0.12, 0.608, 0.515)` | A4 sheet, short edge exposed |
| Ceramic disc | `(0.08, 0.51, 0.518)` | thin circular disc, terracotta orange |
| Bamboo leaf 1 | `(0.25, 0.67, 0.515)` | long axis into the table |
| Bamboo leaf 2 | `(0.42, 0.67, 0.515)` | long axis into the table |
| Label paper | `(0.56, 0.56, 0.515)` | short edge exposed |

The table is on the robot's left side, so the robot mainly approaches from the `-Y` side. Materials should not overlap in top-down projection. Rectangular pieces should expose their short edge toward `-Y`, and long materials should extend into the table toward `+Y`. The grasp sites should be placed on these exposed robot-side edges or front tips, with at least `0.03 m` of overhang beyond the robot-side table edge. The current layout targets about `0.04 m` of exposed edge.

## Scene File Structure

Create new files without overwriting existing peghole scenes:

```text
mujoco_xarm6/
  environment.yml
  scene/
    scene_winejar.xml
    assets/
      materials/
        wine_jar.stl
        lotus_leaf.stl
        bamboo_leaf.stl
  tools/
    generate_scene_assets.py
  test/
    check_material_asset_dimensions.py
    check_material_layout.py
    inspect_winejar_scene.py
```

Responsibilities:

- `environment.yml`: conda environment definition for MuJoCo scene generation and inspection.
- `scene_winejar.xml`: main MuJoCo scene. Includes `xarm6.xml`, defines floor, robot base, wine jar, material table, rigid materials, lights, camera, materials, and sites.
- `generate_scene_assets.py`: deterministic script that generates STL assets for the jar, lotus leaf sector, and bamboo leaf.
- `check_material_asset_dimensions.py`: verifies generated material STL dimensions that affect task geometry.
- `check_material_layout.py`: verifies material top-down placement does not overlap and grasp sites are exposed toward the robot-side table edge.
- `inspect_winejar_scene.py`: loads the scene and prints key body/site positions to verify the model is structurally correct.

## Environment Strategy

Use conda for the first implementation pass. Conda is the lower-friction choice for this stage because MuJoCo viewer and GLFW need local display access. Docker is useful later for reproducibility, but it adds X11/GPU forwarding complexity before the scene itself is proven.

Create:

- `mujoco_xarm6/environment.yml`

The environment should include:

- Python 3.11.
- `mujoco`.
- `numpy`.
- `opencv-python`.
- `gymnasium`.
- `glfw`.

All asset generation and scene inspection commands should run after activating this environment. Do not install packages directly into the system Python environment.

## Validation

Programmatic validation:

- Running `conda env create -f mujoco_xarm6/environment.yml` should create an isolated environment named `winejar-mujoco`.
- Running `conda run -n winejar-mujoco python mujoco_xarm6/test/inspect_winejar_scene.py` should load `scene_winejar.xml` without XML or asset errors.
- Running `conda run -n winejar-mujoco python mujoco_xarm6/test/check_material_asset_dimensions.py` should confirm the generated bamboo leaf is `0.42 m` long and `0.125 m` wide.
- The same dimensions check should confirm the generated wine jar height is `0.46 m`, bottom diameter is `0.25 m`, diameter at `z=0.35 m` is `0.32 m`, and mouth outer diameter is `0.12 m`.
- Running `conda run -n winejar-mujoco python mujoco_xarm6/test/check_material_layout.py` should confirm material footprints do not overlap and all grasp sites are exposed near the robot-side table edge.
- The script should print key site positions:
  - `jar_mouth_center`
  - `jar_label_target`
  - all material center sites
  - all material grasp sites
- The printed coordinates should match the design within small expected tolerances.

Manual validation:

- Open the scene in MuJoCo viewer.
- Confirm the robot is raised by a base.
- Confirm the jar is directly in front of the robot.
- Confirm the material table is on the robot's left side.
- Confirm the materials are ordered by process sequence.
- Confirm material edges visibly overhang toward the robot side.
- Confirm there is no obvious interpenetration among table, jar, robot, and material objects.

## Implementation Notes

The existing scene files should be used as references:

- Use `scene_peghole.xml` as the reference for scene-level structure, lighting, skybox/ground/table material style, and `include file="xarm6.xml"`.
- Use `xarm6.xml` as the robot source. Do not edit it for this first scene unless loading requires a minimal compatible adjustment.
- Keep `scene_peghole.xml`, `xarm6_test.xml`, and `xarm6_peghole.xml` unchanged.

The workspace is currently not a git repository, so the superpowers requirement to commit the design document cannot be completed unless a repository is initialized or the files are moved into a git repo. The design workflow still records this document and performs self-review.

## Future Extensions

After the static scene is accepted, later iterations can add:

- Reachability checks for material grasp sites and jar placement sites.
- Simple scripted pick-and-place trajectories.
- Collision-aware path planning.
- Wrist camera or third-person camera capture validation.
- Rigid placement success metrics.
- Semi-flexible material approximations using multi-body chains or MuJoCo `flex`.
- Label placement against the jar side surface.
- Tying/locking simulation as a separate subsystem.

## Implementation Notes After Build

- The first generated scene keeps the existing `xarm6.xml` and peghole scene files unchanged.
- STL assets are generated as binary STL files because MuJoCo 3.10 failed to decode the initial ASCII STL output.
- The jar uses a visual STL and simple cylinder collision geoms for stable first-pass loading.
- Materials are rigid bodies with center/grasp sites for later planning.
- The material table implementation uses center `(0.00, 0.84, 0.25)` and size `1.30 m x 0.68 m x 0.50 m` so the enlarged bamboo leaves and other materials can be laid out without overlap while keeping the table center aligned with the robot pedestal center along `X`.
- Bamboo leaf dimensions were updated to length `0.42 m` and maximum width `0.125 m`; the leaves are oriented with their long axis into the table and grasp sites at the robot-side front tip.
- Material placement was updated so rectangular materials expose their short edge toward the robot-side `-Y` edge, and long leaves extend into the table along `+Y`.
- The wine jar was updated to height `0.46 m`, bottom diameter `0.25 m`, maximum diameter `0.32 m` at height `0.35 m`, and a `1.5 cm` hollow mouth cylinder with outer diameter `0.12 m`.
- After the jar height update, the robot pedestal top was lowered to `0.20 m` and the material table top was lowered to `0.50 m`.
- The ceramic disc material color was changed to terracotta orange.
