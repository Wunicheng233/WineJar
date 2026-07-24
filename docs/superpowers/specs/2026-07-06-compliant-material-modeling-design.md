# Compliant Material Modeling Design

## Goal

Design a practical path for modeling the wine jar loading materials as compliant, bendable objects instead of pure rigid bodies. The immediate purpose is to make the later tying animation physically believable: bamboo leaves, cropped lotus leaf, and cropped white paper should be able to bend, lift, sag, and gather when the tie-gun head descends and the tie band tightens.

This design does not attempt full botanical or paper material fidelity. It prioritizes a stable MuJoCo animation that communicates the process without hiding impossible rigid-body behavior.

## Current Project Context

The current production scene is `mujoco_xarm6/scene/scene_winejar_production_demo.xml`.

Current material state:

- `preloaded_lotus_leaf` is a rigid cropped mesh fixed under `station_wine_jar`.
- `preloaded_white_paper` is a rigid cropped mesh fixed under `station_wine_jar`.
- `staged_bamboo_leaf_bottom` and `staged_bamboo_leaf_top` are rigid mesh bodies with freejoints.
- The two staged bamboo leaves are pickable by the vacuum arm, but each leaf currently moves as one rigid body.
- The tie-gun model is a simplified vertical ring mechanism with a loaded band on the guide and a lock clip.

The rigid model is acceptable for layout and early robot reach validation, but it cannot show material gathering under the tie-gun.

## MuJoCo Capability Check

MuJoCo 3.x supports deformable objects through `flex` and `flexcomp`. Official documentation describes `flex` as deformable elements that can generate contact, constraint, and passive forces; `flexcomp` is the modeling-time macro that automatically builds the underlying flex structure. 1D flexes represent line/cable-like elements, 2D flexes represent triangle/shell-like surfaces, and 3D flexes represent volumetric soft bodies.

References:

- MuJoCo Modeling: https://mujoco.readthedocs.io/en/stable/modeling.html
- MuJoCo XML Reference, `flexcomp`: https://mujoco.readthedocs.io/en/stable/XMLreference.html

Decision: use MuJoCo `flex/flexcomp` later, but do not make it the first implementation step. Flex contact, thin-sheet tuning, and solver stability can consume a lot of time before the production workflow itself is proven.

## Recommended Strategy

Use a staged approach:

1. **Segmented compliant approximation first**
   - Model each flexible sheet as a chain or grid of small rigid segments.
   - Connect adjacent segments with soft equality constraints or joints.
   - This is not true cloth, but it is stable, inspectable, and good enough for showing bend/gather behavior.

2. **Upgrade selected materials to 2D `flexcomp` later**
   - Once the animation path and tie-gun timing are clear, replace the most important material with a true 2D flex model.
   - Start with one bamboo leaf, not all materials.

3. **Only then consider a deformable tie band**
   - The tie band can later be represented as 1D flex or a segmented cable-like chain.
   - For now, the tie band remains a visual/kinematic element so that material modeling can be debugged separately.

## Material Priority

### Bamboo Leaves

Priority: highest.

Reason: the two bamboo leaves are placed by the vacuum arm and then need to be gathered by the tie-gun. Their behavior most affects whether the final tying animation feels credible.

First implementation:

- Replace each rigid bamboo leaf mesh body with 9 to 13 short overlapping leaf segments along the long axis.
- Each segment is a thin ellipsoid/box/capsule-like rigid body.
- Adjacent segments are connected with bend-limited hinge joints or soft equality constraints.
- The center or one end can be used as the suction attachment target.
- The visual silhouette should still resemble a 42 cm long, 12.5 cm max-width bamboo leaf.

Suggested segmentation:

- Number of segments: `11`
- Segment length: about `0.045 m`
- Segment overlap: small overlap so the leaf does not visually split.
- Width profile: narrow at both ends, widest around the middle.
- Thickness: `0.002 m` to `0.004 m`
- Mass per segment: total leaf mass around `0.01 kg`, distributed over segments.

### Cropped Lotus Leaf

Priority: medium.

Reason: it is already on the jar mouth before the bamboo leaves are placed. It should be able to fold or sag under compression, but it does not need to be manipulated by the vacuum arm.

First implementation:

- Keep the current cropped polygon mesh as a visual reference.
- Add a segmented simplified compliant layer above or replacing it when tying animation begins.
- Use a coarse 2D patch with 5 to 9 pieces.

### Cropped White Paper

Priority: medium.

Reason: paper is thinner and more pliable than the ceramic disc, but for the demo it can share the same coarse segmented-sheet approximation as the cropped lotus leaf.

First implementation:

- Use a 3x3 or 3x2 segmented patch.
- Make it stiffer than the lotus leaf and bamboo leaves.
- Keep it thin and light.

### Ceramic Disc And Metal Weight

Priority: rigid.

Reason: these should remain rigid. The ceramic disc and metal weight are intended to press or stabilize the flexible materials rather than deform.

## First Implementation Scope

The first implementation should only replace the two staged bamboo leaves with segmented compliant leaves.

In scope:

- Add a generator for segmented bamboo leaf MJCF fragments.
- Replace `staged_bamboo_leaf_bottom` and `staged_bamboo_leaf_top` with compliant segmented bodies.
- Preserve existing pick site names:
  - `staged_bamboo_leaf_bottom_pick_site`
  - `staged_bamboo_leaf_top_pick_site`
- Preserve compatibility with the vacuum weld names:
  - `left_suction_weld_leaf_bottom`
  - `left_suction_weld_leaf_top`
- Update inspection to verify each segmented leaf has multiple moving segment bodies.
- Add a simple bend/sag validation: after stepping simulation, segments should not behave as one perfectly rigid body.

Out of scope for first implementation:

- Replacing preloaded lotus and white paper.
- Implementing true `flexcomp`.
- Coupling the tie band to material deformation.
- Final production animation sequence.

## Architecture

Use generated MJCF snippets instead of hand-writing dozens of segment bodies into the main scene.

Proposed files:

```text
mujoco_xarm6/
  tools/
    generate_compliant_material_models.py
  scene/
    compliant_bamboo_leaf_pair.xml
  test/
    inspect_compliant_material_models.py
```

`generate_compliant_material_models.py` responsibilities:

- Generate an include file containing two segmented bamboo leaves.
- Define segment bodies, joints/constraints, geoms, sites, and weld-compatible parent bodies.
- Keep dimensions deterministic and readable.

`compliant_bamboo_leaf_pair.xml` responsibilities:

- Contain only material bodies and their equality constraints.
- Avoid robot, jar, conveyor, or camera definitions.
- Be included by `scene_winejar_production_demo.xml`.

`inspect_compliant_material_models.py` responsibilities:

- Load the full production scene.
- Verify segment count, material support height, named sites, and non-rigid compliance indicators.

## Naming Design

Each leaf should have a stable root body:

- `staged_bamboo_leaf_bottom`
- `staged_bamboo_leaf_top`

Each root contains segment bodies:

- `staged_bamboo_leaf_bottom_seg_00`
- `staged_bamboo_leaf_bottom_seg_01`
- ...
- `staged_bamboo_leaf_top_seg_00`
- `staged_bamboo_leaf_top_seg_01`
- ...

The pick site remains on or near the middle segment:

- `staged_bamboo_leaf_bottom_pick_site`
- `staged_bamboo_leaf_top_pick_site`

The root body should be the body referenced by the suction weld at first. If this proves too rigid, the weld can later target the middle segment instead, but the site names should remain stable.

## Compliance Model Options

### Option A: Hinged Segment Chain

Each segment is a rigid body connected to its neighbor with a hinge joint. The hinge axes are arranged so the leaf can bend upward/downward along its length.

Pros:

- Simple and inspectable.
- Easier to control than true cloth.
- Good visual bending along the long axis.

Cons:

- Does not capture twisting or cross-width folding well.
- Can look segmented if the visual geoms do not overlap.

### Option B: Soft Equality Chain

Each segment is a free body, and adjacent segments are connected by soft weld/equality constraints.

Pros:

- Can bend and twist more naturally.
- Softness can be tuned with `solref`/`solimp`.

Cons:

- More likely to jitter.
- Harder to keep the leaf from stretching apart.

### Option C: 2D Flexcomp Sheet

Use MuJoCo `flexcomp` with a 2D grid or custom shape.

Pros:

- Closest to true thin-sheet behavior.
- Contact is distributed over deformable elements.

Cons:

- Higher tuning burden.
- Harder to keep stable while also interacting with robot tools, jar, and tie-band visuals.
- Less transparent for early debugging.

Recommendation: start with Option A for bamboo leaves. Keep Option C as a later upgrade once the animation sequence is proven.

## Validation

Static validation:

- The production scene loads.
- Both bamboo leaves exist as segmented compliant assemblies.
- Each bamboo leaf has at least 9 segment bodies.
- Pick site names still exist.
- Suction weld names still exist.
- Segment bottoms are supported by the table with no visible floating gap.

Dynamic validation:

- Step simulation for a short period with gravity.
- Verify segments do not explode or drift apart.
- Apply a small downward probe or scripted force to the middle of a leaf.
- Verify the middle segment moves more than the root/end segments, indicating bending rather than rigid-body motion.
- Verify the leaf does not pass through the table immediately.

Manual validation:

- Open the production scene in MuJoCo viewer.
- Confirm the bamboo leaves still look like two stacked leaves.
- Confirm there are no obvious gaps between segments.
- Confirm a small bend/sag is visible or can be produced with a scripted probe.

## Risks And Decisions

Risk: segmented leaves may look too artificial.

Decision: use overlapping, tapered segment geoms and keep the first implementation focused on top-down animation. If segmentation is too visible, add a visual mesh/skin layer later while keeping segmented bodies for physics.

Risk: vacuum pickup could attach to the wrong segment or make the leaf fold badly.

Decision: keep the original pick site names and initially attach to the root/middle segment. Tune after visual inspection.

Risk: true flex might be necessary for convincing tying.

Decision: prove the workflow with segmented compliance first. Move to `flexcomp` only if the segmented approach cannot produce acceptable gathering behavior.

Risk: solver instability.

Decision: start with low segment count and conservative masses/damping. Increase fidelity only after the scene remains stable.

## Acceptance Criteria

The first compliant-material iteration is acceptable when:

- The two staged bamboo leaves are no longer single rigid mesh bodies.
- The scene loads with both robots, jar, conveyor, tie-gun, and segmented leaves.
- The leaves remain supported on the material table without floating.
- The existing vacuum pickup site names still exist.
- A validation script can demonstrate non-rigid bending or sag.
- The model remains stable enough to open in the viewer and inspect interactively.
