# Tie Gun End Effector Design

## Goal

Create a MuJoCo-ready design for the right-side xArm6 automatic tie-gun end effector used in the wine jar production animation. The end effector should be visually understandable and mechanically plausible for a demo: it descends vertically over the wine jar mouth, closes two semicircular jaws into a ring, then shows a tie band shrinking inward around the jar neck/material bundle.

This design is for animation and workflow demonstration first. It should avoid fake object teleportation, but it does not need to simulate the real flexible tie band, pneumatic internals, cutting blade, or cable feed mechanics in full physical detail.

## Current Project Context

The production demo scene is `mujoco_xarm6/scene/scene_winejar_production_demo.xml`.

Current layout:

- Conveyor carries the wine jar along the production line.
- Left xArm6 has a vacuum suction end effector.
- Right xArm6 currently has only a bare flange/TCP marker.
- The right arm will later perform the tying operation after the left arm places bamboo leaves and a metal weight.

The user provided a short product demonstration video. The observed mechanism has a front guide/head with two curved arms that close into a ring-like shape. The project-specific operation should use a vertical top-down approach rather than a side approach.

## Scope

In scope:

- Add a simplified tie-gun model to the right xArm6 end flange.
- Model the end effector with a connector, compact tool body, and two semicircular jaws.
- Represent open and closed jaw states.
- Add a visual tie band that can shrink inward during animation.
- Add reference sites for motion planning and animation alignment.
- Keep the model credible enough that viewers understand the process.

Out of scope:

- Detailed CAD-accurate product geometry.
- True flexible tie-band simulation.
- Internal pneumatic actuator dynamics.
- Real cable-tie feeding and cutting mechanics.
- Real force interaction with bamboo leaves or jar neck.

## Operating Assumption

The tie gun is mounted for vertical use:

- The right arm moves the tie gun above the wine jar mouth.
- The ring plane is horizontal.
- The ring central axis is vertical and aligned with the wine jar centerline.
- The open jaws descend around the material bundle and jar neck.
- The jaws close into a full circular guide.
- The tie band contracts inward until it reaches the neck/material bundle.

This is preferred over a side approach because the production scene already places the right arm beside the conveyor, and a top-down motion is easier to understand, easier to animate, and less likely to collide with the jar belly.

## Geometry Design

Use simple MuJoCo primitives first.

End effector components:

| Component | Suggested Shape | Purpose |
| --- | --- | --- |
| `tie_gun_connector` | short cylinder | attaches to right xArm6 flange |
| `tie_gun_body` | compact box/cylinder combination | visual pneumatic head body |
| `tie_gun_air_port` | small cylinder | suggests pneumatic connection |
| `tie_gun_left_jaw` | semicircle made from capsule segments | left half of closing ring |
| `tie_gun_right_jaw` | semicircle made from capsule segments | right half of closing ring |
| `tie_gun_loaded_band_*` | thin capsule ring segments seated near the jaw guide | tie band loaded in the guide before tightening |
| `tie_gun_band_lock_clip` | small box at the jaw connection | lock/retention point that holds the band until pressure is reached |
| `tie_gun_band_tail` | short capsule segment | visible free tail near the lock clip |

The jaws should look like two curved guide claws rather than two straight gripper fingers. Each jaw can be built from several capsule segments arranged along a semicircle. The full closed ring should have an outer radius around `0.10 m` to `0.12 m`, large enough to clear the wine jar mouth and loose material bundle. In the default static model, the tie band should sit on or very near the jaw guide, with a visible lock clip at one jaw connection. It should not appear as a floating final inner loop. During the later tightening animation, most of the band may detach from the guide and shrink inward, while the lock-clip portion remains held until pressure/release is represented.

Initial dimensions:

- Connector radius: `0.030 m`
- Connector height: `0.030 m`
- Tool body footprint: about `0.10 m x 0.08 m`
- Open jaw ring radius: about `0.105 m`
- Jaw tube radius: about `0.006 m`
- Loaded tie band visual tube radius: about `0.0025 m`
- Loaded tie band guide gap: no more than about `0.010 m` from the jaw guide radius

## Kinematic Design

The first version can use animation-controlled geometry rather than full rigid-body dynamics.

Recommended representation:

- Keep the tie-gun body rigidly attached to `right_bare_end_effector`.
- Add a `tie_gun_center_site` at the center of the jaw ring.
- Add a `tie_gun_approach_site` above the ring center.
- Add a `tie_gun_neck_target_site` reference for alignment with `neck_tie_target_site`.
- Use two jaw state parameters in the later animation script:
  - `jaw_open`: jaws separated, with a visible front gap.
  - `jaw_closed`: jaws form a full ring.
- Use a staged tie-band visual state in the later animation script:
  - `loaded`: band is seated in the jaw guide and held at the lock clip.
  - `tightening`: most of the band detaches from the guide and shrinks inward, while the lock-clip section remains held.
  - `released`: pressure is reached, the lock-clip section releases, and the tightened band remains on the jar neck/material bundle.

For the initial scene model, do not show the final tightened inner band by default. The final tightened band belongs to the animation state after the shrink/release sequence.

## Animation Sequence

The right-arm tying sequence should be:

1. **Approach above jar**
   - Move `tie_gun_center_site` above `jar_mouth_center`.
   - Keep enough clearance above the bamboo leaves and metal weight.

2. **Vertical insertion**
   - Move downward along the jar centerline.
   - Jaws remain open.
   - Ring plane remains horizontal.

3. **Jaw close**
   - Two semicircular jaws close into a circular guide around the material bundle.
   - This should be visible as an explicit mechanism action.

4. **Tie shrink**
   - Most of the visual tie band detaches from the jaw guide and contracts toward the neck radius.
   - The lock-clip section remains held on the guide during contraction.
   - The band stops around the jar neck/material bundle when pressure is reached.

5. **Retract**
   - The lock-clip section releases.
   - Tie gun rises vertically.
   - For demonstration, the tightened tie band may remain around the jar neck as final-state geometry.

## MuJoCo Integration

Preferred file structure:

```text
mujoco_xarm6/
  scene/
    xarm6_right_tie_gun.xml
  tools/
    generate_tie_gun_model.py
  test/
    inspect_tie_gun_model.py
```

Integration strategy:

- Generate or hand-write a prefixed right-arm include that replaces the current bare flange with the tie-gun model.
- Keep names prefixed with `right_` to avoid collisions.
- Preserve `right_link_tcp` or replace it with a clearer TCP site such as `right_tie_gun_center_site`.
- The production scene should include the right-arm tie-gun model only after the static model loads and passes inspection.

The model should keep the right arm mounted on its independent base and should not move the conveyor, jar, or left-arm setup unless workspace validation requires it.

## Validation

Programmatic validation:

- Load the generated tie-gun model alone.
- Load the full production scene with the tie-gun attached.
- Verify required bodies/sites exist:
  - `right_tie_gun_connector`
  - `right_tie_gun_body`
  - `right_tie_gun_center_site`
  - `right_tie_gun_approach_site`
  - `right_tie_gun_neck_target_site`
- Verify the tie-gun ring center can align with `neck_tie_target_site` in the production scene.
- Verify the right arm home pose does not place the tie-gun inside the conveyor, jar, or floor.

Manual validation:

- Open the production scene in MuJoCo viewer.
- Confirm the tie-gun model is visibly attached to the right xArm6 flange.
- Confirm the ring is horizontal and points downward toward the jar mouth.
- Confirm the open jaws are recognizable and not already closed in the default state.
- Confirm the ring radius looks large enough to pass around the jar mouth/material bundle.

## Risks And Decisions

Risk: The visual tie band could look like a fake floating ring.

Decision: In the default model, keep the band seated on the jaw guide with a lock clip and tail. Only create the inward shrinking/final neck band as an animation state, not as a default static inner ring.

Risk: A fully physical cable tie would be complex and may distract from the production animation.

Decision: Use scripted visual radius changes first. This is acceptable for an explanatory animation as long as the robot motion and jaw sequence remain physically plausible.

Risk: The right arm may not reach a top-down tying pose after adding the tool length.

Decision: Validate right-arm reach before creating the final animation. If needed, adjust the right base position slightly or reduce tool length.

## Acceptance Criteria

The design is acceptable when:

- The user can recognize the end effector as a vertical automatic tie-gun head.
- The model has a connector, body, two semicircular jaws, and a tie-band visual.
- The jaws default to an open state.
- The animation plan supports loaded, close, shrink, pressure/release, and final tightened states.
- The right xArm6 can plausibly bring the ring center over the wine jar neck.
- No hidden fixed objects are used to fake a completed tying operation.
