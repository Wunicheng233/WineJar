# xArm SDK Control

This folder contains the Python SDK control path for the wine jar loading project.

Safety rules:

- Read robot state first.
- Never clear `C1` or another safety stop without physically inspecting the cell.
- When `X > 300 mm`, keep TCP `Z <= 300 mm` below the overhead camera.
- Keep exact stops for suction contact and release even when transit blending is enabled.
- Stay at the emergency stop during real-robot experiments.

Environment:

```bash
conda env create -f xarm_sdk_control/environment.yml
conda activate winejar-xarm-sdk
```

Robot IP is stored in:

```text
xarm_sdk_control/config/robot.yaml
```

## Modules

Task scripts are thin orchestration over shared importable modules:

- `robot_motion.py` — connection, preparation, emergency-stop checks, safe reset motion
- `visual_targeting.py` — calibration loading, RGB-D scan pipeline, detection selection
- `cross_place_route.py` — two-shell cycle route construction and task geometry
- `cross_place_motion.py` — motion-profile validation, joint-aware planning, route execution
- `trajectory_speed_planner.py` — joint speed/acceleration scaling and controller preflight
- `pick_test_workflow.py` — attended single-object pick-test profiles and execution
- `vacuum_control.py` / `end_effector_config.py` / `positioning_bias.py` — end-effector I/O, tool config, measured bias
- `realsense_capture.py` — aligned RGB-D capture session
- `geometry.py` / `cli_utils.py` / `run_output.py` / `task_timing.py` — shared pose math, argparse plumbing, run reporting, timeline

## Attended single-object pick test

`20_visual_pick_test.py` is an isolated test workflow for visually targeting,
lifting, and returning one object under direct operator observation. Its
default behavior does not infer whether pickup succeeded and does not change
the validated two-shell production task. The default mode captures one RGB-D
frame, detects the target, prints the guarded route, and commands neither robot
motion nor controller I/O:

```bash
python xarm_sdk_control/scripts/20_visual_pick_test.py \
  --target bamboo \
  --allow-validated-clipped-target

python xarm_sdk_control/scripts/20_visual_pick_test.py \
  --target ceramic
```

During a real test, vacuum remains on while the object is lifted vertically and
held briefly for the operator. The object is then lowered to its original
source, released, and the robot returns home. There is no post-lift camera
capture, model inference, automatic retry, or software pickup verdict. The
ceramic profile uses 21 mm spring compression and lifts only 25 mm vertically.

After the dry-run route and workspace have been inspected, the attended test
form is:

```bash
python xarm_sdk_control/scripts/20_visual_pick_test.py \
  --target ceramic \
  --run \
  --confirm-no-pressure-sensor \
  --require-vision-service \
  --min-x-mm 180 --max-x-mm 520 \
  --min-y-mm -180 --max-y-mm 180 \
  --min-z-mm -85 --max-z-mm 500
```

An optional closed-loop experiment captures another RGB-D frame while the
object is lifted. It classifies the target as `confirmed`, `not_confirmed`, or
`inconclusive`; always returns the tool vertically to the original source
before venting; and captures a fresh targeting frame before every retry. The
following controlled experiment deliberately leaves the pump off on attempt 1,
then replans and performs a normal attempt 2:

```bash
python xarm_sdk_control/scripts/20_visual_pick_test.py \
  --target ceramic \
  --run \
  --verify-pick \
  --simulate-first-pick-failure \
  --max-attempts 2 \
  --confirm-no-pressure-sensor \
  --require-vision-service \
  --compression-mm 21 \
  --min-x-mm 180 --max-x-mm 520 \
  --min-y-mm -180 --max-y-mm 180 \
  --min-z-mm -85 --max-z-mm 500
```

The closed-loop command exits with status `2` if all attempts finish without a
`confirmed` visual pickup. Robot and vacuum cleanup still complete before that
status is returned.

## Two-shell performance experiment

Start the persistent vision service as documented in `vision/README.md`, release
the physical emergency stop, enable the robot in UFactory, and stage two bamboo
shells plus the ceramic disk. Then run:

```bash
conda run --no-capture-output -n winejar-xarm-sdk \
  python xarm_sdk_control/scripts/19_visual_cross_place_two_leaves.py \
  --run \
  --confirm-no-pressure-sensor \
  --require-vision-service \
  --allow-validated-clipped-leaf \
  --record-global-camera \
  --compression-mm 13 \
  --min-score 0.35 \
  --min-x-mm 180 --max-x-mm 520 \
  --min-y-mm -180 --max-y-mm 180 \
  --min-z-mm -85 --max-z-mm 500
```

The stability-aware performance profile defaults to:

- empty transit speed: `300 mm/s`
- loaded transit speed: `220 mm/s`
- vertical speed: `200 mm/s`
- loaded vertical speed: `120 mm/s`
- precision speed: `60 mm/s`
- contact speed: `20 mm/s`
- empty/loaded/precision acceleration: `500/250/200 mm/s^2`
- camera-safe transport clearance: `100 mm`
- approach-to-contact blend radius: `10 mm`
- vacuum settle: `0.9 s`
- release settle: `0.3 s`

The task software rejects values above the profile-specific validated limits.
These conservative task limits are intentionally much lower than the xArm6
controller limits reported by SDK 1.18.4 (`1000 mm/s`, `50000 mm/s^2`,
`180 deg/s`, and about `1145 deg/s^2`). Cartesian commands still pass through
the controller's inverse kinematics and joint-limit enforcement.

Before each leaf cycle, the task samples the exact Cartesian route with
continuous inverse kinematics and estimates joint speed and acceleration.
Defaults reserve 60% joint-speed margin and 60% joint-acceleration margin.

Every vision-generated route is replayed exactly once through the xArm no-motion
path checker before suction or motion. A controller rejection aborts the cycle;
the runtime task does not search for a faster profile or rescale the route.
Loaded lift, rotation, lateral transfer, and placement are exact-stop segments
and are never blended together. Only the empty approach uses safe blending.

For diagnosing a repeatable leaf-2 `only_check` C24 that does not latch as a
controller error, `--skip-second-controller-preflight` skips only that controller
check. Continuous IK, joint-speed/acceleration limiting, workspace limits, the
camera keepout, and the first-cycle controller preflight remain active. Treat
this as an attended experiment option, not a production default.

The reset TCP is at `Z=300 mm`. Each cycle moves directly from its safe start
pose to the source approach. Fine contact and the loaded lift are slower. The
shell keeps its pickup yaw until it has cleared the stacked material; placement
yaw is then interpolated during the exact-stop horizontal transfer. Release occurs at the target layer,
followed by a slow 25 mm retreat. After leaf 1, the robot remains at the ceramic
retract pose while the second image is processed; only leaf 2 returns home.

The RealSense pipeline starts and warms before productive-cycle timing, then
stays open for both scans. Each run also measures the first leaf's SAM2 mask
principal axis before and after placement so commanded TCP yaw can be compared
with the leaf's observed physical rotation.

The second scan accepts lower-confidence bamboo-shell detections because the
remaining shell can have substantially lower contrast after the top shell is
removed. This does not broaden the pickup area: the candidate must still remain
within 80 mm of the source stack measured in scan 1, so the already placed shell
cannot be selected.

Every `result.json` contains a monotonic event timeline and reports productive
cycle time separately from the initial safety reset.

Global-camera recording uses a `640x480 @ 15 fps` RGB-D stream by default to
reduce USB bandwidth while retaining enough temporal detail for review. The
robot motion profile is unchanged. Use `--recording-fps 30` only after the
camera cable and USB controller have passed a sustained recording test.

### Closed-loop pick verification and retry

`19_visual_cross_place_two_leaves.py` can classify every lift from a fresh
RGB-D frame and recover + replan + retry when the pick clearly failed. The
mechanism is opt-in and leaves the validated baseline unchanged when off:

```bash
# production command with verification enabled
python xarm_sdk_control/scripts/19_visual_cross_place_two_leaves.py \
  --run --confirm-no-pressure-sensor \
  --require-vision-service \
  --verify-pick \
  --max-pick-attempts 3 \
  --allow-validated-clipped-leaf \
  --compression-mm 13 --min-score 0.35 \
  --min-x-mm 180 --max-x-mm 520 \
  --min-y-mm -180 --max-y-mm 180 \
  --min-z-mm -85 --max-z-mm 500

# controlled experiment: leaf-1 attempt 1 contacts normally but keeps the
# pump OFF, so the failure-detection + recovery + retry path is exercised
python xarm_sdk_control/scripts/19_visual_cross_place_two_leaves.py \
  --run --confirm-no-pressure-sensor \
  --require-vision-service \
  --verify-pick \
  --simulate-first-pick-failure --max-pick-attempts 2 \
  --allow-validated-clipped-leaf \
  --compression-mm 13 --min-score 0.35 \
  --min-x-mm 180 --max-x-mm 520 \
  --min-y-mm -180 --max-y-mm 180 \
  --min-z-mm -85 --max-z-mm 500
```

Verification policy per lift:

- `confirmed` (leaf seen at the lifted tool position) — continue placement.
- `not_confirmed` (leaf still near its original source height) — descend back
  to the source release height, vent any carried leaf onto the stack, retreat,
  return to the camera-clear reset pose, capture a fresh targeting frame, and
  retry. Every replanned route passes the full joint-aware planner and the
  controller no-motion preflight again.
- `inconclusive` — currently treated as success and recorded, because many
  verified successful picks read as inconclusive with the present vision
  check; the recorded reasons will guide tightening this policy later.

Details:

- `--max-pick-attempts` (1-3, default 3) bounds attempts per leaf. A retry
  targeting scan only accepts a candidate within `--max-second-source-shift-mm`
  of the original source stack, so a leaf that fell elsewhere (or the already
  placed leaf) can never be selected.
- Verification score is adaptive: leaf 1 uses `--min-score`, leaf 2 uses
  `--second-scan-min-score` (the remaining shell has lower contrast after the
  top shell is removed).
- Every attempt's targeting and verification frames, the full verification
  dict, retry decision, and recovery flag are recorded per attempt under
  `data/visual_cross_place/<run>/leaf_<N>_attempt_<NN>/` and in `result.json`
  (`pick_verification` top-level block, `cycles[].pick_attempts[]`,
  `final_pick_verification_status`). Timeline events use
  `leaf_<N>.attempt_<NN>.verification.*` / `.recovery.*` / `.targeting.*`.
- If all attempts for a leaf fail, the task vents at the source, stops,
  writes `failure.json` (with `pick_verification_failed`) plus a
  `result.json` with `"status": "failed"`, and exits with code `2`.

Known v1 limitations (recorded for later analysis):

- Stacked-leaf ambiguity: after a real leaf-1 success the remaining shell
  sits ~3 mm lower at the source and can read as `not_confirmed`, causing a
  conservative vent-back-and-retry (safe, only costs time).
- The suction cup can occlude the carried leaf from the overhead camera,
  producing `inconclusive` (currently treated as success).
- If a leaf drops far from the stack, the retry legitimately picks the
  remaining stacked leaf under the failed leaf's role; the dropped leaf must
  be recovered by the operator.

Generated RGB-D frames, masks, previews, and run logs live under
`xarm_sdk_control/data/`. They are disposable experiment artifacts and are not
required by the control software. Keep `calibration/` and `config/`: those files
contain the active camera transform, end-effector geometry, I/O mapping, and
measured positioning correction.
