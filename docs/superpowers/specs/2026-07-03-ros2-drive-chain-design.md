# ROS2 Drive Chain Design

## Goal

Define a low-risk path for driving the xArm6 with ROS2, starting from fake/simulated control and only later moving to real hardware. The immediate goal is to avoid environment mismatch, package-version conflicts, and unsafe direct motion on the physical arm.

## What MoveIt2 Is

MoveIt2 is not just a single planner. It is a ROS2 manipulation framework around motion planning and execution.

In this project it would provide:

- Robot model loading from URDF/SRDF.
- Joint limits, planning groups, end-effector frames, and kinematics configuration.
- Collision-aware motion planning through planning pipelines such as OMPL.
- Integration with RViz for interactive target testing.
- Trajectory execution through ROS2 controllers when connected to either fake hardware or real hardware.

MoveIt2 does not replace the xArm driver. The xArm ROS2 driver is still responsible for communicating with the robot controller. MoveIt2 produces trajectories; xArm ROS2 executes them through its driver/controller stack.

## Current Local Situation

The local archive `xarm6/xarm_ros-master.zip` is primarily the ROS1 package set. Its README targets Ubuntu 16.04 with ROS Kinetic/Melodic, so it should not be used as the ROS2 main path.

The ROS2 path should use the official `xArm-Developer/xarm_ros2` repository and the branch matching the installed ROS2 distribution.

## Recommended Environment Choice

Two viable combinations:

| Option | OS | ROS2 | xarm_ros2 branch | Risk |
| --- | --- | --- | --- | --- |
| Stable-first | Ubuntu 22.04 | Humble | `humble` | Lower integration risk, more mature ecosystem |
| Matches project proposal | Ubuntu 24.04 | Jazzy | `jazzy` | Matches Ubuntu 24.04, but newer stack has more compatibility risk |

Recommendation: use Ubuntu 22.04 + ROS2 Humble if hardware control reliability is the priority. Use Ubuntu 24.04 + ROS2 Jazzy only if the project requirement strongly depends on Ubuntu 24.04.

Rationale:

- ROS2 Jazzy is the correct ROS2 release for Ubuntu 24.04.
- MoveIt2 provides binary packages for Jazzy on Ubuntu 24.04.
- xArm ROS2 lists Ubuntu 24.04 + Jazzy as a supported/tested environment.
- However, xArm ROS2 issue history shows Jazzy-related problems around `move_group` and `ros2_control`; these may be manageable but should not be ignored.

## Drive Chain Architecture

Use this layered chain:

```text
Task Layer
  wine jar loading state machine
  target poses for material grasp/release

Motion Layer
  MoveIt2 planning scene
  xArm6 planning group
  collision-aware joint trajectory planning

Driver Layer
  xarm_ros2
  xarm_api services
  xarm_controller / ros2_control trajectory execution

Hardware Layer
  xArm6 controller
  gripper
```

MuJoCo remains useful for task layout and geometry validation, but it should not be treated as the ROS2 driver. The ROS2 driver path uses xArm URDF/SRDF and MoveIt2 configuration.

## Phased Verification Plan

### Phase 0: Environment Audit

Before installing anything:

- Confirm OS version with `lsb_release -a`.
- Confirm whether ROS2 is already installed.
- Confirm whether `/opt/ros/<distro>/setup.bash` exists.
- Confirm Python version and apt sources are not mixed across ROS distros.
- Decide one distro only: Humble or Jazzy.

Failure condition:

- Multiple ROS2 distros sourced in the same shell.
- Ubuntu version does not match chosen ROS2 distro.
- Conda environment active while building ROS2 workspace.

### Phase 1: Build xarm_ros2 Without Hardware

Create a clean workspace outside the current MuJoCo project:

```text
~/winejar_ros2_ws
```

Clone:

```bash
git clone https://github.com/xArm-Developer/xarm_ros2.git --recursive -b <ros_distro>
```

Run `rosdep install`, then `colcon build`.

Pass condition:

- Workspace builds.
- `source install/setup.bash` works.
- `ros2 pkg list | grep xarm` shows xArm packages.

### Phase 2: Fake Robot / RViz / MoveIt2

Launch xArm6 model first:

```bash
ros2 launch xarm_description xarm6_rviz_display.launch.py add_gripper:=true
```

Then launch fake MoveIt:

```bash
ros2 launch xarm_moveit_config xarm6_moveit_fake.launch.py add_gripper:=true
```

Pass condition:

- RViz opens.
- xArm6 with gripper appears.
- Planning group can plan to a nearby pose.
- No real robot is connected.

### Phase 3: Real Robot API Only

Connect robot over Ethernet. Start only the API driver:

```bash
ros2 launch xarm_api xarm6_driver.launch.py robot_ip:=<robot_ip>
```

Run low-risk checks:

```bash
ros2 topic echo /xarm/robot_states
ros2 topic echo /joint_states
```

Then enable and set mode/state only after confirming no active errors:

```bash
ros2 service call /xarm/motion_enable xarm_msgs/srv/SetInt16ById "{id: 8, data: 1}"
ros2 service call /xarm/set_mode xarm_msgs/srv/SetInt16 "{data: 0}"
ros2 service call /xarm/set_state xarm_msgs/srv/SetInt16 "{data: 0}"
```

Pass condition:

- Robot state topics publish.
- Services return success.
- Robot remains stationary unless a deliberate command is issued.

### Phase 4: Real Robot MoveIt2

Only after Phase 3 passes, launch real MoveIt:

```bash
ros2 launch xarm_moveit_config xarm6_moveit_realmove.launch.py robot_ip:=<robot_ip> add_gripper:=true
```

First motion should be a very small joint-space move with reduced velocity/acceleration scaling.

Pass condition:

- RViz planning succeeds.
- Execution succeeds.
- Motion is slow, bounded, and collision-free.

### Phase 5: Wine Jar Task Layer

Implement our task node after the ROS2 drive chain is proven.

The node should not directly stream raw joint commands initially. It should call MoveIt2 for planned motion or call xArm API for controlled primitive motions.

## Main Failure Risks

### Environment mismatch

Highest-risk cases:

- Ubuntu 24.04 with ROS2 Humble from unofficial sources.
- Ubuntu 22.04 with ROS2 Jazzy.
- Sourcing both Humble and Jazzy in one terminal.
- Building xarm_ros2 branch `humble` under Jazzy or branch `jazzy` under Humble.
- Building ROS2 workspace while conda is active.

Mitigation:

- Use a dedicated shell script that sources exactly one ROS2 distro.
- Keep MuJoCo conda environment separate from ROS2 workspace.
- Do not install ROS2 Python packages with pip.

### Gazebo mismatch

Gazebo should be avoided in the first ROS2 drive milestone. We do not need it to validate xArm driver + MoveIt2 fake control. Gazebo introduces simulator-version dependencies that are separate from MoveIt2 and xArm API.

### MoveIt2 planning succeeds but hardware execution fails

This can happen when controller mode/state is wrong, network latency is high, or ros2_control controller state is not active.

Mitigation:

- Validate `xarm_api` first.
- Use Ethernet direct connection where possible.
- Keep first motion small and slow.

### Gripper integration

Arm motion and gripper control should be validated separately. Do not block arm-drive validation on gripper behavior.

## Decision

Do not connect the real robot yet.

Next engineering step:

1. Choose target environment: Humble/Ubuntu 22.04 or Jazzy/Ubuntu 24.04.
2. Create a clean ROS2 workspace.
3. Build official `xarm_ros2`.
4. Validate fake xArm6 MoveIt2.
5. Only then test real robot API.

## References

- xArm ROS2 official repository: https://github.com/xArm-Developer/xarm_ros2
- ROS2 Jazzy Ubuntu installation: https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html
- MoveIt2 documentation: https://moveit.picknik.ai/
- MoveIt2 binary install information: https://moveit.ai/install-moveit2/binary/
