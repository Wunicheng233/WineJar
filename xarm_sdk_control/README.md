# xArm SDK Control

This folder contains the Python SDK control path for the wine jar loading project.

Safety rule for the first stage:

- Read robot state first.
- Do not enable motion yet.
- Do not clear errors yet.
- Do not move the arm yet.
- Do not control the gripper yet.

Environment:

```bash
conda env create -f xarm_sdk_control/environment.yml
conda activate winejar-xarm-sdk
```

Robot IP is stored in:

```text
xarm_sdk_control/config/robot.yaml
```
