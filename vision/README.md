# WineJar Vision

This folder is the independent vision environment for GroundingDINO + SAM2.

The intended data flow is:

1. RealSense or MuJoCo saves an RGB-D frame folder.
2. This vision environment runs open-vocabulary detection and segmentation.
3. It writes a JSON result plus mask images.
4. Robot control or simulation code consumes the result.

This keeps the robot-control environment small and reduces dependency conflicts with
PyTorch, CUDA, Transformers, and SAM2.

## Install

Create the environment:

```bash
conda env create -f vision/environment.yml
conda activate winejar-vision
```

Verify PyTorch and CUDA:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

Install SAM2:

```bash
pip install -r vision/requirements-sam2.txt
```

## First offline test

Capture one RGB-D frame in the control environment:

```bash
conda activate winejar-xarm-sdk
python xarm_sdk_control/scripts/13_capture_realsense_frame.py
```

Then run detection and segmentation in the vision environment:

```bash
conda activate winejar-vision
python vision/scripts/01_offline_grounded_sam2.py \
  --frame-dir xarm_sdk_control/data/vision_frames/<timestamp> \
  --text-prompt "ceramic disk. bamboo leaf. paper label." \
  --box-threshold 0.25 \
  --text-threshold 0.20
```

The script writes:

- `vision_result.json`
- `annotated.png`
- `mask_*.png`

## Persistent inference service

For robot cycle-time tests, start the vision service before starting production:

```bash
conda run --no-capture-output -n winejar-vision \
  env HF_HUB_OFFLINE=1 \
  python -u vision/scripts/03_grounded_sam2_service.py
```

Wait for the JSON `"status": "ready"` message. The service keeps GroundingDINO
and SAM2 in GPU memory and listens only on the local Unix socket:

```text
/tmp/winejar_grounded_sam2.sock
```

The robot task automatically uses this service when the socket exists. For a
cycle-time test, pass `--require-vision-service` so a missing service fails the
task instead of silently falling back to cold one-shot inference.

The service never connects to or commands the robot. It reads a saved RGB-D
frame directory and writes the same `vision_result.json`, `annotated.png`, and
`mask_*.png` artifacts as the offline script.

Model loading is startup overhead and is not production cycle time. The robot
task records the productive cycle from the first RGB-D capture after reset to
the final safe return in its `result.json` timeline.

Stop the service with `Ctrl+C` after experiments. Its Unix socket is removed on
clean shutdown.
