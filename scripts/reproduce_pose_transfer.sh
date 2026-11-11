#!/usr/bin/env bash
# Offline only. Uses existing environments/assets; never installs or operates devices.
set -euo pipefail
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ $# != 1 ]]; then
  printf 'Usage: bash %s NEW_OUTPUT_DIRECTORY\n' "$0" >&2
  exit 2
fi
task_output=$(realpath -m -- "$1")
if [[ -e "$task_output" ]]; then
  printf 'Output already exists; preserving it: %s\n' "$task_output" >&2
  exit 2
fi
cd -- "$task_root"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
conda run --no-capture-output -n winejar-vision python scripts/run_fixed_camera_calibration_study.py --output "$task_output/calibration"
conda run --no-capture-output -n winejar-stereo python scripts/run_body_relocalization_study.py --output "$task_output/body"
conda run --no-capture-output -n winejar-stereo python scripts/run_body_partial_view_ablation.py --source "$task_output/body"
conda run --no-capture-output -n winejar-vision python scripts/run_charuco_eye_to_hand_check.py --output "$task_output/charuco"
conda run --no-capture-output -n winejar-vision python scripts/refine_charuco_eye_to_hand_check.py --source "$task_output/charuco"
conda run --no-capture-output -n winejar-vision python scripts/combine_body_and_handeye_errors.py --bundle "$task_output"
conda run --no-capture-output -n winejar-vision python scripts/summarize_pose_transfer_study.py --bundle "$task_output" --calibration "$task_output/calibration"
conda run --no-capture-output -n winejar-stereo python scripts/export_body_transfer_viewer.py --bundle "$task_output"
DISPLAY=${DISPLAY:-:0} conda run --no-capture-output -n winejar-stereo python scripts/verify_pose_transfer_study.py --bundle "$task_output" --calibration "$task_output/calibration"
printf 'Offline reproduction complete: %s\n' "$task_output"
