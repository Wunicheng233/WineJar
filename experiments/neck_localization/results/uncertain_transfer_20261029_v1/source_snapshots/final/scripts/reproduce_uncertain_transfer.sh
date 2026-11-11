#!/usr/bin/env bash
# Offline confirmation loop only; no hardware, new training or package installs.
set -euo pipefail
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
[[ $# == 1 ]] || { printf 'Usage: bash %s NEW_OUTPUT_DIRECTORY\n' "$0" >&2; exit 2; }
task_output=$(realpath -m -- "$1")
[[ ! -e "$task_output" ]] || { printf 'Preserving existing output: %s\n' "$task_output" >&2; exit 2; }
cd -- "$task_root"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
stereo=(conda run --no-capture-output -n winejar-stereo python)
vision=(conda run --no-capture-output -n winejar-vision python)
"${stereo[@]}" scripts/prepare_uncertain_handeye.py --config experiments/neck_localization/configs/uncertain_transfer_20261029_v2.json --output "$task_output/confirmation_inputs"
"${stereo[@]}" scripts/prepare_nominal_poe.py --inputs "$task_output/confirmation_inputs"
"${vision[@]}" scripts/run_uncertain_handeye.py --inputs "$task_output/confirmation_inputs" --output "$task_output/confirmation_baselines" --phase test
"${vision[@]}" scripts/run_poe_camera_calibration.py --inputs "$task_output/confirmation_inputs" --output "$task_output/confirmation_poe" --phase test
"${vision[@]}" scripts/run_poe_camera_calibration.py --inputs "$task_output/confirmation_inputs" --output "$task_output/confirmation_poe_block" --phase test --block-robust
for method in baselines poe poe_block; do
  "${stereo[@]}" scripts/evaluate_uncertain_handeye.py --inputs "$task_output/confirmation_inputs" --fits "$task_output/confirmation_$method" --output "$task_output/confirmation_evaluation_$method"
done
"${vision[@]}" scripts/run_simple_calibration_controls.py --inputs "$task_output/confirmation_inputs" --output "$task_output/simplicity_controls"
"${stereo[@]}" scripts/evaluate_uncertain_handeye.py --inputs "$task_output/confirmation_inputs" --fits "$task_output/simplicity_controls" --output "$task_output/simplicity_evaluation"
"${vision[@]}" scripts/check_rendered_robot_board.py --inputs "$task_output/confirmation_inputs" --output "$task_output/rendered_board" --stage texture
"${stereo[@]}" scripts/check_rendered_robot_board.py --inputs "$task_output/confirmation_inputs" --output "$task_output/rendered_board" --stage render
"${vision[@]}" scripts/fit_partial_rendered_board.py --inputs "$task_output/confirmation_inputs" --rendered "$task_output/rendered_board" --output "$task_output/partial_rendered_board"
"${stereo[@]}" scripts/evaluate_uncertain_handeye.py --inputs "$task_output/confirmation_inputs" --fits "$task_output/partial_rendered_board" --output "$task_output/partial_rendered_evaluation"
"${stereo[@]}" scripts/run_bounded_transport_study.py --source experiments/neck_localization/results/pose_transfer_20261028_v3 --output "$task_output/bounded_transport"
"${stereo[@]}" scripts/export_calibration_arrival.py --bundle "$task_output" --output "$task_output/arrival_viewer_v2"
"${stereo[@]}" scripts/export_calibration_arrival.py --bundle "$task_output" --output "$task_output/arrival_simple_controls" --include-simple-controls
"${vision[@]}" scripts/summarize_uncertain_transfer.py --bundle "$task_output"
"${stereo[@]}" scripts/verify_uncertain_transfer.py --bundle "$task_output"
"${stereo[@]}" scripts/verify_uncertain_transfer.py --bundle "$task_output" --viewer-name arrival_simple_controls --audit-output "$task_output/simple_GL_audit"
"${stereo[@]}" scripts/audit_calibration_transfer.py --bundle "$task_output" --output "$task_output/final_audit"
printf 'Offline confirmation reproduction complete: %s\n' "$task_output"
