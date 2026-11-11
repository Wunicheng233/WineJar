#!/usr/bin/env bash
# Fully offline reproduction; refuses existing output. No camera-layout search.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
output_dir="${1:?Usage: bash scripts/run_support_feedback_stage1.sh NEW_OUTPUT_DIRECTORY}"
if [[ -e "$output_dir" ]]; then
    printf 'Refusing to overwrite: %s\n' "$output_dir" >&2
    exit 1
fi
mkdir -p -- "$output_dir"
output_dir="$(cd -- "$output_dir" && pwd)"
sim_environment="${WINEJAR_SIM_ENV:-embodied-sim}"
export OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl MPLBACKEND=Agg
sim_python() { conda run --no-capture-output -n "$sim_environment" python "$@"; }
sim_python -m pytest -q tests/neck_localization/test_support_feedback.py \
  tests/neck_localization/test_joint_profile.py tests/neck_localization/test_depth_profile.py \
  tests/neck_localization/test_jar_model.py tests/neck_localization/test_scene_preparation.py > "$output_dir/checks.log"
sim_python scripts/prepare_neck_scene.py --config experiments/neck_localization/configs/preparation_v2.json \
  --output "$output_dir/development_a" > "$output_dir/development_a.log"
sim_python scripts/prepare_neck_scene.py --config experiments/neck_localization/configs/short_neck_heldout_v1.json \
  --output "$output_dir/development_b" > "$output_dir/development_b.log"
# Previously inspected six test cases are now development only.
sim_python scripts/prepare_neck_scene.py --config experiments/neck_localization/configs/joint_profile_test_v1.json \
  --output "$output_dir/development_c" > "$output_dir/development_c.log"
sim_python scripts/run_support_feedback.py --scenes "$output_dir/development_a" "$output_dir/development_b" \
  "$output_dir/development_c" --output "$output_dir/development" > "$output_dir/development.log"
sim_python scripts/freeze_support_feedback.py --development "$output_dir/development" \
  --test-config experiments/neck_localization/configs/support_feedback_test_v1.json \
  --output "$output_dir/freeze.json"
sim_python scripts/prepare_neck_scene.py --config experiments/neck_localization/configs/support_feedback_test_v1.json \
  --output "$output_dir/test_scenes" > "$output_dir/test_scenes.log"
sim_python scripts/run_support_feedback.py --scenes "$output_dir/test_scenes" --split test \
  --freeze "$output_dir/freeze.json" --output "$output_dir/test" > "$output_dir/test.log"
sim_python scripts/report_support_feedback.py --development "$output_dir/development" \
  --test "$output_dir/test" --output "$output_dir/report"
printf 'Offline comparison complete: %s/report/tables.md\n' "$output_dir"
