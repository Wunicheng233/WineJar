#!/usr/bin/env bash
# Rebuild v2 development/fresh test; first-stage negative results stay archived.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
output_dir="${1:?Usage: bash scripts/run_task_validated_stage2.sh NEW_OUTPUT_DIRECTORY}"
if [[ -e "$output_dir" ]]; then
    printf 'Refusing to overwrite: %s\n' "$output_dir" >&2
    exit 1
fi
mkdir -p -- "$output_dir"
output_dir="$(cd -- "$output_dir" && pwd)"
sim_environment="${WINEJAR_SIM_ENV:-embodied-sim}"
export OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl MPLBACKEND=Agg
sim_python() { conda run --no-capture-output -n "$sim_environment" python "$@"; }
sim_python -m pytest -q tests/neck_localization/test_task_validated_output.py \
  tests/neck_localization/test_reliable_output.py > "$output_dir/checks.log"
# The first v1 test is now development, never labelled unseen for v2.
sim_python scripts/prepare_neck_scene.py --config experiments/neck_localization/configs/reliable_output_test_v1.json \
  --output "$output_dir/development_scenes" > "$output_dir/development_scenes.log"
sim_python scripts/run_task_validated_output.py --scenes "$output_dir/development_scenes" \
  --output "$output_dir/development" --write-freeze "$output_dir/freeze.json" > "$output_dir/development.log"
sim_python scripts/prepare_neck_scene.py --config experiments/neck_localization/configs/reliable_task_fallback_test_v2.json \
  --output "$output_dir/test_scenes" > "$output_dir/test_scenes.log"
sim_python scripts/run_task_validated_output.py --scenes "$output_dir/test_scenes" --split test \
  --freeze "$output_dir/freeze.json" --output "$output_dir/test" > "$output_dir/test.log"
sim_python scripts/report_reliable_output.py --development "$output_dir/development" \
  --test "$output_dir/test" --primary-v2 --output "$output_dir/report"
printf 'Three-camera task-validation comparison complete: %s/report/tables.md\n' "$output_dir"
