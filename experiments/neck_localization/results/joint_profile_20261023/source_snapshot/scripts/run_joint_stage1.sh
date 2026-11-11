#!/usr/bin/env bash
# Bounded fixed-rig joint-estimation comparison. Offline, no installation/hardware.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
output_dir="${1:?Usage: bash scripts/run_joint_stage1.sh NEW_OUTPUT_DIRECTORY}"
if [[ -e "$output_dir" ]]; then
    printf 'Refusing to overwrite: %s\n' "$output_dir" >&2
    exit 1
fi
mkdir -p -- "$output_dir"
output_dir="$(cd -- "$output_dir" && pwd)"
sim_environment="${WINEJAR_SIM_ENV:-embodied-sim}"
export MUJOCO_GL=egl MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1
sim_python() { conda run --no-capture-output -n "$sim_environment" python "$@"; }
sim_python -m pytest -q tests/neck_localization/test_joint_profile.py \
    tests/neck_localization/test_depth_profile.py tests/neck_localization/test_jar_model.py \
    tests/neck_localization/test_scene_preparation.py > "$output_dir/checks.log"
sim_python scripts/prepare_neck_scene.py \
    --config experiments/neck_localization/configs/preparation_v2.json \
    --output "$output_dir/development_a" > "$output_dir/development_a.log"
# The old, already-inspected holdout is now ONLY development/regression data.
sim_python scripts/prepare_neck_scene.py \
    --config experiments/neck_localization/configs/short_neck_heldout_v1.json \
    --output "$output_dir/development_b" > "$output_dir/development_b.log"
sim_python scripts/run_joint_profile.py --scenes "$output_dir/development_a" "$output_dir/development_b" \
    --output "$output_dir/development" > "$output_dir/development.log"
sim_python scripts/freeze_joint_profile.py \
    --test-config experiments/neck_localization/configs/joint_profile_test_v1.json \
    --development "$output_dir/development" --output "$output_dir/freeze.json" > "$output_dir/freeze.log"
sim_python scripts/prepare_neck_scene.py \
    --config experiments/neck_localization/configs/joint_profile_test_v1.json \
    --output "$output_dir/test_scenes" > "$output_dir/test_scenes.log"
sim_python scripts/run_joint_profile.py --scenes "$output_dir/test_scenes" --split test \
    --freeze "$output_dir/freeze.json" --output "$output_dir/test" > "$output_dir/test.log"
sim_python scripts/report_joint_profile.py --development "$output_dir/development" \
    --test "$output_dir/test" --freeze "$output_dir/freeze.json" \
    --output "$output_dir/report" > "$output_dir/report.log"
sim_python scripts/diagnose_joint_profile_locality.py --scenes "$output_dir/development_a" \
    --output "$output_dir/report/locality_ablation.json" > "$output_dir/locality_ablation.log"
printf 'Completed offline fixed-rig comparison. Report: %s/report/tables.md\n' "$output_dir"
