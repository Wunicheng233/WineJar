#!/usr/bin/env bash
# Reproduce this bounded offline loop, with no hardware or installation calls.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
output_dir="${1:?Usage: bash scripts/run_short_neck_stage1.sh NEW_OUTPUT_DIRECTORY}"
if [[ -e "$output_dir" ]]; then
    printf 'Refusing to overwrite: %s\n' "$output_dir" >&2
    exit 1
fi
mkdir -p -- "$output_dir"
output_dir="$(cd -- "$output_dir" && pwd)"
sim_environment="${WINEJAR_SIM_ENV:-embodied-sim}"
export MUJOCO_GL=egl MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1
sim_python() { conda run --no-capture-output -n "$sim_environment" python "$@"; }

sim_python -m pytest -q tests/neck_localization/test_jar_model.py \
    tests/neck_localization/test_scene_preparation.py tests/neck_localization/test_depth_profile.py \
    > "$output_dir/checks.log"
sim_python scripts/prepare_neck_scene.py \
    --config experiments/neck_localization/configs/preparation_v2.json \
    --output "$output_dir/development_scenes" > "$output_dir/development_scene.log"
sim_python scripts/prepare_neck_scene.py \
    --config experiments/neck_localization/configs/short_neck_heldout_v1.json \
    --output "$output_dir/heldout_scenes" > "$output_dir/heldout_scene.log"
sim_python scripts/run_depth_profile_baseline.py --scenes "$output_dir/development_scenes" \
    --output "$output_dir/development" --camera-sets cam_a cam_a,cam_b all \
    > "$output_dir/development_cli.log"
sim_python scripts/run_depth_profile_baseline.py --scenes "$output_dir/heldout_scenes" \
    --output "$output_dir/heldout" --split heldout --camera-sets cam_a cam_a,cam_b all \
    > "$output_dir/heldout_cli.log"
sim_python scripts/export_neck_surface_masks.py --scenes "$output_dir/heldout_scenes" \
    > "$output_dir/exterior_masks.log"
sim_python scripts/run_depth_profile_baseline.py --scenes "$output_dir/heldout_scenes" \
    --output "$output_dir/diagnostic" --split diagnostic --camera-sets cam_a cam_a,cam_b all \
    --evidence exterior_diagnostic > "$output_dir/exterior_cli.log"
sim_python scripts/report_depth_profile_baseline.py --development "$output_dir/development" \
    --heldout "$output_dir/heldout" --diagnostic "$output_dir/diagnostic" \
    --model-scenes "$output_dir/development_scenes" --heldout-scenes "$output_dir/heldout_scenes" \
    --output "$output_dir/report" > "$output_dir/report.log"
sim_python scripts/diagnose_depth_profile_geometry.py \
    --scenes "$output_dir/development_scenes" "$output_dir/heldout_scenes" \
    --records "$output_dir/heldout/records.json" --output "$output_dir/report/oracle_diagnostics.json" \
    > "$output_dir/oracle_diagnostic.log"
sim_python scripts/visualize_depth_profile_case.py --scenes "$output_dir/heldout_scenes" \
    --records "$output_dir/heldout/records.json" --case broad_reverse_tilt --camera-set cam_a \
    --output "$output_dir/report/case_walkthroughs/broad_reverse_tilt_cam_a" \
    > "$output_dir/false_accept_visualization.log"
sim_python scripts/visualize_depth_profile_case.py --scenes "$output_dir/heldout_scenes" \
    --records "$output_dir/heldout/records.json" --case tilted_flat --camera-set cam_a,cam_b \
    --output "$output_dir/report/case_walkthroughs/tilted_flat_cam_a_b" \
    > "$output_dir/rejection_visualization.log"
printf 'Completed offline loop. Report: %s/report/README.md\n' "$output_dir"
