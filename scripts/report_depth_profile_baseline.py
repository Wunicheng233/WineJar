#!/usr/bin/env python3
"""Archive the small ideal-depth loop and draw evaluation-only diagnostic plots."""
import argparse
import gzip
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def table(summary, title, evidence):
    lines = [f"### {title}", "", "|方法|视图|N/已解|成功/拒绝/错误接受|圆心中位/P95/最大 mm|法向中位/P95/最大 °|直径中位/P95/最大 mm|计算中位/P95/最大 ms|",
             "|---|---|---|---|---|---|---|---|"]
    for r in summary:
        if r["evidence"] != evidence:
            continue
        metrics = []
        for key in ("center_mm", "normal_deg", "diameter_mm", "runtime_ms"):
            metrics.append("/".join("—" if r[key+"_"+suffix] is None else f"{r[key+'_'+suffix]:.3f}"
                                    for suffix in ("median", "p95", "max")))
        rates = "/".join(f"{100*r[key]:.0f}%" for key in
                         ("success_rate", "rejection_rate", "false_accept_rate"))
        lines.append(f"|{r['method']}|{r['factor'].split('/')[-1]}|{r['n']}/{r['solved_n']}|{rates}|"+"|".join(metrics)+"|")
    lines += ["", "法向/切向分解（相对真值法向；法向分量取绝对值）：", "",
              "|方法|视图|法向中位/P95/最大 mm|切向中位/P95/最大 mm|",
              "|---|---|---|---|"]
    for r in summary:
        if r["evidence"] != evidence:
            continue
        metrics = []
        for key in ("normal_component_mm", "tangent_mm"):
            metrics.append("/".join("—" if r[key+"_"+suffix] is None else f"{r[key+'_'+suffix]:.3f}"
                                    for suffix in ("median", "p95", "max")))
        lines.append(f"|{r['method']}|{r['factor'].split('/')[-1]}|"+"|".join(metrics)+"|")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--model-scenes", type=Path, required=True)
    parser.add_argument("--heldout-scenes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tables = []
    for name, source in [("development", args.development), ("heldout", args.heldout),
                         ("exterior_diagnostic", args.diagnostic)]:
        folder = args.output / name
        folder.mkdir()
        for path in source.glob("*.json"):
            if path.name == "records.json":
                (folder / "records.json.gz").write_bytes(gzip.compress(path.read_bytes(), mtime=0))
            else:
                shutil.copy2(path, folder / path.name)
        evidence = "exterior_diagnostic" if name == "exterior_diagnostic" else "rgb"
        tables.append(table(json.loads((source / "summary.json").read_text()), name, evidence))
    (args.output / "tables.md").write_text("\n\n".join(tables)+"\n")
    (args.output / "README.md").write_text("""# 短颈酒坛：模型与首轮理想深度闭环

4个开发模型、4个预先冻结估计器后生成的留出模型；固定工位，一/二/三视图。
RGB前景提取和整坛GT掩码分列；另有事后外表面GT掩码诊断。均为准确相机、无传感器噪声。
估计器以口沿平面方向作为局部颈轴代理，再从观测半径剖面寻找目标范围，不能推广为任意歪颈均可解。

- [全部分位数、成功/拒绝/错误接受及法向/切向统计](tables.md)
- [留出案例误差图](heldout_errors.png)
- [最差单视图案例的剖面诊断](single_view_failure.png)
- 一键脚本另生成[错误接受的四步诊断图](case_walkthroughs/broad_reverse_tilt_cam_a/steps.png)及[拒绝案例诊断图](case_walkthroughs/tilted_flat_cam_a_b/steps.png)。
- [正常坛型](review/reference_shallow_overview.png)
- [局部歪颈](review/tilted_neck_overview.png)与[近景](review/tilted_neck_neck.png)
- `development/`、`heldout/`、`exterior_diagnostic/`：压缩逐次记录、汇总、参数和来源散列。
- `development_scenes/`、`heldout_scenes/`：配置、清单和仅用于评价的目标真值。
- `source_snapshot/`：本轮计算/生成代码快照，正式维护入口仍在项目模块与scripts中。
- 运行独立诊断后，`oracle_diagnostics.json`记录真值中心/轴替换及范围定义差异；这些成绩不是普通估计结果。

中心4 mm源于历史目标；法向2°、直径2 mm只是分析阈值。分位数包含错误接受的已解案例，
未解拒绝不伪造误差；每个条件只有4个案例，P95/最大值不是总体保证。计算耗时不含采集、磁盘读取和渲染。
没有估计不确定性；平面残差不能代表绝对精度。未经抗噪、真实非圆/非平口沿或实机有效性验证。
留出案例已用于事后错误诊断，不能再用于下一版方法的未见最终评价。
""")
    for name, source in [("development_scenes", args.model_scenes), ("heldout_scenes", args.heldout_scenes)]:
        folder = args.output / name
        folder.mkdir()
        for filename in ("effective_config.json", "manifest.json"):
            shutil.copy2(source / filename, folder / filename)
        evaluation = folder / "evaluation_only"
        evaluation.mkdir()
        for target in source.glob("*/evaluation_only/target.json"):
            shutil.copy2(target, evaluation / (target.parent.parent.name+"_target.json"))

    review = args.output / "review"
    review.mkdir()
    for case, image in [("reference_shallow", "overview"), ("reference_shallow", "front"),
                        ("tilted_neck", "overview"), ("tilted_neck", "neck")]:
        shutil.copy2(args.model_scenes / case / "review" / f"{image}.png", review / f"{case}_{image}.png")
    records = json.loads((args.heldout / "records.json").read_text())
    methods = [("rim_guided_profile", "Mouth-axis proxy", "#16759c"),
               ("vertical_profile", "Fixed vertical axis", "#bd632f")]
    sets = ["cam_a", "cam_a,cam_b", "all"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    for ax, metric, label, threshold in zip(axes,
            ["center_mm", "normal_deg", "diameter_mm"],
            ["Center error (mm)", "Normal angle (deg)", "Diameter error (mm)"], [4, 2, 2]):
        for j, (method, title, color) in enumerate(methods):
            for i, camera_set in enumerate(sets):
                rows = [r for r in records if r["evidence"] == "rgb" and r["method"] == method
                        and r["camera_set"] == camera_set and r[metric] is not None]
                x = np.full(len(rows), i+(j-.5)*.22)
                ax.scatter(x, [r[metric] for r in rows], s=32, color=color,
                           marker="o" if j == 0 else "x", label=title if i == 0 else None)
        ax.axhline(threshold, linestyle="--", color="grey", linewidth=1)
        ax.set(xticks=[0, 1, 2], xticklabels=["1 view", "2 views", "3 views"], ylabel=label)
        ax.grid(axis="y", alpha=.20)
    axes[0].legend(fontsize=8)
    fig.suptitle("Held-out smoke test: 4 jars, RGB extraction + ideal depth, true cameras\n"
                 "All solved attempts shown; unsolved rejections are reported in the tables", fontsize=10)
    fig.savefig(args.output / "heldout_errors.png", dpi=160)
    plt.close(fig)

    # Show the worst solved one-view proxy case, including its failure.
    candidates = [r for r in records if r["evidence"] == "rgb" and r["method"] == "rim_guided_profile"
                  and r["camera_set"] == "cam_a" and r["center_mm"] is not None]
    worst = max(candidates, key=lambda r: r["center_mm"])
    truth = json.loads((args.heldout_scenes / worst["case"] / "evaluation_only/target.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
    for ax, camera_set in zip(axes, ["cam_a", "all"]):
        row = next(r for r in records if r["case"] == worst["case"] and r["evidence"] == "rgb"
                   and r["method"] == "rim_guided_profile" and r["camera_set"] == camera_set)
        diag = row["diagnostics"]
        s = np.array(diag["axial_from_observed_mouth_m"])*1000
        ax.plot(s, np.array(diag["radius_m"])*1000, ".", alpha=.5, label="Observed radial quantiles")
        ax.plot(s, np.array(diag["smoothed_radius_m"])*1000, label="Smoothed profile")
        interval = np.array(diag["near_min_interval_from_mouth_m"])*1000
        ax.axvspan(*interval, color="#16759c", alpha=.15, label="Selected near-min band")
        true_axial = (np.array(truth["center_m"])-diag["observed_mouth_center_m"]) @ np.array(row["estimate"]["normal"])*1000
        ax.axvline(true_axial, color="#bd632f", linestyle="--", label="GT center projected on estimated axis")
        ax.axhline(truth["radius_m"]*1000, color="#bd632f", linestyle=":")
        ax.set(xlabel="Axial position from estimated mouth (mm)", ylabel="Radius (mm)",
               title=f"{camera_set}: center error {row['center_mm']:.2f} mm")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=6.5)
    fig.suptitle(worst["case"]+" — evaluation-only diagnosis; GT never supplied to estimator", fontsize=10)
    fig.savefig(args.output / "single_view_failure.png", dpi=160)
    plt.close(fig)
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir()
    root = Path(__file__).resolve().parents[1]
    for name in ("depth_profile.py", "jar_model.py", "scene_preparation.py", "geometry.py", "evaluation.py"):
        shutil.copy2(root / "neck_localization" / name, snapshot / name)
    shutil.copy2(Path(__file__), snapshot / Path(__file__).name)
    for name in ("run_depth_profile_baseline.py", "prepare_neck_scene.py", "export_neck_surface_masks.py",
                 "diagnose_depth_profile_geometry.py", "run_short_neck_stage1.sh", "visualize_depth_profile_case.py"):
        shutil.copy2(root / "scripts" / name, snapshot / name)
    print(str(args.output.resolve()))


if __name__ == "__main__":
    main()
