# 5 mm环带几何审计，2026-10-23

正式结果位于 `verified/`。根目录为首轮审计，`reproduction/`为独立完整复跑。
48个既有冻结预测复用，不是新估计器未见测试；两种轴向边界均为假设。
无噪声、无厚度/接触模型、无输送/机器人动作。

- `verified/records.json.gz`：原V2预测字段不变，追加band与evaluation-only band_evaluation。
- `summary.json/csv`：5种宽度×2种边界×3组，共30组。
- `cases.csv`：5 mm逐例两边界评估。
- `interval_diagnostic.png`：1常规+6代理越界例，真值轴对齐，仅事后解释。
- `band_*.png/json`：3例实际MuJoCo窗口截图及输入散列。
- `viewer_validation.json`：B切换后近景/3路图像实际像素变化、GT访问与RGB未变检查。
- `reproduction_check.json`：48条记录与表/图复现，原13个冻结源码未变。
- `source_snapshot/`：本阶段代码快照。

完整定义、结果、限制及查看/复现命令：
`docs/research/neck_localization/finite_band_stage1.md`。
