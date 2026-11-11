# 夜间偏差补偿研究验证

正式报告：docs/research/neck_localization/overnight_uncertainty_results_20261024.md。

- inputs_and_sources.json：开发216+首次测试256+第二测试80，共552组补偿后深度、RGB和K/T逐项核对。
- 550组最终拟合点逐点对应保存的最终拟合输入；首次测试2条没有几何候选，未伪造点集。
- full_reproduction.json：最新8场景重渲染、40原法+80补偿记录全量独立复现，除耗时外相同。
- tests_summary.txt：本机实际测试结果摘要。
- source_manifest.json / source_snapshot/：本阶段模块、CLI、配置和测试快照。

补偿后深度不是原始相机测量。原始有偏输入保留在各raw运行目录；源码不读取真实偏差/尺寸/目标来求解。
所有偏差值、先验与协方差均为明确假设，不是实测设备性能，不宣称实机精度或完整置信度。
