# 固定轴向深度偏差第一步，2026-10-23

12个已看模型颈长>=6 mm场景，9档假设偏差，108次；不是新未见测试或设备实测。
带宽3 mm，原算法与阈值未调。完整数据在runs/depth_bias_stage1_20261023_v1/，原资产链接保持依赖。

- summary.csv / cases.csv：全部/常规/压力汇总与逐例指标。
- depth_bias_sensitivity.png：正确输出/拒绝/错误接受、尺寸敏感度与单机偏差诊断。
- paired_diameter_slope.json：11个双侧有解案例的直径变化斜率（不是通用布局定律）。
- common_p2_wrong_diameter.png：实际窗口，nominal_002被接受但直径误差3.106 mm。
- common_p2_actual_pointcloud.png：同案例实际带偏差点云，未用理想点云替代。
- input_viewer_validation.json：108组深度/RGB/K/T与94组拟合点逐点核查。
- limited_reproduction.json：12个零对照与3类代表重算，未完整第二次跑108条。
- source_snapshot/：本阶段执行、报告、扰动模块、查看器与计划快照。

机理、官方资料、参数假设与查看/复现命令：
docs/research/neck_localization/depth_bias_stage1.md。
