# 优化前理论核查，2026-10-22

本目录不含新定位精度结果，也没有相机位置搜索或传感器噪声实验。
[分析报告](../../../../docs/research/neck_localization/optimization_analysis.md)给出来源、模型、导数、局限与方案取舍。

[linearization_checks.json](linearization_checks.json)是局部导数的有限差分核对及预先指定的圆弧退化反例。
单位径向方差和已知角度证据只是解析假设；特征值/条件数不能解释为RealSense毫米精度。
种子20261022；复现入口为scripts/analyze_neck_observability.py，输出路径必须不存在。
