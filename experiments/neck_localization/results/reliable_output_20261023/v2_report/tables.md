# 目标范围验证备用解：第二次独立评价

首批新测试暴露核心跨度限制：v1三视图46/48、旧法47/48；原结果保持冻结。
首批48例现仅作为v2开发，另锁新种子32常规+16压力；本轮只评价当前三相机主工况。
对照frozen_region、adaptive_validated（冻结v1）、task_validated_output（v2）。
v2先运行v1；必要时用训练点得到备用轴，以完整目标范围、留出点、分相机残差、信息和候选一致性验证。
不要求备用解具有3mm内部轴核心；目标范围至少1.5mm，残差0.4mm/留出均值0.2mm限保持。
留出点不能校验共同标定偏差；尚无5mm带宽/输送/机器人验证。

## 新测试

成功=接受且圆心≤4 mm/法向≤2°/直径≤2 mm，后两项仍为分析阈值；拒绝和错误输出计失败。
误差含已求出的拒绝候选，未求出不填零；误差/耗时为中位数 / P95 / 最大。

|组|视图|方法|成功/拒绝/错误接受/总数|圆心mm|法向°|直径mm|耗时ms|>1000ms|
|---|---|---|---|---|---|---|---|---|
|nominal|all|adaptive_validated|32/0/0/32|0.836 / 1.210 / 1.297|0.027 / 0.117 / 0.152|0.020 / 0.039 / 0.040|331.910 / 414.454 / 480.041|0|
|nominal|all|frozen_region|32/0/0/32|0.832 / 1.205 / 1.306|0.021 / 0.117 / 0.263|0.021 / 0.039 / 0.041|383.651 / 489.793 / 601.642|0|
|nominal|all|task_validated_output|32/0/0/32|0.836 / 1.210 / 1.297|0.027 / 0.117 / 0.152|0.020 / 0.039 / 0.040|332.486 / 414.750 / 475.079|0|
|stress|all|adaptive_validated|12/4/0/16|0.675 / 1.161 / 1.303|0.021 / 0.056 / 0.060|0.012 / 0.065 / 0.071|586.164 / 786.539 / 802.521|0|
|stress|all|frozen_region|7/9/0/16|0.494 / 0.858 / 0.869|0.011 / 0.033 / 0.035|0.010 / 0.070 / 0.074|319.126 / 446.910 / 449.867|0|
|stress|all|task_validated_output|16/0/0/16|0.490 / 1.109 / 1.303|0.015 / 0.061 / 0.064|0.014 / 0.063 / 0.071|623.500 / 842.880 / 889.569|0|

1000 ms仅为离线说明线，真实工位预算未给定；不含采集、传输、机器人执行。

|组|视图|方法|XYZ最大绝对误差mm|沿法向/切向最大mm|
|---|---|---|---|---|
|nominal|all|adaptive_validated|0.107 / 0.117 / 1.290|1.297 / 0.016|
|nominal|all|frozen_region|0.106 / 0.116 / 1.299|1.306 / 0.017|
|nominal|all|task_validated_output|0.107 / 0.117 / 1.290|1.297 / 0.016|
|stress|all|adaptive_validated|0.191 / 0.184 / 1.288|1.303 / 0.004|
|stress|all|frozen_region|0.099 / 0.142 / 0.858|0.869 / 0.004|
|stress|all|task_validated_output|0.191 / 0.184 / 1.288|1.303 / 0.058|

拒绝/错误接受：

- stress_001 / all / frozen_region: 拒绝 axis_core_touches_candidate_boundary
- stress_002 / all / frozen_region: 拒绝 feedback_not_converged
- stress_003 / all / frozen_region: 拒绝 near_minimum_interval_touches_observed_search_boundary
- stress_006 / all / frozen_region: 拒绝 axis_core_has_insufficient_extent
- stress_006 / all / adaptive_validated: 拒绝 continuous_core_insufficient_extent
- stress_007 / all / frozen_region: 拒绝 axis_core_touches_candidate_boundary
- stress_008 / all / adaptive_validated: 拒绝 continuous_core_insufficient_extent
- stress_011 / all / frozen_region: 拒绝 near_minimum_interval_touches_observed_search_boundary
- stress_012 / all / frozen_region: 拒绝 axis_core_has_insufficient_extent
- stress_012 / all / adaptive_validated: 拒绝 continuous_core_insufficient_extent
- stress_013 / all / frozen_region: 拒绝 near_minimum_interval_touches_observed_search_boundary
- stress_014 / all / frozen_region: 拒绝 axis_core_has_insufficient_extent
- stress_014 / all / adaptive_validated: 拒绝 continuous_core_insufficient_extent

## 开发（非未见测试）

成功=接受且圆心≤4 mm/法向≤2°/直径≤2 mm，后两项仍为分析阈值；拒绝和错误输出计失败。
误差含已求出的拒绝候选，未求出不填零；误差/耗时为中位数 / P95 / 最大。

|组|视图|方法|成功/拒绝/错误接受/总数|圆心mm|法向°|直径mm|耗时ms|>1000ms|
|---|---|---|---|---|---|---|---|---|
|nominal|all|adaptive_validated|32/0/0/32|0.373 / 1.210 / 1.393|0.015 / 0.076 / 0.287|0.012 / 0.036 / 0.056|271.610 / 412.798 / 483.658|0|
|nominal|all|frozen_region|32/0/0/32|0.370 / 1.216 / 1.395|0.012 / 0.069 / 0.304|0.012 / 0.036 / 0.054|329.388 / 483.804 / 543.228|0|
|nominal|all|task_validated_output|32/0/0/32|0.373 / 1.210 / 1.393|0.015 / 0.076 / 0.287|0.012 / 0.036 / 0.056|273.440 / 410.938 / 478.487|0|
|stress|all|adaptive_validated|14/2/0/16|0.581 / 0.915 / 0.925|0.020 / 0.053 / 0.065|0.016 / 0.058 / 0.073|322.080 / 692.837 / 704.712|0|
|stress|all|frozen_region|15/1/0/16|0.543 / 0.912 / 0.921|0.009 / 0.053 / 0.065|0.016 / 0.057 / 0.075|340.348 / 442.569 / 496.342|0|
|stress|all|task_validated_output|16/0/0/16|0.554 / 0.914 / 0.925|0.010 / 0.051 / 0.065|0.016 / 0.055 / 0.073|324.441 / 886.643 / 933.643|0|

1000 ms仅为离线说明线，真实工位预算未给定；不含采集、传输、机器人执行。

|组|视图|方法|XYZ最大绝对误差mm|沿法向/切向最大mm|
|---|---|---|---|---|
|nominal|all|adaptive_validated|0.084 / 0.104 / 1.393|1.392 / 0.024|
|nominal|all|frozen_region|0.083 / 0.104 / 1.395|1.395 / 0.027|
|nominal|all|task_validated_output|0.084 / 0.104 / 1.393|1.392 / 0.024|
|stress|all|adaptive_validated|0.154 / 0.163 / 0.913|0.925 / 0.009|
|stress|all|frozen_region|0.108 / 0.162 / 0.909|0.921 / 0.013|
|stress|all|task_validated_output|0.154 / 0.163 / 0.913|0.925 / 0.058|

拒绝/错误接受：

- stress_004 / all / adaptive_validated: 拒绝 continuous_core_insufficient_extent
- stress_008 / all / adaptive_validated: 拒绝 continuous_core_insufficient_extent
- stress_009 / all / frozen_region: 拒绝 near_minimum_interval_touches_observed_search_boundary

## 扩张与备用路径（新测试）

|组|视图|28mm接受|42mm接受|56mm接受|备用固定权重接受|无输出或拒绝|
|---|---|---|---|---|---|
|nominal|cam_a|0|0|0|0|0|
|nominal|cam_a,cam_b|0|0|0|0|0|
|nominal|all|32|0|0|0|0|
|stress|cam_a|0|0|0|0|0|
|stress|cam_a,cam_b|0|0|0|0|0|
|stress|all|7|5|0|0|4|
