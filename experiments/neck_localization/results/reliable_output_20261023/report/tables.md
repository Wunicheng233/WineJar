# 正确输出率改进：理想观测阶段

开发36个已看随机坛型；新测试48个在实现前锁定，开发后冻结；同一圆截面模型家族、固定相机、准确K/T。
三方法：frozen_region（旧区域）、stable_local（连续核心+阻尼+留出验证）、adaptive_validated（再加范围扩张/验证备用解）。
新两方法25%确定性留出点不参与轴/曲线目标，但初始ROI/口沿和支撑检查仍使用全部观测；不是独立相机或独立标定验证。
更改了支持模型和验证结构，没有靠真值放行、删除所有质量检查或调高误差限。
尚不包含5 mm带宽、相机误差、输送或机械臂端到端任务。

## 新测试

成功=接受且圆心≤4 mm/法向≤2°/直径≤2 mm，后两项仍为分析阈值；拒绝和错误输出计失败。
误差含已求出的拒绝候选，未求出不填零；误差/耗时为中位数 / P95 / 最大。

|组|视图|方法|成功/拒绝/错误接受/总数|圆心mm|法向°|直径mm|耗时ms|>1000ms|
|---|---|---|---|---|---|---|---|---|
|nominal|all|adaptive_validated|32/0/0/32|0.373 / 1.210 / 1.393|0.015 / 0.076 / 0.287|0.012 / 0.036 / 0.056|275.929 / 413.207 / 481.273|0|
|nominal|all|frozen_region|32/0/0/32|0.370 / 1.216 / 1.395|0.012 / 0.069 / 0.304|0.012 / 0.036 / 0.054|337.938 / 485.155 / 548.034|0|
|nominal|all|stable_local|32/0/0/32|0.373 / 1.210 / 1.393|0.015 / 0.076 / 0.287|0.012 / 0.036 / 0.056|273.845 / 412.320 / 482.089|0|
|nominal|cam_a|adaptive_validated|30/2/0/32|0.130 / 1.288 / 1.355|0.021 / 0.542 / 0.764|0.011 / 0.039 / 0.087|149.408 / 508.166 / 755.340|0|
|nominal|cam_a|frozen_region|27/5/0/32|0.096 / 1.274 / 1.372|0.024 / 0.520 / 0.717|0.012 / 0.055 / 0.098|167.490 / 280.296 / 438.004|0|
|nominal|cam_a|stable_local|30/2/0/32|0.130 / 1.288 / 1.355|0.021 / 0.542 / 0.764|0.011 / 0.039 / 0.087|147.255 / 237.385 / 694.626|0|
|nominal|cam_a,cam_b|adaptive_validated|32/0/0/32|0.382 / 1.282 / 1.424|0.019 / 0.193 / 1.893|0.010 / 0.052 / 0.234|201.640 / 299.799 / 443.342|0|
|nominal|cam_a,cam_b|frozen_region|30/2/0/32|0.103 / 1.267 / 1.406|0.016 / 0.108 / 0.264|0.010 / 0.032 / 0.068|235.970 / 413.472 / 529.508|0|
|nominal|cam_a,cam_b|stable_local|31/1/0/32|0.102 / 1.268 / 1.424|0.017 / 0.116 / 0.258|0.009 / 0.033 / 0.069|200.402 / 300.365 / 435.242|0|
|stress|all|adaptive_validated|14/2/0/16|0.581 / 0.915 / 0.925|0.020 / 0.053 / 0.065|0.016 / 0.058 / 0.073|327.928 / 689.636 / 694.113|0|
|stress|all|frozen_region|15/1/0/16|0.543 / 0.912 / 0.921|0.009 / 0.053 / 0.065|0.016 / 0.057 / 0.075|338.246 / 442.146 / 497.331|0|
|stress|all|stable_local|13/3/0/16|0.562 / 0.916 / 0.925|0.013 / 0.054 / 0.065|0.017 / 0.059 / 0.073|286.402 / 348.107 / 365.085|0|
|stress|cam_a|adaptive_validated|13/3/0/16|0.378 / 1.130 / 1.165|0.160 / 0.309 / 0.324|0.010 / 0.053 / 0.054|198.447 / 326.465 / 337.374|0|
|stress|cam_a|frozen_region|12/4/0/16|0.139 / 0.991 / 1.040|0.029 / 0.250 / 0.279|0.022 / 0.056 / 0.061|162.991 / 267.027 / 327.765|0|
|stress|cam_a|stable_local|12/4/0/16|0.345 / 1.107 / 1.165|0.111 / 0.310 / 0.324|0.016 / 0.053 / 0.054|143.853 / 253.793 / 260.780|0|
|stress|cam_a,cam_b|adaptive_validated|14/2/0/16|0.550 / 0.933 / 0.955|0.036 / 0.116 / 0.137|0.020 / 0.064 / 0.078|254.667 / 492.173 / 495.120|0|
|stress|cam_a,cam_b|frozen_region|15/1/0/16|0.534 / 0.928 / 0.940|0.009 / 0.107 / 0.114|0.020 / 0.065 / 0.079|253.421 / 322.234 / 356.224|0|
|stress|cam_a,cam_b|stable_local|13/3/0/16|0.533 / 0.935 / 0.955|0.035 / 0.117 / 0.137|0.022 / 0.065 / 0.078|206.168 / 287.866 / 301.747|0|

1000 ms仅为离线说明线，真实工位预算未给定；不含采集、传输、机器人执行。

|组|视图|方法|XYZ最大绝对误差mm|沿法向/切向最大mm|
|---|---|---|---|---|
|nominal|all|adaptive_validated|0.084 / 0.104 / 1.393|1.392 / 0.024|
|nominal|all|frozen_region|0.083 / 0.104 / 1.395|1.395 / 0.027|
|nominal|all|stable_local|0.084 / 0.104 / 1.393|1.392 / 0.024|
|nominal|cam_a|adaptive_validated|0.110 / 0.110 / 1.355|1.355 / 0.040|
|nominal|cam_a|frozen_region|0.104 / 0.106 / 1.372|1.372 / 0.041|
|nominal|cam_a|stable_local|0.110 / 0.110 / 1.355|1.355 / 0.040|
|nominal|cam_a,cam_b|adaptive_validated|0.333 / 0.135 / 1.421|1.424 / 0.233|
|nominal|cam_a,cam_b|frozen_region|0.079 / 0.100 / 1.403|1.406 / 0.021|
|nominal|cam_a,cam_b|stable_local|0.082 / 0.101 / 1.421|1.424 / 0.020|
|stress|all|adaptive_validated|0.154 / 0.163 / 0.913|0.925 / 0.009|
|stress|all|frozen_region|0.108 / 0.162 / 0.909|0.921 / 0.013|
|stress|all|stable_local|0.108 / 0.163 / 0.913|0.925 / 0.009|
|stress|cam_a|adaptive_validated|0.188 / 0.211 / 1.146|1.165 / 0.011|
|stress|cam_a|frozen_region|0.127 / 0.183 / 1.024|1.040 / 0.008|
|stress|cam_a|stable_local|0.125 / 0.211 / 1.146|1.165 / 0.011|
|stress|cam_a,cam_b|adaptive_validated|0.156 / 0.170 / 0.940|0.955 / 0.008|
|stress|cam_a,cam_b|frozen_region|0.109 / 0.168 / 0.925|0.940 / 0.008|
|stress|cam_a,cam_b|stable_local|0.110 / 0.170 / 0.940|0.955 / 0.008|

拒绝/错误接受：

- nominal_002 / cam_a / frozen_region: 拒绝 axis_core_touches_candidate_boundary
- nominal_002 / cam_a / stable_local: 拒绝 continuous_core_touches_candidate_boundary
- nominal_002 / cam_a / adaptive_validated: 拒绝 continuous_core_touches_candidate_boundary
- nominal_002 / cam_a,cam_b / frozen_region: 拒绝 axis_core_touches_candidate_boundary
- nominal_002 / cam_a,cam_b / stable_local: 拒绝 continuous_core_touches_candidate_boundary
- nominal_007 / cam_a / frozen_region: 拒绝 axis_core_touches_candidate_boundary
- nominal_010 / cam_a / frozen_region: 拒绝 feedback_not_converged
- nominal_014 / cam_a / frozen_region: 拒绝 axis_core_touches_candidate_boundary
- nominal_018 / cam_a,cam_b / frozen_region: 拒绝 feedback_not_converged
- nominal_024 / cam_a / frozen_region: 拒绝 axis_core_touches_candidate_boundary
- nominal_024 / cam_a / stable_local: 拒绝 continuous_core_touches_candidate_boundary
- nominal_024 / cam_a / adaptive_validated: 拒绝 near_minimum_interval_touches_observed_search_boundary
- stress_003 / cam_a / frozen_region: 拒绝 axis_core_touches_candidate_boundary
- stress_004 / cam_a / stable_local: 拒绝 continuous_core_insufficient_extent
- stress_004 / cam_a / adaptive_validated: 拒绝 continuous_core_insufficient_extent
- stress_004 / cam_a,cam_b / stable_local: 拒绝 continuous_core_insufficient_extent
- stress_004 / cam_a,cam_b / adaptive_validated: 拒绝 continuous_core_insufficient_extent
- stress_004 / all / stable_local: 拒绝 continuous_core_insufficient_extent
- stress_004 / all / adaptive_validated: 拒绝 continuous_core_insufficient_extent
- stress_007 / cam_a / frozen_region: 拒绝 axis_core_touches_candidate_boundary
- stress_008 / cam_a / stable_local: 拒绝 continuous_core_insufficient_extent
- stress_008 / cam_a / adaptive_validated: 拒绝 continuous_core_insufficient_extent
- stress_008 / cam_a,cam_b / stable_local: 拒绝 continuous_core_insufficient_extent
- stress_008 / cam_a,cam_b / adaptive_validated: 拒绝 continuous_core_insufficient_extent
- stress_008 / all / stable_local: 拒绝 continuous_core_insufficient_extent
- stress_008 / all / adaptive_validated: 拒绝 continuous_core_insufficient_extent
- stress_009 / cam_a / frozen_region: 拒绝 axis_core_touches_candidate_boundary
- stress_009 / cam_a / stable_local: 拒绝 continuous_core_touches_candidate_boundary
- stress_009 / cam_a,cam_b / frozen_region: 拒绝 near_minimum_interval_touches_observed_search_boundary
- stress_009 / cam_a,cam_b / stable_local: 拒绝 continuous_core_touches_candidate_boundary
- stress_009 / all / frozen_region: 拒绝 near_minimum_interval_touches_observed_search_boundary
- stress_009 / all / stable_local: 拒绝 continuous_core_touches_candidate_boundary
- stress_012 / cam_a / frozen_region: 拒绝 no_supported_near_upward_mouth_plane
- stress_012 / cam_a / stable_local: 拒绝 no_supported_near_upward_mouth_plane
- stress_012 / cam_a / adaptive_validated: 拒绝 no_supported_near_upward_mouth_plane

## 开发（非未见测试）

成功=接受且圆心≤4 mm/法向≤2°/直径≤2 mm，后两项仍为分析阈值；拒绝和错误输出计失败。
误差含已求出的拒绝候选，未求出不填零；误差/耗时为中位数 / P95 / 最大。

|组|视图|方法|成功/拒绝/错误接受/总数|圆心mm|法向°|直径mm|耗时ms|>1000ms|
|---|---|---|---|---|---|---|---|---|
|nominal|all|adaptive_validated|24/0/0/24|0.946 / 1.076 / 1.151|0.022 / 0.090 / 0.094|0.014 / 0.025 / 0.029|317.620 / 437.571 / 476.294|0|
|nominal|all|frozen_region|23/1/0/24|0.944 / 1.075 / 1.151|0.029 / 0.079 / 0.099|0.014 / 0.025 / 0.029|355.366 / 534.604 / 631.683|0|
|nominal|all|stable_local|24/0/0/24|0.946 / 1.076 / 1.151|0.022 / 0.090 / 0.094|0.014 / 0.025 / 0.029|318.172 / 435.887 / 475.241|0|
|nominal|cam_a|adaptive_validated|23/1/0/24|0.526 / 1.363 / 1.623|0.148 / 0.583 / 0.715|0.021 / 0.074 / 0.084|163.688 / 308.046 / 380.218|0|
|nominal|cam_a|frozen_region|19/5/0/24|0.470 / 1.241 / 1.665|0.112 / 0.379 / 0.754|0.014 / 0.063 / 0.087|173.358 / 325.265 / 532.708|0|
|nominal|cam_a|stable_local|23/1/0/24|0.526 / 1.363 / 1.623|0.148 / 0.583 / 0.715|0.021 / 0.074 / 0.084|164.341 / 308.106 / 381.540|0|
|nominal|cam_a,cam_b|adaptive_validated|24/0/0/24|0.836 / 1.058 / 1.200|0.044 / 0.179 / 0.209|0.013 / 0.030 / 0.044|225.277 / 303.642 / 360.493|0|
|nominal|cam_a,cam_b|frozen_region|22/2/0/24|0.878 / 1.078 / 1.188|0.038 / 0.135 / 0.173|0.017 / 0.034 / 0.038|259.010 / 325.175 / 411.756|0|
|nominal|cam_a,cam_b|stable_local|24/0/0/24|0.836 / 1.058 / 1.200|0.044 / 0.179 / 0.209|0.013 / 0.030 / 0.044|223.938 / 304.101 / 358.892|0|
|stress|all|adaptive_validated|12/0/0/12|0.946 / 1.220 / 1.222|0.017 / 0.123 / 0.139|0.015 / 0.080 / 0.100|277.918 / 678.587 / 916.734|0|
|stress|all|frozen_region|11/1/0/12|0.925 / 1.226 / 1.228|0.006 / 0.135 / 0.166|0.015 / 0.081 / 0.097|346.457 / 552.161 / 563.117|0|
|stress|all|stable_local|11/1/0/12|0.924 / 1.221 / 1.222|0.015 / 0.125 / 0.139|0.016 / 0.081 / 0.100|277.659 / 471.707 / 486.004|0|
|stress|cam_a|adaptive_validated|5/7/0/12|0.491 / 1.230 / 1.241|0.036 / 0.196 / 0.220|0.015 / 0.044 / 0.051|199.385 / 400.799 / 427.561|0|
|stress|cam_a|frozen_region|3/9/0/12|0.570 / 1.194 / 1.263|0.034 / 0.087 / 0.093|0.008 / 0.010 / 0.010|132.357 / 237.000 / 282.291|0|
|stress|cam_a|stable_local|4/8/0/12|0.328 / 1.129 / 1.241|0.025 / 0.089 / 0.098|0.013 / 0.046 / 0.051|117.093 / 202.890 / 257.792|0|
|stress|cam_a,cam_b|adaptive_validated|10/2/0/12|0.979 / 1.215 / 1.277|0.057 / 0.099 / 0.101|0.015 / 0.069 / 0.090|234.512 / 512.591 / 584.809|0|
|stress|cam_a,cam_b|frozen_region|8/4/0/12|0.923 / 1.232 / 1.271|0.040 / 0.078 / 0.088|0.015 / 0.075 / 0.093|235.546 / 352.769 / 359.480|0|
|stress|cam_a,cam_b|stable_local|9/3/0/12|0.926 / 1.222 / 1.277|0.053 / 0.087 / 0.097|0.015 / 0.072 / 0.090|195.611 / 329.413 / 344.668|0|

1000 ms仅为离线说明线，真实工位预算未给定；不含采集、传输、机器人执行。

|组|视图|方法|XYZ最大绝对误差mm|沿法向/切向最大mm|
|---|---|---|---|---|
|nominal|all|adaptive_validated|0.073 / 0.099 / 1.150|1.151 / 0.017|
|nominal|all|frozen_region|0.076 / 0.097 / 1.151|1.151 / 0.016|
|nominal|all|stable_local|0.073 / 0.099 / 1.150|1.151 / 0.017|
|nominal|cam_a|adaptive_validated|0.081 / 0.081 / 1.623|1.623 / 0.050|
|nominal|cam_a|frozen_region|0.080 / 0.081 / 1.664|1.665 / 0.040|
|nominal|cam_a|stable_local|0.081 / 0.081 / 1.623|1.623 / 0.050|
|nominal|cam_a,cam_b|adaptive_validated|0.083 / 0.091 / 1.197|1.200 / 0.018|
|nominal|cam_a,cam_b|frozen_region|0.081 / 0.090 / 1.185|1.187 / 0.019|
|nominal|cam_a,cam_b|stable_local|0.083 / 0.091 / 1.197|1.200 / 0.018|
|stress|all|adaptive_validated|0.184 / 0.179 / 1.208|1.222 / 0.018|
|stress|all|frozen_region|0.184 / 0.180 / 1.213|1.228 / 0.018|
|stress|all|stable_local|0.184 / 0.179 / 1.208|1.222 / 0.018|
|stress|cam_a|adaptive_validated|0.182 / 0.108 / 1.227|1.241 / 0.008|
|stress|cam_a|frozen_region|0.165 / 0.111 / 1.248|1.263 / 0.006|
|stress|cam_a|stable_local|0.158 / 0.108 / 1.227|1.241 / 0.008|
|stress|cam_a,cam_b|adaptive_validated|0.186 / 0.187 / 1.260|1.277 / 0.012|
|stress|cam_a,cam_b|frozen_region|0.154 / 0.187 / 1.254|1.271 / 0.009|
|stress|cam_a,cam_b|stable_local|0.186 / 0.187 / 1.260|1.277 / 0.012|

拒绝/错误接受：

- nominal_002 / all / frozen_region: 拒绝 feedback_not_converged
- nominal_003 / cam_a / frozen_region: 拒绝 axis_core_touches_candidate_boundary
- nominal_003 / cam_a,cam_b / frozen_region: 拒绝 axis_core_touches_candidate_boundary
- nominal_004 / cam_a / frozen_region: 拒绝 axis_core_touches_candidate_boundary
- nominal_005 / cam_a / frozen_region: 拒绝 axis_core_touches_candidate_boundary
- nominal_013 / cam_a / frozen_region: 拒绝 axis_core_touches_candidate_boundary
- nominal_013 / cam_a,cam_b / frozen_region: 拒绝 axis_core_touches_candidate_boundary
- nominal_015 / cam_a / frozen_region: 拒绝 no_supported_near_upward_mouth_plane
- nominal_015 / cam_a / stable_local: 拒绝 no_supported_near_upward_mouth_plane
- nominal_015 / cam_a / adaptive_validated: 拒绝 no_supported_near_upward_mouth_plane
- stress_000 / cam_a / frozen_region: 拒绝 axis_core_touches_candidate_boundary
- stress_000 / cam_a / stable_local: 拒绝 continuous_core_touches_candidate_boundary
- stress_000 / cam_a / adaptive_validated: 拒绝 near_minimum_interval_touches_observed_search_boundary
- stress_000 / cam_a,cam_b / frozen_region: 拒绝 axis_core_touches_candidate_boundary
- stress_000 / cam_a,cam_b / stable_local: 拒绝 continuous_core_touches_candidate_boundary
- stress_000 / cam_a,cam_b / adaptive_validated: 拒绝 near_minimum_interval_touches_observed_search_boundary
- stress_001 / cam_a / frozen_region: 拒绝 axis_core_touches_candidate_boundary
- stress_001 / cam_a / stable_local: 拒绝 continuous_core_touches_candidate_boundary
- stress_001 / cam_a,cam_b / frozen_region: 拒绝 axis_core_touches_candidate_boundary
- stress_001 / cam_a,cam_b / stable_local: 拒绝 continuous_core_touches_candidate_boundary
- stress_001 / all / frozen_region: 拒绝 axis_core_touches_candidate_boundary
- stress_001 / all / stable_local: 拒绝 continuous_core_touches_candidate_boundary
- stress_002 / cam_a / frozen_region: 拒绝 axis_core_has_insufficient_extent
- stress_003 / cam_a / frozen_region: 拒绝 axis_core_touches_candidate_boundary
- stress_003 / cam_a / stable_local: 拒绝 continuous_core_touches_candidate_boundary
- stress_003 / cam_a / adaptive_validated: 拒绝 near_minimum_interval_touches_observed_search_boundary
- stress_004 / cam_a / frozen_region: 拒绝 no_supported_near_upward_mouth_plane
- stress_004 / cam_a / stable_local: 拒绝 no_supported_near_upward_mouth_plane
- stress_004 / cam_a / adaptive_validated: 拒绝 no_supported_near_upward_mouth_plane
- stress_005 / cam_a / frozen_region: 拒绝 no_supported_near_upward_mouth_plane
- stress_005 / cam_a / stable_local: 拒绝 no_supported_near_upward_mouth_plane
- stress_005 / cam_a / adaptive_validated: 拒绝 no_supported_near_upward_mouth_plane
- stress_005 / cam_a,cam_b / frozen_region: 拒绝 no_supported_near_upward_mouth_plane
- stress_005 / cam_a,cam_b / stable_local: 拒绝 no_supported_near_upward_mouth_plane
- stress_005 / cam_a,cam_b / adaptive_validated: 拒绝 no_supported_near_upward_mouth_plane
- stress_008 / cam_a / frozen_region: 拒绝 axis_core_touches_candidate_boundary
- stress_008 / cam_a / stable_local: 拒绝 continuous_core_touches_candidate_boundary
- stress_008 / cam_a / adaptive_validated: 拒绝 continuous_core_touches_candidate_boundary
- stress_008 / cam_a,cam_b / frozen_region: 拒绝 axis_core_touches_candidate_boundary
- stress_009 / cam_a / frozen_region: 拒绝 no_supported_near_upward_mouth_plane
- stress_009 / cam_a / stable_local: 拒绝 no_supported_near_upward_mouth_plane
- stress_009 / cam_a / adaptive_validated: 拒绝 no_supported_near_upward_mouth_plane
- stress_010 / cam_a / frozen_region: 拒绝 axis_core_touches_candidate_boundary
- stress_010 / cam_a / stable_local: 拒绝 continuous_core_touches_candidate_boundary
- stress_010 / cam_a / adaptive_validated: 拒绝 near_minimum_interval_touches_observed_search_boundary

## 扩张与备用路径（新测试）

|组|视图|28mm接受|42mm接受|56mm接受|备用固定权重接受|无输出或拒绝|
|---|---|---|---|---|---|
|nominal|cam_a|30|0|0|0|2|
|nominal|cam_a,cam_b|32|0|0|1|0|
|nominal|all|32|0|0|0|0|
|stress|cam_a|12|1|0|0|3|
|stress|cam_a,cam_b|13|1|0|0|2|
|stress|all|13|1|0|0|2|
