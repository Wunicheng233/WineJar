# 冻结算法的随机几何审计

24例假设常规范围、12例压力范围；不是实测工厂分布。固定相机、准确K/T、理想深度。
每例两方法×三视图组合；全部尝试计入分母，误差包含已求出但拒绝的候选，未求出不填零。
单元格为中位数 / P95 / 最大；4 mm/2°/直径2 mm为分析阈值，后两项尚未工艺定案。

|组|相机|方法|成功/拒绝/错误接受/总数|圆心mm|法向°|直径mm|耗时ms|
|---|---|---|---|---|---|---|---|
|nominal|all|fixed_equal_trust|23/1/0/24|0.964 / 1.112 / 1.186|0.222 / 0.575 / 0.624|0.016 / 0.034 / 0.053|288.939 / 373.382 / 442.216|
|nominal|all|region_equal_trust|23/1/0/24|0.944 / 1.075 / 1.151|0.029 / 0.079 / 0.099|0.014 / 0.025 / 0.029|346.906 / 520.186 / 621.299|
|nominal|cam_a|fixed_equal_trust|23/1/0/24|0.715 / 1.575 / 2.071|0.273 / 0.701 / 1.342|0.102 / 0.329 / 0.348|148.036 / 321.970 / 451.594|
|nominal|cam_a|region_equal_trust|19/5/0/24|0.470 / 1.241 / 1.665|0.112 / 0.379 / 0.754|0.014 / 0.063 / 0.087|172.106 / 325.696 / 486.694|
|nominal|cam_a,cam_b|fixed_equal_trust|24/0/0/24|0.883 / 1.196 / 1.254|0.232 / 0.572 / 0.709|0.053 / 0.103 / 0.213|212.976 / 259.951 / 321.090|
|nominal|cam_a,cam_b|region_equal_trust|22/2/0/24|0.878 / 1.078 / 1.188|0.038 / 0.135 / 0.173|0.017 / 0.034 / 0.038|255.300 / 317.309 / 409.525|
|stress|all|fixed_equal_trust|11/1/0/12|0.916 / 1.231 / 1.237|0.166 / 0.274 / 0.300|0.013 / 0.075 / 0.091|291.183 / 460.781 / 469.391|
|stress|all|region_equal_trust|11/1/0/12|0.925 / 1.226 / 1.228|0.006 / 0.135 / 0.166|0.015 / 0.081 / 0.097|338.943 / 540.717 / 555.130|
|stress|cam_a|fixed_equal_trust|5/7/0/12|0.845 / 0.961 / 0.969|0.263 / 0.403 / 0.414|0.059 / 0.154 / 0.173|125.882 / 216.442 / 231.403|
|stress|cam_a|region_equal_trust|3/9/0/12|0.570 / 1.194 / 1.263|0.034 / 0.087 / 0.093|0.008 / 0.010 / 0.010|131.482 / 238.239 / 285.099|
|stress|cam_a,cam_b|fixed_equal_trust|9/3/0/12|0.817 / 1.257 / 1.323|0.250 / 0.540 / 0.542|0.061 / 0.086 / 0.089|198.258 / 297.262 / 301.581|
|stress|cam_a,cam_b|region_equal_trust|8/4/0/12|0.923 / 1.232 / 1.271|0.040 / 0.078 / 0.088|0.015 / 0.075 / 0.093|231.356 / 346.566 / 353.505|

## 拒绝和错误接受

- nominal_002 / all / region_equal_trust: 拒绝 feedback_not_converged; center=1.0376977368253573, normal=0.017497735958612416, diameter=0.01659504826587166
- nominal_003 / cam_a / region_equal_trust: 拒绝 axis_core_touches_candidate_boundary; center=None, normal=None, diameter=None
- nominal_003 / cam_a,cam_b / region_equal_trust: 拒绝 axis_core_touches_candidate_boundary; center=None, normal=None, diameter=None
- nominal_004 / cam_a / region_equal_trust: 拒绝 axis_core_touches_candidate_boundary; center=None, normal=None, diameter=None
- nominal_005 / cam_a / region_equal_trust: 拒绝 axis_core_touches_candidate_boundary; center=None, normal=None, diameter=None
- nominal_008 / all / fixed_equal_trust: 拒绝 weighted_axis_model_mismatch; center=1.10800708768156, normal=0.6239749496188498, diameter=0.03436070264455082
- nominal_013 / cam_a / region_equal_trust: 拒绝 axis_core_touches_candidate_boundary; center=None, normal=None, diameter=None
- nominal_013 / cam_a,cam_b / region_equal_trust: 拒绝 axis_core_touches_candidate_boundary; center=None, normal=None, diameter=None
- nominal_015 / cam_a / fixed_equal_trust: 拒绝 no_supported_near_upward_mouth_plane; center=None, normal=None, diameter=None
- nominal_015 / cam_a / region_equal_trust: 拒绝 no_supported_near_upward_mouth_plane; center=None, normal=None, diameter=None
- stress_000 / cam_a / fixed_equal_trust: 拒绝 near_minimum_interval_touches_observed_search_boundary; center=None, normal=None, diameter=None
- stress_000 / cam_a / region_equal_trust: 拒绝 axis_core_touches_candidate_boundary; center=None, normal=None, diameter=None
- stress_000 / cam_a,cam_b / fixed_equal_trust: 拒绝 near_minimum_interval_touches_observed_search_boundary; center=None, normal=None, diameter=None
- stress_000 / cam_a,cam_b / region_equal_trust: 拒绝 axis_core_touches_candidate_boundary; center=None, normal=None, diameter=None
- stress_001 / cam_a / fixed_equal_trust: 拒绝 near_minimum_interval_touches_observed_search_boundary; center=None, normal=None, diameter=None
- stress_001 / cam_a / region_equal_trust: 拒绝 axis_core_touches_candidate_boundary; center=None, normal=None, diameter=None
- stress_001 / cam_a,cam_b / fixed_equal_trust: 拒绝 near_minimum_interval_touches_observed_search_boundary; center=None, normal=None, diameter=None
- stress_001 / cam_a,cam_b / region_equal_trust: 拒绝 axis_core_touches_candidate_boundary; center=None, normal=None, diameter=None
- stress_001 / all / fixed_equal_trust: 拒绝 near_minimum_interval_touches_observed_search_boundary; center=None, normal=None, diameter=None
- stress_001 / all / region_equal_trust: 拒绝 axis_core_touches_candidate_boundary; center=None, normal=None, diameter=None
- stress_002 / cam_a / region_equal_trust: 拒绝 axis_core_has_insufficient_extent; center=None, normal=None, diameter=None
- stress_003 / cam_a / fixed_equal_trust: 拒绝 near_minimum_interval_touches_observed_search_boundary; center=None, normal=None, diameter=None
- stress_003 / cam_a / region_equal_trust: 拒绝 axis_core_touches_candidate_boundary; center=None, normal=None, diameter=None
- stress_004 / cam_a / fixed_equal_trust: 拒绝 no_supported_near_upward_mouth_plane; center=None, normal=None, diameter=None
- stress_004 / cam_a / region_equal_trust: 拒绝 no_supported_near_upward_mouth_plane; center=None, normal=None, diameter=None
- stress_005 / cam_a / fixed_equal_trust: 拒绝 no_supported_near_upward_mouth_plane; center=None, normal=None, diameter=None
- stress_005 / cam_a / region_equal_trust: 拒绝 no_supported_near_upward_mouth_plane; center=None, normal=None, diameter=None
- stress_005 / cam_a,cam_b / fixed_equal_trust: 拒绝 no_supported_near_upward_mouth_plane; center=None, normal=None, diameter=None
- stress_005 / cam_a,cam_b / region_equal_trust: 拒绝 no_supported_near_upward_mouth_plane; center=None, normal=None, diameter=None
- stress_008 / cam_a / region_equal_trust: 拒绝 axis_core_touches_candidate_boundary; center=None, normal=None, diameter=None
- stress_008 / cam_a,cam_b / region_equal_trust: 拒绝 axis_core_touches_candidate_boundary; center=None, normal=None, diameter=None
- stress_009 / cam_a / fixed_equal_trust: 拒绝 no_supported_near_upward_mouth_plane; center=None, normal=None, diameter=None
- stress_009 / cam_a / region_equal_trust: 拒绝 no_supported_near_upward_mouth_plane; center=None, normal=None, diameter=None
- stress_010 / cam_a / fixed_equal_trust: 拒绝 near_minimum_interval_touches_observed_search_boundary; center=None, normal=None, diameter=None
- stress_010 / cam_a / region_equal_trust: 拒绝 axis_core_touches_candidate_boundary; center=None, normal=None, diameter=None

精确参数见config.json和scenes/*/model_manifest.json；按结果分类的查看索引见review_index.json。
图库红线为区域策略三视图估计，青色虚线为评估真值；没有结果时只画真值并标拒绝。
本次只审计现有圆截面模型家族，不包含椭圆、不规则凹槽、真实反光或深度偏差。
样本来自假设随机分布，经验P95与零失败均不能作为工厂性能保证。
