# 短颈酒坛：模型与首轮理想深度闭环

4个开发模型、4个预先冻结估计器后生成的留出模型；固定工位，一/二/三视图。
RGB前景提取和整坛GT掩码分列；另有事后外表面GT掩码诊断。均为准确相机、无传感器噪声。
估计器以口沿平面方向作为局部颈轴代理，再从观测半径剖面寻找目标范围，不能推广为任意歪颈均可解。

- [全部分位数、成功/拒绝/错误接受及法向/切向统计](tables.md)
- [MuJoCo场景、相机、原始输入与细圆线](inspection/scene_inputs.png)
- [单视图错误接受的直观对照](inspection/failure_comparison.png)与[拒绝案例](inspection/rejected_case.png)
- [修复窗口呈现后的大近景错误接受对照](inspection/window_verified_large_neck_v2.png)（实际窗口回读；旧图是CPU合成预览）
- [实际点云按相机来源着色](inspection/point_cloud_sources.png)与[算法工位裁剪的点云大近景](inspection/point_cloud_station.png)；定位结果未变
- [留出案例误差图](heldout_errors.png)
- [最差单视图案例的剖面诊断](single_view_failure.png)
- [错误接受的四步诊断图](case_walkthroughs/broad_reverse_tilt_cam_a/steps.png)与[拒绝案例诊断图](case_walkthroughs/tilted_flat_cam_a_b/steps.png)
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
