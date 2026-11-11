# SAM2固定前景适配结果的独立只读审查

四组正式结果均完成且validation通过、零输入/后端异常。逐组FoundationStereo与SimSense实际使用相同mask；缓存、RGB、PNG哈希复核通过；四组冻结源码哈希完全一致且当前未改变。

此处旧12坛和新8坛均已被此前版本评分。本轮是固定SAM2协议的配对输入适配，不能再称新的独立测试；未根据这些结果选mask或调参数。

## 主中心指标与接受状态

各格为对应组中的数量；轴向带容纳仅指已标注颈段的轴向几何，未包含径向接触、碰撞、标定或机器人误差。

| 数据 | 深度 | IR | 几何 | N | 候选中心≤4mm | 接受且中心≤4mm | 接受且中心≤3mm | 接受且轴向带容纳 |
|---|---|---|---|---:|---:|---:|---:|---:|
| audit | fs | clean | standard_sor | 12 | 12 | 12 | 11 | 9 |
| audit | fs | clean | visible_profile | 12 | 12 | 12 | 12 | 9 |
| audit | fs | shared_noise | standard_sor | 12 | 12 | 12 | 12 | 10 |
| audit | fs | shared_noise | visible_profile | 12 | 12 | 12 | 12 | 10 |
| audit | simsense | clean | standard_sor | 12 | 12 | 12 | 12 | 11 |
| audit | simsense | clean | visible_profile | 12 | 12 | 12 | 12 | 12 |
| audit | simsense | shared_noise | standard_sor | 12 | 12 | 12 | 12 | 11 |
| audit | simsense | shared_noise | visible_profile | 12 | 12 | 12 | 12 | 11 |
| confirmation | fs | clean | standard_sor | 8 | 8 | 8 | 8 | 7 |
| confirmation | fs | clean | visible_profile | 8 | 8 | 7 | 6 | 5 |
| confirmation | fs | shared_noise | standard_sor | 8 | 8 | 8 | 8 | 7 |
| confirmation | fs | shared_noise | visible_profile | 8 | 8 | 8 | 8 | 7 |
| confirmation | simsense | clean | standard_sor | 8 | 8 | 8 | 7 | 8 |
| confirmation | simsense | clean | visible_profile | 8 | 8 | 8 | 7 | 6 |
| confirmation | simsense | shared_noise | standard_sor | 8 | 8 | 8 | 8 | 8 |
| confirmation | simsense | shared_noise | visible_profile | 8 | 8 | 8 | 7 | 6 |

## 有界案例复核

- 确认集`_004`：FoundationStereo/shared_noise/visible_profile原有拒绝被修复，接受False→True；圆心0.667→0.589mm、法向0.710→0.569°，轴向带余量1.589→1.758mm。
- 确认集`_005`：FoundationStereo两几何、两IR条件的圆心仍≤3mm，但法向2.430–3.643°，轴向带余量−1.538至−1.439mm；SAM2没有解决该处朝向与整带越界。不能用中心通过代表整带通过。
- 同一`_005`的SimSense标准SOR/shared_noise法向2.187→0.450°，轴向带余量−1.219→0.357mm；成熟传统匹配也可能在该项更好，不能把残差直接归因于所有成熟方法均无能力。
- 确认集`_006`：FoundationStereo/clean/visible_profile中心0.660mm、法向0.348°、直径误差0.003mm的候选被完整拟合曲面包含门槛拒绝（最大外越界2.395px）；应区分目标参数与上下文曲面验证。未调整2px门槛。

原始160条SAM2正式记录与同输入v1记录的分项汇总、哈希和`_004/_005`配对明细见`SAM2_paired_readonly_review.json`。复现只读审查：`conda run -n winejar-stereo python scripts/audit_geometry_foreground_pairing.py --output NEW_FILE.json`。
