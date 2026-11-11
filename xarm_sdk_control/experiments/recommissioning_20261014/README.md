# 2026-10-14 工位恢复代码

本目录把原来只保存在Git忽略的`data/recommissioning/2026-10-14/`中的有用代码纳入版本管理。
共享路径、读状态及散列检查提取到`context.py`，调整导入与候选文件位置；没有修改下压、路径、速度或吸附判据。

| 文件 | 状态 |
|---|---|
| `resume_three_picks.py` | 原实现曾完成纸/陶瓷和竹壳吸起放回；仍绑定当时场景与ROI |
| `full_loading_recovery.py` | 完整取放草稿，仅生成过clear计划；没有完整实机执行记录 |

共同前提：活动TCP116、相机013222073103、10月14日实验标定。当前日期或工位变化
不能只靠文件名适配。旧图、旧像素、固定观察位都必须重新复核。
两者import/`--help`不会连接设备；正常生成计划会连接机器人读取状态和查询逆解/限位；
`--execute`会运动，取放会操作泵阀。不要把计划命令当作纯离线命令。

```bash
conda run -n winejar-xarm-sdk python \
  xarm_sdk_control/experiments/recommissioning_20261014/resume_three_picks.py --help
conda run -n winejar-xarm-sdk python \
  xarm_sdk_control/experiments/recommissioning_20261014/full_loading_recovery.py --help
```

## 单物料恢复器

保留原始行为：`--material paper_ceramic`或`bamboo`，通过`--frame`选择已复核新帧。
竹壳可显式传`--compression-mm 8`，这是总下压量，不与3 mm累加。
paper/ceramic默认无额外下压。路径按10 mm采样，检查IK、TCP/关节限位与相邻关节变化。
固定ROI只适合原场景；更换摆放后必须更新/复核候选选择，不能仅换frame路径。
不允许同一执行日志重复执行；输入散列、起始位姿和TCP有一致性检查。

## 完整恢复草稿

按单个leg规划，输入新帧、源/目标像素、material、可选yaw和独立output目录。
材料是`clear/paper/ceramic/shell_1/shell_2`；竹壳下压8 mm，其余0 mm。
草稿还含固定观察位、层厚估计、释放净空和较保守速度，尚未在新工位完整实测。
第一片后需要重新拍源料堆；草稿本身不负责检测“剩余一片”和验证最终交叉关系。
执行前还应评审压力丢失时的停机/载荷保持、物料与机械臂扫掠空间及最后释放高度。

## 标定与数据来源

两个恢复器显式使用`calibration/experimental/20261014/global_camera_candidate.json`，
不加载`visual_position_bias.json`，不按文件名自动选择另一份外参。
原始数据仍在`xarm_sdk_control/data/recommissioning/2026-10-14/`，不随Git迁移。
该目录不存在时，先按[数据管理](../../../docs/DATA_MANAGEMENT.md)迁移需要的输入。
关键执行日志已另存[证据快照](../../../docs/evidence/2026-10-14/index.json)。

这里的源码恢复不等于恢复动作已经执行。当前停点与下一步以[Agent交接](../../../docs/HANDOFF.md)为准。
