# xArm SDK Control

本目录是 WineJar 项目的实机控制层。根目录 [README](../README.md) 是当前
硬件状态和现场运行命令的权威说明；本文只记录代码结构和维护边界。

## 设计原则

- `scripts/` 只负责命令行参数、任务编排和结果落盘。
- 可复用的视觉、运动、真空和重试逻辑放在顶层 Python 模块中。
- dry-run 与单元测试不得发送机械臂运动或 GPIO 指令。
- 实机入口必须显式给出工作空间边界，并在运动前执行相机避让检查。
- 当前已经实机跑通的 `21_complete_loading.py` 不做无收益的大规模改写。

## 模块职责

| 模块 | 职责 |
|---|---|
| `robot_motion.py` | 连接、状态检查、准备、相机安全复位 |
| `visual_targeting.py` | 标定加载、RGB-D 扫描、检测分类和目标选择 |
| `cross_place_route.py` | 吸取、搬运、放置和撤离的路径几何 |
| `cross_place_motion.py` | 运动剖面、关节感知规划、路由执行和压力持续检查 |
| `cross_place_retry.py` | 吸附失败后的恢复、重扫描、重规划和有限重试 |
| `trajectory_speed_planner.py` | IK 采样、速度缩放、端点回退和控制器预检 |
| `pick_test_workflow.py` | 单物吸取与源到目标搬运测试 |
| `suction_sensor.py` | CI0 压力开关读取、超时和去抖 |
| `vacuum_control.py` | CO0/CO1 真空状态机和安全关断 |
| `end_effector_config.py` | 执行器 YAML 解析与参数校验 |
| `realsense_capture.py` | 对齐 RGB-D 会话、录制和断线重试 |
| `positioning_bias.py` | 人工测量的基坐标补偿 |
| `cli_utils.py` | 公共 CLI 参数和工作空间检查 |
| `run_output.py` | `result.json` / `failure.json` |
| `task_timing.py` | 单调时钟事件时间线 |
| `geometry.py` | 位姿和角度辅助函数 |

## 任务入口

- `19_visual_cross_place_two_leaves.py`：只验证两片竹壳交叉放置。
- `20_visual_pick_test.py`：按物料类型执行吸起、悬停、放回；适合硬件调试。
- `21_complete_loading.py`：完整上料生产入口，当前实机主流程。

早期的非接触接近脚本和失败的荷叶竖直运输原型已经删除。Git 历史仍保留这些
实验，不应把它们重新复制回生产代码。

## 当前完整流程参数

`21_complete_loading.py` 的稳定优先默认值：

| 参数 | 值 |
|---|---:|
| 空载水平速度 | `250 mm/s` |
| 载荷水平速度 | `180 mm/s` |
| 空载竖直速度 | `170 mm/s` |
| 载荷竖直速度 | `100 mm/s` |
| 精确运动速度 | `50 mm/s` |
| 接触速度 | `15 mm/s` |
| 空载/载荷/精确加速度 | `400/200/160 mm/s²` |
| 关节速度与加速度利用率 | `0.35/0.35` |
| 搬运净空 | `100 mm` |
| 白纸/陶瓷/竹壳下压 | `17/7/13 mm` |
| 竹壳目标角 | `+45/-60 deg` |

路径规划会先尝试密集连续 IK；靠近腕部奇异位形时，xArm 求解器可能切换
J4/J6 分支，此时自动退回端点插值估算。控制器 `only_check` 的 C24 结果当前
作为警告记录，因为现场已经多次出现“预检 C24、实际路径正常完成”的情况；
真正的运动指令返回非零仍会立即中止任务。

## 压力策略

- 所有物料在接触后都必须通过 CI0 初次吸附确认。
- 陶瓷片和竹壳在载荷段结束后继续强制确认。
- 白纸因材料透气和边缘漏气，第一次载荷压力丢失记为警告，后续载荷段不再
  重复轮询；这一例外只适用于当前白纸和当前压力阈值。
- 吸附失败的偏移序列为 `(0,0) -> (0,-5) -> (5,0) mm`，最多三次。

## 测试

```bash
conda activate winejar-xarm-sdk
python -m pytest -q xarm_sdk_control/tests
python -m compileall -q xarm_sdk_control
```

测试使用假机械臂、假真空控制器和临时目录，不连接实机。修改安全边界、真空
状态机、路径顺序、重试恢复或配置解析时，必须同步增加测试。

## 配置与产物

- `config/end_effector.yaml`：TCP、弹簧行程、CO0/CO1、CI0。
- `config/jar_position.yaml`：固定坛口教点。
- `config/visual_position_bias.json`：当前相机与双吸盘的人工补偿。
- `calibration/global_camera_*.json`：固定相机外参。
- `data/`：可清理运行产物，代码不读取历史任务目录。

不要把标定文件、当前配置或模型权重误当成普通实验数据删除。
