# WineJar

基于 xArm6、Intel RealSense D435i、GroundingDINO 和 SAM2 的酒坛视觉上料项目。
系统使用固定全局 RGB-D 相机识别物料，通过 Python SDK 驱动 xArm6，并以
真空压力开关确认吸附状态。

## 当前状态

截至 2026-08-30，实机已经连续完成以下流程：

1. 荷叶预先铺在固定酒坛坛口；
2. 识别并放置白纸；
3. 识别并放置陶瓷片；
4. 重新识别已经露出的双层竹壳；
5. 将两片竹壳交叉放置到陶瓷片上；
6. 机械臂安全复位，气泵和电磁阀关闭。

最新完整实验的四种物料均在第一次尝试中通过压力确认，总时长约 81.3 秒。
其中包含一次 RealSense USB 自动重连和一次较慢的保守轨迹估算，当前目标是
先保证稳定性，再继续优化节拍。

仍未解决：荷叶本身面积大、折叠后容易展开，尚不能稳定从料台吸起。因此当前
生产入口使用 `--skip-lotus`，把荷叶视为已经铺在坛口。

动画仿真不属于本仓库，维护在
[Wunicheng233/Winejar-mujoco](https://github.com/Wunicheng233/Winejar-mujoco)。

## 仓库结构

```text
.
├── docs/                  设计与标定记录
├── vision/                GroundingDINO + SAM2 独立推理环境和常驻服务
├── xarm6/                 xArm 用户手册与开发者手册
├── xarm_sdk_control/      实机控制、标定、任务脚本和单元测试
│   ├── calibration/       当前全局相机外参，不可删除
│   ├── config/            机器人、执行器、坛口教点和视觉补偿
│   ├── scripts/           可直接运行的标定、诊断和任务入口
│   ├── tests/             不连接实机的 pytest 测试
│   └── data/              本地实验产物，已被 Git 忽略
└── 项目策划.docx          原始项目策划
```

## 当前硬件配置

| 项目 | 当前值 |
|---|---|
| 机器人 | UFACTORY xArm6，Python SDK 1.18.4 |
| 相机 | Intel RealSense D435i，固定全局俯视 |
| 末端 | 单弹簧双吸盘，双吸盘中点与法兰中心轴基本对齐 |
| TCP | 法兰轴向下 `113 mm` |
| 弹簧最大压缩 | `20 mm` |
| CO0 | 真空泵，`1` 为启动 |
| CO1 | 真空通路电磁阀，`1` 为吸取通路，`0` 为放气 |
| CI0 | ZSE30A-01-N 压力开关，`0` 表示真空达到阈值 |
| 坛口教点 | `(300, -200, 65) mm`，见 `xarm_sdk_control/config/jar_position.yaml` |
| 基础视觉补偿 | `(-5, +44, 0) mm`，见 `xarm_sdk_control/config/visual_position_bias.json` |
| 陶瓷片额外补偿 | `+8 mm X`，仅在完整上料脚本中应用 |

当前完整流程的默认下压量：白纸 `17 mm`、陶瓷片 `7 mm`、竹壳 `13 mm`。
这些值与当前双吸盘、相机位姿和料台高度绑定，移动硬件后必须重新验证。

## 环境安装

控制与 RealSense 环境：

```bash
conda env create -f xarm_sdk_control/environment.yml
conda activate winejar-xarm-sdk
```

视觉环境：

```bash
conda env create -f vision/environment.yml
conda activate winejar-vision
pip install -r vision/requirements-sam2.txt
```

模型权重保存在本机 `vision/models/`，不进入 Git。首次使用需要联网下载，之后
可用 `HF_HUB_OFFLINE=1` 完全离线运行。

## 启动视觉服务

在独立终端中启动常驻推理服务：

```bash
conda activate winejar-vision
HF_HUB_OFFLINE=1 python -u vision/scripts/03_grounded_sam2_service.py
```

看到 `"status": "ready"` 后保持该终端运行。服务把 GroundingDINO 和 SAM2
常驻显存，通过本地 Unix socket `/tmp/winejar_grounded_sam2.sock` 接收任务，
不会连接或控制机械臂。

## 完整实机流程

现场布置：

- 荷叶已经铺在固定酒坛坛口；
- 两片竹壳叠放，白纸放在竹壳上方；
- 陶瓷片单独放在相机视野内；
- 竹壳料区必须位于机器人基坐标约
  `X=[400,540] mm, Y=[50,180] mm`；
- 相机、酒坛、机械臂底座和料台均不得移动。

先在 UFactory 中释放急停并使能机械臂，然后在控制环境运行：

```bash
conda activate winejar-xarm-sdk
python xarm_sdk_control/scripts/21_complete_loading.py \
  --run --skip-lotus \
  --require-vision-service --allow-validated-clipped-leaf \
  --serial 014122072372 \
  --min-x-mm 180 --max-x-mm 520 \
  --min-y-mm -250 --max-y-mm 180 \
  --min-z-mm -85 --max-z-mm 500
```

任务顺序为白纸、陶瓷片、竹壳 1、竹壳 2。竹壳放置角分别为 `+45` 和
`-60` 度，用于补偿竹壳与吸盘之间的实际转动滑移。脚本开始时自动移动到
相机安全复位位姿；中断后可用 `--start-material` 从某一物料继续，但必须由
现场人员确认前序层确实已经放好。

## 安全约束

- 所有实机命令必须显式提供 X/Y/Z 工作空间边界。
- 实验期间必须有人留在急停按钮旁，不得无人运行。
- 当 `X > 300 mm` 时，TCP 高度不得超过 `Z=300 mm`，避免碰撞全局相机。
- 接触、吸附确认和释放使用精确停止；只有空载安全接近段允许轨迹混合。
- 吸附异常时优先保持真空并执行安全恢复，避免物料突然掉落。
- 控制器错误 C1 表示控制箱急停被按下，必须先检查现场再解除。
- 相机、TCP 或料台移动后，旧标定和人工偏差补偿均不再可信。

## 压力确认和重试

泵启动后，CI0 必须在 2 秒内连续 50 ms 保持为 `0`，才判定吸附成功。
失败时任务会回到料堆、放气、撤离、重新扫描并换吸点，最多尝试 3 次。
陶瓷片和竹壳在每个载荷运动段后继续检查压力。

白纸存在少量稳定漏气：初次吸附确认仍为强制条件，但搬运途中一次压力丢失
只记录为警告；该策略已经在现场验证成功。后续若更换吸盘或压力阈值，应重新
评估这一例外。

## 常用脚本

| 脚本 | 用途 |
|---|---|
| `01_read_status.py` | 只读机械臂状态 |
| `02_prepare_robot.py` | 准备并使能机械臂 |
| `11_collect_global_camera_sample.py` | 采集固定相机标定样本 |
| `12_solve_global_camera_calibration.py` | 求解固定相机外参 |
| `13_capture_realsense_frame.py` | 保存一帧对齐 RGB-D |
| `15_vacuum_io_check.py` | 检查 CO0/CO1，不移动机械臂 |
| `18_calibrate_visual_position_bias.py` | 测量视觉系统偏差 |
| `19_visual_cross_place_two_leaves.py` | 双竹壳交叉放置实验 |
| `20_visual_pick_test.py` | 单物料吸起、悬停和放回测试 |
| `21_complete_loading.py` | 当前完整上料生产入口 |
| `22_suction_sensor_check.py` | 只读 CI0 压力开关 |

单物吸取示例：

```bash
python xarm_sdk_control/scripts/20_visual_pick_test.py --material ceramic
python xarm_sdk_control/scripts/20_visual_pick_test.py --material paper
python xarm_sdk_control/scripts/20_visual_pick_test.py --material bamboo
```

默认是 dry-run；需要实机运动时必须增加 `--run`、视觉服务要求和明确的工作
空间边界。

## 测试与数据

测试不连接相机或机械臂：

```bash
conda activate winejar-xarm-sdk
python -m pytest -q xarm_sdk_control/tests
```

`xarm_sdk_control/data/` 中的 RGB-D、掩码、预览图和运行 JSON 都是可再生成的
本地产物，不参与程序运行，也不会推送到 GitHub。需要复盘时保留相关任务的
单个时间戳目录即可；`calibration/` 与 `config/` 不是实验产物，禁止随数据清理。

## 已知限制与下一步

1. 荷叶尚不能稳定从料台吸起，当前采用预铺方案。
2. 完整流程稳定性已有一次完整复验，仍需要更多重复实验统计成功率。
3. 当前 81 秒节拍包含保守轨迹估算和一次相机重连，仍有优化空间。
4. 视觉补偿目前主要来自一次人工对齐，后续应增加多点校验并统计残差。
5. 压力开关只能确认真空是否达到阈值，不能测量连续真空度。

完整的环境配置、标定、现场命令、参数依据、迁移和验收说明见
[项目实施与复现指南](docs/WineJar项目实施与复现指南.md)。更详细的代码模块说明见
[xarm_sdk_control/README.md](xarm_sdk_control/README.md)，视觉服务说明见
[vision/README.md](vision/README.md)。
