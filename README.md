# WineJar

基于 xArm6、RealSense D435i、GroundingDINO 和 SAM2 的酒坛柔性物料视觉上料项目。
任务目标：依次把荷叶、白纸、陶瓷片、两片叠放的竹壳从桌面料堆吸起，按固定教点
交叉叠放到酒坛坛口。

本文档同时是**项目交接说明**：当前实机进展、架构、以及留给后续接手者
（Codex）的已知问题与下一步，见文末「已知问题与下一步」。

## 当前实机进展（2026-08-22）

已验证可行：

- 全局相机 RGB-D 标定、视觉定位偏差补偿（`visual_position_bias.json`）
- GroundingDINO 开放词汇检测 + SAM2 分割（逐物单短语提示词，防止标签合并）
- 陶瓷片 / 白纸 吸起-悬停-放回闭环（脚本 20，实机成功）
- 双竹壳交叉放置（脚本 19）第一片吸取、放置、二次扫描均走通
- 真空压力开关吸起确认链路完整：真实吸起 ~50ms 内确认；受控失败（泵不开）
  → 自动恢复 → 重规划 → 重试成功，实机验证过
- 搬移途中失压的持续检查（loaded 段后复查）曾真实捕捉到片 2 中途脱落

尚未完成（按优先级）：

1. **片 2（底层竹壳）抬升阶段反复脱落**。压缩量加到 16mm、重试偏移
   （(0,0) → (0,-5) → (5,0)mm）都试过仍会漏气。疑似竹壳弯曲弧面与吸盘刚性
   唇口不贴合，属硬件/物料问题。约束：**不能翻面**（上料对正反面有要求）。
2. **荷叶吸不起来**。四种物料里最难吸的，换吸盘、调压缩、调吸点均未成功。
3. **完整上料（脚本 21）尚未端到端跑通**。荷叶/白纸/陶瓷单段都验证过，
   双壳段受问题 1 阻塞。
4. 视觉复核 `inconclusive` 目前视为成功（用户拍板：成功吸起常被判
   inconclusive），视觉可靠后再收紧。

## 仓库结构

```text
.
├── docs/                  项目设计、标定与实施文档
├── vision/                GroundingDINO + SAM2 独立视觉环境与服务
├── xarm6/                 xArm 用户与开发者手册
├── xarm_sdk_control/      实机控制、标定、视觉抓取与测试（主工作目录）
│   ├── config/            robot.yaml / end_effector.yaml / jar_position.yaml
│   │                      / visual_position_bias.json（不可删）
│   ├── scripts/           任务入口脚本 01~22（下面有说明）
│   ├── tests/             130 个 pytest（假臂夹具，不动机器人）
│   ├── data/              实验产物（帧、掩码、result.json），可随时清理
│   ├── calibration/       相机标定文件（不可删）
│   └── *.py               共享模块（下面有说明）
└── 项目策划.docx          原始项目策划
```

`xarm_sdk_control/README.md` 还有一份英文版模块与机制细述（部分章节
滞后于最新代码，以本文件为准）。

## 环境配置

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

**踩坑记录（交接必读）：**

- `conda run` 包装对 torch 环境是坏的（加载不到 torch）。视觉服务必须用
  解释器绝对路径直接启动，且必须 `setsid nohup`，否则随 shell 退出而死：

  ```bash
  setsid nohup env HF_HUB_OFFLINE=1 \
    /home/wunicheng/anaconda3/envs/winejar-vision/bin/python -u \
    vision/scripts/03_grounded_sam2_service.py \
    > /tmp/vision_service.log 2>&1 < /dev/null & disown
  ```

  服务在 `/tmp/winejar_grounded_sam2.sock` 上就绪后打印
  `"status": "ready"`。控制侧脚本用 `--require-vision-service` 强制依赖它。
- 控制侧测试直接用解释器路径跑（任意 cwd 均可，夹具自动定位仓库根）：

  ```bash
  /home/wunicheng/anaconda3/envs/winejar-xarm-sdk/bin/python \
    -m pytest -q xarm_sdk_control/tests
  ```

- 模型权重在 `vision/models/`（不入库）；首次运行需联网下载，离线用
  `HF_HUB_OFFLINE=1`。
- RealSense 偶尔 USB 掉线：物理重插线即可，服务会自动重启一次。

## 硬件约定

| 项 | 值 |
|---|---|
| 机械臂 | xArm6，SDK 1.18.4，IP 在 `config/robot.yaml` |
| 相机 | RealSense D435i 固定全局俯视 |
| CO0 | 真空泵继电器 |
| CO1 | 真空/放气通路电磁阀 |
| CI0 | ZSE30A-01-N 真空压力开关（NPN 三线）：**0 = 真空达标（吸起成功）**，1 = 无真空 |
| 吸盘 TCP | 法兰中心轴向下 120mm（单吸盘，`end_effector.yaml`） |
| 默认压缩量 | 13mm（竹壳；陶瓷 6mm、白纸 1mm、荷叶 2mm 见脚本 21 参数） |
| 酒坛坛口 | 固定教点 (300, -150, 65)mm，存 `config/jar_position.yaml` |

气泵真值表：

| 状态 | CO0 气泵 | CO1 通路阀 |
|---|---:|---:|
| 待机 | 0 | 0 |
| 吸取与保持 | 1 | 1 |
| 主动释放 | 0 | 0 |

## 共享模块（架构）

任务脚本只做编排，逻辑都在可导入模块里（tests/ 用假臂直接测它们）：

| 模块 | 职责 |
|---|---|
| `robot_motion.py` | 连接、准备、急停检查、安全复位 |
| `visual_targeting.py` | 标定加载、RGB-D 扫描、检测物分类与选择 |
| `cross_place_route.py` | 双壳循环与完整上料的路由几何构造 |
| `cross_place_motion.py` | 路由执行：速度剖面校验、关节感知规划、**吸起确认 + loaded 段持续检查** |
| `cross_place_retry.py` | **吸起失败 → 恢复 → 重规划 → 重试**的通用循环 |
| `trajectory_speed_planner.py` | 密集逆解采样规划 + 端点插值回退规划 + 控制器预检 |
| `pick_test_workflow.py` | 脚本 20 单物吸起测试的剖面与执行 |
| `suction_sensor.py` | CI0 压力开关读取、去抖、超时（M4） |
| `vacuum_control.py` / `end_effector_config.py` | 泵阀 I/O、执行器参数 |
| `positioning_bias.py` | 视觉定位偏差补偿 |
| `realsense_capture.py` | RGB-D 采集会话 |
| `geometry.py` / `cli_utils.py` / `run_output.py` / `task_timing.py` | 位姿数学、argparse 公共参数、运行报告、事件时间线 |

## 任务脚本

| 脚本 | 用途 |
|---|---|
| `01_read_status.py` / `02_prepare_robot.py` | 只读状态 / 准备 |
| `11/12/13` | 全局相机采样 / 标定解算 / RealSense 帧采集 |
| `15_vacuum_io_check.py` | 泵阀 I/O 检查（无运动） |
| `16_visual_vacuum_pick.py` | 单片竹壳非接触接近检查 |
| `18_calibrate_visual_position_bias.py` | 视觉偏差标定 |
| `19_visual_cross_place_two_leaves.py` | 双竹壳交叉放置（核心实验） |
| `20_visual_pick_test.py` | 单物料吸起-悬停-放回验证（`--material`） |
| `21_complete_loading.py` | 完整封坛上料（5 段） |
| `22_suction_sensor_check.py` | 只读打印 CI0，带时间戳 |

### 双竹壳交叉放置（19）

```bash
python xarm_sdk_control/scripts/19_visual_cross_place_two_leaves.py \
  --run \
  --require-vision-service \
  --allow-validated-clipped-leaf \
  --compression-mm 16 \
  --min-score 0.35 \
  --min-x-mm 180 --max-x-mm 520 \
  --min-y-mm -180 --max-y-mm 180 \
  --min-z-mm -85 --max-z-mm 500
```

- 默认路径（不带 `--verify-pick`）已经走压力开关确认 + 重试循环
- `--simulate-first-pick-failure --max-pick-attempts 3`：受控实验——片 1 第一次
  接触时泵不开，用真实传感器失败驱动恢复+重试链路（无需 `--verify-pick`）
- 二次扫描阈值 `--second-scan-min-score` 默认 **0.10**（底层壳对比度低；
  0.20 曾漏检 0.187 分的片 2）。80mm 近源门限保证不会误选已放置的壳
- `--verify-pick`：可选视觉复核（inconclusive 视为成功，见已知问题 4）

### 单物料吸起测试（20）

```bash
python xarm_sdk_control/scripts/20_visual_pick_test.py --material ceramic
python xarm_sdk_control/scripts/20_visual_pick_test.py --material lotus-leaf
python xarm_sdk_control/scripts/20_visual_pick_test.py --material paper
python xarm_sdk_control/scripts/20_visual_pick_test.py --material bamboo
```

一次开场扫描，检测画面中**所有**该物料并按置信度逐个吸起-悬停-放回原位。
每件最多 3 次尝试，重试按偏移序列换吸点；传感器未确认吸起即安全中止
（退出码 2）。

### 完整封坛上料（21）

现场布置：A 堆 = 两片竹壳叠放；B 堆 = 从上到下荷叶、白纸、陶瓷片。

```bash
# dry-run（无运动）→ 检查通过后加 --run 实机
python xarm_sdk_control/scripts/21_complete_loading.py \
  --require-vision-service \
  --min-x-mm 180 --max-x-mm 520 \
  --min-y-mm -180 --max-y-mm 180 \
  --min-z-mm -85 --max-z-mm 500
```

5 段：荷叶 → 白纸 → 陶瓷片 → 竹壳 1 → 竹壳 2，全部叠放到坛口教点。
层高累积：荷叶 4 / 白纸 0.5 / 陶瓷 5 / 竹壳 3mm（均可调）。
`--max-legs N` 只跑前 N 段；`--intermediate-home` 每段回位。
每段独立走压力确认 + 重试循环（`--max-pick-attempts` 默认 3）；
全部失败 → result.json(failed) + failure.json + 退出码 2。

## 吸起确认与重试机制（M4 核心）

流程：泵开 → 静止接触态轮询 CI0（10ms 采样，**50ms 连续成功去抖**，
**2s 超时**）→ 确认后抬升；每个 loaded 段结束后**复查一次压力**
（持续检查，捕捉中途脱落）。

未确认 → 记 `suction_failed` → 保持吸附状态安全恢复（回到料堆放气、
后撤、回位）→ 新扫描重规划 → 换吸点重试（偏移序列 (0,0) → (0,-5) →
(5,0)mm）。19/21 走重试；20 直接中止退出码 2。

重试目标选择统一加 **80mm 近源门限**：候选必须距原料堆 ≤80mm，防止误选
已放置物料或掉落他处的物料。

已知 v1 局限：传感器只在静止接触态与段间轮询，**段内（运动中）失压
不实时检测**；`--verify-pick` 视觉复核仍可覆盖该盲区。

## 运动规划（含奇异点问题）

每片动作前先做连续逆解采样，估算关节速度/加速度并缩放；随后 xArm
无动作路径预检。**预检失败现在是警告而非中止**（`ControllerPreflightError`
被捕获降级——真实 C24 曾反复出现但实机执行正常，用户拍板放宽）。

home 位形（roll ≈ 178.5°）靠近腕部奇异点：控制器 IK 求解器对近乎相同的
输入会间歇翻转 J4/J6 分支，密集采样因此出现假「IK discontinuity」和
0.00mm/s² 曲率噪声告警。对策（`trajectory_speed_planner.py`）：

1. 估计失败重试 3 次；
2. 仍失败则回退 **`plan_cartesian_route_speeds_endpoint`**：只在段端点解 IK
   （锚定上一端点），段内线性关节插值，损失曲率上限但免疫分支翻转。
   实机已用回退规划成功执行过片 1 路线。

可选后续优化：IK 链做 3 点滑动平均后再算曲率，进一步压密集规划的假告警。

## 视觉管线要点

- 每类物料用**单短语提示词**分别扫描（"lotus leaf." / "paper label." /
  "ceramic disk." / "bamboo leaf."），多短语会触发 GroundingDINO 标签合并
- `visual_targeting.detection_object_kind` 按关键词唯一匹配分类物料
- 二次扫描（已取走一片后）允许更低置信度：`--second-scan-min-score 0.10`
- 标定与偏差补偿与实验台布局绑定：**移动相机/机械臂/执行器后必须重新
  标定验证**（`visual_position_bias.json`）

## 已知问题与下一步（交接给 Codex）

按优先级：

1. **片 2 抬升脱落（最优先）**：16mm 压缩 + 偏移重试仍漏气。疑似壳弧面与
   吸盘唇口不贴合。方向：换唇口更软的吸盘/柔性密封垫；吸点改选弧面较平
   处；或与用户讨论斜吸。**禁止翻面**。
2. **荷叶吸不起**：多种压缩/吸点组合失败。方向：更大吸盘、贴补强纸、
   降低吸取抬升速度。
3. **脚本 21 端到端验证**：单段都已验证，双壳段被问题 1 阻塞；修好后按
   dry-run → 实机流程跑通 5 段。
4. **视觉复核收紧**：`inconclusive` 现视为成功（用户拍板，视觉可靠后改），
   `pick_verification.py` 与 `--verify-pick` 的验证路径仍保留。
5. 可选：运动中失压的中断式连续监测（线程轮询 CI0）；IK 3 点平滑。
6. 待办清理（需用户确认后再删）：`validate_pick_test_args` 死代码、
   `--skip-second-controller-preflight`（预检已降级为警告，此参数仅剩
   消音作用）。

## 安全规则

- 实机命令必须明确提供工作空间边界（X/Y/Z min/max），否则拒绝执行
- 实机实验必须在急停按钮旁有人值守
- 控制器错误 `C1` = 控制箱急停被按下：**必须先物理检查释放**再重新使能
- 相机避让：X > 300mm 时 TCP 的 Z 必须 ≤ 300mm
- 接触与释放用精确停止（exact stop），不参与转角混合
- 吸附状态下发生异常优先保持真空，避免物料突然掉落
- 吸附失败后机器臂保持吸附状态完成安全恢复（回到料堆放气），防止误判
  紧急停机

## 实验产物

`xarm_sdk_control/data/` 下的运行记录（RGB-D 帧、掩码、预览图、
`result.json`/`failure.json`、时间线）是复盘用的可清理产物，代码不依赖。
复盘数据在 `data/visual_cross_place/<run>/` 与 `data/visual_pick_test/<run>/`。
