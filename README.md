# WineJar

基于 xArm6、RealSense D435i、GroundingDINO 和 SAM2 的酒坛柔性物料视觉上料项目。

当前实机流程能够完成：

- 固定全局相机 RGB-D 采集与坐标标定
- GroundingDINO 开放词汇检测
- SAM2 像素级分割
- 竹壳与陶瓷片三维定位
- xArm6 视觉引导运动
- 自制单吸盘的泵与真空/放气切换阀控制
- 单片竹壳吸放
- 两片叠放竹壳二次视觉定位与交叉放置

MuJoCo 仿真与演示动画维护在独立仓库：
[Wunicheng233/Winejar-mujoco](https://github.com/Wunicheng233/Winejar-mujoco)。

## 仓库结构

```text
.
├── docs/                  项目设计、标定与实施文档
├── vision/                GroundingDINO + SAM2 独立视觉环境
├── xarm6/                 xArm 用户与开发者手册
├── xarm_sdk_control/      实机控制、标定、视觉抓取与测试
└── 项目策划.docx          原始项目策划
```

采集图像、深度图、分割结果和运行日志不会提交到 Git。它们分别生成在
`xarm_sdk_control/data/` 与 `vision/data/`，可在实验复盘完成后直接清理。
模型权重保存在 `vision/models/`，不提交 Git，但运行视觉服务时必须保留。

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

视觉模型首次运行时由 Hugging Face 下载。不要把模型权重提交到仓库。

## 硬件约定

- xArm6 控制器地址默认存放在 `xarm_sdk_control/config/robot.yaml`
- RealSense 型号：Intel RealSense D435i
- `CO0`：真空泵继电器
- `CO1`：真空/放气通路电磁阀
- 当前单吸盘 TCP：法兰中心轴向下 `120 mm`
- 当前没有真空压力传感器

GPIO 真值表：

| 状态 | CO0 气泵 | CO1 通路阀 |
|---|---:|---:|
| 待机 | 0 | 0 |
| 吸取与保持 | 1 | 1 |
| 主动释放 | 0 | 0 |
| 释放完成 | 0 | 0 |

执行器参数统一存放在 `xarm_sdk_control/config/end_effector.yaml`。当前单吸盘与弹簧金具已经实机验证，默认压缩量为 `13 mm`；更换吸盘或金具后必须重新确认该值。

## 常用流程

只读检查机械臂：

```bash
python xarm_sdk_control/scripts/01_read_status.py
```

检查真空 I/O，不启用机械臂运动：

```bash
python xarm_sdk_control/scripts/15_vacuum_io_check.py --test status
python xarm_sdk_control/scripts/15_vacuum_io_check.py --test cycle --hold-seconds 2
```

单片竹壳非接触接近检查（无真空动作，仅下降至吸点上方并悬停观察）：

```bash
python xarm_sdk_control/scripts/16_visual_vacuum_pick.py \
  --run \
  --allow-clipped-leaf-for-approach \
  --min-x-mm 180 --max-x-mm 520 \
  --min-y-mm -180 --max-y-mm 180 \
  --min-z-mm -75 --max-z-mm 430 \
  --transit-speed 70 \
  --descent-speed 20 \
  --acc 240
```

接触式吸放任务由 `19_visual_cross_place_two_leaves.py`（两片交叉放置）与
`20_visual_pick_test.py`（有人值守单物吸起测试）承担。

两片竹壳交叉放置：

```bash
python xarm_sdk_control/scripts/19_visual_cross_place_two_leaves.py \
  --run \
  --confirm-no-pressure-sensor \
  --require-vision-service \
  --allow-validated-clipped-leaf \
  --compression-mm 13 \
  --min-score 0.35 \
  --min-x-mm 180 --max-x-mm 520 \
  --min-y-mm -180 --max-y-mm 180 \
  --min-z-mm -85 --max-z-mm 500
```

脚本默认将速度按工况分层：空载搬运 `300 mm/s`、带料搬运 `220 mm/s`、
普通竖直 `200 mm/s`、带料竖直抬升 `120 mm/s`、精细动作 `60 mm/s`、
接触 `20 mm/s`。第一片放置后机械臂停留在陶瓷片上方完成第二次视觉，
不执行中间复位；RealSense pipeline 在两次扫描之间保持运行。

每片竹壳动作前，程序会先做连续逆解采样，估算各关节速度和加速度，
随后调用 xArm 控制器进行一次无动作路径预检。任何超速、碰撞、限位或
规划错误都会在真空开启和机械臂运动前终止；程序不会在实机任务中自动
搜索或改写速度参数。

## 安全规则

- 实机命令必须明确提供工作空间边界。
- 第一次运行新路径时，操作者必须位于急停按钮可及位置。
- 控制器错误 `C1` 表示控制箱急停按钮被按下，必须先物理释放再重新使能。
- 没有压力传感器时，程序无法确认物料是否吸牢。
- 吸附状态下发生异常时，程序会优先保持真空，避免物料突然掉落。
- 标定文件和 `visual_position_bias.json` 与当前实验台布局绑定；移动相机、机械臂或执行器后必须重新验证。

## 测试

在任意目录运行（测试夹具会自动定位仓库根目录）：

```bash
conda run -n winejar-xarm-sdk python -m pytest -q xarm_sdk_control/tests
```
