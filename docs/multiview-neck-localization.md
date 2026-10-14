# 三相机坛口定位、离线验收与重新标定

更新：2026-09-09。本文对应 `31_estimate_mouth_height.py` 的 schema 3，
不是早期矿泉水瓶或体素重建方案的精度报告。本流程不控制机械臂。

## 1. 当前现场状态

用户已报告相机位置发生改变。移动前的两组外参、阵列与工位坐标只能用于
对应历史图像，不能用于当前现场。三台 D435i 仍可通过 USB 枚举：

| 角色 | 序列号 | 历史内参文件，仅作复算输入（位于 `xarm_sdk_control/calibration/multiview_neck/`） |
|---|---|---|
| 俯视 overhead | 013222073103 | `overhead_intrinsics_20260904_164948_818.json` |
| 侧视 side_a | 142122070566 | `side_a_intrinsics_20260901_220838_167.json` |
| 侧视 side_b | 011422070041 | `side_b_intrinsics_20260901_221208_025_filtered.json` |

上述内参对应 RGB 640x480，但本次数值检查发现三者的图像边缘畸变反解异常，
不再直接用于新外参采集。四参数替代模型目前只是候选，见 2.0 和第 10 节。
仅改变相机位置通常不需要重做内参；更换相机、镜头、
分辨率或其他影响投影的光学设置，则应重新确认内参。设备绑定依据序列号，
不是 USB 插入顺序。

`scene_status.json` 保存当前场景 ID、准入状态和批准的阵列数值指纹。`26` 在打开流
之前检查：必须是当前场景、显式验收的阵列、绑定的内外参、合格的内参数值模型。
缺少记录、历史阵列、未知阵列及未验收候选均拒绝；改文件名不能绕过检查。
它不是相机位移检测器，移动相机后仍需主动使旧结果失效。
新采集和工位结果记录阵列指纹，定位时检查是否混用了几何配置。
缺少指纹的历史数据仍允许离线回放，但无法获得同等的来源一致性保证。

## 2. 重新标定的顺序

调整安装位置时，先打开不依赖任何标定的实时 RGB 预览：

```bash
conda activate winejar-xarm-sdk
python xarm_sdk_control/scripts/37_preview_three_cameras.py
```

点击预览窗口后，`1/2/3` 放大俯视/A/B，`0` 返回三路总览，`S` 保存当前截图，
`Q` 或 Esc 关闭并释放相机。首次收到三路稳定图像也会自动保存截图到
`data/multiview_neck/live_previews/`。这些帧不是严格同步采集，不能作为测量数据。
出现 `NO LIVE FRAME` 表示该画面已过期。开始 `35` 标定前必须先关闭预览。

先固定相机和支架，撤开遮挡物。两台相机必须在每次采样期间观察同一块静止的
刚性标定板；不同样本之间移动和倾斜板，相机不要动。ChArUco 支持局部可见，
但不能接受模糊、反光、过少或几乎共线的角点。不要通过大幅放松阈值凑够样本。

当前板：9x9 **方格**，方格边长 50 mm，marker 边长 37 mm，字典 DICT_4X4_50。
9x9 不是内角点数量。应核对实物长度和板面平整度；打印规格并不等于已校验尺度。

### 2.0 先检查内参，再固定本轮内参版本

从仓库根目录运行以下纯离线命令，使用保存的角点，不打开相机：

```bash
conda activate winejar-xarm-sdk
CAL=xarm_sdk_control/calibration/multiview_neck
python xarm_sdk_control/scripts/40_evaluate_camera_intrinsics.py \
  --intrinsics "$CAL/overhead_intrinsics_20260904_164948_818.json" \
  --intrinsics "$CAL/side_a_intrinsics_20260901_220838_167.json" \
  --intrinsics "$CAL/side_b_intrinsics_20260901_221208_025_filtered.json"
```

输出位于 `data/multiview_neck/intrinsic_evaluation/<时间戳>/`，每台各有
`*_brown4.json`、`*_brown5.json`，总报告为 `evaluation.json`。加 `--redetect`
会从原始 RGB 重新检测，检测失败直接报错，不悄悄删除样本。

本次已生成候选目录如下；重新运行后，应将 `INTR` 改为实际输出目录：

```bash
INTR=xarm_sdk_control/data/multiview_neck/intrinsic_evaluation/20260909_195835_286
```

以下外参命令使用其中通过数值检查的 Brown4 候选，最终还需独立验证。
两组外参必须使用同一份俯视内参数值。不要在外参求解后替换内参文件内容，
即使序列号相同也不行；更换内参必须重新求外参。

若需要重新拍内参，继续使用 `23`，例如俯视相机：

```bash
python xarm_sdk_control/scripts/23_calibrate_camera_intrinsics.py \
  --camera-id overhead --serial 013222073103 \
  --pattern-cols 11 --pattern-rows 8 --square-size-mm 20 --model brown4
```

`brown4` 估计 k1、k2、p1、p2，将 k3 固定为 0；`brown5` 额外估计 k3。
不是参数越多越准确。标定板必须平整，覆盖中心、边缘、不同距离和倾角；
不要只用中心附近的小范围图像。外参所用板不必与内参所用板相同。

### 2.1 俯视与侧视 A

以下命令均从仓库根目录运行：

```bash
conda activate winejar-xarm-sdk
CAL=xarm_sdk_control/calibration/multiview_neck
python xarm_sdk_control/scripts/35_calibrate_camera_pair_charuco.py \
  --reference-id overhead --reference-serial 013222073103 \
  --reference-intrinsics "$INTR/overhead_brown4.json" \
  --target-id side_a --target-serial 142122070566 \
  --target-intrinsics "$INTR/side_a_brown4.json" \
  --squares-x 9 --squares-y 9 --square-size-mm 50 --marker-size-mm 37 \
  --dictionary DICT_4X4_50
```

窗口内 `S` 保存合格样本，至少 12 组才能按 `C` 求解，默认 30 组自动结束采集。
`Q` 或 Esc 取消。建议采满姿态多样的 30 组，覆盖实际坛口附近的空间，而非只在
一个位置轻微晃动。2026-09-09 起，采集默认要求每台至少 20 个角点、角点凸包
占图像面积至少 3%，单目 PnP 重投影误差不超过 1 px；这些不是整体精度保证。
保存前还检查连续静止、角点分布和重复姿态，具体参数见文末新增说明。

### 2.2 俯视与侧视 B

```bash
python xarm_sdk_control/scripts/35_calibrate_camera_pair_charuco.py \
  --reference-id overhead --reference-serial 013222073103 \
  --reference-intrinsics "$INTR/overhead_brown4.json" \
  --target-id side_b --target-serial 011422070041 \
  --target-intrinsics "$INTR/side_b_brown4.json" \
  --squares-x 9 --squares-y 9 --square-size-mm 50 --marker-size-mm 37 \
  --dictionary DICT_4X4_50
```

两轮之间不能移动俯视相机。无需机械臂，也无需测量标定板到 TCP 的距离。
输出带时间戳的新文件，不覆盖旧结果。

普通棋盘也可以，以下是 B 组命令；A 组替换角色、序列号和目标内参即可：

```bash
python xarm_sdk_control/scripts/24_calibrate_camera_pair_extrinsics.py \
  --reference-id overhead --reference-serial 013222073103 \
  --reference-intrinsics "$INTR/overhead_brown4.json" \
  --target-id side_b --target-serial 011422070041 \
  --target-intrinsics "$INTR/side_b_brown4.json" \
  --pattern-cols 11 --pattern-rows 8 --square-size-mm 20
```

普通棋盘要求完整可见，且内角点行列一奇一偶，以便用黑白格消除 180 度编号
歧义。颜色判断不确定就拒绝当前帧；不要把行列同奇偶的板硬套进这个流程。

### 2.3 验证外参并生成新阵列

将下列尖括号替换为两次采集实际输出的路径：

```bash
python xarm_sdk_control/scripts/36_validate_charuco_extrinsics.py \
  --pair <新的A外参JSON> --pair <新的B外参JSON> --all-folds --method compare
python xarm_sdk_control/scripts/25_create_multiview_camera_rig.py \
  --reference-intrinsics "$INTR/overhead_brown4.json" \
  --pair-extrinsics <新的A外参JSON> --pair-extrinsics <新的B外参JSON>
```

`36` 留出每第五组样本，只用其余样本重新求外参，再检查留出样本的跨相机
重投影误差，以及共同角点三角化后的相邻棋格距离。内参没有一同留出，且棋盘
实际尺寸仍是输入假设，不能将该结果当作坛口的最大绝对误差。
`25` 默认仅生成候选，不改变当前批准状态，不能直接用于 `26`。允许静态采集
还需要下面的新图像独立验收；旧文件缺少数值绑定时，`36` 可显式加
`--allow-legacy-input` 做历史诊断，但不能用该参数绕过新阵列的准入。

### 2.4 新图像独立验收与显式启用

拟合完成后，固定一块新的静止板进行独立检查。普通棋盘和 ChArUco 都支持，
实际板规格自动从外参文件读取：

```bash
python xarm_sdk_control/scripts/38_check_stationary_charuco_pair.py \
  --pair <待检查的外参JSON> --samples 5
```

使用外参文件记录的相机序列号、分辨率和板规格，不连接机器人。每帧输出单目
PnP 残差与双向跨相机残差；覆盖图的绿圈为检测角点，红十字为旧外参预测角点。
结果保存在 `data/multiview_neck/stationary_validation/`，不会重新拟合或覆盖外参。
单个姿态的连续帧只能检查稳定性，需要另外移动并固定板以覆盖不同姿态。
返回 0 仅表示完成采集和计算，不表示标定精度合格；角点检测不全返回 3，保留
失败记录及原图。2026-09-09 的 B 组静止检查约 8 px，当前仍待复核，不能跳过。

**每个相机对分别采至少五个不同姿态，每个姿态至少三帧。** 每次运行 `38` 前
移动并固定板，覆盖实际坛口区域不同位置、高度和倾角；不能只连续运行五遍而不
移动板。程序以至少 20 mm 平移或 5 度转角区分姿态，但这个最低条件并不证明
全工位覆盖，空间分布仍需人工检查。每次保留 `validation.json` 路径。

确认验收门槛后，显式提交全部报告，例如：

```bash
python xarm_sdk_control/scripts/25_create_multiview_camera_rig.py \
  --reference-intrinsics "$INTR/overhead_brown4.json" \
  --pair-extrinsics <新的A外参JSON> --pair-extrinsics <新的B外参JSON> \
  --accept-for-capture --max-cross-rmse-px 1.5 --max-cross-p95-px 3.0 \
  --validation-report <A姿态1的validation.json> \
  --validation-report <A姿态2的validation.json> \
  --validation-report <A姿态3的validation.json> \
  --validation-report <A姿态4的validation.json> \
  --validation-report <A姿态5的validation.json> \
  --validation-report <B姿态1的validation.json> \
  --validation-report <B姿态2的validation.json> \
  --validation-report <B姿态3的validation.json> \
  --validation-report <B姿态4的validation.json> \
  --validation-report <B姿态5的validation.json>
```

这里 1.5/3.0 px 是演示门槛，不是已经确认的工厂标准，也不是毫米保证。
程序没有默认替操作者选择这两个验收门槛，必须明确提供。所有样本都必须通过，
不会丢弃失败帧后只检查平均数；同时检查两路单目 RMSE 不大于 1 px、数值绑定、
场景一致性、验证原图哈希，以及是否复用了拟合原图或重复证据。

启用成功会先备份 `scene_status.json`，再原子更新批准的阵列指纹。原始内参、
外参和报告不覆盖。此状态只允许静态视觉采集，不代表可以控制机械臂。
本次软件修正没有执行启用，当前仍是 `calibration_required`。

### 2.5 重新采空场、建立高度零面，再拍酒坛

```bash
python xarm_sdk_control/scripts/26_capture_three_camera_frame.py \
  --rig <新阵列JSON> --output-root xarm_sdk_control/data/multiview_neck/new_backgrounds
python xarm_sdk_control/scripts/30_calibrate_workcell_plane.py \
  --rig <新阵列JSON> --capture <新空场目录> --roi <承载面内部x,y,width,height>
python xarm_sdk_control/scripts/26_capture_three_camera_frame.py \
  --rig <新阵列JSON> --output-root xarm_sdk_control/data/multiview_neck/new_jars
```

空场 ROI 应只含实际支承平面，不含坛体、板边、台阶或另一个高度的桌面。
默认平面阈值 2 mm、最低内点率 0.85、最大 RMS 1.5 mm；失败时先检查有效深度
和 ROI，不应为了通过而盲目放宽。旧场景曾用 3 mm/0.75 的门限，不能照搬到新场景。
输出 Z=0 为承载面，+Z 向上；该坐标不是机器人基坐标，仍需独立建立两者的变换。

## 3. 为什么使用 GroundingDINO + SAM2，但不让它们直接测高度

本项目复用已有模型和加载接口，没有训练或手写替代检测网络。
GroundingDINO 根据文本找酒坛，SAM2 根据检测框提取物体掩码。
其原始用途分别是开放词汇检测与提示式分割，参见
[GroundingDINO 官方实现](https://github.com/IDEA-Research/GroundingDINO) 和
[SAM2 官方实现](https://github.com/facebookresearch/sam2)。

本次实际分割中，侧视 A 的背景干扰显著减少，但侧视 B 仍把后方白色管道连入了
酒坛掩码。这说明语义掩码适合确定分析范围，不等于毫米级边界，也不自带米制坐标。
最终几何必须检查真实图像边缘、多视图重投影和观测之间的一致性。

```bash
conda activate winejar-vision
HF_HUB_OFFLINE=1 python vision/scripts/04_segment_multiview_jar.py \
  --capture <新酒坛采集目录> --prompt "ceramic jar. pottery vase." --min-score 0.25
```

模型在本次进程中只加载一次，再处理三张图。输出每台的 `mask.png`、`overlay.png`
与总 `segmentation.json`。模型环境与控制环境保持分离；没有增加机器人实时依赖。

## 4. 当前几何算法

1. 俯视图：在酒坛掩码内寻找封闭的暗色开口，拟合整圈椭圆并细化灰度边缘。
   不再用最高深度点的均值代替坛口中心，避免把中心推向口沿一侧。
2. 侧视图：提取真实上边缘、左右轮廓和局部坛肩斜率变化。被误分入的管道若顶端
   位于预测窗外，就拒绝其观测，而不是裁出一条假的水平口沿。
3. 三视图联合拟合：以共享的三维圆心、平面法向、内半径及壁厚表示内外口沿，
   将圆投影到三张图上，以像素残差求解几何参数。允许坛口平面倾斜。
4. 采用有边界的鲁棒最小二乘，降低少数异常边缘的影响；三个初始倾斜方向用于
   缓解局部最优。每视图平衡权重，不让俯视图因为边缘点多而压过侧视图。
5. 对坛肩候选像素使用完整 `(u,v)` 视线和拟合轴的切线几何换算三维高度。
   不再把整幅侧视图的某一行对应到中心轴上的一个高度。
6. 检查各视图残差、切线半径误差、两台侧视边界差和 RGB/深度差异，保留被拒绝
   的点与原因。深度只用于初始化及独立交叉检查，不作为精确无误的高度约束。

投影、去畸变、PnP 与三角化使用
[OpenCV calib3d](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)；
非线性优化调用 [SciPy least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)
的 `soft_l1` 损失。这是针对工位设计的几何测量组合，不是复刻某一篇“酒坛定位”论文。

**模型限制：** 当前假设近似圆形、内外沿近似共面同心，且近颈段可近似为圆柱。
平面法向是拟合口沿的法向，不是独立测得的整段坛颈中心轴。明显椭圆、局部翘曲、
歪颈而口沿与颈轴不垂直时，不能直接套用这一解释。侧视主要使用朝下的图像纵轴
分析局部轮廓，尚未支持任意大滚转相机姿态。封盖、遮住开口或严重遮挡应报失败。

## 5. 输出、参数与回放

控制环境新增依赖 `scipy>=1.11,<2`，已记录在 `environment.yml`。
已有环境可用 `python -m pip install 'scipy>=1.11,<2'` 补装；本次验证版本为 1.17.1。

历史数据的完整回放：

```bash
conda activate winejar-xarm-sdk
python xarm_sdk_control/scripts/31_estimate_mouth_height.py \
  --config xarm_sdk_control/config/neck_localization_20260908.json
```

该配置只引用相机移动前的保存数据，Git 不包含这些 RGB-D；在另一台电脑需要转移
相应数据或使用自己的新采集。当前掩码元数据按原采集路径绑定，搬迁目录时也须
同步其 `capture_dir`，不能把掩码与另一帧配用。新数据可不建配置，直接指定：

```bash
python xarm_sdk_control/scripts/31_estimate_mouth_height.py \
  --rig <新阵列JSON> --workcell-frame <新工位JSON> --capture <新酒坛目录> \
  --masks <新分割目录> \
  --roi overhead:<x,y,width,height> --roi side_a:<x,y,width,height> \
  --roi side_b:<x,y,width,height>
```

也可用 `--background-capture` 代替 `--masks`，但必须使用同机位、同场景、可比曝光
的空背景。当前优先使用模型掩码。ROI 必须随场景重选，不能继续用历史像素框。
配置内路径相对配置文件；命令行路径相对当前工作目录。

| 配置参数 | 默认值 | 作用及选择依据 |
|---|---:|---|
| `minimum_height_mm` / `maximum_height_mm` | 30 / 600 | 相对承载面的初始化高度筛选；随实际酒坛高度调整，不是预设输出 |
| `profile_window_mm` | 26 | 侧视局部坛肩搜索窗；须覆盖颈肩过渡，过大易混入坛体，过小可能没有转折 |
| `edge_search_px` | 1.5 | 围绕掩码边界的灰度细化范围；扩大可能追到纹理而非物体边缘 |
| `maximum_edge_fit_error_px` | 1.2 | 局部折线拟合误差门限；用于拒绝无可靠边界的观测 |
| `maximum_tangent_radius_error_mm` | 4 | 边界视线与拟合圆柱的径向不一致门限；不是允许的最终定位误差 |
| `maximum_neck_spread_mm` | 5 | 两台侧视坛肩估计的差异门限；超限保留结果并标记复核 |
| `maximum_rim_rmse_px` | 2 | 每台相机口沿重投影 RMSE 门限；低残差不能排除共同系统误差 |
| `maximum_depth_disagreement_mm` | 8 | 深度初始高度与 RGB 平面中心高度的差异告警；倾斜也可能造成差异，需查图 |
| `bootstrap_samples` | 20 | 对已选边缘重采样拟合，检查条件稳定性；设 0 可关闭，不影响主拟合 |
| `random_seed` | 7 | 使重采样结果可复现，不影响现场相机标定 |

这些是当前实验的诊断参数，不是工厂验收标准。不要只为让输出变成 `0` 而放宽。
低层还有限制：内半径搜索为初值的 0.65~1.4 倍、壁厚 1~40 mm、圆心 XY/Z
搜索范围为初值附近 120/80 mm。触及边界或无法观测时应复查输入而非输出硬凑。

关键输出严格区分：

- `rim_center_workcell_mm`：拟合口沿平面中心，Z 不是倾斜口沿最高点。
- `rim_plane_normal_workcell` / `rim_plane_equation_workcell_m`：工位系法向和平面方程。
- `rim_z_extent_mm`：倾斜外沿各处的最低/最高 Z，不是颈部范围。
- `rim_to_shoulder_axial_length_mm`：从口沿平面沿法向到坛肩转折的总距离。
- `binding_band_mm`：当前为 null；凸起口沿下的扎带窄槽需要独立证据，不能用上述总距离替代。
- `conditional_uncertainty`：仅固定标定和边界选择下的图像点重采样范围，不含标定、分割偏差、非圆形与平面误差。

返回码 0 为几何检查通过，3 为结果已保存但需复核，2 为输入/拟合失败。
`absolute_accuracy_validated` 和 `robot_motion_allowed` 均为 false，不因返回 0 自动解除。
输出有三张全幅覆盖图和三张 4 倍局部放大图；未使用生成式图像增强。

## 6. 本次离线对照结果

输入：`jar_capture_20260908/20260908_152937_039`，掩码：
`jar_masks/20260908_160714_206061`，均在 `xarm_sdk_control/data/multiview_neck/`。
保留的优化报告：`mouth_heights/20260908_204407_993/mouth_height.json`。

| 项目 | 原方法/当前结果 |
|---|---|
| 旧方法输出 | 顶高 494.84 mm，所谓“颈长”33.13 mm；含经验宽度比例，不是可靠窄槽测量 |
| 新口沿平面中心 Z | 491.27 mm，与旧最高表面指标定义不同，不能直接当作误差减少了 3.57 mm |
| 拟合内径 / 外径 | 102.41 / 119.51 mm，尚未用卡尺验证 |
| 平面倾角 / 外沿 Z 范围 | 4.79 度 / 486.27~496.26 mm，均是模型估计 |
| 口沿至坛肩总距离 | 16.70 mm，不代表用户所说不超过 10 mm 的扎带窄槽 |
| 俯视 / A / B 重投影 RMSE | 1.14 / 0.34 / 1.12 px |
| 两台侧视坛肩距离差 | 3.37 mm；四个候选中只有两个通过全部门限，仍有明显不确定性 |
| RGB 与深度初始高度差 | 约 9.01 mm，超过 8 mm 告警门限，因此为 `needs_review` |
| 重采样中心 Z 的条件 95% 区间 | 490.96~492.06 mm；不能解读为绝对误差约 1 mm |

历史 ChArUco 留出验证：每组 30 帧中的 6 帧留出，24 帧重新求解。
A/B 跨相机 RMSE 分别 0.805/0.946 px，相邻棋格长度绝对误差 P95 为
1.20/0.365 mm。这未显示明显尺度倍数错误，但不能证明坛口指标的绝对准确性。
报告为 `calibration_validation/20260908_204855_375/validation.json`。

验收下一步：重新标定后固定相机，先拍裸坛；实测承载面到口沿的高度、口沿外径，
并明确“窄槽”上下边界。测试不同位置/朝向和多只坛，统计偏差、P95、失败率及
错误通过率，单独检验歪颈和遮挡。当前不承诺最终最大误差。

## 7. 结构体检与数据管理

- `31` 仅负责参数、输入输出，删除了原脚本重复的深度/轮廓分支。
- `neck_observation.py` 负责二维证据，`rim_geometry.py` 负责投影与拟合，
  `mouth_height_pipeline.py` 负责数据编排、质检、可视化；不把模型环境混入控制环境。
- `mouth_height.py` 仅保留初始化所需的稳定深度高度与坐标变换，移除旧半径比例
  融合和轮廓行高查表实现。历史 `28/29` 保留用于旧结果回放，不进入新主线。
- 本次检查不改动已经跑通的机械臂动作、真空 IO 和上料参数。
- 删除可重算的诊断缓存和未被配置/文档引用的调参结果；保留全部原始 RGB-D、
  标定样本、模型权重、有效标定、主要回归结果和这次优化前后的对照。
- 本次清理 `multiview_neck/diagnostics/` 和 `mouth_heights/` 内的
  `20260902_145425_033`、`20260902_195500_074`、`20260902_195513_000`、
  `20260908_204035_093`，约 6.5 MiB；最后一项为本次无重采样的重复调试输出。
- `data/` 约 426 MiB，其中大量是不可凭空再采到的原始标定样本，不应为追求
  清理数字而删除。代码测试不需要相机或机械臂。

```bash
conda activate winejar-xarm-sdk
python -m pytest -q xarm_sdk_control/tests
```

几何测试覆盖倾斜口沿、噪声/离群点、多相机高低视角、切线高度换算、暗口中心、
无坛肩时拒绝输出、背景管道误入掩码和旧阵列失效保护。合成测试检验实现正确性，
不能替代真实场景的毫米级验收。

## 8. 2026-09-09 标定求解与采样检查改进

### 8.1 对照实验与结论

比较对象为 A 组 `overhead_from_side_a_charuco_extrinsics_20260908_213535_459.json`
以及 B 组重采的 `overhead_from_side_b_charuco_extrinsics_20260909_122602_438.json`。
旧文件不修改，实验输出的变换仅放在报告里，没有发布为新的现场阵列。

`pose_average` 是原来的独立 PnP、位姿一致性筛选、平均变换；`stereo` 在同样保留
的训练样本上，按共同的棋盘三维角点对齐两路观测，再调用 OpenCV
`stereoCalibrateExtended`，固定内参，联合优化共享角点的重投影误差。
两路局部可见的角点可以不同，但每组至少需要 8 个共同、非共线角点，否则明确报错，
不会悄悄回退另一种方法。接口将 OpenCV 的目标相机相对参考相机变换求逆，继续遵守
仓库的 `T_reference_from_target` 约定。参考
[OpenCV 双目标定说明](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)。

采用五个交错留出划分：每次 24 组训练、6 组测试，30 组都恰好进入一次测试。
每个划分的两种方法使用相同的保留训练组，测试组不依据输出误差事后删除。
池化 RMSE 按角点数量加权平方误差计算，不是简单平均五个 RMSE。

| 指标 | 位姿平均 | 联合标定 |
|---|---:|---:|
| A：五折池化跨相机 RMSE | 约 1.02 px | 约 1.03 px |
| B：五折池化跨相机 RMSE | 2.203 px | 2.207 px |
| B：原第 5 个留出划分 RMSE | 2.014 px | 1.969 px |
| B：全量拟合后回放独立静止图像 RMSE | 0.831 px | 0.794 px |

结论：联合标定没有稳定的整体优势，不能宣传为精度升级。因此默认仍为
`pose_average`，`stereo` 作为显式可选方法保留。B 的个别留出组仍有较大误差。
静止图像只有一个姿态，其余数据也来自同一采集会话；这些不是全工位独立验收，
更不是毫米误差上限。板面不平、内参偏差、成像质量和采样期间运动仍需实测排查。

两相机组五折报告：`data/multiview_neck/calibration_validation/20260909_124200_066/validation.json`。
B 组含独立静止图像的完整报告：`data/multiview_neck/calibration_validation/20260909_124632_868/validation.json`。
上述路径相对于 `xarm_sdk_control/`。

从仓库根目录复现实验，不连接任何设备：

```bash
conda activate winejar-xarm-sdk
python xarm_sdk_control/scripts/36_validate_charuco_extrinsics.py \
  --pair xarm_sdk_control/calibration/multiview_neck/overhead_from_side_b_charuco_extrinsics_20260909_122602_438.json \
  --method compare --all-folds --allow-legacy-input \
  --stationary-capture xarm_sdk_control/data/multiview_neck/stationary_validation/20260909_122200_790
```

`--method compare` 同时计算两种方法；只提供 `--method stereo` 仅跑联合方法。
不加 `--all-folds` 时仍使用原有每第五组留出的划分。静止回放重新用全部标定样本
拟合一次，只在 fold 4 的记录中输出，避免混同训练子集与完整标定。

### 8.2 采样检查

采样检查集中在 `calibration_sampling.py`。`24/35` 现在是两个薄入口，共用
`pair_calibration_session.py` 的窗口、按键和保存流程，不引入新的相机框架。
按 S 被拒绝时窗口显示具体原因，成功样本记录各路实测质量和阈值。

| 参数 | 新默认值 | 依据与限制 |
|---|---:|---|
| `--min-corners-per-camera` | 20 | 当前板有 64 个内角点，避免只凭少数角点求位姿；可按局部可见范围调整，最低 8 |
| `--min-board-area-ratio` | 0.03 | 角点凸包至少占画面 3%，避免目标太小；历史样本主要高于该值 |
| `--min-corner-spread-ratio` | 0.12 | 中心化二维角点的较小/较大奇异值比，拦截近共线或极度压扁的分布 |
| `--max-pnp-reprojection-error-px` | 1.0 | 保留原阈值，但检查前移到保存之前；不再先存明显高残差样本 |
| `--hold-still-seconds` | 0.6 | 两路需连续稳定一段时间；不是严格硬件同步 |
| `--max-still-motion-px` | 1.0 | 与静止窗口起点相比，共同角点位移的 P90 不超过 1 px；累积慢漂移也会重置计时 |
| `--min-sample-shift-px` | 12 | 若两路都与某个已保存样本近似不变，拒绝重复采样；仍需人工主动覆盖不同高度和倾角 |
| `--max-board-area-ratio` | 0.75 | 避免板占满画面或角点过于贴边；不是覆盖所有边缘的证明 |
| `--max-pair-skew-ms` | 100 | 粗略拒绝不同步/陈旧帧；优先使用同域全局时间，否则只能检查主机接收时差 |

像素阈值按当前 640x480 给定，更换分辨率后应重新评估。丢失角点、时间间断或明显
移动会重置静止窗口。单目残差和角点分布不能完全代替专门的模糊、反光或板面翘曲检测；
本次没有添加未经数据验证的锐度阈值，也无法从历史单帧追溯采样时是否真正静止。
采样检查通过不代表外参验收通过，仍必须做留出和独立复核。

`35` 原有相机和板参数命令可继续使用。需要实验性联合求解时，额外增加
`--method stereo`；默认不变。原始标定文件不会被这些默认值或离线比较覆盖。
## 9. 普通棋盘离线复核（2026-09-09）

侧视 B 的普通棋盘实验发现两个独立问题：固定 11 像素半窗口的
`cornerSubPix` 会在小尺寸/倾斜棋盘上拉偏角点；SB 与普通检测回退的角点
编号还可能相差 180 度。第 13 组同时受到了这两个问题影响。

修正方式：按角点最小间距限制细化窗口；对本次 11×8 内角点棋盘，根据
多个内部黑白格的对比度投票统一编号，低对比度或意见不一致时拒绝。
编号判断不使用外参或验证残差，不再只依赖首角点附近的一个区域。
内角点行列数同奇偶的普通棋盘不能用此黑白格规则消除方向歧义。

可复现命令（离线，不打开设备，不覆盖原始标定）：

```bash
python xarm_sdk_control/scripts/39_reprocess_chessboard_pair.py \
  --pair xarm_sdk_control/calibration/multiview_neck/overhead_from_side_b_extrinsics_20260909_145232_523.json
python xarm_sdk_control/scripts/36_validate_charuco_extrinsics.py \
  --pair xarm_sdk_control/data/multiview_neck/reprocessed_pairs/20260909_150236_601/pair.json \
  --all-folds --method compare --allow-legacy-input
```

每次重处理生成独立目录，第二条命令应使用第一条实际输出的路径。
上述已完成运行的报告：
`xarm_sdk_control/data/multiview_neck/calibration_validation/20260909_150255_820/validation.json`。

- 30 组均通过单相机 1 px 门槛，最终位姿一致性筛选保留 25 组。
- 五折验证让全部 30 组各作为一次测试样本，不删除测试异常值。
- 原位姿平均法总体跨视角 RMSE 1.3174 px，各折 0.9741～1.7924 px。
- 固定内参的联合立体求解总体 RMSE 1.3864 px，暂不替换默认方法。
- 原始未修正结果为 33.3708 px，包含编号不一致造成的巨大对应误差，
  不能把这一变化理解成实际物理精度改善了同样倍数。
- 当前修复曾参考这批图像进行诊断，因此即使做了留出验证，也必须再采集
  新的独立姿态验收；这些数值不是坛口高度的毫米误差上限。
- 新候选未写入现场 rig，尚未通过独立验收。原始图片、原始标定均保留。

## 10. 系统性修正与离线结果（2026-09-09 晚）

### 10.1 内参不能只看一个拟合 RMS

全画面检查在 33x25 像素网格上执行“去畸变、再投影”，要求最大闭合误差不大于
0.25 px，并在覆盖视场的半径范围采样径向映射导数，要求始终为正。
这是数值模型健康检查，不是成像真值检查，也不是对所有连续像素的数学证明。
依据是 OpenCV 对畸变模型单调性、可逆性的说明：
[OpenCV 相机标定模型](https://docs.opencv.org/4.12.0/d9/d0c/group__calib3d.html)。

本次对原保存内参角点做五折比较：只用训练折拟合 K/D，留出折只估计板位姿；
没有按留出误差删样本。Brown4 在所有折及全量拟合中通过数值检查，Brown5 未通过。

| 相机 | Brown5 留出局部 RMSE | Brown4 留出局部 RMSE | 本轮结论 |
|---|---:|---:|---|
| overhead | 0.165 px | 0.180 px | Brown4 作为独立验证候选 |
| side_a | 0.233 px | 0.242 px | Brown4 作为独立验证候选 |
| side_b | 0.217 px | 0.231 px | Brown4 作为独立验证候选 |

更低的局部残差并未带来更可靠的全画面模型，因此不按训练误差最小直接挑模型。
这些内参仍可能有印刷尺寸、板面平整度、样本覆盖及共同系统偏差，不能宣称已标定
完成。报告位于 `intrinsic_evaluation/20260909_195835_286/evaluation.json`。

### 10.2 B 组普通棋盘的交叉验证

保留角点修正，使用 Brown4 重求同一批 30 组的外参。五折均测试全部留出角点：

| 指标 | 修正角点 + 原内参 | 修正角点 + Brown4 候选 |
|---|---:|---:|
| 位姿平均：五折池化跨相机 RMSE | 1.317 px | 1.004 px |
| 联合立体：五折池化跨相机 RMSE | 1.386 px | 1.040 px |

默认仍为位姿平均，不能仅为“算法更复杂”切换联合求解。约 24% 的下降只描述这批
数据的像素残差，不能解释为现场高度误差同比下降，更不能推导最大毫米误差。
A 组随后也完成了 ChArUco 原图重算：位姿平均五折 RMSE 为 1.028 px，联合立体
为 1.036 px，与原先约 1.02/1.03 px 基本相当，没有显示明显精度提升。
候选为 `reprocessed_pairs/20260909_201534_757/pair.json`，报告为
`calibration_validation/20260909_201548_033/validation.json`。两组都没有新姿态的
现场独立验收，不能据此直接启用全阵列。

候选为 `reprocessed_pairs/20260909_200151_059/pair.json`，报告为
`calibration_validation/20260909_200551_208/validation.json`；本节结果路径均相对
`xarm_sdk_control/data/multiview_neck/`。纯离线复现命令：

```bash
python xarm_sdk_control/scripts/39_reprocess_chessboard_pair.py \
  --pair "$CAL/overhead_from_side_b_extrinsics_20260909_145232_523.json" \
  --reference-intrinsics "$INTR/overhead_brown4.json" \
  --target-intrinsics "$INTR/side_b_brown4.json"
python xarm_sdk_control/scripts/36_validate_charuco_extrinsics.py \
  --pair <上一条实际输出的pair.json> --all-folds --method compare
```

没有场景记录的历史采集默认不能进入新阵列。`39 --scene-id <当前场景ID>` 是
操作者对“这些原图属于当前且未移动的相机布置”的明确声明，不是程序自动检测。
不确定是否移动过就重采，不能靠填入当前 ID 让历史数据冒充新场景。
`39` 虽保留历史文件名，现已支持两种板，直接把 `--pair` 换成 A 组 ChArUco 文件、
`--target-intrinsics` 换成 `side_a_brown4.json` 即可。默认仍为 `pose_average`，
需要对照时可显式选择 `--method stereo`，不另建一套重复的离线脚本。

### 10.3 数据和结构约定

- `calibration_data.py` 统一内参加载、数值快照、哈希和结果写入；快照绑定序列号、
  分辨率、K 和 D，不把标定板规格或文件名当作相机投影模型的一部分。
- `calibration_target.py` 统一两种板的角点 ID、图像点和板坐标。普通棋盘的方向
  不确定时拒绝，外参不能靠错误编号凑出一个低残差。
- `calibration_sampling.py` 只做采样检查；`pair_calibration_session.py` 统一
  `24/35` 的现场流程，避免两套脚本的阈值、保存和静止逻辑不断分叉。
- `calibration_validation.py` 只做独立检查和验收；留出比较不等于独立新图像验收。
- 新 pair/rig 为 schema 2，加载时核对内参数值和源外参，阵列内嵌变换也必须与源
  外参一致。新文件拒绝覆盖；传统无绑定文件只保留显式离线诊断能力。
- 新配对采样每保存一组就写入原 RGB、SHA256、角点、帧号、时间域、接收时刻及
  admission 指标。即使取消会话，也保留 `session.json` 和 `sample_XX.json`，
  但不会把未求解会话标为有效外参。
- 当前 RGB 没有硬件同步；相同全局时钟域时比较设备时间戳，否则记录的只是主机
  接收时差，不能冒充曝光同步。即使粗时差过关，也必须保证板静止。
- `39` 默认只复用原图路径和重算数值，不再复制生成整套角点覆盖图；确实需要时
  加 `--save-overlays`。这轮保留历史原图与对照报告，没有删除原始测量证据。
- `data/` 被 Git 忽略；他人复现真实结果需要另行转移原始样本及结果目录，不能只
  下载代码就期待上述绝对路径存在。生成候选不自动写入正式配置或启用状态。

相机位置改变后，应为物理布置使用新的唯一 `scene_id`，将注册表 `status` 设为
`calibration_required`，并清空 `approved_rig_fingerprint`。保留旧注册表备份。
不改 ID 不能让程序知道相机发生位移；程序不具备自动识别支架移动的能力。

### 10.4 最后的精度验收仍需实物

独立棋盘检查合格后，重新建立承载面，再用独立量具测量的参照件检查高度。
至少覆盖实际工作区中心/边缘、低/中/高三种高度，每种条件多次采集，保留全部
失败和超差结果。量具不确定度必须显著小于希望验证的容差。

先统一比较指标：`rim_center_workcell_mm[2]` 是拟合口沿平面的中心高度，不是
倾斜口沿的最高点；板格三角化长度也不能代替“承载面到坛口”的高度检验。
记录每个条件的测量值、真值和量具不确定度，再汇总偏差、RMSE、绝对误差 P95、
本批最大绝对误差、失败率。固定变换下的重采样区间不能替代这一步。

在实际酒坛上还应验证口沿形状偏离圆形、歪颈、反光、不同材质和局部遮挡。
本批最大误差不是所有未来工况的理论上限。软件目前不自动批准绝对精度或机器人
运动；相机到工位、工位到机器人、机器人轨迹的验收是不同的环节。
