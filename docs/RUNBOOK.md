# 环境与现场恢复手册

2026-10-29标定/跨工位传递：[结果、适用条件、失败和完整复现](engineering/uncertain_transfer_20261029.md)。以下从任意目录打开原MuJoCo界面的新静态枪头到位对照，不连接设备：

```bash
conda run --no-capture-output -n winejar-stereo \
  python /home/wunicheng/Git/WineJar/scripts/view_calibration_arrival.py --ground-truth --large-neck

# 新目录完整确认闭环，已有输入/环境须保留，不重新训练旧前端
bash /home/wunicheng/Git/WineJar/scripts/reproduce_uncertain_transfer.sh /tmp/winejar_uncertain_transfer_NEW
```

鼠标切原Huber/不确定性/关节映射/整图加权；红是视觉目标、青是真颈圆、黄是实际枪头内圆/轴。右图和拟合点是识别t0，当前仅静态到位，修正后的下降路径和最终锁紧未验证。默认显示通过例，加`--case new8__confirmation_nominal_001`看旧带失败；新失败种子的完整命令见报告。运行环境需按脚本分工，cv5的stereo不支持hand-eye接口，标定求解使用已有vision环境。

相机到机械臂/下游坛身对照的离线复现和原MuJoCo查看命令见[传递验证报告](engineering/pose_transfer_engineering_20261028.md)。酒厂汇报素材在`experiments/factory_briefing_20261028/output/`，含6张16:9高清PNG、可编辑PPT和简短讲解，未连接设备。

本文以仓库根目录为工作目录。实机命令用于后续现场工作；本次整理只运行离线检查。

2026-10-21当前任务为[坛颈圆环离线研究](../experiments/neck_localization/README.md)。
本阶段不执行本文的设备读取、采集、标定或运动命令；以下保留作将来现场手册。

2026-10-28最新：预设启停到位误差、独立虚拟标定、姿态/半径组合及工具净空结果见[本轮报告与完整复现命令](engineering/cross_station_progress_20261028.md)。无编码器主场景配置`experiments/neck_localization/configs/timed_stop_20261028_v1.json`，结果`cross_station_20261028_v3/timed_stop_main`，幅度均为假设。新组合沿用原MuJoCo界面：

```bash
conda run --no-capture-output -n winejar-stereo \
  python /home/wunicheng/Git/WineJar/scripts/view_tying_stage.py \
  --viewer /home/wunicheng/Git/WineJar/experiments/neck_localization/results/cross_station_20261028_v3/viewer \
  --condition new8/fusion/clean --case confirmation_nominal_003 --phase close --ground-truth
```

`confirmation_nominal_001`是保留带越界负例。该查看束仍演示理想工位传递，随机到位误差只在保存数组/图表中比较；旧无参默认不改，不把动画当真实扎紧。

2026-10-28三工位离线查看（原MuJoCo界面、保存输入，不连设备）：

```bash
conda run --no-capture-output -n winejar-stereo \
  python /home/wunicheng/Git/WineJar/scripts/view_tying_stage.py --ground-truth
```

左下工位按钮/播放：识别→遮盖→扎口上方→下降→闭合→收带演示。默认保留新8失败例；看几何通过例加`--case confirmation_nominal_003 --phase close`。右侧三机RGB/深度始终是识别t0保存输入，后续不重新观测颈部。收带动画不代表锁紧/剪断完成。批量重建、输入依赖、完整指标和假设见[三工位报告](engineering/three_station_tying_20261028.md)。原`scripts/view_structural_study.py`及默认束保留。

## 1. 安装与依赖分工

```bash
conda env create -f xarm_sdk_control/environment.yml
conda env create -f vision/environment.yml
conda run -n winejar-vision pip install -r vision/requirements-sam2.txt
```

控制环境负责SDK、RealSense、OpenCV、NumPy/SciPy、YAML和pytest；视觉环境负责
PyTorch/CUDA、GroundingDINO、SAM2、Transformers。不要为了跑测试合并环境。
本机环境在 `~/anaconda3/envs/winejar-xarm-sdk` 与 `~/anaconda3/envs/winejar-vision`。
这些绝对路径只描述本机，可用 `conda run -n ...` 替代。

模型位于 `vision/models/`，被Git忽略。GroundingDINO还可能使用用户目录下的
Hugging Face缓存。换机要迁移权重/缓存或按视觉文档下载，不能只复制代码。
`vision/requirements-sam2.txt` 是SAM2安装依据。

```bash
conda run -n winejar-vision python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
conda run -n winejar-xarm-sdk python -m pytest -q xarm_sdk_control/tests
conda run -n winejar-vision python -m unittest discover -s vision/tests
python tools/project_maintenance.py audit
```

## 2. 只读诊断与预览

以下两个脚本读取机械臂，但不运动；先查看其`--help`确认本版本参数：

```bash
conda run -n winejar-xarm-sdk python xarm_sdk_control/scripts/01_read_status.py
conda run -n winejar-xarm-sdk python xarm_sdk_control/scripts/22_suction_sensor_check.py --help
```

核对IP、控制器错误/急停、使能、模式、队列、位姿与实际TCP。
配置中的116 mm不能替代控制器的实际活动TCP；之前开机曾出现0或146 mm。
界面使能后脚本仍报急停时，应核对同一IP与实际错误码，不靠猜测自动清错。

腕部实时预览：

```bash
bash 预览手眼相机.sh
# 自定义环境：WINEJAR_PYTHON=/path/to/python bash 预览手眼相机.sh
```

全局只采图，不连接机器人：

```bash
conda run -n winejar-xarm-sdk python xarm_sdk_control/scripts/13_capture_realsense_frame.py \
  --serial 013222073103 --output-root xarm_sdk_control/data/vision_frames
```

先关闭占用相机的预览或其它采集进程。保存的frame目录包含color、对齐原始depth、
深度预览和frame.json；计算必须读取原始depth与其depth_scale，不能用伪彩图测量。

## 3. 视觉服务与离线分析

```bash
conda run --no-capture-output -n winejar-vision python -u vision/scripts/03_grounded_sam2_service.py
```

等待服务输出ready后保持终端运行。默认Unix socket为`/tmp/winejar_grounded_sam2.sock`。
需要完整本地缓存时可设置`HF_HUB_OFFLINE=1`。服务不会控制机械臂。

单帧离线推理，`<新采集目录>`替换为真实frame目录：

```bash
conda run --no-capture-output -n winejar-vision python vision/scripts/01_offline_grounded_sam2.py \
  --frame-dir '<新采集目录>' \
  --text-prompt 'white paper. ceramic disk. bamboo leaf. lotus leaf.' \
  --box-threshold 0.25 --output-min-score 0.25
```

必须检查掩码和叠图：纸曾被模型误判为陶瓷，竹壳和荷叶也可能组合误检。
圆陶瓷优先复核完整轮廓中心；竹壳不能从被机械臂遮挡的残缺轮廓直接求中心。
荷叶可截断用于本次放置，但要选可见内部、足以承托物料的区域，不吸取荷叶。

## 4. 标定与工具参数

- 当前实验候选：`calibration/experimental/20261014/`，不要仅按文件日期自动启用。
- 手眼采样：`scripts/32_calibrate_wrist_hand_eye.py`只读机器人，板固定、腕部相机随臂变化。
  普通棋盘11×8内角点、20 mm。保存实际TCP并反算法兰；不能把TCP位姿当法兰位姿。
- 新全局候选由固定棋盘和新手眼推导。原始样本、求解诊断与源脚本归档位置见数据文档。
- 俯视相机/底座相对位置变化后需重做全局外参。腕部安装不动不必仅因底座移动就强制重做手眼，
  但应独立检查；本项目10月14日已重新求解，不要未经核查又回用9月模型。
- 成功的吸取参数：纸/陶瓷额外下压0；叠放竹壳相对视觉表面总共8 mm。
  下压量是材料接触策略，不能用来掩盖整幅深度尺度或外参误差。

三相机复标和独立验收方法见 [技术说明](multiview-neck-localization.md)，当前暂停该任务。

## 5. 实机恢复顺序与异常

1. 新图、新状态、新计划；检查源点、放置点、工具长度与现场扫掠空间。
2. 实时插值检查IK、TCP限位、关节限位与关节连续性。几何检查不是完整碰撞检测。
3. 保持原来的材料参数先完成白纸→陶瓷→第一片竹壳。每步拍照验证物料离台与放置。
4. 第一片移走后重新识别剩余竹壳；检查是否连带多片，避免第二次去目标区抓已放置物料。
5. 第二片转到交叉方向放置，最后检查摆放、释放、撤离、泵和阀状态。

气路：CO0泵、CO1真空选择阀，CI0=0为成功。建立真空后要满足50 ms连续确认，
2秒超时则在源处泄压并撤离；不能未经真空确认就载物横移。
载物途中失去确认要停止继续搬运并判断物料状态，不能无条件关泵让可能的载荷掉落。
压力开关不提供连续真空度，也不能确认吸盘贴合位置、离台高度或抓了几片。

旧`21_complete_loading.py`会复位并读取旧工位配置，不能照历史命令直接运行。
[恢复草稿](../xarm_sdk_control/experiments/recommissioning_20261014/README.md)已纳入版本管理，
但完整取放路线未执行，接手后必须先审查再使用。

## 6. 每次实验必须留下的内容

保存输入帧、相机序列号和内参、标定路径与散列、实际TCP、起始位姿、路线检查、
每段返回码、CI0时间或读数、抬升和放置照片、异常与最终位姿/IO/载荷状态。
失败也保留。最终结果写清“计划/预查/已运动/吸起/完整放置”哪个阶段。

更新`CURRENT_STATE.md`和必要的证据索引；不要改写历史原始日志。
