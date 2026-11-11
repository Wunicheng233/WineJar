# 当前定位系统机理与操作说明

2026-10-22，针对`depth_profile.py`的已运行基线。本文解释当前实现，不能把后续候选方案当作已实现功能。
本轮补充MuJoCo交互查看器与中间步骤可视化，没有调整估计器或重做新的精度实验。
后续10月23日新增独立的[局部联合估计器与对照报告](joint_profile_stage1.md)，本文下方算法细节仍专指冻结旧基线。

## 先看直观场景与输入

桌面终端运行：

```bash
cd /home/wunicheng/Git/WineJar
conda run --no-capture-output -n embodied-sim python scripts/view_neck_localization.py
```

左侧是真实MuJoCo渲染的酒坛、传送带与三台相机示意，左下是可独立旋转的颈部近景；
红色细圆线是估计目标圆，箭头是其正法向（朝坛口）。箭头绘制长度固定75 mm，只表示方向。
这是近最细范围中点处的圆，不是口沿外圆。结果线以透视投影画在表面之上，背面圆弧也可见，
目的是便于检查完整圆；它们不是检测到的图像边缘。
相机外形仅示意，位置/朝向来自场景，相机前短视锥由实际K计算；短视锥长度不是工作距离。
这些标记只加入查看器的`MjvScene`，没有加入测量模型，也不会改变原始输入。

右侧直接读取三台相机保存的原始RGB；标为“参与估计”或“仅预览”。D切换同一批输入的光轴Z深度，
伪彩固定0.45–1.35 m，范围外截色；用于显示的伪彩不是估计器输入。
左侧自由旋转不改变固定测量相机、右侧输入或已经保存的估计。
查看器只检查既有结果，不运行新估计，不把显示真值送入算法。

| 操作 | 作用 |
|---|---|
| 左键拖动 / 右键拖动 / 滚轮 | 在主场景或颈部近景各自旋转 / 平移 / 缩放；RGB区域不响应 |
| 1 / 2 / 3 | 读取cam_a / cam_a+cam_b / 三台相机的既有结果；其余图像仍保留预览 |
| 左右方向键 | 切换四个酒坛案例 |
| O | 在右侧输入显示副本上叠加结果，原始文件不变 |
| D | RGB / 深度显示切换 |
| G | 默认关闭；开启青色虚线真值，仅用于评估比较 |
| N | 放大/还原颈部近景；放大后仍可独立旋转、平移和缩放 |
| P / C / V | 实体/点云；点云RGB原色/相机来源色；全坛前景/算法工位裁剪 |
| M / L | 在当前记录已有方法间切换；显示新联合/固定轴样条实际拟合点，旧方法无此记录时明确回退 |
| R / Esc | 复位观察视角 / 退出 |

默认`broad_reverse_tilt`三视图。按1并打开G，即可检查约6.05 mm错误接受：
算法内部仍显示接受，独立评估明确显示超过4 mm。拒绝案例不绘制估计圆或法向。
也可直接启动对应案例：

```bash
conda run --no-capture-output -n embodied-sim python scripts/view_neck_localization.py \
  --case broad_reverse_tilt --camera-set cam_a --overlay --ground-truth
conda run --no-capture-output -n embodied-sim python scripts/view_neck_localization.py \
  --case tilted_flat --camera-set cam_a,cam_b
```

静态预览：[场景与原始输入](../../../experiments/neck_localization/results/short_neck_20261022/inspection/scene_inputs.png)、
[错误接受与真值对照](../../../experiments/neck_localization/results/short_neck_20261022/inspection/failure_comparison.png)、
[拒绝案例](../../../experiments/neck_localization/results/short_neck_20261022/inspection/rejected_case.png)。

需要截图时追加`--snapshot /absolute/new/path.png`，输出PNG和来源/显示设置JSON；路径必须不存在。
隐藏截图窗口仍需要桌面DISPLAY及GLFW/OpenGL，不能把它当成无显示服务器的EGL批处理命令。
已验证多种视口宽高比下的叠线投影、输入只读性、切换/拒绝/窗口缩放；估计器散列保持冻结。

### 窗口显示修复（2026-10-22）

用户截图暴露了旧版显示缺陷：CPU合成图有圆/箭头/文字，但`mjr_drawPixels`沿用三维渲染的
深度测试状态，窗口里的主景和近景会漏掉这些内容。此前PNG取自CPU合成图，未检出实际显示问题。
现在二维呈现时暂时关闭深度测试/混合/裁剪，完成后恢复GL状态；截图改为回读实际窗口后缓冲区。
已用强制`GL_NEVER`的回归检查和三个颈部观察角度验证：实际窗口像素与完整合成图逐像素一致。
原始场景、定位算法和输入没有改变。旧进程需关闭后重新启动才会使用修复。

大近景错误接受对照（截图来自实际窗口）：

```bash
conda run --no-capture-output -n embodied-sim python scripts/view_neck_localization.py \
  --camera-set cam_a --ground-truth --large-neck
```

普通三视图大近景用`--large-neck`即可，真值仍默认关闭。

### 查看实际点云

```bash
conda run --no-capture-output -n embodied-sim python scripts/view_neck_localization.py \
  --point-cloud --large-neck
```

点来自原始保存的光轴深度、K/T与普通RGB前景，不读取模型表面或目标标签；不补洞、不插值、
不做额外ICP配准。P切换实体与点云，C切原始RGB色与相机来源色（和右侧边框同色），
V切全坛可见前景与旧定位器实际使用的宽工位裁剪范围。1/2/3选择的相机决定显示哪些点。
全坛前景也只有原始相机视场中的可见部分，不等于完整坛体；部分坛底原本不在输入图像里。
每相机最多60000个显示点，确定性抽稀只影响显示，界面报告显示数/原数；估计器输入不变。
工位裁剪后、显示抽稀前的点数组已与冻结定位器逐项核对。
[相机来源色预览](../../../experiments/neck_localization/results/short_neck_20261022/inspection/point_cloud_sources.png)、
[工位裁剪大近景](../../../experiments/neck_localization/results/short_neck_20261022/inspection/point_cloud_station.png)。
点云使用与MuJoCo一致的反向Z深度与投影规则，新增检查覆盖投影位置和近远点遮挡。

下一阶段的联合估计、相机位置理论和现实边界见[优化分析](optimization_analysis.md)。
联合法已完成的小实验、新记录加载方式及M/L对照见[联合首轮](joint_profile_stage1.md)。
V仍显示旧宽工位范围，不等于新方法实际局部点；新方法请按L，显示点数/坐标已逐项核对。

### 相机布局的来源与局限

当前是人工选择的基线布局，没有做布局优化、安装约束测量或实机内参复刻。
配置在`experiments/neck_localization/configs/preparation_v2.json`与`short_neck_heldout_v1.json`。
相机都看向固定名义点`[0,0,0.59] m`，不会按各坛真实目标位置跟随。
世界原点为名义坛底，Z向上；方位角从+X向+Y增加，以下俯视角等于相机相对名义点的仰角。

| 相机 | 方位角 | 俯视角 | 到名义点距离 | 世界坐标约值（m） |
|---|---:|---:|---:|---|
| cam_a | −50° | 10° | 0.8 m | [0.5064, −0.6035, 0.7289] |
| cam_b | 50° | 10° | 0.8 m | [0.5064, 0.6035, 0.7289] |
| cam_c | 180° | 8° | 0.8 m | [−0.7922, 0, 0.7013] |

共同设置为640×480、垂直视场50°、理想针孔；这些不是RealSense实测参数。
选择意图是分散观察方位、覆盖酒坛上部，作为一/二/三视图对照起点；没有证据说明该布局最优。
焦距约514.7 px，在约0.8 m处，10 mm尺寸的理想正投影粗估约6.4 px；实际受倾斜和遮挡影响。
这说明短颈在当前输入里很小，不能用人审放大近景掩盖测量输入的采样限制。

几何分析预计：侧视有利于观察外颈，但口沿平面更接近侧缘观察；增加俯视可改善口沿平面支持，
同时可能让凸唇遮住更多槽部。这是待实验验证的权衡。距离/视场影响目标像素数，
方位分布影响圆弧覆盖及误差方向；真实设备上还需考虑深度随距离/入射角变化、相机间相关偏差与标定误差。
目前只证明当前理想设置下的结果，不能推出三相机优于其他布局。
后续若讨论布局，应保持算法和测试坛型一致，分别扫描俯仰角、距离、方位分布，
联合报告有效支持、圆心/法向/直径误差、拒绝和错误接受；尚未执行这轮扫描。

## 系统结构

```text
MuJoCo生成器 → RGB、米制光轴深度、准确相机参数
                    ↓
RGB前景筛选 → 像素反投影 → 世界系点云合并
                    ↓
观测上部平面 → 平面点凸包拟圆 → 口沿中心和法向
                    ↓
固定参考轴 → 径向分位数剖面 → 近最细连续范围 → 中点处目标圆
                    ↓
算法接受/拒绝 → 独立评估器与目标真值比较
```

MuJoCo负责静态几何、光照、RGB和深度渲染。定位算法使用NumPy/OpenCV/SciPy离线运行；
当前没有实机采集接口、机械臂控制、学习网络或实时在线服务。

多相机当前是深度点云融合：每台相机独立反投影到同一个世界系，直接拼接点云，再共同拟合。
没有跨视图RGB特征匹配、目标圆三角化、相机参数联合优化或逐相机等权；点数多的视图自然贡献更多点。

## 输入、输出与真值

普通输入为1/2/3组640×480 RGB和同网格float32深度，以及每台相机的K和`T_world_from_camera`。
深度已经是米制光轴Z，不是射线距离或RealSense原始Z16；当前RGB/深度共用理想针孔相机。
没有畸变、真实RGB/IR基线或主动双目成像误差。
世界系原点位于名义坛底中心，Z向上；相机系采用OpenCV的X向右、Y向下、Z向前。

反投影：`p_C=Z*K^-1*[u,v,1]`，再以`p_W=R_WC*p_C+t_WC`变到世界系。
固定工位筛选Z为0.40–0.76 m、XY半径0.28 m；这是宽工位先验，不是每坛目标真值ROI。
估计器没有接收每坛真实坛高、真实槽径或目标圈坐标。相机参数当前使用生成器准确值，未做虚拟板标定。

输出`Estimate`包含世界系圆心`center`（m）、单位法向`normal`、半径`radius`（m），直径由2r得到；
还有`accepted`、`reasons`、计算耗时和诊断字段。范围端点/长度保存在`diagnostics`，坐标沿口沿参考轴。
法向取世界Z正方向，在当前近朝上工况下表示朝坛口。
`residual`是口沿平面RMS残差（m），不是目标定位误差；协方差和条件数目前未估计。

`accepted`只说明内部检查通过。`truth_pass`由独立评估器根据真值计算，仅仿真可得；
中心4 mm、法向2°、直径2 mm为当前分析阈值，后两者未确定为工艺标准。
普通估计结束后才读取`evaluation_only/target.json`。oracle诊断和图中GT叠加均单独标记。

## 当前算法逐步做什么

1. **前景。** RGB转HSV，取S<100且V>35的像素，再保留最大连通域。利用灰/陶色坛与蓝背景的差异。
   这是整个酒坛前景，没有检测目标槽、没有二维椭圆检测。暗颈部也可能被此阈值错误去掉。
2. **点云。** 用掩码中的有效深度反投影，变到世界系，宽工位裁剪后合并。合计少于300点拒绝。
3. **口沿平面。** 这里指凸起口沿最上部环形表面的近似平面，与下方扎带目标圆平面不同。
   RANSAC随机选少量点提出平面，反复比较哪些候选得到更多近邻点支持，从而减轻离群点影响。
   找观测Z最高点，从最高6 mm区域提出平面。RANSAC试450次，候选法向与竖直夹角≤30°，
   内点距离阈值0.18 mm；随后在上部35 mm区域反复扩展内点、SVD细化4次。
   这个阈值针对当前理想深度调试，未用实机数据定标。最高点离群也尚未处理。
4. **口沿中心。** 将平面内点投到二维，求凸包，把凸包顶点作为外圆候选；先代数圆初值，再软L1圆拟合。
   完整环面的凸包近似外圆；部分环面会出现封口直线或其他非外圆边界，这些也进入了拟合。
5. **参考轴。** `rim_guided_profile`把口沿中心o和法向n固定，轴为`o+s*n`。
   当前没有独立拟合颈轴，也没有后续联合调整轴线；这是假设口沿与目标颈部同轴、平行。
   `vertical_profile`对照则保持同一口沿中心，强制n=[0,0,1]。
6. **半径剖面。** 对每个点p计算`s=(p-o)·n`与`rho=||p-o-s*n||`。
   搜索口沿下2–60 mm，每0.5 mm分箱；≥8点的箱取rho的85%分位数作为半径。
   没有给各层重新拟圆或重估圆心。较大分位数意在减轻内壁混入，但不是实际外表面分离。
   支持箱之间间隔>2.5 mm拒绝；小空档插值，再以5点二次Savitzky–Golay平滑。
7. **范围和圆。** 找平滑剖面最小值，以`r_min+1 mm`为半径上限，取包含全局最小值的那段连续范围。
   阈值交点线性插值，轴向中点`s_mid=(s_lo+s_hi)/2`，输出`c=o+s_mid*n`、`n`、`r(s_mid)`。
   1 mm是半径容差，不是轴向允许误差。多个分离候选目前没有完善的歧义判别。
8. **质量检查。** 检查平面支持/面积、口沿半径宽先验、剖面点数/空档、范围是否碰搜索边界、
   目标是否比凸唇窄等。不是经过校准的置信度，仍有错误接受。

这套实现把形状变化压缩成沿一条固定轴的半径曲线。口沿参考错位时，后面无法靠剖面自行纠正轴的位置。
这是一次前向估计过程，没有从目标圆结果回头调整参考轴的闭环或联合优化。
本轮模型口沿与直颈共轴，尚未验证非平口沿、两者不平行、弯颈或不圆。

## 中间计算诊断图（需要分析原因时再看）

现有图在`experiments/neck_localization/results/short_neck_20261022/case_walkthroughs/`：

- `broad_reverse_tilt_cam_a/steps.png`：单视图错误接受。口沿中心偏5.99 mm；目标圆心6.05 mm、法向2.44°、直径7.48 mm。
- `tilted_flat_cam_a_b/steps.png`：双视图主动拒绝，原因`unsupported_gap_in_radius_profile`。

图中左上为实际RGB、提取轮廓与中心投影；右上为平面点、凸包、拟合口沿和GT；
左下为半径剖面与选中范围；右下为每箱点数，红色区域是超过2.5 mm的支持空档。
第二例无目标输出，所以没有伪造估计圈。GT仅用于后处理诊断，不存在于估计器输入图。
当前未注入传感器深度空洞，图中的支持空档来自可见性、图像筛选和采样等。

可从文件链接点击，也可在图形桌面终端运行（下面所有命令从仓库根目录开始）：

```bash
cd /home/wunicheng/Git/WineJar
xdg-open experiments/neck_localization/results/short_neck_20261022/case_walkthroughs/broad_reverse_tilt_cam_a/steps.png
xdg-open experiments/neck_localization/results/short_neck_20261022/case_walkthroughs/tilted_flat_cam_a_b/steps.png
```

## 完整复现与单独查看

```bash
cd /home/wunicheng/Git/WineJar
winejar_review_run=/home/wunicheng/Git/WineJar/experiments/neck_localization/runs/review_20261022_01
bash scripts/run_short_neck_stage1.sh "$winejar_review_run"
```

目录必须不存在；已存在时换后缀，不删除旧结果。默认使用现有`embodied-sim`环境。
脚本执行17项检查，生成4开发+4留出场景，完成120次方法/视图/证据组合，输出报告和两类失败过程图。
这是静态渲染与离线估计，不会启动实时窗口。日志保存在新目录根部，结果可打开：

```bash
xdg-open "$winejar_review_run/report/heldout_errors.png"
xdg-open "$winejar_review_run/report/case_walkthroughs/broad_reverse_tilt_cam_a/steps.png"
xdg-open "$winejar_review_run/report/case_walkthroughs/tilted_flat_cam_a_b/steps.png"
```

复现后的目录为`development_scenes/`、`heldout_scenes/`（XML/网格/RGB-D）、
`development/`、`heldout/`（逐次records.json与汇总）、`diagnostic/`（外表面GT诊断）、`report/`。
统计中的GT条件与普通RGB条件分列，不能混称实际提取效果。

只重做某个已有记录的中间步骤图，不重新生成场景或调算法：

```bash
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 conda run --no-capture-output -n embodied-sim \
  python scripts/visualize_depth_profile_case.py \
  --scenes "$winejar_review_run/heldout_scenes" \
  --records "$winejar_review_run/heldout/records.json" \
  --case broad_reverse_tilt --camera-set cam_a \
  --output "$winejar_review_run/my_single_view_steps"
```

第二例改为`--case tilted_flat --camera-set cam_a,cam_b`，并用另一个新输出目录。
脚本核对冻结估计器散列，防止用改过的算法重建旧记录的中间步骤。

只运行普通RGB条件的估计器，不重渲染：

```bash
OPENBLAS_NUM_THREADS=1 conda run --no-capture-output -n embodied-sim \
  python scripts/run_depth_profile_baseline.py \
  --scenes "$winejar_review_run/heldout_scenes" \
  --output "$winejar_review_run/rgb_only" --split heldout \
  --evidence rgb --camera-sets cam_a cam_a,cam_b all
```

图形桌面下可单独打开MuJoCo模型（已核对本机viewer支持`--mjcf`参数）：

```bash
env -u MUJOCO_GL conda run --no-capture-output -n embodied-sim python -m mujoco.viewer \
  --mjcf="$winejar_review_run/heldout_scenes/broad_reverse_tilt/scene.xml"
```

查看器用于观察模型/相机，定位结果仍由离线脚本产生，不会自动实时叠加在查看器里。
上面这条`mujoco.viewer`命令是通用原始模型查看器；本文开头的`view_neck_localization.py`
才是带相机标记、输入图像和已有结果叠线的专用入口。

一键复现生成新目录后，可查看新数据：

```bash
conda run --no-capture-output -n embodied-sim python scripts/view_neck_localization.py \
  --scenes "$winejar_review_run/heldout_scenes" \
  --records "$winejar_review_run/heldout/records.json"
```

## 适合讨论调整的三个关键位置

1. 用口沿中心/法向固定颈轴是否合适，还是应由颈部局部几何独立估计/联合调整。
2. 每层用径向85%分位数是否足够，如何分开口内壁和外壁、处理有限圆弧及中心偏差。
3. 最细范围的搜索域和端点：生成器限制在标注颈段，算法的宽域阈值可能包含相邻肩/唇，
   平颈例即便解析几何无噪声也有0.94–1.05 mm中点差。这是定义差异，不能靠平滑或更多相机自动消除。

相机/坛型可在`configs/preparation_v2.json`修改，算法参数在`depth_profile.py:ProfileSettings`。
改方法后旧留出检查会因源码散列不匹配而拒绝；探索应标为development并建立新的未见评价案例，
不应仅修改旧散列来继续宣称原留出是未见最终测试。
