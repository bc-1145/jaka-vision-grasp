# 深度点云、CAD 匹配与翻正放置

台面单件自动复现入口：[`一键台面竖放.bat`](../一键台面竖放.bat)，流程、监督条件和检查命令见[台面自动竖放说明](README_台面自动竖放.md)。

本文中的启动文件、配置及命令路径均相对于工作区根目录（本文件的上一级）。[返回工作区入口](../README.md)。

`pick_place_3d.py` 使用 NU4000 原生点云，直接读取 `螺栓.FCStd` 中 Body 的最终 BREP。当前 `matching.strategy = "clutter"` 启用 `clutter_matching.py` 的堆叠多实例匹配：平面移除和聚类后，同时生成模型尺寸的局部窗口，利用杆身半径与法向估计圆柱轴线，搜索头部方向和轴向位置，再用鲁棒 ICP 对齐 CAD。独立点云簇仍保留 PCA 与 FPFH 备用配准；`strategy = "legacy"` 可恢复原算法做同帧对比。此几何约束针对当前连续轴对称的中空法兰管，杆身外径取模型的 21 mm，不使用夹爪的 8.65 mm 指令值作为几何尺寸。

新算法从完整场景重分配每个模型支持的观测点，允许在一个粘连簇内找出多件零件。可见性校验利用实际深度前景识别遮挡；缺失深度不当作遮挡证据。原有场景点、可见面、头部及杆身支持率阈值保留，头部完全不可见或剩余证据太少时仍拒绝。额外比较细管观测点的圆柱拟合残差与平面拟合残差，过滤更符合平面边缘的候选。重叠的模型表面和重复分配的观测点用于合并重复位姿；轴对称模型绕自身轴的转角不唯一，不重复计数。新的匹配分数以可见表面支持率、拟合残差和净空计算。

实时预览每帧用当前深度重新拟合上一帧位姿，校验通过才保留，同时轮换搜索新的局部区域；旧零件消失时清除对应结果，跟踪失效时重新完整搜索。右侧区分分割簇数、候选区域数、实际配准次数和通过校验的实例数。跟踪提示只用于预览，执行用扫描仍重新采集并绑定机器人拍照位姿。

可用只访问相机的回放工具对比固定数据，输出诊断 JSON 和轮廓图，均保存到指定目录，不覆盖抓取方案：

```powershell
.\.camera-runtime\python.exe diagnostics/evaluate_matching.py --directory pick_results/algorithm_evaluation --capture 3 --strategy legacy
.\.camera-runtime\python.exe diagnostics/evaluate_matching.py --directory pick_results/algorithm_evaluation --strategy clutter
.\.camera-runtime\python.exe diagnostics/evaluate_matching.py --directory pick_results/algorithm_evaluation --strategy clutter --track
```

当前现场配置为法兰相机（`camera.mount: eye_in_hand`）。已将本次识别位置记录在 `pick_results/observation_pose.json` 和配置的 `robot.observation_pose` 中。每次扫描还会把当时的 TCP 位姿、工具号和用户坐标系号保存在候选结果内；换算使用拍照时的位姿，扫描期间机器人移动会使该次扫描失效。相机预览可以保持打开。

工件可以散乱放置。手眼标定用于建立相机、机械臂和夹爪之间的坐标关系，安装及 TCP 定义不变时不需对每件工件重新标定。候选净空计算包含完整场景中的平面、未成簇点及同一簇中不符合当前 CAD 表面的点，避免把相接工件整体当作一件。此数值仅用于候选排序，不代表已验证夹指碰撞或解决全部堆叠遮挡情况。

放置模式为 `placement.mode: home_tcp`。`一键回Home.bat` 与一键视觉抓取使用 `robot_poses.json` 的 Home；通用 `pick_place_3d.py` 规划器使用 `pick_place_config.json` 的 `placement.home_pose`。两处独立保存，修改 Home 后应核对放置配置和已生成方案。此模式不从料框平面推测放置面，也不需要 `platform_z_mm`。实际执行仍需核对 Home 释放位置可承接工件。

当前抓取部位固定为杆身中部：CAD 杆身外径 21 mm，大头外径约 34 mm。抓点取杆身轴向范围 3.6–51.2 mm 的中点 Z=27.4 mm，搜索接近方向时不再把抓点移向两端；张开指令 20 mm，细管夹持指令为用户指定的 8.65 mm。8.65 mm 是执行器指令值，不能当作杆身外径或实际指间距。

点击「生成杆身抓取方案」，或运行 `python pick_place_3d.py plan-results`，可生成 `pick_results/plan.json`、可读的 `grasp_plan.md` 及带抓点标记的 `grasp_preview.png`。标定未完成时也会生成相机坐标下的预览方案，机器人抓取位姿保持空值，界面禁止执行。新扫描会清除上一份方案，避免使用旧抓点。

双击 `启动3D抓取软件.bat` 可打开桌面软件。界面提供标定状态、机器人状态、点云俯视预览、匹配候选表、独立三维查看、运动规划和带双重确认的真机执行。桌面界面与相机算法使用两个独立运行时，避免 Inuitive Python 3.10 SDK 与 Tkinter 运行时冲突。

默认只进行识别和预览，不发送机械臂或夹爪指令：

```powershell
.\.camera-runtime\python.exe pick_place_3d.py inspect-model
.\.camera-runtime\python.exe pick_place_3d.py validate-config
.\.camera-runtime\python.exe pick_place_3d.py scan --visualize
.\.camera-runtime\python.exe pick_place_3d.py run --visualize
```

识别结果保存在 `pick_results/`：原始场景、移除平台后的点云、最佳 CAD 对齐点云、全部候选 JSON，以及标定完整时生成的运动计划 JSON。预览窗口中灰色是场景，绿色是选中物体，红色是 CAD 对齐结果。

## 真机执行前标定

没有标定板时，可点「无板手眼标定（不用标定板）」采集同一静止料框的多个视角，详见 [无板标定说明](README_无板标定.md)。程序只采集，不主动移动机械臂；求解候选须复核，夹持中心偏移仍需单独确定。

在 `pick_place_config.json` 中补齐以下实测值后，`validate-config` 才会显示 `execution: READY`：

- `camera.tcp_T_camera`：相机坐标到当前 TCP 坐标的 4×4 手眼变换，长度单位 mm；`camera.calibration_tool_id` 必须对应标定时使用的工具号。改变相机安装或 TCP 定义后需重新标定。固定相机模式才使用 `camera.base_T_camera`。
- `gripper.tcp_to_grasp_center_mm`：当前机器人 TCP 到两指实际夹持中心的向量，使用 TCP 坐标系，单位 mm。
- `gripper.measured_max_object_mm`：当前手指安装方式下实测可夹最大外径。模型杆部外径为 21 mm；设定的 0～20 mm 是执行器位置命令，不是实际指间距。
- `placement.platform_z_mm`：仅在 `platform` 模式下，用于定义放置平台的机器人基座 Z 高度。

相机标定可用点选向导求解。先在点云窗口中依次 Shift+左键选择至少四个固定特征点，关闭窗口后，按相同顺序手动点动机器人，让已准确定义的当前 TCP 原点接触这些点；程序只读取 TCP，不主动移动机器人。法兰相机模式将使用采集时的 TCP 位姿，把求出的基座外参转换成手眼矩阵，并绑定工具号。最大残差超过 3 mm 时拒绝保存；坐标原点无法实际触及时，需要已有标定数据或另行使用标定目标，不能用夹指末端替代未知的 TCP 原点：

```powershell
.\.camera-runtime\python.exe pick_place_3d.py calibrate-camera --points 4
.\.camera-runtime\python.exe pick_place_3d.py calibrate-platform
```

测量当前 TCP 原点到实际夹持中心的向量，以及当前手指安装方式可容纳的最大工件外径后，可用命令写入。以下数字只是格式示例，不能直接用于真机：

```powershell
.\.camera-runtime\python.exe pick_place_3d.py configure-tool --offset 0 0 100 --max-object 24
```

手眼变换、夹持中心和实际夹持间距未知时，程序只允许视觉预览。`platform` 模式还要求平台高度。真机执行还要求显式参数：

```powershell
.\.camera-runtime\python.exe pick_place_3d.py run --execute --confirm EXECUTE
```

## 一键视觉抓取

双击 `一键视觉抓取.bat`，启动时必须已在 Home。程序先沿已配置的上料路线到达 `robot_poses.json` 的 `loading` 料框位置，采集彩色及深度图、重新扫描匹配 CAD，并严格选择本帧匹配分最高的零件。该零件须达到抓取匹配分 0.78、外部净空 10 mm，并通过逆解、关节范围和路线净空检查；不满足时停止，不自动改选其他零件。之后张开夹爪至 20 mm，慢速到达细管抓点，检查实际夹持中心的位置、高度和夹爪方向，满足条件后才收至 8.65 mm。夹紧后沿接近路线退出料框，保存提起后的彩色及深度复核图，再复用“一键回 Home”的分段路线返回，保持夹爪闭合。

当前坐标采用统一的光学约定。空爪开合录像确认闭合轴沿 TCP Y；规划器使用配置中的实际闭合轴，并补偿夹持中心的横向偏移。三轮指令行程与图像运动联合估计的夹持中心为 TCP `[-22.33, -1.08, 166.42]` mm，已完成一次从台面夹起并竖放的实物验证；该估计仍依赖指令差对应实际总行程的假设，不等于独立实测标定。8.65 mm 继续仅表示执行器指令，不表示实际指间距。第五轴完整路线限制在约 ±120° 内；接近工件时降速。没有有效候选、扫描后机器人位置变化，或细管抓点未对准时停止，不继续闭合。闭合指令回读只代表输出设定一致，不代表已确认夹住一个零件。可使用 `一键视觉抓取.bat --manual-verify`，在提起后查看图片并输入 YES 才继续回 Home；未确认时保持夹爪状态并停止。

台面竖放使用独立的 `robot.table_observation_pose`，保留用户指定的高处总览位；不修改 Home。抓取前先回此处、重新扫描，再规划管身中部抓取。`diagnostics/stand_table_part.py` 的 `overview` 返回总览，`plan --target-dir <本次扫描目录>` 生成路线，`approach` 保存实际下爪前画面；复核零件与爪头对齐后才能继续 `pick`，抬升确认实物夹持后才能执行 `place`。结果保存在 `pick_results/table_upright/stand_run.json`，完成翻正并松开后退回总览位，再检查最终图像。

`diagnostics/record_gripper_calibration.py` 在总览位记录三轮空爪开合的彩色、深度视频、原始深度 PNG 和时间戳，只操作机器人夹爪，不发送机械臂运动。录像中每帧的数值明确标为指令值；相机失帧或机械臂被移动时停止后续开合，不重试不确定的夹爪写入。

命令行只读检查：

```powershell
.\.camera-runtime\python.exe -X utf8 one_click_pick.py --check
```

JAKA 的机器人姿态、工具姿态和安全区域始终有效；控制器拒绝某段运动时，程序停止且不会绕过或修改安全设置。规划器会排除本机已验证会触发 `983057` 的第五轴和大角度数学逆解。

通用 `pick_place_3d.py` 执行器仍使用独立的放置流程：从采集时的位姿开始，张开、接近、抓取、抬升、人工确认夹持，再移动到配置的 home 释放位姿并退回。执行前核对当前位置和工具坐标系仍与扫描时一致，每一步检查故障和实际 TCP；该执行器抬升后需人工输入 `HOLD`。它与上述保持夹持回 Home 的一键流程不同。

当直接旋转超过配置的角度限制时，计划会被阻止。现场示教并填写 `fixture` 中的放入、取出姿态以及物体被定位夹爪夹住时的基座坐标姿态后，可启用第二只定位夹爪作为中转；程序据此重新计算二次抓取后的物体/TCP 关系，再执行最终翻正放置。
