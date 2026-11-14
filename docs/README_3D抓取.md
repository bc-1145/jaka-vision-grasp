# 深度点云、CAD 匹配与翻正放置

本文中的启动文件、配置及命令路径均相对于工作区根目录（本文件的上一级）。[返回工作区入口](../README.md)。

`pick_place_3d.py` 使用 NU4000 原生点云，直接读取 `螺栓.FCStd` 中 Body 的最终 BREP，经过平面移除、聚类、FPFH 粗配准和点到面 ICP 精配准，选出综合匹配度和周围空间最好的实例。

当前现场配置为法兰相机（`camera.mount: eye_in_hand`）。已将本次识别位置记录在 `pick_results/observation_pose.json` 和配置的 `robot.observation_pose` 中。每次扫描还会把当时的 TCP 位姿、工具号和用户坐标系号保存在候选结果内；换算使用拍照时的位姿，扫描期间机器人移动会使该次扫描失效。相机预览可以保持打开。

工件可以散乱放置。手眼标定用于建立相机、机械臂和夹爪之间的坐标关系，安装及 TCP 定义不变时不需对每件工件重新标定。候选净空计算包含完整场景中的平面、未成簇点及同一簇中不符合当前 CAD 表面的点，避免把相接工件整体当作一件。此数值仅用于候选排序，不代表已验证夹指碰撞或解决全部堆叠遮挡情况。

放置模式为 `placement.mode: home_tcp`。`一键回Home.bat` 与一键视觉抓取使用 `robot_poses.json` 的 Home；通用 `pick_place_3d.py` 规划器使用 `pick_place_config.json` 的 `placement.home_pose`。两处独立保存，修改 Home 后应核对放置配置和已生成方案。此模式不从料框平面推测放置面，也不需要 `platform_z_mm`。实际执行仍需核对 Home 释放位置可承接工件。

当前抓取部位明确为杆身：CAD 杆身外径 21 mm，大头外径约 34 mm。规划在杆身轴向范围内比较多个抓点，按完整场景净空排序；张开指令 20 mm，杆身夹持指令使用用户已确认的 9.43 mm。9.43 mm 是执行器指令值，不能当作杆身外径或实际指间距。

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

双击 `一键视觉抓取.bat`。程序会先垂直提高到 340 mm 净空高度，再转到 `robot_poses.json` 中保存的 `photo_observation` 拍照点；重新采集深度、匹配螺栓模型，并从全部识别结果中选择匹配合格且有手指净空的工件。第一阶段策略不要求侧抓或最终工件姿态，而是优先沿相机视线顶向/斜顶向接近；夹持中心按现场几何位于法兰 -Z 方向约 180 mm，并把第五轴完整路线限制在现场控制器接受的约 ±120° 内。自由空间动作与“一键回 Home”使用相同快速档：直线 1500 mm/s、8000 mm/s²，关节 180°/s、720°/s²，速度倍率 100%。接近散料和最终放置时自动降速。夹紧后沿原接近路线退出，转到 Home 上方、下降松开，再提高离开。

命令行只读检查：

```powershell
.\.camera-runtime\python.exe -X utf8 one_click_pick.py --check
```

JAKA 的机器人姿态、工具姿态和安全区域始终有效；控制器拒绝某段运动时，程序停止且不会绕过或修改安全设置。规划器会排除本机已验证会触发 `983057` 的第五轴和大角度数学逆解。

法兰相机模式从采集时的位姿开始，依次张开、接近、抓取、抬升、确认夹持、移动到 home 释放位姿、退回。执行前会核对当前位置和工具坐标系仍与扫描时一致，旧的无位姿记录的匹配结果需重新扫描。每一步检查控制器故障、等待队列结束并核对实际 TCP。由于夹爪只有输出设定回读，没有夹持成功反馈，抬升后必须人工输入 `HOLD` 才继续。

当直接旋转超过配置的角度限制时，计划会被阻止。现场示教并填写 `fixture` 中的放入、取出姿态以及物体被定位夹爪夹住时的基座坐标姿态后，可启用第二只定位夹爪作为中转；程序据此重新计算二次抓取后的物体/TCP 关系，再执行最终翻正放置。
