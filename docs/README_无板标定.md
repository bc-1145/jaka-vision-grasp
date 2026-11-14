# 不用标定板的手眼采集

本文中的启动文件、配置及命令路径均相对于工作区根目录（本文件的上一级）。[返回工作区入口](../README.md)。

用当前料框、工件和周围固定结构作为三维参照。不需要把工件摆整齐，也不需要购买棋盘格。整个采集期间，料框和里面的工件必须保持不动。

打开「3D 视觉抓取」，点击「无板手眼标定（不用标定板）」。可选择自动轨迹采集，或手动改变角度逐次采集。下面的单次采集按钮只读取位姿与点云。

1. 第一组已经在当前识别位置采集。
2. 用示教器手动改变相机观察角度，停稳后点「采集当前姿态」。相邻画面应保留大部分相同场景，尽量包含料框边角和立体工件。
3. 采集至少 8 组，建议 8～12 组。需要绕两个不同方向改变角度；只平移、只绕同一轴旋转或重复同一姿态不够。手动流程由操作者调整姿态；自动流程见下方。
4. 点「求解并验证」。程序检查点云重叠、非平面结构、机器人姿态可观测性、拟合误差、保留数据的验证误差和结果稳定性。未通过时不会采用结果。

求解结果保存在 `calibration_data/markerless/handeye_candidate.json`，供复核；不会自动写入 `pick_place_config.json` 或解除抓取限制。当前新增流程已通过算法与合成场景测试，尚需完成多角度真机采集验证，不能保证当前料框一定能提供足够可靠的配准。反光、遮挡、单一平面或场景被碰动，都可能导致失败。

这一步解决相机与当前 TCP 的安装关系；夹持中心偏移仍需单独确认。9.43 mm 已记录为本模型杆身的夹持指令，并不表示夹持中心的位置。

采集文件和机器人位姿保存在 `calibration_data/markerless/`。如果相机安装、工具定义或场景发生变化，需要另建采集目录，不能混用数据。

命令行：

```powershell
.\.camera-runtime\python.exe markerless_handeye.py capture
.\.camera-runtime\python.exe markerless_handeye.py status
.\.camera-runtime\python.exe markerless_handeye.py solve
```

方法把静止场景的三维配准所得相机相对运动，与机器人读出的 TCP 相对运动组合，求解手眼关系。相关基础见 [Open3D 相机运动估计](https://www.open3d.org/docs/release/tutorial/pipelines/rgbd_odometry.html) 和 [OpenCV 手眼标定的多轴运动条件](https://docs.opencv.org/5.0/main_modules/calib.html)。

## 自动移动采集

`calibration_motion.py plan` 读取当前机器人位姿，生成中心姿态加 8 个方向的倾斜姿态，最大倾角 6°。每个方向分成 8 个小步，经控制器逆解、正解回代及关节插补检查后保存。每个观察位采完按同一路径返回本次起点，再换方向，不调用 HOME 脚本。

```powershell
.\.camera-runtime\python.exe -X utf8 calibration_motion.py plan
```

输出 `calibration_data/auto_trajectory.json` 和 `.png`。规划只读取状态和调用运动学计算，**没有发送运动**。关节限位、碰撞净空及负载未核实，不能把逆解成功当作可直接执行。

现场数据保存在 `calibration_data/calibration_site.json`，初始留空，不会臆造尺寸：

- `attachment_radius_mm`：法兰中心到相机、夹具、支架所有刚性附件的最远距离。
- `attachment_clearance_mm`：起点附件与障碍物的最小净空。程序按旋转位移上界加 10 mm 余量检查；这不是整条机械臂的碰撞模型。
- `arm_and_cables_clear`：结合具体轨迹和关节范围核实整条机械臂、附件、线缆与障碍物和自身的净空。
- `joint_limits_deg`：实际控制器配置的六组 `[下限, 上限]`，程序要求至少 2° 余量。
- `verified_payload`：已在控制器正确配置的附件 `mass`（kg）与 `centroid`（法兰坐标 mm）。程序只读取核对，不猜测或自动修改负载。
- `measurement_source`：现场测量或操作者提供的依据；`reviewed_plan_sha256`：对已核实轨迹 `json.dumps(plan, sort_keys=True).encode()` 的 SHA256。改轨迹必须重新核实现场数据。

这些值需现场资料支持，不能为了让程序运行而填默认值。当前实机读取的负载为 0 kg，现场条件尚不完整。

条件完整后使用 GUI 的「自动移动并采集」，或：

```powershell
.\.camera-runtime\python.exe -X utf8 calibration_motion.py run
```

执行前重新计算并对照轨迹，每小步使用关节速度 1°/s、加速度 2°/s²；一次只发一个有限运动指令，检测实际关节、TCP 和停稳状态后才采集。通信响应丢失不会重发运动，故障会尝试 `stop_program` 并验证停稳，停止未确认会明确报错。相机采集失败或取消会结束本轮，不会继续回程。软件监控通过网络工作，不能代替机器人安全系统和现场急停。

正常完成采集后回到本次起点，求解候选手眼矩阵。自动采集使用新的 `calibration_data/auto_时间/` 目录，保留各帧位姿、点云、轨迹、现场数据、采集日志和执行报告，不混入历史场景。标定结果不会自动解除抓取执行限制。

旧版控制器的 10001 端口只允许一个控制客户端；自动运行前需断开其它夹具控制程序，避免互相抢占连接。相机预览可以保持开启。接口依据：[JAKA 官方 TCP/IP 协议](https://www.jaka.com/docs/legacy172/en/guide/1.7.2/tcpip.html)。
