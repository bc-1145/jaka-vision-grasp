# 相机实时预览

本文中的启动文件、配置及命令路径均相对于工作区根目录（本文件的上一级）。[返回工作区入口](../README.md)。

双击 `start_color_depth_viewer.bat`，显示 Inuitive NU4000 的彩色和深度画面。

双击 `start_live_model_preview.bat`（或运行 `.camera-runtime\python.exe color_depth_viewer.py --live-model`）开启持续 CAD 匹配。Color / Depth / Model Pose 并排显示：相机采集进程同时提供彩色、深度和原生 XYZ，独立匹配进程重复执行现有 CAD 注册与几何校验，右侧显示匹配分数、XYZ、RPY、RMSE、输入图像年龄及计算耗时。识别频率取决于实际计算时间，并非每个视频帧都有新结果。通过校验的模型轮廓和坐标轴投影到两路画面；匹配结果超过显示有效期、相机失联或目标区域深度发生明显变化时隐藏旧标记。没有合格匹配会明确显示未匹配。该模式不读取或发送机器人、夹爪控制命令，也不修改标定参数。诊断结果与最近画面单独保存到 `pick_results/live_model/status.json`、`preview.png`，不会覆盖用于抓取的扫描和方案。

轮廓采用带黑色描边的黄色线条，旁边显示候选编号及 CAD 分数。显示有效期为本次匹配耗时加 15 秒，避免计算超过 15 秒时刚生成的轮廓就被隐藏；显示前仍检查实时深度是否与输入一致、相机是否在线。最近一次实际显示轮廓的截图保存在 `pick_results/live_model/contours.png`。

Color / Depth / Model Pose 从左到右并排显示，右侧独立面板显示最近一次 CAD 扫描的匹配数量、分数、相机坐标 XYZ（毫米）、RPY（度）及结果年龄。先在抓取界面点击“扫描并匹配 CAD”；检测实现在 `pick_place_3d.py`，结果为 `pick_results/matches.json`。预览每秒读取一次结果，这是扫描快照，移动相机或工件后需重新扫描，不代表逐帧跟踪。

若要将模型原点和 XYZ 坐标轴投影到彩色画面，需要在 `pick_place_config.json` 的 `camera.color_projection` 配置真实标定参数：`image_size` 为 `[宽, 高]`，`matrix` 为彩色相机的 3×3 内参矩阵，`distortion` 为 OpenCV 畸变系数，`color_T_camera` 为点云相机坐标到彩色相机坐标的 4×4 变换（平移单位毫米）。缺失参数或分辨率不匹配时仅显示数值和提示，不猜测像素位置。坐标轴长度为 30 mm，X 红、Y 绿、Z 蓝。

正常启动没有自动关闭时限。点窗口右上角关闭按钮、按 Q / Esc，或在控制台按 Ctrl+C 才退出。

相机未连接、取帧超时或采集进程异常时，预览窗口会保留并显示等待或重连状态。保留的上一帧会标注画面已过去的时间。单次 InuService 取帧超时先重试，连续超时或停止出帧则重启采集；重连间隔逐步增加，最多 15 秒。相机恢复后自动显示新画面。

采集在独立进程运行，避免 SDK 阻塞或原生库异常导致窗口一同关闭。程序关闭时会停止自己的采集进程，不修改机械臂或夹具状态。

诊断日志位于 `logs/camera_viewer.log` 和 `logs/camera_capture.log`，每个日志达到约 2 MB 后轮转，保留三份历史文件。

`--check-frames N` 仅用于诊断：验证 N 对帧后退出，30 秒未完成时报错退出。日常预览无需添加这个参数。

Color 和 Depth 中的黄色轮廓由 CAD 的真实三角面投影生成，保留头部与杆身的台阶，不再使用凸包。XYZ 轴用红、绿、蓝显示。当前相机的出厂投影标定保存在 `camera.color_projection` 和 `camera.depth_projection`。标记为扫描候选，移动相机或工件后需重新扫描。

匹配增加了场景点贴合率、CAD 可见表面覆盖率、头部及杆身覆盖率验证，阈值在配置的 `matching.minimum_*_support` 中。覆盖不足或总分未达阈值时不输出候选。`pick_results/matching_diagnostics.json` 记录候选被拒绝的原因；没有通过验证时会清除上次的 `best_model.ply`，避免三维窗口继续显示旧模型。

检测范围由 `camera.roi` 控制，当前深度范围扩大到 150–1000 mm。`camera.excluded_boxes_mm` 排除随腕部相机固定的夹爪体积；该范围依据当前夹爪观测设置，更换相机安装位置或夹爪后需重新核实。检测范围中的工件若进入排除体积，也不会参与匹配。

当前配置使用 `camera.point_cloud_frame = "opencv_optical"`：采集时将 SDK 原生点云的 Y 取反，统一为 X 向右、Y 向下、Z 向前。手眼矩阵已使用同一坐标约定重新求解，结果见 `calibration_data/optical_frame_candidate.json`。原生坐标生成的旧匹配结果不得与新矩阵混用，需重新扫描。此前依据错误投影设置的夹爪排除框已清空。

一键抓取顺序为：Home → 到上料处/料框 → 彩色及深度视觉选件 → 全开夹爪（20 mm）→ 对准细管的位置、高度和方向 → 闭合至 8.65 mm → 提起退出料框并保存复核图 → 保持夹持回 Home。识别不到有效抓点时停在识别阶段。手动拍照接近点保留经控制器 IK 检查的 40 mm 退离距离；一键抓取使用 `loading` 上料处选件。
