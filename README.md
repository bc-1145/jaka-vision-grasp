# 机械臂视觉抓取工作区

JAKA 机械臂、Z-EFG-20P 夹具和 Inuitive NU4000 深度相机的控制、识别与抓取程序。

## 常用入口

在本目录双击以下脚本：

| 功能 | 启动文件 |
| --- | --- |
| 图形化 3D 抓取软件 | [启动3D抓取软件.bat](启动3D抓取软件.bat) |
| 彩色 / 深度相机预览 | [start_color_depth_viewer.bat](start_color_depth_viewer.bat) |
| 双夹爪控制台 | [启动夹具控制台.bat](启动夹具控制台.bat) |
| 回到保存的 Home | [一键回Home.bat](一键回Home.bat) |
| 一键视觉抓取 | [一键视觉抓取.bat](一键视觉抓取.bat) |
| 命令行 3D 识别预览 | [启动3D识别抓取.bat](启动3D识别抓取.bat) |

Home 保存在 [robot_poses.json](robot_poses.json) 的 `home` 中，以文件内的数值为准。回位流程先只动 J3，让 TCP Z 升高到 `first_lift_z`（当前 340 mm），再对位到 Home 上方，最后下降到 Home；已到 Home 时直接结束。若单独移动 J3 无法达到抬升高度，则停止并报告错误。J3 的旋转方向根据当前姿态计算。

“上料处”单独保存在同文件的 `loading` 中，关节角 `J1～J6 = [0, 0, -90, 0, -90, 0]°`。

## 目录与配置

| 位置 | 内容 |
| --- | --- |
| 根目录 `*.py` | 运行中的应用、机械臂 / 夹具控制、相机采集和算法模块 |
| `robot_poses.json` | Home、Home 上方对位关节、首段抬升高度、上料处和拍照点 |
| `pick_place_config.json` | 相机标定、模型匹配、抓取及放置参数 |
| `gripper_devices.json` | 当前夹爪设备及 IO 配置 |
| `model.FCStd`、`螺栓.FCStd` | CAD 模型；两个文件内容相同，分别保留供测试和当前配置使用 |
| [docs/](docs/) | 操作与标定说明 |
| [diagnostics/](diagnostics/) | 设备发现记录、历史 IO 映射与安装诊断日志 |
| [installers/](installers/) | JAKA 安装包 |
| `calibration_data/` | 标定采集、轨迹和求解结果 |
| `pick_results/` | 点云、匹配候选、抓取方案及执行报告 |
| `logs/` | 程序运行日志 |
| `tests/` | 自动化测试 |
| `JAKA/` | 厂商原始 SDK 和示例 |
| `.runtime/` | 桌面软件和夹具控制运行环境 |
| `.camera-runtime/` | 相机 SDK 和 3D 算法运行环境 |
| `.driver-setup/`、`.runtime-setup/` | 驱动及运行环境安装、诊断资料 |

运行入口、程序、有效配置、模型、运行环境和标定 / 抓取数据沿用原路径。整理仅将四份说明文档移入 `docs/`，根目录 JAKA 安装包移入 `installers/`，三份诊断资料移入 `diagnostics/`，全部保留。

## 详细说明

- [3D 识别与抓取](docs/README_3D抓取.md)
- [夹具控制](docs/README_夹具控制.md)
- [无板标定](docs/README_无板标定.md)
- [相机预览](docs/README_相机预览.md)

所有说明中的命令都从工作区根目录运行。夹具发现工具的 `--inspect` 输出写入 `diagnostics/device_discovery.json`；`--save-config` 仍保存根目录 `gripper_devices.json`。

## 本地检查

以下命令仅运行离线测试：

```powershell
python -m unittest tests.test_arm_control -v
.\.camera-runtime\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

完整测试需要 3D 依赖及支持 Tkinter 的运行环境。`.camera-runtime` 用于相机和 3D 算法；桌面界面测试可单独使用 `.runtime` 运行。缺少桌面环境时，可使用根目录 `setup-runtime.ps1` 安装。
