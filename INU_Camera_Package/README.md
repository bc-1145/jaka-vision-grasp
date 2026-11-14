# INU NU4000 相机独立包

仅包含 Inuitive NU4000B0 彩色/深度相机相关文件，适用于 **64 位 Windows**。不包含机械臂、夹爪、标定数据或历史采集图片。

## 在新电脑上安装

将压缩包完整解压，不要只解压 `bin\inu_viewer.exe`。路径可以自定义，启动脚本不依赖原电脑目录。

### 1. 安装 USB 驱动

右键以管理员身份运行：

```text
installers\InuDriver_3.0.7.0-1.09_Setup_X64.exe
```

安装后连接相机 USB。设备管理器中应出现 `NU4000B0`，硬件 ID 为 `VID_2959&PID_4001`。

### 2. 安装 InuDev SDK / InuService

运行：

```text
installers\InuDev_Pro_4.16.0006.30_X64.exe
```

建议保持默认安装路径。该安装包提供相机必需的 `InuService`。安装完成后重新插拔一次相机 USB，并确认服务正在运行：

```powershell
Get-Service InuService
```

`Status` 应为 `Running`。如果目标电脑已经安装同版本驱动和 InuDev SDK，可以跳过前两步。

### 3. 启动预览

双击：

```text
run_viewer.bat
```

预览程序使用本包 `bin` 目录中的 DLL，无需设置 `PATH`，也不需要安装 Python。

## 预览界面操作

| 按键 | 功能 |
| --- | --- |
| `S` | 保存当前彩色图和 16 位深度图到 `captures\` |
| `A` | 深度量程自动 / 手动切换 |
| `+` / `-` | 调整手动深度量程 |
| `M` | 切换深度配色 |
| `F` | 画面旋转 180°（默认开启） |
| `R` | 切换全分辨率 / 流畅模式 |
| `P` / `空格` | 暂停 / 继续 |
| `Esc` / `Q` | 退出 |

默认输出：彩色 **1600×1200**、深度 **1080×720**。深度 PNG 的像素单位为毫米，`0` 表示无效像素。

## Python 使用（可选）

随包绑定只支持 **64 位 Python 3.14**，其他 Python 版本不能直接加载该 `.pyd`。

```bat
py -3.14 -m pip install numpy opencv-python
run_python_viewer.bat
```

也可以不安装 NumPy / OpenCV，直接运行连接和取帧测试：

```bat
py -3.14 ..\tests\manual\test_camera.py
```

测试成功会打印彩色、深度分辨率以及“测试通过”。

Python 接口位于 `pythonsdk\api\`，扩展为 `pythonsdk\_inudev.cp314-win_amd64.pyd`。`../tests/manual/test_camera.py` 开头的路径设置展示了如何从任意目录加载随包 DLL 和绑定；不要将 `.pyd` 单独移走。

## 文件说明

```text
run_viewer.bat              C++ 预览启动器（推荐）
run_python_viewer.bat       Python 预览启动器
../tests/manual/test_camera.py 连接及彩色/深度取帧测试
color_depth_viewer.py       Python 预览源码
bin\                       预览程序及随包运行库
pythonsdk\                 Python 3.14 x64 绑定
installers\                官方 USB 驱动和 InuDev 4.16 安装包
source\                    C++ 查看器及 Python 绑定源码（仅重建时需要）
```

## 连不上时

1. 关闭所有正在使用相机的程序，重新插拔 USB，优先使用主板上的 USB 3.x 接口。
2. 在 `services.msc` 中重新启动 `InuService`。
3. 确认设备管理器里的 `NU4000B0` 没有感叹号。
4. 再运行 `run_viewer.bat`；Python 用户可运行 `../tests/manual/test_camera.py` 查看具体错误。
