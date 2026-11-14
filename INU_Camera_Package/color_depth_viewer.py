# -*- coding: utf-8 -*-
"""
Inuitive NU4000 彩色 + 深度实时可视化
=====================================

依赖:
    - Inuitive InuDev SDK (Windows x64 版, 或 Linux 版)
      默认路径: C:/Program Files/Inuitive/InuDev/    (Windows)
                /opt/Inuitive/InuDev/                (Linux)
      也可以用环境变量 INUITIVE_PATH 或 --sdk 指定。
    - Python 绑定: 本目录 pythonsdk/ 下自编译的 _inudev 扩展
      (官方 4.x 安装包不带 pythonsdk, 详见 README)
    - pip install opencv-python numpy

用法:
    python color_depth_viewer.py
    python color_depth_viewer.py --sdk "D:/Inuitive/InuDev"
    python color_depth_viewer.py --min-depth 200 --max-depth 4000

按键:
    q / Esc   退出
    s         保存当前一对图像 (color_XXXX.png / depth_XXXX.png)
    a         切换深度范围自动/手动
    + / -     手动模式下调整最大深度 (每次 250 mm)
    m         切换深度伪彩色 colormap

注意:
    Inuitive 深度流输出的单位是毫米 (uint16), 0 表示无效/无回波。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


def find_sdk_dir(cli_arg: str | None) -> Path:
    candidates = []
    if cli_arg:
        candidates.append(cli_arg)
    env = os.environ.get("INUITIVE_PATH")
    if env:
        candidates.append(env)
    if sys.platform == "win32":
        candidates += [r"C:/Program Files/Inuitive/InuDev",
                       r"C:/Program Files (x86)/Inuitive/InuDev"]
    else:
        candidates += ["/opt/Inuitive/InuDev", os.path.expanduser("~/Inuitive/InuDev")]

    for c in candidates:
        p = Path(c)
        if p.is_dir():
            return p
    raise SystemExit(
        "找不到 Inuitive SDK 目录。请用 --sdk 或 INUITIVE_PATH 指定, 例如:\n"
        '  set INUITIVE_PATH=C:\\Program Files\\Inuitive\\InuDev\n'
        "  python color_depth_viewer.py"
    )


def prepare_runtime(sdk_dir: Path) -> None:
    """把 SDK 的 python 包和原生动态库加进搜索路径。"""
    py_sdk = sdk_dir / "pythonsdk"
    bin_dir = sdk_dir / "bin"
    if not py_sdk.is_dir():
        # 官方 4.x SDK 安装包不含 pythonsdk，使用本目录随附的绑定
        py_sdk = Path(__file__).resolve().parent / "pythonsdk"
    if not py_sdk.is_dir():
        raise SystemExit(f"SDK 目录不完整, 缺少 {sdk_dir / 'pythonsdk'} (也没有找到本地 pythonsdk)")
    if sys.platform == "win32":
        # 让 InuStreams.dll 能找到同目录下的依赖 DLL（句柄要保留，否则立即失效）
        if hasattr(os, "add_dll_directory") and bin_dir.is_dir():
            _DLL_DIR_HANDLES.append(os.add_dll_directory(str(bin_dir)))
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    else:
        os.environ["LD_LIBRARY_PATH"] = str(bin_dir) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    sys.path.insert(0, str(py_sdk))
    os.environ["INUITIVE_PATH"] = str(sdk_dir)


_DLL_DIR_HANDLES = []


# --------------------------------------------------------------------------------------
# 图像转换
# --------------------------------------------------------------------------------------

def frame_to_color(frame):
    """把 ImageFrame (BGRA/RGBA/BGR/RGB) 转成可显示的三通道 uint8。"""
    h, w, bpp = frame.height, frame.width, frame.bytes_per_pixel
    if h == 0 or w == 0:
        return None
    arr = _raw_bytes(frame, bpp)
    if arr is None:
        arr = np.asarray(frame.buffer)

    if bpp == 4:
        img = arr.reshape(h, -1, 4)[:, :w, :]
        return _four_channel_to_bgr(img, frame.format)
    if bpp == 3:
        img = arr.reshape(h, -1, 3)[:, :w, :]
        # 3 通道的 RGB/BGR 取决于流格式, 默认按 BGR 显示并在下方提供开关
        return img
    # 1/2 字节的彩色流(如 YUV/Bayer)暂不支持
    return None


def frame_to_depth(frame):
    """把 ImageFrame (DEPTH, uint16/毫米) 转成 h x w 的 numpy 数组。"""
    h, w, bpp = frame.height, frame.width, frame.bytes_per_pixel
    if h == 0 or w == 0:
        return None
    if bpp != 2:
        return None

    arr = _raw_bytes(frame, 2)
    if arr is not None and arr.size % h == 0:
        stride = arr.size // h
        if stride >= w:
            return arr.reshape(h, stride)[:, :w]
        return None

    # 退路: 用 SDK 自己的 buffer 属性
    try:
        arr = np.asarray(frame.buffer)
        if arr.size == h * w:
            arr = arr.astype(np.uint32) & 0xFFFF
            return arr.astype(np.uint16).reshape(h, w)
    except Exception:
        pass
    return None


def _raw_bytes(frame, itemsize: int):
    """绕过 SDK 的 dtype 假设, 直接按原始字节读取帧缓冲。"""
    target = getattr(frame, "image_frame", None)
    if target is None:
        return None
    try:
        buf = np.frombuffer(target, dtype=np.uint8)
    except Exception:
        return None
    if buf.size % itemsize:
        return None
    return buf.view(np.uint16 if itemsize == 2 else np.uint8)


def _four_channel_to_bgr(img, fmt):
    import cv2
    # api.shared.ImageFormat: BGRA / RGBA 两个 4 通道格式
    name = getattr(fmt, "name", str(fmt))
    if "RGBA" in name:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def render_depth(depth, min_mm, max_mm, colormap):
    """uint16 毫米 -> 伪彩色 BGR。0 视为无效, 显示为黑色。"""
    import cv2
    d = depth.astype(np.float32)
    span = max(1.0, float(max_mm - min_mm))
    norm = np.clip((d - min_mm) * 255.0 / span, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, colormap)
    color[d <= 0] = (0, 0, 0)
    return color


def center_distance(depth) -> float:
    h, w = depth.shape
    patch = depth[h // 2 - 10:h // 2 + 10, w // 2 - 10:w // 2 + 10]
    valid = patch[patch > 0]
    return float(np.median(valid)) if valid.size else 0.0


# --------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Inuitive NU4000 color+depth viewer")
    ap.add_argument("--sdk", help="Inuitive SDK 根目录 (含 bin/ 和 pythonsdk/)")
    ap.add_argument("--min-depth", type=int, default=200, help="深度显示下限 mm (默认 200)")
    ap.add_argument("--max-depth", type=int, default=4000, help="深度显示上限 mm (默认 4000)")
    ap.add_argument("--scale", type=float, default=1.0, help="窗口缩放 (默认 1.0)")
    args = ap.parse_args()

    sdk_dir = find_sdk_dir(args.sdk)
    prepare_runtime(sdk_dir)
    print(f"[i] SDK: {sdk_dir}")

    try:
        import cv2
    except ImportError:
        raise SystemExit("缺少 opencv-python, 请先: pip install opencv-python numpy")

    from api.errors import ErrorCode, Error          # noqa: E402
    from api.inu_sensor import InuSensor, ConnectionState  # noqa: E402
    from api.image_stream import ImageStream, ImageOutputFormat, ImagePostProcessing  # noqa: E402
    from api.depth_stream import DepthStream, DepthOutputFormat, DepthPostProcessing  # noqa: E402
    from api.shared import StreamType, ImageFormat   # noqa: E402

    state = {"connected": False, "error": ""}

    def sensor_cb(_self, sensor, connection_state, error):
        state["connected"] = connection_state == ConnectionState.CONNECTED
        if error is not None and getattr(error, "code", None) not in (None, ErrorCode.STATUS_OK):
            state["error"] = f"{error.code} {getattr(error, 'description', '')}"
        print(f"[sensor] state={connection_state} error={error}")

    sensor = InuSensor()
    print("[i] 初始化传感器 ...")
    try:
        hw_info = sensor.init()
        channels_size, hw_info = sensor.start(hw_info)
        sensor.register = sensor_cb
    except Exception as exc:
        raise SystemExit(
            f"[x] 传感器初始化失败: {exc}\n"
            "    常见原因: 相机没有枚举 (USB 线/口问题), 或 InuService 没有运行, "
            "或 SDK 版本与固件不匹配。"
        )

    color_stream = sensor.create_stream(StreamType.GENERAL_CAMERA)
    color_stream.init(ImageOutputFormat.BGRA, ImagePostProcessing.NONE_PROCESSING)
    color_stream.start()

    depth_stream = sensor.create_stream(StreamType.DEPTH)
    depth_stream.init(DepthOutputFormat.DEPTH, DepthPostProcessing.DEFAULT)
    depth_stream.start()
    print("[i] 彩色流 + 深度流已启动, 按 q/Esc 退出")

    win = "Inuitive NU4000  color | depth"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    min_mm = args.min_depth
    max_mm = args.max_depth
    auto_range = False
    colormaps = [cv2.COLORMAP_JET, cv2.COLORMAP_TURBO, cv2.COLORMAP_INFERNO]
    cm_idx = 0

    saved = 0
    last_t = time.time()
    fps = 0.0

    try:
        while True:
            try:
                color_frame = color_stream.frame
                depth_frame = depth_stream.frame
            except KeyboardInterrupt:
                break

            color_img = frame_to_color(color_frame) if color_frame is not None else None
            depth_arr = frame_to_depth(depth_frame) if depth_frame is not None else None

            if depth_arr is not None:
                if auto_range:
                    valid = depth_arr[depth_arr > 0]
                    if valid.size:
                        max_mm = int(np.percentile(valid, 99))
                        max_mm = max(max_mm, min_mm + 100)
                depth_img = render_depth(depth_arr, min_mm, max_mm, colormaps[cm_idx])
                center_mm = center_distance(depth_arr)
                cv2.putText(depth_img, f"{center_mm/1000.0:.2f} m  [{min_mm}-{max_mm}mm "
                                       f"{'auto' if auto_range else 'manual'}]",
                            (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.drawMarker(depth_img, (depth_arr.shape[1] // 2, depth_arr.shape[0] // 2),
                               (255, 255, 255), cv2.MARKER_CROSS, 16, 1)
            else:
                depth_img = None

            if color_img is not None:
                h, w = color_img.shape[:2]
                cv2.putText(color_img, f"COLOR {w}x{h}", (12, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            if color_img is not None and depth_img is not None and color_img.shape[:2] == depth_img.shape[:2]:
                left, right = color_img, depth_img
            else:  # 尺寸不一致时各自缩放, 并排显示
                left = color_img if color_img is not None else _placeholder(640, 480, "no color")
                right = depth_img if depth_img is not None else _placeholder(640, 480, "no depth")
                left = cv2.resize(left, (640, 480))
                right = cv2.resize(right, (640, 480))

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-3, now - last_t))
            last_t = now

            canvas = cv2.hconcat([left, right])
            cv2.putText(canvas, f"{fps:5.1f} FPS   ['q' quit  's' save  'a' auto  '+'/'-' range  'm' colormap]",
                        (12, canvas.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            if args.scale != 1.0:
                canvas = cv2.resize(canvas, None, fx=args.scale, fy=args.scale)
            cv2.imshow(win, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("s") and color_img is not None and depth_arr is not None:
                saved += 1
                cv2.imwrite(f"color_{saved:04d}.png", color_img)
                cv2.imwrite(f"depth_{saved:04d}.png", depth_arr)
                print(f"[i] saved color_{saved:04d}.png / depth_{saved:04d}.png")
            elif key == ord("a"):
                auto_range = not auto_range
            elif key == ord("+") or key == ord("="):
                max_mm += 250
            elif key == ord("-"):
                max_mm = max(min_mm + 250, max_mm - 250)
            elif key == ord("m"):
                cm_idx = (cm_idx + 1) % len(colormaps)

            if state["error"]:
                print(f"[!] {state['error']}")
                state["error"] = ""
    finally:
        print("[i] 停止流 ...")
        for s in (depth_stream, color_stream):
            try:
                s.stop()
                s.terminate()
            except Exception as exc:
                print(f"[!] 停止流时出错: {exc}")
        try:
            sensor.register = None
            sensor.stop()
            sensor.terminate()
        except Exception as exc:
            print(f"[!] 停止传感器时出错: {exc}")
        cv2.destroyAllWindows()

    return 0


def _placeholder(w, h, text):
    import cv2
    img = np.zeros((h, w, 3), np.uint8)
    cv2.putText(img, text, (w // 2 - 60, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
    return img


if __name__ == "__main__":
    sys.exit(main())
