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
import logging
import multiprocessing
import os
import sys
import time
import subprocess
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
    runtime = Path(__file__).resolve().parent / ".camera-runtime" / "python.exe"
    if sys.platform == "win32" and sys.version_info[:2] != (3, 10) and runtime.is_file():
        return subprocess.call([str(runtime), str(Path(__file__).resolve()), *sys.argv[1:]])
    ap = argparse.ArgumentParser(description="Inuitive NU4000 color+depth viewer")
    ap.add_argument("--sdk", help="Inuitive SDK 根目录 (含 bin/ 和 pythonsdk/)")
    ap.add_argument("--min-depth", type=int, default=200, help="深度显示下限 mm (默认 200)")
    ap.add_argument("--max-depth", type=int, default=4000, help="深度显示上限 mm (默认 4000)")
    ap.add_argument("--scale", type=float, default=1.0, help="窗口缩放 (默认 1.0)")
    ap.add_argument("--check-frames", type=int, default=0, help="Exit after verifying this many color/depth frame pairs")
    ap.add_argument("--live-model", action="store_true", help="Continuously match native XYZ to CAD; no robot/gripper commands")
    args = ap.parse_args()

    from pose_overlay import PoseOverlay
    root = Path(__file__).resolve().parent
    pose_overlay = PoseOverlay(root / "pick_results", root / "pick_place_config.json", live=args.live_model)

    sdk_dir = find_sdk_dir(args.sdk)
    prepare_runtime(sdk_dir)
    print(f"[i] SDK: {sdk_dir}")

    try:
        import cv2
    except ImportError:
        raise SystemExit("缺少 opencv-python, 请先: pip install opencv-python numpy")

    from camera_capture import CameraCapture, configure_logging
    configure_logging("camera_viewer.log")
    log = logging.getLogger(__name__)
    log.info("Camera preview opened; SDK=%s", sdk_dir)
    capture = CameraCapture(sdk_dir, with_points=args.live_model)
    live_matcher = None
    if args.live_model:
        from live_model_preview import LiveModelMatcher
        live_matcher = LiveModelMatcher(root / "pick_place_config.json", root / "pick_results" / "live_model")
        print("[i] Live CAD preview only: no robot or gripper I/O", flush=True)
    win = "Inuitive NU4000  color | depth | model pose"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, _placeholder(1280, 480, "Connecting to camera..."))

    min_mm = args.min_depth
    max_mm = args.max_depth
    auto_range = False
    colormaps = [cv2.COLORMAP_JET, cv2.COLORMAP_TURBO, cv2.COLORMAP_INFERNO]
    cm_idx = 0

    verified_frames = 0
    frame_deadline = time.monotonic() + 30
    saved = 0
    last_t = time.monotonic()
    last_report = last_t
    last_color = last_depth = None
    fps = 0.0

    try:
        while True:
            frames = capture.poll()
            if frames is not None:
                last_color, last_depth = frames
                verified_frames += 1
                now = time.monotonic()
                fps = 0.9 * fps + 0.1 / max(1e-3, now - last_t)
                last_t = now
                if now - last_report >= 30:
                    log.info("Preview active: displayed_pairs=%s source_frames=%s fps=%.1f",
                             verified_frames, capture.sequence, fps)
                    last_report = now
                if verified_frames == 1:
                    print(f"[i] Frames received: color={last_color.shape}, depth={last_depth.shape}, valid_depth={np.count_nonzero(last_depth)}", flush=True)
            if args.check_frames and verified_frames < args.check_frames and time.monotonic() > frame_deadline:
                raise RuntimeError("Camera verification timed out after 30 seconds; see logs/camera_capture.log")
            if live_matcher is not None:
                result = live_matcher.poll()
                if result is not None:
                    source_depth = (live_matcher.source_images[1]
                                    if live_matcher.source_images is not None else None)
                    pose_overlay.set_live_result(result, source_depth)
                    print(f"[i] Live CAD: matches={len(result['matches'])} "
                          f"fit={result.get('duration_s', 0):.2f}s error={result.get('error', '')}", flush=True)
                if last_color is not None and last_depth is not None and not live_matcher.busy:
                    cloud = capture.poll_points()
                    if cloud is not None:
                        sequence, stamp, points = cloud
                        live_matcher.submit(points, stamp, sequence, last_color, last_depth)
            # Overlays must not modify the last raw image or an SDK-owned buffer.
            color_img = last_color.copy() if last_color is not None else None
            depth_arr = last_depth

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

            panel_height = color_img.shape[0] if color_img is not None and depth_img is not None and color_img.shape[:2] == depth_img.shape[:2] else 480
            pose_panel = pose_overlay.draw(color_img, height=panel_height, depth_image=depth_img,
                                           current_depth=depth_arr, camera_live=capture.status == "Live")
            if color_img is not None and depth_img is not None and color_img.shape[:2] == depth_img.shape[:2]:
                left, right = color_img, depth_img
            else:  # 尺寸不一致时各自缩放, 并排显示
                left = color_img if color_img is not None else _placeholder(640, 480, "no color")
                right = depth_img if depth_img is not None else _placeholder(640, 480, "no depth")
                left = cv2.resize(left, (640, 480))
                right = cv2.resize(right, (640, 480))

            canvas = cv2.hconcat([left, right, pose_panel])
            cv2.putText(canvas, f"{fps:5.1f} FPS   ['q' quit  's' save  'a' auto  '+'/'-' range  'm' colormap]",
                        (12, canvas.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            if capture.status != "Live":
                age = "" if last_color is None else f" - last image {time.monotonic() - last_t:.1f}s ago"
                cv2.rectangle(canvas, (0, 40), (left.shape[1] + right.shape[1], 80), (0, 0, 0), -1)
                cv2.putText(canvas, capture.status + age, (12, 67),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 190, 255), 2, cv2.LINE_AA)
            if args.scale != 1.0:
                canvas = cv2.resize(canvas, None, fx=args.scale, fy=args.scale)
            cv2.imshow(win, canvas)
            if live_matcher is not None and result is not None:
                result["displayed_contours"] = pose_overlay.outline_count
                cv2.imwrite(str(root / "pick_results" / "live_model" / "preview.png"), canvas)
                if pose_overlay.outline_count:
                    cv2.imwrite(str(root / "pick_results" / "live_model" / "contours.png"), canvas)
                log.info("Live CAD display: matches=%s contours=%s input_age=%.1fs",
                         len(result["matches"]), pose_overlay.outline_count,
                         time.time() - result.get("captured_unix", time.time()))

            if args.check_frames and verified_frames >= args.check_frames:
                if not np.any(depth_arr):
                    raise RuntimeError("Depth frames contain no valid measurements")
                check_dir = Path(__file__).resolve().parent / ".runtime-setup"
                check_dir.mkdir(exist_ok=True)
                cv2.imwrite(str(check_dir / "camera-check.png"), canvas)
                print(f"[i] Verified {verified_frames} color/depth frame pairs", flush=True)
                break
            key = cv2.waitKey(20) & 0xFF
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
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

    except KeyboardInterrupt:
        log.info("Preview interrupted by user")
    finally:
        log.info("Closing camera preview")
        try:
            capture.close()
        finally:
            if live_matcher is not None:
                live_matcher.close()
            cv2.destroyAllWindows()

    return 0


def _placeholder(w, h, text):
    import cv2
    img = np.zeros((h, w, 3), np.uint8)
    cv2.putText(img, text, (w // 2 - 60, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
    return img


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
