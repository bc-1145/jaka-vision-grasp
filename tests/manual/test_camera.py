# -*- coding: utf-8 -*-
"""使用随包 DLL 和 Python 绑定，测试 NU4000 初始化及彩色/深度取帧。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

def main() -> None:
    ROOT = Path(__file__).resolve().parents[2] / "INU_Camera_Package"
    BIN = ROOT / "bin"
    DLL_HANDLE = None

    os.environ["INUITIVE_PATH"] = str(ROOT)
    os.environ["PATH"] = str(BIN) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        DLL_HANDLE = os.add_dll_directory(str(BIN))
    sys.path.insert(0, str(ROOT / "pythonsdk"))

    from api.depth_stream import DepthOutputFormat  # noqa: E402
    from api.image_stream import ImageOutputFormat  # noqa: E402
    from api.inu_sensor import InuSensor  # noqa: E402
    from api.shared import StreamType  # noqa: E402

    sensor = InuSensor()
    color = None
    depth = None
    try:
        print("[1/3] 初始化相机...", flush=True)
        hw = sensor.init()
        sizes, _ = sensor.start(hw)

        print("[2/3] 启动彩色和深度流...", flush=True)
        color = sensor.create_stream(StreamType.GENERAL_CAMERA)
        color.init(ImageOutputFormat.BGRA)
        color.start()
        depth = sensor.create_stream(StreamType.DEPTH)
        depth.init(DepthOutputFormat.DEPTH)
        depth.start()

        print("[3/3] 等待图像帧...", flush=True)
        color_frame = color.frame
        depth_frame = depth.frame
        if color_frame is None or depth_frame is None:
            raise RuntimeError("2 秒内没有收到完整的彩色/深度帧")

        print(f"彩色: {color_frame.width} x {color_frame.height}")
        print(f"深度: {depth_frame.width} x {depth_frame.height}")
        print(f"通道: {sizes['channels']}")
        print("测试通过。")
    finally:
        if depth is not None:
            depth.stop()
            depth.terminate()
        if color is not None:
            color.stop()
            color.terminate()
        sensor.close()


if __name__ == "__main__":
    main()
