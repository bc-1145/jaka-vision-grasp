# -*- coding: utf-8 -*-
"""
Inuitive InuDev 4.x 的 Python 绑定 (pythonic 包装层)
=====================================================

本目录下的 ``_inudev`` 扩展模块是直接基于本机安装的 InuDev 4.16 C++ SDK
编译出来的（官方 4.x 安装包不包含 pythonsdk，只有 Inuitive 支持渠道才有）。
这里按 InuDev 5.x Python API 的命名习惯提供了 ``api`` 包：

    from api.errors import ErrorCode, Error
    from api.inu_sensor import InuSensor, ConnectionState
    from api.image_stream import ImageStream, ImageOutputFormat
    from api.depth_stream import DepthStream, DepthOutputFormat
    from api.shared import StreamType, ImageFormat

它只是 ``_inudev`` 的薄封装，行为与 ``color_depth_viewer.py`` 期望的一致。

导入本包之前需要让 ``InuStreams.dll`` 能被找到：把 SDK 的 ``bin`` 目录加入
DLL 搜索路径（下面的代码会读 ``INUITIVE_PATH``，否则用默认安装路径）。
"""

from __future__ import annotations

import os
import sys

_dll_handles = []


def _setup_native_paths() -> None:
    """让 _inudev.pyd 能找到 InuStreams.dll 及其依赖。"""
    candidates = [
        os.environ.get("INUITIVE_PATH"),
        r"C:\Program Files\Inuitive\InuDev",
        r"C:\Program Files (x86)\Inuitive\InuDev",
        "/opt/Inuitive/InuDev",
    ]
    bin_dir = None
    for root in candidates:
        if not root:
            continue
        path = os.path.join(root, "bin")
        if os.path.isdir(path):
            bin_dir = path
            break
    if bin_dir is None:
        return
    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        try:
            # 句柄必须保留，否则目录会立刻被移除
            _dll_handles.append(os.add_dll_directory(bin_dir))
        except OSError:
            pass


_setup_native_paths()

from api.shared import ImageFormat, StreamType  # noqa: E402,F401
from api.errors import Error, ErrorCode  # noqa: E402,F401
from api.inu_sensor import ConnectionState, InuSensor  # noqa: E402,F401

__all__ = [
    "ImageFormat",
    "StreamType",
    "Error",
    "ErrorCode",
    "InuSensor",
    "ConnectionState",
]
