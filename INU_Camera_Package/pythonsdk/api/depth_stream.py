# -*- coding: utf-8 -*-
"""深度流（对应官方 InuDev 5.x 的 api.depth_stream）。"""

from __future__ import annotations

from enum import IntEnum

from .image_stream import ImageFrame


class DepthOutputFormat(IntEnum):
    DEFAULT = 0
    RAW = 1
    BGRA = 2
    RGBA = 3
    DEPTH = 4
    RGB = 5


class DepthPostProcessing(IntEnum):
    DEFAULT = 0
    REGISTRATION = 1


_FORMAT_NAMES = {
    DepthOutputFormat.DEFAULT: "DEFAULT",
    DepthOutputFormat.RAW: "RAW",
    DepthOutputFormat.BGRA: "BGRA",
    DepthOutputFormat.RGBA: "RGBA",
    DepthOutputFormat.DEPTH: "DEPTH",
    DepthOutputFormat.RGB: "RGB",
}


class DepthStream:
    def __init__(self, native, parent=None):
        self._native = native
        self._parent = parent

    def init(self, output_format=DepthOutputFormat.DEPTH,
             post_processing=DepthPostProcessing.DEFAULT):
        try:
            name = _FORMAT_NAMES[DepthOutputFormat(output_format)]
        except (ValueError, KeyError):
            name = "DEPTH"
        self._native.init(format=name)

    def start(self, *args, **kwargs):
        self._native.start()

    def stop(self):
        self._native.stop()

    def terminate(self):
        self._native.terminate()

    @property
    def frame(self):
        """最新深度帧；像素为 uint16 毫米，0 表示无效。"""
        native = self._native.get_frame(2000)
        return ImageFrame(native) if native is not None else None
