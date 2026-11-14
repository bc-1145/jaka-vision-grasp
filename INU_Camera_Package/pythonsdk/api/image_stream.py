# -*- coding: utf-8 -*-
"""彩色图像流（对应官方 InuDev 5.x 的 api.image_stream）。"""

from __future__ import annotations

from enum import IntEnum

from .shared import ImageFormat


class ImageOutputFormat(IntEnum):
    DEFAULT = 0
    RAW = 1
    BGRA = 2
    RGBA = 3
    BGR = 4


class ImagePostProcessing(IntEnum):
    NONE_PROCESSING = 0
    REGISTERED = 1
    UNDISTORTED = 2
    GAMMA_CORRECT = 4


_FORMAT_NAMES = {
    ImageOutputFormat.DEFAULT: "DEFAULT",
    ImageOutputFormat.RAW: "RAW",
    ImageOutputFormat.BGRA: "BGRA",
    ImageOutputFormat.RGBA: "RGBA",
    ImageOutputFormat.BGR: "BGR",
}


def _format_from_code(code: int) -> ImageFormat | int:
    try:
        return ImageFormat(code)
    except ValueError:
        return code


class ImageFrame:
    """单帧彩色图像。

    属性：``width`` / ``height`` / ``bytes_per_pixel`` / ``format`` /
    ``frame_index`` / ``timestamp``（纳秒）。

    ``image_frame`` / ``buffer`` 返回帧数据的 ``memoryview``，可以直接
    ``np.frombuffer(frame.image_frame, np.uint8)``。
    """

    def __init__(self, native):
        self._native = native
        self._data = native.data
        self.width = native.width
        self.height = native.height
        self.bytes_per_pixel = native.bytes_per_pixel
        self.format = _format_from_code(native.format_code)
        self.frame_index = native.frame_index
        self.timestamp = native.timestamp_ns
        self.valid = native.valid

    @property
    def image_frame(self):
        return memoryview(self._data)

    @property
    def buffer(self):
        return memoryview(self._data)

    def tobytes(self) -> bytes:
        return self._data

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return (f"<ImageFrame {self.width}x{self.height} "
                f"{getattr(self.format, 'name', self.format)} "
                f"#{self.frame_index}>")


class ImageStream:
    def __init__(self, native, parent=None):
        self._native = native
        self._parent = parent

    def init(self, output_format=ImageOutputFormat.BGRA,
             post_processing=ImagePostProcessing.NONE_PROCESSING):
        try:
            name = _FORMAT_NAMES[ImageOutputFormat(output_format)]
        except (ValueError, KeyError):
            name = "BGRA"
        # 4.x 的 Init(eRegistered/...) 会连带启动深度流，这里只设置输出格式
        self._native.init(format=name)

    def start(self, *args, **kwargs):
        self._native.start()

    def stop(self):
        self._native.stop()

    def terminate(self):
        self._native.terminate()

    @property
    def frame(self):
        native = self._native.get_frame(2000)
        return ImageFrame(native) if native is not None else None
