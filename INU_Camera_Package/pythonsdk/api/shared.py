# -*- coding: utf-8 -*-
"""共享的枚举类型。"""

from enum import IntEnum


class StreamType(IntEnum):
    """InuSensor.create_stream() 可创建的流类型。"""

    GENERAL_CAMERA = 1
    TRACKING = 2
    STEREO = 3
    DEPTH = 4
    FEATURES_TRACKING = 5


class ImageFormat(IntEnum):
    """帧缓冲的像素格式。"""

    EMPTY = 0
    DEPTH = 1
    BGR = 2
    BGRA = 3
    DISPARITY = 4
    RGBA = 7
    RGB = 13
