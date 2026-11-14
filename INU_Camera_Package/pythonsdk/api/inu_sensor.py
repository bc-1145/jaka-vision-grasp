# -*- coding: utf-8 -*-
"""传感器对象（对应官方 InuDev 5.x 的 api.inu_sensor）。"""

from __future__ import annotations

from enum import IntEnum

import _inudev

from .errors import Error, ErrorCode


class ConnectionState(IntEnum):
    UNKNOWN_CONNECTION_STATE = 0
    CONNECTED = 1
    SERVICE_DISCONNECTED = 4
    DISCONNECTED = 5


def _error_from_native(code: int, description: str):
    if code == 0:
        return None
    try:
        code = ErrorCode(code)
    except ValueError:
        pass
    return Error(code, description)


class InuSensor:
    """Inuitive 传感器。

    用法与官方 5.x 绑定保持一致::

        sensor = InuSensor()
        hw_info = sensor.init()
        channels_size, hw_info = sensor.start(hw_info)
        ...
        sensor.stop()
        sensor.terminate()
    """

    def __init__(self):
        self._native = _inudev.Sensor()
        self._hw_info = None
        self._register = None

    # -- 连接状态回调 -----------------------------------------------------
    @property
    def register(self):
        return self._register

    @register.setter
    def register(self, callback):
        self._register = callback
        self._native.set_state_callback(None if callback is None else self._on_state)

    def _on_state(self, state, code, description):
        callback = self._register
        if callback is None:
            return
        try:
            state = ConnectionState(state)
        except ValueError:
            pass
        try:
            callback(self, self, state, _error_from_native(code, description))
        except Exception:
            # 回调里的异常不应该打断 SDK 线程
            import traceback

            traceback.print_exc()

    # -- 生命周期 ---------------------------------------------------------
    def init(self, *args, **kwargs):
        """初始化设备，返回硬件信息 dict（channels / sensors）。"""
        self._hw_info = self._native.init()
        return self._hw_info

    def start(self, hw_info=None, binning: bool = False, fps: int = 0):
        """按通道启动设备。

        binning=True 使用传感器 binning 模式（800x600 / 544x360，≈30 FPS）；
        否则使用全分辨率（1600x1200 / 1080x720，≈8 FPS）。
        """
        channels_size = self._native.start(binning=binning, fps=fps)
        if hw_info is None:
            hw_info = self._hw_info
        return channels_size, hw_info

    def stop(self):
        self._native.stop()

    def terminate(self):
        self._native.terminate()

    def close(self):
        self._native.close()

    # -- 流 ---------------------------------------------------------------
    def create_stream(self, stream_type, channel: int | None = None):
        from .shared import StreamType

        kind = StreamType(stream_type)
        if kind == StreamType.DEPTH:
            from .depth_stream import DepthStream

            return DepthStream(self._native.create_depth_stream())
        if kind == StreamType.GENERAL_CAMERA:
            from .image_stream import ImageStream

            return ImageStream(self._native.create_image_stream())
        raise ValueError(f"暂不支持的流类型: {stream_type}")

    # -- 状态 -------------------------------------------------------------
    @property
    def connection_state(self) -> ConnectionState:
        try:
            return ConnectionState(self._native.connection_state)
        except ValueError:
            return ConnectionState.UNKNOWN_CONNECTION_STATE

    @property
    def hw_info(self):
        return self._hw_info
