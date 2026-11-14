# -*- coding: utf-8 -*-
"""错误码与异常类型。"""

from enum import IntEnum


class ErrorCode(IntEnum):
    STATUS_OK = 0
    INTERNAL_ERROR = 1
    INIT_ERROR = 2
    TIMEOUT_ERROR = 3
    STATE_ERROR = 4
    OPERATION_CANCELED = 5
    UNEXPECTED_PARAMETER_VALUE = 6
    DUAL_SENSOR_IDENTIFICATION_ERROR = 7
    INVALID_FRAME = 8
    SERVICE_PROCESS_FAILURE = 11
    SENSOR_DETECTION_FAILURE = 12


class Error(Exception):
    """与官方 API 类似的错误对象。"""

    def __init__(self, code=ErrorCode.STATUS_OK, description=""):
        self.code = code
        self.description = description
        super().__init__(f"{code} {description}".strip())
