"""Z-EFG-20P control using the supplied arm_control.py protocol."""

from __future__ import annotations

import math
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any

import arm_control as sdk
from gripper_discovery import read_controller_status
from gripper_transport import send_command


MAX_TARGET = sdk.GRIPPER_OPEN
MAX_OPENING_MM = 20.0
_CONTROLLER_LOCKS: dict[str, threading.RLock] = {}
_LOCK_REGISTRY = threading.Lock()


@dataclass(frozen=True)
class GripperProfile:
    name: str = "机器人夹爪"
    high_index: int = 3
    low_index: int = 4
    unit_id: int = 1
    device_id: str = ""

    def __post_init__(self):
        if (type(self.high_index) is not int or type(self.low_index) is not int
                or not 1 <= self.high_index < 65535 or self.low_index != self.high_index + 1):
            raise ValueError("夹爪位置映射必须是一对相邻的有效输出寄存器。")
        if type(self.unit_id) is not int or not 1 <= self.unit_id <= 247:
            raise ValueError("夹爪从站 ID 必须在 1～247 之间。")


def profiles_from_status(status: dict) -> list[GripperProfile]:
    """Derive position outputs from the live extension setup and pin offsets.

    Each configured AO range maps Modbus position addresses 2 and 3 to
    one-based JAKA extended AO indexes. Never assume the second starts at 5.
    """
    extio = status.get("extio", status)
    try:
        setup, pinmap = extio["setup"], extio["pinmap"]
        if len(setup) != extio["num"] or len(pinmap) != len(setup):
            raise ValueError("扩展 IO 设备数量和映射表不一致。")
        profiles = []
        used_indexes = set()
        device_ids = set()
        for index, device in enumerate(setup):
            first_register, count = device[3][3]
            offset = pinmap[index][3]
            if (any(type(number) is not int for number in (first_register, count, offset))
                    or offset < 0 or first_register < 0 or count < 0):
                raise ValueError("扩展 IO 地址或数量无效。")
            if not first_register <= 2 < 3 < first_register + count:
                continue
            high = offset + 2 - first_register + 1
            identity = device[4]
            if not isinstance(identity, str) or not identity or identity in device_ids:
                raise ValueError("扩展设备标识缺失或重复。")
            profile = GripperProfile(
                name=str(device[1]) or f"夹爪 {index + 1}",
                high_index=high, low_index=high + 1,
                unit_id=device[2][2], device_id=identity)
            if used_indexes.intersection((profile.high_index, profile.low_index)):
                raise ValueError("夹爪位置输出映射重叠。")
            used_indexes.update((profile.high_index, profile.low_index))
            device_ids.add(identity)
            profiles.append(profile)
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("无法解析控制器的扩展 IO 映射。") from exc
    if not profiles:
        raise ValueError("没有找到映射位置寄存器 2、3 的夹爪设备。")
    return profiles


def validate_target(value: Any) -> float:
    try:
        target = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("开度目标必须是数字。") from exc
    if not math.isfinite(target) or not 0 <= target <= MAX_TARGET:
        raise ValueError(f"开度目标必须在 0～{MAX_TARGET:g} 之间。")
    return target


def validate_opening(value: Any) -> float:
    try:
        opening = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("开合毫米数必须是数字。") from exc
    if not math.isfinite(opening) or not 0 <= opening <= MAX_OPENING_MM:
        raise ValueError(f"开合毫米数必须在 0～{MAX_OPENING_MM:g} mm 之间。")
    return opening


def opening_to_words(value: Any) -> tuple[int, int]:
    """Encode millimeters as float32, high word then low word.

    HITBOT Z-EFG-20P manual: position is float32 in mm. The supplied
    SDK maps the position pair onto extended analog outputs 3 and 4.
    20.0 mm -> 0x41A00000 -> (16800, 0), matching sdk.GRIPPER_OPEN.
    """
    return struct.unpack(">HH", struct.pack(">f", validate_opening(value)))


def words_to_opening(high: Any, low: Any) -> float:
    words = []
    for word in (high, low):
        try:
            number = float(word)
        except (TypeError, ValueError) as exc:
            raise ValueError("开度寄存器必须是 16 位整数。") from exc
        if not math.isfinite(number) or not number.is_integer() or not 0 <= number <= 65535:
            raise ValueError("开度寄存器必须是 0～65535 的整数。")
        words.append(int(number))
    opening = struct.unpack(">f", struct.pack(">HH", *words))[0]
    return validate_opening(opening)


@dataclass(frozen=True)
class Snapshot:
    state: dict
    high: float
    low: float


class WriteUncertainError(RuntimeError):
    """A write was attempted; its physical result must not be assumed."""


class GripperClient:
    def __init__(self, host: str = sdk.ROBOT_IP, simulated: bool = False,
                 timeout: float = 3.0, profile: GripperProfile | None = None):
        self.host = host.strip()
        if not self.host or any(c.isspace() for c in self.host):
            raise ValueError("请输入正确的机器人 IP 或主机名。")
        self.simulated = simulated
        self.timeout = timeout
        self.profile = profile or GripperProfile()
        with _LOCK_REGISTRY:
            self._controller_lock = _CONTROLLER_LOCKS.setdefault(self.host, threading.RLock())
        self._sim_target = 0.0
        self._sim_low = 0.0

    def _call(self, command: dict) -> dict:
        reply = send_command(command, timeout=self.timeout, ip=self.host)
        if not isinstance(reply, dict):
            raise RuntimeError("机器人返回的数据不是 JSON 对象。")
        code = str(reply.get("errorCode", "")).strip()
        if code not in ("0", "0x0", "OK"):
            raise RuntimeError(
                f"{command['cmdName']}: errorCode={code or '缺失'}，"
                f"{reply.get('errorMsg', '')}"
            )
        return reply

    def _read(self, index: int) -> float:
        value = self._call({"cmdName": "get_analog_output", "type": 2,
                            "index": index}).get("value")
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"寄存器 {index} 返回无效数值：{value!r}") from exc
        if not math.isfinite(result):
            raise RuntimeError(f"寄存器 {index} 返回非有限数值。")
        return result

    def snapshot(self) -> Snapshot:
        with self._controller_lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> Snapshot:
        if self.simulated:
            return Snapshot({"mode": "模拟", "enable": True, "power": True},
                            self._sim_target, self._sim_low)
        state = self._call({"cmdName": "get_robot_state"})
        return Snapshot(state, self._read(self.profile.high_index), self._read(self.profile.low_index))

    def _validate_mapping(self):
        if self.simulated or not self.profile.device_id:
            return
        current = profiles_from_status(read_controller_status(self.host, self.timeout))
        matching = next((item for item in current if item.device_id == self.profile.device_id), None)
        if (matching is None or matching.high_index != self.profile.high_index
                or matching.low_index != self.profile.low_index
                or matching.unit_id != self.profile.unit_id):
            raise RuntimeError(f"{self.profile.name} 的 IO 映射已变化，已阻止操作；请重新识别夹爪。")

    def connect(self) -> Snapshot:
        with self._controller_lock:
            self._validate_mapping()
            return self.snapshot()

    def set_target(self, value: Any, *, settle_time: float = 0.3) -> Snapshot:
        """Legacy SDK high-word operation; the GUI uses set_opening_mm()."""
        target = validate_target(value)
        if not target.is_integer():
            raise ValueError("原始寄存器目标必须是整数。")
        return self._write_words(int(target), 0, settle_time=settle_time)

    def set_opening_mm(self, value: Any, *, settle_time: float = 0.3) -> Snapshot:
        high, low = opening_to_words(value)
        return self._write_words(high, low, settle_time=settle_time)

    def _write_words(self, high: int, low: int, *, settle_time: float) -> Snapshot:
        with self._controller_lock:
            return self._write_words_locked(high, low, settle_time=settle_time)

    def _write_words_locked(self, high: int, low: int, *, settle_time: float) -> Snapshot:
        if self.simulated:
            self._sim_target = float(high)
            self._sim_low = float(low)
            return self.snapshot()

        self._validate_mapping()

        # This preflight only checks faults exposed by this SDK. It does not
        # enable, power on, reset, or move the robot arm.
        motion = self._call({"cmdName": "get_motion_state"})
        for key in ("estop", "collision", "soft_limit"):
            flag = motion.get(key)
            if flag is not None and str(flag).strip().lower() not in (
                    "0", "false", "0.0", ""):
                raise RuntimeError(f"机器人报告 {key}={flag}，已阻止本次写入。")

        high_index, low_index = self.profile.high_index, self.profile.low_index
        stage = f"写入寄存器 {high_index}"
        try:
            # Preserve the exact order from sdk.gripper(). These two writes
            # are not atomic and must never be retried automatically.
            self._call({"cmdName": "set_analog_output", "type": 2,
                        "index": high_index, "value": float(high)})
            stage = f"写入寄存器 {low_index}"
            self._call({"cmdName": "set_analog_output", "type": 2,
                        "index": low_index, "value": float(low)})
            stage = "回读寄存器"
            # Live dragging reads the output setpoints immediately. Neither
            # delayed nor immediate readback confirms physical finger position.
            if settle_time > 0:
                time.sleep(settle_time)
            result = self.snapshot()
            if result.high != high:
                raise RuntimeError(f"寄存器 {high_index} 回读 {result.high:g}，目标 {high}")
            if result.low != low:
                raise RuntimeError(f"寄存器 {low_index} 回读 {result.low:g}，目标 {low}")
            return result
        except Exception as exc:
            raise WriteUncertainError(
                f"{stage}时出错：{exc}。夹具可能已动作，结果尚未确认；"
                "请检查现场并重新连接回读，程序不会自动重试。"
            ) from exc
