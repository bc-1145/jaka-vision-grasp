"""Read physical Z-EFG-20P feedback through existing JAKA input mappings.

No actuator, bus-configuration or controller commands are sent. AO setpoints
are deliberately never inspected: they are not physical feedback.

Protocol: https://cn.hitbot.cc/hitbot/document/Z-EFG/Z-EFG-20P/Z-EFG-20P.pdf
0x0040 initialization (5=ready), 0x0041 grip state (0=arrived, 1=moving,
2=holding, 3=dropped), 0x0042..43 actual position float32 in mm,
0x0046..47 actual motor current float32 in A, high word first.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import struct
import time

from gripper_backend import GripperProfile
from gripper_discovery import read_controller_status


PROTOCOL_SOURCE = "https://cn.hitbot.cc/hitbot/document/Z-EFG/Z-EFG-20P/Z-EFG-20P.pdf"
SOURCE = "jaka_extio_input_registers"
STATE_NAMES = {0: "arrived", 1: "moving", 2: "holding", 3: "dropped"}
CONFIG = Path(__file__).with_name("gripper_devices.json")


def _robot_profile():
    data = json.loads(CONFIG.read_text(encoding="utf-8-sig"))
    entries = [item for item in data["devices"] if item.get("unit_id") == 1]
    if len(entries) != 1:
        raise ValueError("Robot gripper unit_id=1 is not uniquely configured")
    return data["host"], GripperProfile(**entries[0])


def _word(value):
    if isinstance(value, bool):
        raise ValueError("Boolean register value")
    number = float(value)
    if not math.isfinite(number) or not number.is_integer() or not 0 <= number <= 65535:
        raise ValueError("Feedback register is not a finite unsigned 16-bit word")
    return int(number)


def _float32(words):
    result = struct.unpack(">f", struct.pack(">HH", *[_word(x) for x in words]))[0]
    if not math.isfinite(result):
        raise ValueError("Nonfinite physical feedback")
    return float(result)


def feedback_from_status(status, profile, *, created_unix=None):
    """Decode a fresh status frame; only the identified device's AI range counts.

    ``available`` means valid initialized grip-state feedback is present.
    Position/current may independently be missing. ``holding`` is None when
    the feedback is unavailable, uninitialized or malformed, never inferred
    from current, output registers, target position, or missing data.
    """
    result = dict(available=False, grip_state=None, state_name=None,
                  holding=None, dropped=None, actual_position_mm=None,
                  current_a=None, initialization_state=None, reason="",
                  source=SOURCE, protocol_source=PROTOCOL_SOURCE,
                  created_unix=time.time() if created_unix is None else created_unix,
                  field_available=dict(state=False, position=False, current=False,
                                       initialization=False),
                  device_id=profile.device_id, unit_id=profile.unit_id,
                  command_is_measured_gap=False)
    try:
        if not profile.device_id:
            raise ValueError("Device identity is missing; cannot select physical feedback")
        extio = status.get("extio", status)
        setup, pinmap, values = extio["setup"], extio["pinmap"], extio["status"]
        if len(setup) != extio["num"] or len(pinmap) != len(setup):
            raise ValueError("Extension IO setup/pinmap count mismatch")
        matches = [(index, device) for index, device in enumerate(setup)
                   if device[4] == profile.device_id]
        if len(matches) != 1:
            raise ValueError("Configured gripper identity is absent or duplicated")
        index, device = matches[0]
        if device[2][2] != profile.unit_id:
            raise ValueError("Configured gripper unit ID changed")
        first_register, count = device[3][2]
        offset = pinmap[index][2]
        if (any(type(value) is not int for value in (first_register, count, offset))
                or min(first_register, count, offset) < 0):
            raise ValueError("Invalid physical input-register mapping")
        result["input_mapping"] = dict(first_register=first_register,
                                        register_count=count, input_offset=offset)
        if extio.get("mode") != 1:
            result["reason"] = "扩展 IO 未运行，反馈可能是缓存值"
            return result

        def read_words(address, length):
            if not first_register <= address or address + length > first_register + count:
                return None
            start = offset + address - first_register
            if len(values) <= 2 or len(values[2]) < start + length:
                raise ValueError("Mapped input registers are absent from the status frame")
            return [_word(value) for value in values[2][start:start + length]]

        initialization = read_words(0x0040, 1)
        state = read_words(0x0041, 1)
        position = read_words(0x0042, 2)
        current = read_words(0x0046, 2)
        if initialization is not None:
            result["initialization_state"] = initialization[0]
            result["field_available"]["initialization"] = True
        if state is not None:
            if state[0] not in STATE_NAMES:
                raise ValueError("Unknown physical grip-state code")
            result.update(grip_state=state[0], state_name=STATE_NAMES[state[0]])
            result["field_available"]["state"] = True
        if position is not None:
            actual = _float32(position)
            if not 0 <= actual <= 20.05:
                raise ValueError("Physical position is outside the gripper stroke")
            result["actual_position_mm"] = actual
            result["field_available"]["position"] = True
        if current is not None:
            amperes = _float32(current)
            if abs(amperes) > 10:
                raise ValueError("Physical current is implausible; check register word order")
            result["current_a"] = amperes
            result["field_available"]["current"] = True
        if state is None:
            result["reason"] = "夹持状态 0x0041 未映射到输入；AO 开度设定值不能确认夹持"
        elif initialization is None:
            result["reason"] = "初始化状态 0x0040 未映射，无法确认反馈已初始化"
        elif initialization[0] != 5:
            result["reason"] = "夹爪反馈未初始化完成"
        else:
            result.update(available=True, holding=state[0] == 2, dropped=state[0] == 3,
                          reason="physical grip-state feedback")
    except (KeyError, IndexError, TypeError, ValueError, OverflowError) as exc:
        result.update(available=False, holding=None, dropped=None,
                      reason="无法验证夹爪真实反馈：" + str(exc))
    return result


def read_feedback(host=None, profile=None, *, status=None, timeout=2.0):
    """Read status port only. Safe inside a caller's existing controller lease."""
    if profile is None:
        configured_host, profile = _robot_profile()
        host = host or configured_host
    host = host or "10.5.5.100"
    try:
        frame = read_controller_status(host, timeout) if status is None else status
        return feedback_from_status(frame, profile)
    except (OSError, TimeoutError, ValueError) as exc:
        result = feedback_from_status({}, profile)
        result["reason"] = "夹爪反馈读取失败：" + str(exc)
        return result


class GripperFeedback:
    def __init__(self, host=None, profile=None, *, reader=None):
        self.host, self.profile = host, profile
        self.reader = reader

    def snapshot(self):
        if self.reader is not None:
            return self.reader()
        return read_feedback(self.host, self.profile)

    def wait_for_hold(self, timeout=2.0, stable_samples=3, interval=.1):
        if timeout <= 0 or stable_samples < 2 or interval <= 0:
            raise ValueError("Hold verification requires a positive timeout and at least two samples")
        deadline = time.monotonic() + timeout
        samples, consecutive, last_stamp = [], 0, None
        while time.monotonic() < deadline:
            sample = self.snapshot()
            samples.append(sample)
            if not sample.get("available"):
                return dict(confirmed=False, reason=sample.get("reason", "feedback unavailable"), samples=samples)
            if sample.get("dropped"):
                return dict(confirmed=False, reason="physical drop feedback", samples=samples)
            stamp = sample.get("created_unix")
            fresh = isinstance(stamp, (int, float)) and math.isfinite(stamp) and abs(time.time() - stamp) <= 2
            if not fresh or (last_stamp is not None and stamp <= last_stamp):
                return dict(confirmed=False, reason="feedback is stale or repeated", samples=samples)
            last_stamp = stamp
            consecutive = consecutive + 1 if sample.get("holding") is True else 0
            if consecutive >= stable_samples:
                return dict(confirmed=True, reason="stable physical grip-state feedback", samples=samples)
            time.sleep(interval)
        return dict(confirmed=False, reason="physical holding state was not stable before timeout", samples=samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    host, profile = _robot_profile()
    status = read_controller_status(host, 3)
    result = dict(created_unix=time.time(), host=host, profile=asdict(profile),
                  extio=status.get("extio"), feedback=read_feedback(host, profile, status=status),
                  motion_sent=False, configuration_changed=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
