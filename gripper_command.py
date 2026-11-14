"""One-shot robot gripper command with checked mapping and output readback."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from controller_lease import controller_lease
from gripper_backend import GripperClient, GripperProfile, opening_to_words, words_to_opening
from gripper_transport import close_connections


CONFIG = Path(__file__).with_name("gripper_devices.json")


def robot_profile():
    data = json.loads(CONFIG.read_text(encoding="utf-8-sig"))
    matches = [item for item in data["devices"] if item.get("unit_id") == 1]
    if len(matches) != 1:
        raise ValueError("必须唯一配置从站 ID 为 1 的机器人夹爪")
    profile = GripperProfile(**matches[0])
    if not profile.device_id:
        raise ValueError("机器人夹爪缺少设备标识，无法核对实际 IO 映射")
    return data["host"], profile


def main():
    parser = argparse.ArgumentParser(description="独立设置机器人夹爪开度")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--open", action="store_true", help="松开到 20 mm")
    action.add_argument("--mm", type=float, help="设置开度指令，范围 0～20 mm")
    parser.add_argument("--check", action="store_true", help="仅检查配置，不连接或操作夹具")
    parser.add_argument("--settle-seconds", type=float, default=0.0,
                        help="写入开度后等待的秒数，0～5；不是实际开度反馈")
    args = parser.parse_args()
    if not math.isfinite(args.settle_seconds) or not 0 <= args.settle_seconds <= 5:
        parser.error("等待时间必须为 0～5 秒")
    opening = 20.0 if args.open else args.mm
    high, low = opening_to_words(opening)
    host, profile = robot_profile()
    if args.check:
        print(json.dumps({"device": profile.name, "host": host, "unit_id": profile.unit_id,
                          "opening_mm": opening, "indexes": [profile.high_index, profile.low_index],
                          "words": [high, low], "settle_seconds": args.settle_seconds},
                         ensure_ascii=False, indent=2))
        return 0
    with controller_lease(timeout=8.0):
        try:
            client = GripperClient(host, profile=profile)
            snapshot = client.set_opening_mm(opening, settle_time=args.settle_seconds)
            actual = words_to_opening(snapshot.high, snapshot.low)
            print(f"{profile.name}：已发送 {opening:.2f} mm 指令，回读设定值 {actual:.2f} mm。", flush=True)
        finally:
            close_connections()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"夹具操作未完成：{exc}", flush=True)
        raise SystemExit(2)
