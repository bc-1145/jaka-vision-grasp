"""Read-only discovery of JAKA controllers on a directly attached bench LAN."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import ipaddress
import json
from pathlib import Path
import socket
import time

import arm_control as sdk


def read_controller_status(host: str, timeout: float = 3.0) -> dict:
    """Read one status frame, preserving GBK device names on this controller."""
    deadline = time.monotonic() + timeout
    decoder = json.JSONDecoder()
    buffer = b""
    with socket.create_connection((host, sdk.STATUS_PORT), timeout=timeout) as connection:
        while time.monotonic() < deadline:
            connection.settimeout(max(0.01, deadline - time.monotonic()))
            chunk = connection.recv(65536)
            if not chunk:
                break
            buffer += chunk
            try:
                text = buffer.decode("utf-8")
            except UnicodeDecodeError:
                text = buffer.decode("gb18030", errors="replace")
            start = text.find("{")
            if start < 0:
                continue
            try:
                result, _ = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                continue
            if not isinstance(result, dict):
                raise ValueError("机器人状态帧不是对象。")
            return result
    raise TimeoutError(f"未收到 {host}:{sdk.STATUS_PORT} 的完整状态帧。")


def discover_controllers(subnet: str = "10.5.5.0/24") -> list[dict]:
    network = ipaddress.IPv4Network(subnet, strict=False)
    if network.prefixlen < 24:
        raise ValueError("只支持 /24 或更小的机器人网段。")

    def inspect(host):
        address = str(host)
        try:
            with socket.create_connection((address, sdk.CMD_PORT), timeout=0.35):
                pass
            reply = sdk.cmd({"cmdName": "get_robot_state"}, timeout=1.2, ip=address)
            if isinstance(reply, dict) and reply.get("cmdName") == "get_robot_state":
                return {"host": address, "state": reply}
        except (OSError, ValueError):
            pass
        return None

    with ThreadPoolExecutor(max_workers=20) as pool:
        return [result for result in pool.map(inspect, network.hosts()) if result is not None]


def inspect_controller(host: str) -> dict:
    report = {"host": host, "state": sdk.cmd({"cmdName": "get_robot_state"}, ip=host)}
    try:
        report["status"] = read_controller_status(host, timeout=2)
    except Exception as exc:
        report["status_error"] = str(exc)
    report["analog_outputs"] = []
    for index in range(16):
        try:
            reply = sdk.cmd({"cmdName": "get_analog_output", "type": 2, "index": index},
                            timeout=1, ip=host)
        except Exception as exc:
            reply = {"error": str(exc)}
        report["analog_outputs"].append({"index": index, "reply": reply})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="只读搜索机器人和扩展 IO")
    parser.add_argument("--subnet", default="10.5.5.0/24")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--save-config", action="store_true", help="保存真实 IO 映射供双夹爪界面使用")
    args = parser.parse_args()
    devices = discover_controllers(args.subnet)
    print(json.dumps(devices, ensure_ascii=False, indent=2))
    if args.save_config:
        from dataclasses import asdict
        from gripper_backend import profiles_from_status
        if len(devices) != 1:
            raise RuntimeError("请确认机器人网段中有且仅有一台目标控制器。")
        host = devices[0]["host"]
        profiles = profiles_from_status(read_controller_status(host))
        with open("gripper_devices.json", "w", encoding="utf-8") as file:
            json.dump({"host": host, "devices": [asdict(item) for item in profiles]},
                      file, ensure_ascii=False, indent=2)
        print("SAVED gripper_devices.json", len(profiles), "grippers")
    if args.inspect:
        reports = [inspect_controller(item["host"]) for item in devices]
        report_path = Path(__file__).resolve().parent / "diagnostics" / "device_discovery.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as file:
            json.dump(reports, file, ensure_ascii=False, indent=2)
        for report in reports:
            print("HOST", report["host"])
            print("STATUS_KEYS", list(report.get("status", {})))
            print("EXTENDED_AO", json.dumps(report["analog_outputs"], ensure_ascii=False))
