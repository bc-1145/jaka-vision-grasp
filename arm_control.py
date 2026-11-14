# -*- coding: utf-8 -*-
"""Minimal JAKA TCP/IP (port 10001) client for this bench.

Robot: JAKA 6-axis cobot at 10.5.5.100
Gripper: Z-EFG-20P on the extended IO module, driven through analog output
         registers 3 (high word) + 4 (low word); 16800 = fully open (~20 mm),
         0 = fully closed.

Examples:
    python arm_control.py state
    python arm_control.py move-rel 0 0 30 0 0 0 --speed 10
    python arm_control.py move-abs -57.6 -144.1 343.9 -175.7 -3.0 90.2
    python arm_control.py gripper open
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
from pathlib import Path

ROBOT_IP = "10.5.5.100"
CMD_PORT = 10001
STATUS_PORT = 10000

GRIPPER_OPEN = 16800.0
GRIPPER_CLOSE = 0.0
POSES_FILE = Path(__file__).resolve().with_name("robot_poses.json")


def cmd(obj: dict, timeout: float = 6.0, ip: str = ROBOT_IP) -> dict:
    s = socket.socket()
    s.settimeout(timeout)
    try:
        s.connect((ip, CMD_PORT))
        s.sendall(json.dumps(obj).encode() + b"\n")
        buf = b""
        depth = 0
        started = False
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                chunk = s.recv(65536)
            except socket.timeout:
                break
            if not chunk:
                break
            buf += chunk
            for ch in chunk.decode("utf-8", "replace"):
                if ch == "{":
                    depth += 1
                    started = True
                elif ch == "}":
                    depth -= 1
            if started and depth == 0:
                break
        return json.loads(buf.decode("utf-8", "replace"))
    finally:
        s.close()


def call(obj: dict, timeout: float = 6.0) -> dict:
    r = cmd(obj, timeout)
    code = str(r.get("errorCode", "")).strip()
    if code not in ("0", "0x0", "OK"):
        raise RuntimeError(f"{obj.get('cmdName')} failed: errorCode={code} {r.get('errorMsg')}")
    return r


def get_state() -> dict:
    return call({"cmdName": "get_robot_state"})


def get_status_snapshot(timeout: float = 6.0, ip: str = ROBOT_IP) -> dict:
    """Read one complete status frame from the port 10000 push stream."""
    s = socket.socket()
    s.settimeout(timeout)
    dec = json.JSONDecoder()
    buf = b""
    try:
        s.connect((ip, STATUS_PORT))
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                chunk = s.recv(65536)
            except socket.timeout:
                break
            if not chunk:
                break
            buf += chunk
            text = buf.decode("utf-8", "replace")
            start = text.find("{")
            if start < 0:
                continue
            try:
                obj, _ = dec.raw_decode(text[start:])
                return obj
            except json.JSONDecodeError:
                continue
        raise TimeoutError(f"no complete status frame on {ip}:{STATUS_PORT}")
    finally:
        s.close()


def get_tcp() -> list:
    return call({"cmdName": "get_actual_tcp_pos"})["position"]


def get_joints() -> list:
    return call({"cmdName": "get_actual_joint_pos"})["position"]


def forward_kinematics(joints) -> list:
    return call({"cmdName": "kine_forward",
                 "jointPosition": list(joints)})["cartPosition"]


def get_rapid_rate() -> float:
    return float(call({"cmdName": "get_rapid_rate"})["value"])


def set_rapid_rate(value: float):
    rate = float(value)
    if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
        raise ValueError("rapid rate must be between 0 and 1")
    return call({"cmdName": "rapid_rate", "rate_value": rate})


def get_motion_state() -> dict:
    return call({"cmdName": "get_motion_state"})


def move_rel(dxyzabc, speed=10.0, accel=20.0, tol=0.0):
    return call({"cmdName": "moveL", "relFlag": 1, "cartPosition": list(dxyzabc),
                 "speed": speed, "accel": accel, "tol": tol})


def move_abs(pose, speed=10.0, accel=20.0, tol=0.0,
             ori_vel=180.0, ori_acc=720.0):
    return call({"cmdName": "moveL", "relFlag": 0, "cartPosition": list(pose),
                 "speed": speed, "accel": accel, "tol": tol,
                 "ori_vel": ori_vel, "ori_acc": ori_acc})


def move_joint_abs(joints, speed=180.0, accel=720.0, tol=0.0):
    return call({"cmdName": "joint_move", "relFlag": 0,
                 "jointPosition": list(joints), "speed": speed,
                 "accel": accel, "tol": tol})


def load_pose(name: str = "home") -> list[float]:
    with POSES_FILE.open("r", encoding="utf-8-sig") as f:
        poses = json.load(f)
    try:
        pose = poses[name]["position"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"pose {name!r} is missing from {POSES_FILE}") from exc
    if not isinstance(pose, list) or len(pose) != 6:
        raise RuntimeError(f"pose {name!r} must contain six values")
    return [float(v) for v in pose]


def load_home() -> tuple[list[float], float]:
    with POSES_FILE.open("r", encoding="utf-8-sig") as f:
        poses = json.load(f)
    try:
        home = poses["home"]
        pose = [float(value) for value in home["position"]]
        approach_z = float(home["approach_z"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"home.position and home.approach_z are required in {POSES_FILE}") from exc
    if len(pose) != 6:
        raise RuntimeError("home.position must contain six values")
    if not all(math.isfinite(value) for value in (*pose, approach_z)):
        raise RuntimeError("home pose values must be finite")
    if approach_z <= pose[2]:
        raise RuntimeError("home.approach_z must be higher than the home Z value")
    return pose, approach_z


def load_home_motion() -> tuple[float, float, float, float, float]:
    with POSES_FILE.open("r", encoding="utf-8-sig") as f:
        home = json.load(f)["home"]
    try:
        speed = float(home["speed"])
        accel = float(home["accel"])
        joint_speed = float(home["joint_speed"])
        joint_accel = float(home["joint_accel"])
        rapid_rate = float(home["rapid_rate"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("home speed, accel and rapid_rate are required") from exc
    if (not math.isfinite(speed) or speed <= 0 or
            not math.isfinite(accel) or accel <= 0 or
            not math.isfinite(joint_speed) or joint_speed <= 0 or
            not math.isfinite(joint_accel) or joint_accel <= 0 or
            not math.isfinite(rapid_rate) or not 0 < rapid_rate <= 1):
        raise RuntimeError("home motion parameters are invalid")
    return speed, accel, joint_speed, joint_accel, rapid_rate


def load_home_joint() -> list[float]:
    with POSES_FILE.open("r", encoding="utf-8-sig") as f:
        home = json.load(f)["home"]
    try:
        joints = [float(value) for value in home["approach_joint"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("home.approach_joint is required") from exc
    if len(joints) != 6 or not all(math.isfinite(value) for value in joints):
        raise RuntimeError("home.approach_joint must contain six finite values")
    return joints


def load_joint3_lift_config() -> tuple[float, float, float]:
    with POSES_FILE.open("r", encoding="utf-8-sig") as f:
        home = json.load(f)["home"]
    try:
        target_z = float(home["first_lift_z"])
        step = float(home["joint3_search_step"])
        maximum_delta = float(home["joint3_max_delta"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("home joint-3 lift settings are required") from exc
    if (not all(math.isfinite(value) for value in (target_z, step, maximum_delta)) or
            step <= 0 or maximum_delta <= 0):
        raise RuntimeError("home joint-3 lift settings are invalid")
    return target_z, step, maximum_delta


def plan_joint3_lift(current_joints, target_z, step=1.0, maximum_delta=100.0):
    """Find a joint-3-only target whose FK raises TCP Z monotonically."""
    current = [float(value) for value in current_joints]
    if len(current) != 6 or not all(math.isfinite(value) for value in current):
        raise ValueError("current joints must contain six finite values")
    start_pose = forward_kinematics(current)
    if start_pose[2] >= target_z:
        return current, start_pose
    candidates = []
    samples = int(math.floor(maximum_delta / step))
    for direction in (-1.0, 1.0):
        previous_z = float(start_pose[2])
        for index in range(1, samples + 1):
            target = list(current)
            target[2] = current[2] + direction * index * step
            # JAKA Zu joint 3 has no useful solution near or beyond a folded
            # +/-145 degree posture; do not ask the controller to cross it.
            if not -144.0 <= target[2] <= 144.0:
                break
            try:
                pose = forward_kinematics(target)
            except Exception:
                break
            z = float(pose[2])
            if z < previous_z - 0.5:
                break
            previous_z = z
            if z >= target_z:
                candidates.append((index * step, target, pose))
                break
    if not candidates:
        raise RuntimeError(
            f"joint 3 alone cannot raise TCP Z from {start_pose[2]:.3f} "
            f"to {target_z:.3f} mm from the current posture")
    _delta, target, pose = min(candidates, key=lambda item: item[0])
    return target, pose


def _pose_error(actual: list[float], target: list[float]) -> tuple[float, float]:
    position_error = max(abs(a - b) for a, b in zip(actual[:3], target[:3]))
    rotation_error = max(abs((a - b + 180.0) % 360.0 - 180.0)
                         for a, b in zip(actual[3:], target[3:]))
    return position_error, rotation_error


def move_verified(pose, speed=10.0, accel=20.0, timeout=120.0):
    result = move_abs(pose, speed, accel)
    code = str(result.get("errorCode", "")).strip()
    if code not in ("0", "0x0", "OK"):
        raise RuntimeError(f"moveL rejected: {result}")
    if not wait_inpos(timeout=timeout):
        raise TimeoutError(f"motion did not finish within {timeout:g} seconds")
    actual = get_tcp()
    position_error, rotation_error = _pose_error(actual, list(pose))
    if position_error > 5.0 or rotation_error > 2.0:
        raise RuntimeError(f"robot stopped away from target: actual={actual}")
    return actual


def move_joint_verified(joints, speed=180.0, accel=720.0, timeout=120.0):
    result = move_joint_abs(joints, speed, accel)
    code = str(result.get("errorCode", "")).strip()
    if code not in ("0", "0x0", "OK"):
        raise RuntimeError(f"joint_move rejected: {result}")
    if not wait_inpos(timeout=timeout):
        raise TimeoutError(f"joint motion did not finish within {timeout:g} seconds")
    actual = get_joints()
    error = max(abs((a - b + 180.0) % 360.0 - 180.0)
                for a, b in zip(actual, joints))
    if error > 2.0:
        raise RuntimeError(f"robot stopped away from joint target: actual={actual}")
    return actual


def move_home_staged(speed=10.0, accel=20.0, joint_speed=180.0,
                     joint_accel=720.0, timeout=120.0):
    """Return home by lifting, joint-aligning at clearance, then descending."""
    home, configured_z = load_home()
    approach_joint = load_home_joint()
    lift_z, joint3_step, joint3_max_delta = load_joint3_lift_config()
    motion = get_motion_state()
    if (motion.get("estop") or motion.get("collision") or motion.get("soft_limit") or
            motion.get("queue") or motion.get("active_queue") or not motion.get("inpos")):
        raise RuntimeError(f"robot has an active/faulted motion state: {motion}")
    approach_pose = [home[0], home[1], configured_z, *home[3:]]
    print(f"home: {home}")
    print(f"configured approach Z: {configured_z:g}; first lift Z: {lift_z:g}")
    actual = get_tcp()
    position_error, rotation_error = _pose_error(actual, home)
    if position_error <= 0.5 and rotation_error <= 0.5:
        print("HOME: already at target")
        return actual
    if actual[2] < lift_z:
        current_joints = get_joints()
        lift_joint, planned_pose = plan_joint3_lift(
            current_joints, lift_z, joint3_step, joint3_max_delta)
        print(f"JOINT-3 Z LIFT: J3 {current_joints[2]:.3f} -> {lift_joint[2]:.3f}; "
              f"planned Z {actual[2]:.3f} -> {planned_pose[2]:.3f}")
        lifted_joints = move_joint_verified(lift_joint, joint_speed, joint_accel, timeout)
        actual = get_tcp()
        if actual[2] < lift_z - 2.0:
            raise RuntimeError(
                f"lift waypoint did not reach the required height: actual Z={actual[2]:.3f}")
        print(f"JOINT-3 Z LIFT OK: joints={[round(value, 3) for value in lifted_joints]} "
              f"tcp={[round(value, 3) for value in actual]}")
    else:
        print(f"JOINT-3 Z LIFT: skipped; current Z={actual[2]:.3f} is already clear")

    print(f"JOINT ALIGN: {approach_joint}")
    actual_joints = get_joints()
    joint_error = max(abs((a - b + 180.0) % 360.0 - 180.0)
                      for a, b in zip(actual_joints, approach_joint))
    if joint_error > 0.5:
        actual_joints = move_joint_verified(
            approach_joint, joint_speed, joint_accel, timeout)
        print(f"JOINT ALIGN OK: {[round(value, 3) for value in actual_joints]}")
    actual = get_tcp()
    position_error, rotation_error = _pose_error(actual, approach_pose)
    if position_error > 5.0 or rotation_error > 2.0:
        raise RuntimeError(
            "taught approach joints no longer match the configured approach pose; "
            f"actual={actual}")

    print(f"DESCEND: {home}")
    actual = move_verified(home, speed, accel, timeout)
    print(f"DESCEND OK: {[round(value, 3) for value in actual]}")
    return get_tcp()


def write_word(index: int, value: float):
    return call({"cmdName": "set_analog_output", "type": 2, "index": index,
                 "value": float(value)})


def read_word(index: int):
    return call({"cmdName": "get_analog_output", "type": 2, "index": index}).get("value")


def gripper(value: float):
    """value: 0 = closed, 16800 = open (gripper target position register)."""
    write_word(3, value)
    write_word(4, 0.0)
    time.sleep(0.3)
    return read_word(3)


def wait_inpos(timeout=30.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        ms = get_motion_state()
        if ms.get("estop") or ms.get("collision") or ms.get("soft_limit"):
            raise RuntimeError(f"motion fault: {ms}")
        if ms.get("inpos") and not ms.get("queue") and not ms.get("active_queue"):
            return True
        time.sleep(0.2)
    return False


def print_state():
    try:
        st = get_status_snapshot()
    except Exception as exc:  # 10000 端口不可用时退回命令口
        raw = get_state()
        st = {"enabled": raw.get("enable"), "powered_on": raw.get("power"),
              "errmsg": raw.get("errorMsg")}
        print(f"warning: status stream unavailable ({exc}); using get_robot_state")
    tcp = get_tcp()
    joints = get_joints()
    key = {k: st.get(k) for k in ("enabled", "powered_on", "inpos", "errmsg",
                                  "protective_stop", "emergency_stop", "rapidrate",
                                  "motion_mode", "task_state")}
    print("state:", json.dumps(key, ensure_ascii=False))
    print("tcp  :", [round(v, 3) for v in tcp])
    print("joint:", [round(v, 3) for v in joints])


def main() -> int:
    ap = argparse.ArgumentParser(description="JAKA arm helper")
    sub = ap.add_subparsers(dest="action", required=True)

    sub.add_parser("state")

    p = sub.add_parser("home", help="move to the home pose in robot_poses.json")
    p.add_argument("--speed", type=float)
    p.add_argument("--accel", type=float)

    p = sub.add_parser("move-rel", help="relative moveL: dx dy dz da db dc")
    p.add_argument("d", nargs=6, type=float)
    p.add_argument("--speed", type=float, default=10.0)
    p.add_argument("--accel", type=float, default=20.0)

    p = sub.add_parser("move-abs", help="absolute moveL: x y z a b c")
    p.add_argument("p", nargs=6, type=float)
    p.add_argument("--speed", type=float, default=10.0)
    p.add_argument("--accel", type=float, default=20.0)

    p = sub.add_parser("gripper", help="open/close/set target (0..16800)")
    p.add_argument("value")

    args = ap.parse_args()

    if args.action == "state":
        print_state()
    elif args.action == "home":
        state = get_status_snapshot()
        if not state.get("enabled") or not state.get("powered_on"):
            raise RuntimeError("robot is not powered on and enabled")
        if state.get("protective_stop") or state.get("emergency_stop"):
            raise RuntimeError(f"robot is not ready: {state.get('errmsg', '')}")
        pose, _approach_z = load_home()
        (configured_speed, configured_accel, configured_joint_speed,
         configured_joint_accel, configured_rate) = load_home_motion()
        speed = args.speed if args.speed is not None else configured_speed
        accel = args.accel if args.accel is not None else configured_accel
        previous_rate = get_rapid_rate()
        print(f"motion: speed={speed:g} mm/s accel={accel:g} mm/s^2 "
              f"rapid_rate={configured_rate:g}")
        try:
            if abs(previous_rate - configured_rate) > 1e-6:
                set_rapid_rate(configured_rate)
            actual = move_home_staged(
                speed, accel, configured_joint_speed, configured_joint_accel,
                timeout=120.0)
        finally:
            if abs(previous_rate - configured_rate) > 1e-6:
                try:
                    set_rapid_rate(previous_rate)
                    print(f"rapid rate restored to {previous_rate:g}")
                except Exception as exc:
                    print(f"warning: unable to restore rapid rate: {exc}")
        print("actual:", [round(v, 3) for v in actual])
        position_error, rotation_error = _pose_error(actual, pose)
        if position_error > 5.0 or rotation_error > 2.0:
            raise RuntimeError("robot stopped, but the actual pose does not match home")
        print("HOME OK")
    elif args.action == "move-rel":
        print("before:", [round(v, 2) for v in get_tcp()])
        r = move_rel(args.d, args.speed, args.accel)
        print("sent:", r.get("errorCode"), r.get("errorMsg"))
        wait_inpos()
        print("after :", [round(v, 2) for v in get_tcp()])
    elif args.action == "move-abs":
        print("before:", [round(v, 2) for v in get_tcp()])
        r = move_abs(args.p, args.speed, args.accel)
        print("sent:", r.get("errorCode"), r.get("errorMsg"))
        wait_inpos()
        print("after :", [round(v, 2) for v in get_tcp()])
    elif args.action == "gripper":
        v = args.value.lower()
        target = GRIPPER_OPEN if v == "open" else GRIPPER_CLOSE if v == "close" else float(v)
        print("before:", read_word(3))
        print("set  :", gripper(target))
    return 0


if __name__ == "__main__":
    sys.exit(main())
