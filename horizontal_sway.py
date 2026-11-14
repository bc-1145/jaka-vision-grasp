"""One-minute Cartesian level sway with visibly large wrist motion."""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation, Slerp

import arm_control as arm
import calibration_motion as safety
from controller_lease import controller_lease
from gripper_discovery import read_controller_status
from one_click_loading import full_speed

ROOT = arm.POSES_FILE.parent
REPORT = ROOT / "logs" / "horizontal_sway.json"
DURATION = 58.0
LINEAR_ACCEL = 400.0
MAX_LINEAR_SPEED = 250.0
ORIENTATION_SPEED = 90.0
ORIENTATION_ACCEL = 360.0
MAX_JOINT_SPEED = 120.0
MAX_JOINT_ACCEL = 600.0
DEFAULT_CYCLES = 9
HEIGHT_TOLERANCE = 1.0
POLL = 0.05
# A small feedback/settling allowance, never an instruction to accelerate.
ENDPOINT_GRACE = 0.35
TOTAL_GRACE = 0.5
FEEDBACK_GRACE = 0.1
PLANNED_HEIGHT_TOLERANCE = 0.1
WRIST_AMPLITUDES = (20.0, 15.0, 30.0)
JOINT_LIMITS = ((-175, 175), (-175, 175), (-144, 144), (-175, 175), (-115, 115), (-355, 355))


class TempoError(ValueError):
    """Timing can be retried at a slower tempo; geometry errors cannot."""


def validate(amplitude, cycles):
    if not math.isfinite(amplitude) or not 1 <= amplitude <= 200:
        raise ValueError("单侧幅度必须在 1～200 mm 之间")
    if type(cycles) is not int or not 1 <= cycles <= 12:
        raise ValueError("往返次数必须为 1～12 次")


def endpoint_joints(origin, joints, displacement):
    """Solve X displacement on the circle made by rotating base joint J1."""
    radius = math.hypot(origin[0], origin[1])
    desired_x = origin[0] + displacement
    if radius < 1 or abs(desired_x) >= radius:
        raise ValueError("当前水平半径不足以完成此左右幅度，请调整起点或减小幅度")
    bearing = math.atan2(origin[0], -origin[1])
    angle = math.asin(desired_x / radius)
    candidates = [math.degrees(a + 2 * math.pi * turn - bearing)
                  for a in (angle, math.pi - angle) for turn in (-1, 0, 1)]
    delta = min(candidates, key=abs)
    target = list(joints)
    target[0] += delta
    if abs(delta) > 60 or not -175 <= target[0] <= 175:
        raise ValueError("J1 摆幅超出本脚本的保守范围；请调整起点或减小幅度")
    return target


def leg_speed(distance, seconds):
    # Leave time for TCP commands and arrival confirmation. The controller
    # supplies the Cartesian acceleration profile; no servo streaming.
    travel_time = seconds - 0.15
    discriminant = travel_time ** 2 - 4 * distance / LINEAR_ACCEL
    if travel_time <= 0 or discriminant <= 0:
        raise TempoError("当前摆幅无法在设定节奏内平滑完成，请减少往返次数")
    speed = 2 * distance / (travel_time + math.sqrt(discriminant))
    if not 0 < speed <= MAX_LINEAR_SPEED:
        raise TempoError("所需 TCP 速度超出本脚本限制")
    return speed


def compensate_height(robot, target, height):
    """Find a nearby J3 correction for a large wrist endpoint."""
    target = list(target)
    center = target[2]

    def error(j3):
        q = list(target)
        q[2] = j3
        return float(robot.forward(q)[2]) - height

    original_error = error(center)
    if abs(original_error) <= 1e-7:
        return target
    for distance in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0):
        for direction in (-1, 1):
            candidate = center + direction * distance
            if not -144 <= candidate <= 144:
                continue
            if error(candidate) * original_error <= 0:
                target[2] = brentq(error, min(center, candidate), max(center, candidate), xtol=1e-8)
                return target
    raise ValueError("J3 无法在 15° 内补偿大幅摆腕端点高度，请调整起点")


def coordinated_targets(robot, origin, joints, amplitude):
    targets = {"center": list(joints)}
    poses = {"center": list(origin)}
    bend_direction = 1 if joints[4] >= 0 else -1
    for name, direction in (("left", -1), ("right", 1)):
        target = list(joints)
        target[3] += direction * WRIST_AMPLITUDES[0]
        target[4] += bend_direction * WRIST_AMPLITUDES[1]
        target[5] -= direction * WRIST_AMPLITUDES[2]
        if not (-175 <= target[3] <= 175 and -115 <= target[4] <= 115
                and -355 <= target[5] <= 355):
            raise ValueError("腕部配合动作超出本脚本的关节范围")
        target = compensate_height(robot, target, origin[2])
        before_yaw = robot.forward(target)
        target = endpoint_joints(before_yaw, target, origin[0] + direction * amplitude - before_yaw[0])
        pose = robot.forward(target).tolist()
        if abs(pose[0] - origin[0] - direction * amplitude) > 0.1 or abs(pose[2] - origin[2]) > 0.01:
            raise RuntimeError("腕部补偿后的端点不满足横向幅度或高度要求")
        pose[2] = origin[2]
        targets[name], poses[name] = target, pose
    return targets, poses


def sample_path(robot, start_pose, end_pose, start_joints, end_joints):
    """Validate the Cartesian path actually sent to MoveL, with continuous IK."""
    start_pose, end_pose = np.asarray(start_pose), np.asarray(end_pose)
    distance = float(np.linalg.norm(end_pose[:3] - start_pose[:3]))
    angle = safety.pose_error(start_pose, end_pose)[1]
    samples = max(2, math.ceil(distance / 2.5), math.ceil(angle))
    fractions = np.linspace(0, 1, samples + 1)
    rotations = Slerp([0, 1], Rotation.from_euler("xyz", [start_pose[3:], end_pose[3:]], degrees=True))
    reference = np.asarray(start_joints)
    path = [reference]
    height_error = 0.0
    for fraction in fractions[1:]:
        pose = start_pose * (1 - fraction) + end_pose * fraction
        pose[2] = start_pose[2]
        pose[3:] = rotations(fraction).as_euler("xyz", degrees=True)
        q = safety.vector(robot.inverse(pose, reference))
        if np.max(np.abs(q - reference)) > 8:
            raise ValueError("逆解发生关节跳转，已阻止大幅摆腕")
        if any(not low <= value <= high for value, (low, high) in zip(q, JOINT_LIMITS)):
            raise ValueError("大幅摆腕轨迹超出关节规划范围")
        fk = robot.forward(q)
        position_error, rotation_error = safety.pose_error(fk, pose)
        height_error = max(height_error, abs(float(fk[2]) - start_pose[2]))
        if position_error > 0.1 or rotation_error > 0.1 or height_error > PLANNED_HEIGHT_TOLERANCE:
            raise ValueError("大幅摆腕的等高轨迹正逆解不一致")
        path.append(q)
        reference = q
    if np.max(np.abs(reference - end_joints)) > 0.1:
        raise ValueError("逆解到达了不同关节圈数或腕部姿态，已阻止执行")
    path = np.asarray(path)
    return {"start_pose": start_pose.tolist(), "pose": end_pose.tolist(),
            "start_joints": list(start_joints), "joints": list(end_joints),
            "joint_min": np.min(path, axis=0).tolist(), "joint_max": np.max(path, axis=0).tolist(),
            "path_samples": samples + 1, "max_height_error_mm": height_error,
            "_path": path, "_fractions": fractions, "_distance": distance, "_angle": angle}


def time_leg(sample, seconds):
    """Evaluate tempo on the already validated geometry without more robot IO."""
    path, fractions = sample["_path"], sample["_fractions"]
    distance, angle = sample["_distance"], sample["_angle"]
    speed = leg_speed(distance, seconds)
    if angle / distance * speed > ORIENTATION_SPEED or angle / distance * LINEAR_ACCEL > ORIENTATION_ACCEL:
        raise TempoError("大幅摆腕的姿态速度或加速度超过限制，请减少往返次数或调整起点")
    first = np.gradient(path, fractions, axis=0, edge_order=2)
    second = np.gradient(first, fractions, axis=0, edge_order=2)
    velocities = np.minimum(speed, np.sqrt(2 * LINEAR_ACCEL * distance * np.minimum(fractions, 1 - fractions)))
    ramp_distance = speed ** 2 / (2 * LINEAR_ACCEL)
    accelerations = np.where(fractions * distance < ramp_distance, LINEAR_ACCEL,
                            np.where((1 - fractions) * distance < ramp_distance, -LINEAR_ACCEL, 0))
    joint_speed = np.max(np.abs(first * velocities[:, None] / distance), axis=0)
    joint_accel = np.max(np.abs(second * (velocities[:, None] / distance) ** 2
                                + first * accelerations[:, None] / distance), axis=0)
    if np.max(joint_speed) > MAX_JOINT_SPEED or np.max(joint_accel) > MAX_JOINT_ACCEL:
        raise TempoError("采样关节速度或加速度过大，请减少往返次数或调整起点")
    return {**{key: value for key, value in sample.items() if not key.startswith("_")}, "speed": speed,
            "sampled_joint_speed_deg_s": joint_speed.tolist(),
            "sampled_joint_accel_deg_s2": joint_accel.tolist()}


def sample_leg(robot, start_pose, end_pose, start_joints, end_joints, seconds):
    return time_leg(sample_path(robot, start_pose, end_pose, start_joints, end_joints), seconds)


def make_plan(robot, amplitude=200.0, cycles=None):
    auto_tempo = cycles is None
    if cycles is None:
        cycles = DEFAULT_CYCLES
    validate(amplitude, cycles)
    status = read_controller_status(arm.ROBOT_IP, timeout=5)
    safety.check_state(status)
    start = robot.motion_snapshot()
    safety.check_motion_state(start)
    joints = list(start["joint_actual_position"])
    origin = robot.forward(joints).tolist()
    if any(a > b for a, b in zip(safety.pose_error(origin, start["actual_position"]), (0.5, 0.2))):
        raise RuntimeError("当前 TCP 与正解坐标不一致，已阻止摆动")
    # Keep the requested visible wrist amplitudes. A failed plan never silently
    # shrinks them into a barely visible movement.
    targets, poses = coordinated_targets(robot, origin, joints, amplitude)
    requested_cycles = cycles
    samples = []
    previous_name = "center"
    for name in ("right", "center", "left", "center"):
        sample = sample_path(robot, poses[previous_name], poses[name], targets[previous_name], targets[name])
        sample["name"] = name
        samples.append(sample)
        previous_name = name
    rejected_tempos = []
    for candidate in (range(cycles, 0, -1) if auto_tempo else (cycles,)):
        try:
            routes = [time_leg(sample, DURATION / candidate / 4) for sample in samples]
        except TempoError as exc:
            if not auto_tempo:
                raise
            rejected_tempos.append({"cycles": candidate, "reason": str(exc)})
        else:
            cycles = candidate
            break
    else:
        raise TempoError("当前位置在保持摆幅和速度限制时没有可用节奏；请调整起点。"
                         + rejected_tempos[-1]["reason"])
    period = DURATION / cycles
    legs = []
    for index in range(cycles * 4):
        arrival = (index + 1) * period / 4
        legs.append({**routes[index % 4], "arrival_s": arrival})
    end = robot.motion_snapshot()
    safety.check_motion_state(end)
    if max(abs(a - b) for a, b in zip(end["joint_actual_position"], joints)) > 0.1:
        raise RuntimeError("规划期间起点改变，请重新启动")
    return {"origin_pose": origin, "origin_joints": joints, "amplitude_mm": amplitude,
            "duration_s": DURATION, "cycles": cycles, "motion": "cartesian_level_wrist_sway",
            "tempo_mode": "auto" if auto_tempo else "explicit", "requested_cycles": requested_cycles,
            "rejected_tempos": rejected_tempos,
            "height_mm": origin[2], "endpoint_poses": poses, "legs": legs,
            "path_fk_samples": sum(route["path_samples"] for route in routes),
            "max_planned_height_error_mm": max(route["max_height_error_mm"] for route in routes),
            "wrist_amplitudes_deg": list(WRIST_AMPLITUDES),
            "j3_compensation_deg": {name: q[2] - joints[2] for name, q in targets.items()},
            "collision_clearance_checked": False}


def check_envelope(state, plan, leg):
    safety.check_motion_state(state, stationary=False)
    actual = state["joint_actual_position"]
    if abs(state["actual_position"][2] - plan["height_mm"]) > HEIGHT_TOLERANCE:
        raise RuntimeError("高度偏离超过 1 mm，已停止摆动")
    for index, (q, low, high) in enumerate(zip(actual, leg["joint_min"], leg["joint_max"])):
        if not low - 0.8 <= q <= high + 0.8:
            raise RuntimeError(f"J{index + 1} 偏离当前配合动作段，已停止摆动")
    start, end = np.asarray(leg["start_pose"]), np.asarray(leg["pose"])
    position = np.asarray(state["actual_position"][:3])
    vector = end[:3] - start[:3]
    fraction = float(np.clip(np.dot(position - start[:3], vector) / np.dot(vector, vector), 0, 1))
    if np.linalg.norm(position - (start[:3] + fraction * vector)) > 2:
        raise RuntimeError("TCP 偏离等高摆动路径，已停止")


def execute(robot, plan, report, clock=time.monotonic, sleep=time.sleep):
    before = robot.motion_snapshot()
    safety.check_motion_state(before)
    if max(abs(a - b) for a, b in zip(before["joint_actual_position"], plan["origin_joints"])) > 0.1:
        raise RuntimeError("执行起点改变，请重新启动")
    started = clock()
    report["motion_attempted"] = False
    report["leg_results"] = []
    hard_deadline = started + plan["duration_s"] + TOTAL_GRACE
    longest_leg_s = 0.0
    completed_legs = 0
    try:
        for index, leg in enumerate(plan["legs"]):
            start_s = plan["legs"][index - 1]["arrival_s"] if index else 0.0
            # Wait before dispatch, not by fetching another snapshot at the
            # previous leg's deadline (which charged its latency to this leg).
            while clock() < started + start_s:
                sleep(min(POLL, started + start_s - clock()))
            slot_s = leg["arrival_s"] - start_s
            distance = float(np.linalg.norm(np.asarray(leg["pose"][:3]) - leg["start_pose"][:3]))
            travel_s = distance / leg["speed"] + leg["speed"] / LINEAR_ACCEL
            # Start another out-and-back only while at the captured center and
            # with room to return, using the slowest observed leg as a budget.
            at_center = max(abs(a - b) for a, b in zip(
                leg["start_joints"], plan["origin_joints"])) <= 0.1
            if index and at_center and index + 1 < len(plan["legs"]):
                next_leg = plan["legs"][index + 1]
                next_distance = float(np.linalg.norm(
                    np.asarray(next_leg["pose"][:3]) - next_leg["start_pose"][:3]))
                next_travel_s = next_distance / next_leg["speed"] + next_leg["speed"] / LINEAR_ACCEL
                pair_budget = (max(travel_s + POLL, longest_leg_s)
                               + max(next_travel_s + POLL, longest_leg_s) + POLL)
                if hard_deadline - POLL - clock() < pair_budget:
                    report["ended_before_next_pair"] = {
                        "next_leg": index + 1, "pair_budget_s": pair_budget,
                        "remaining_s": started + plan["duration_s"] - clock()}
                    break
            # Each segment gets its own bounded timeout; accumulated settling
            # latency must not shorten a later segment's travel allowance.
            deadline = min(clock() + slot_s + ENDPOINT_GRACE,
                           hard_deadline - FEEDBACK_GRACE)
            timing = {"leg": index + 1, "scheduled_start_s": start_s,
                      "scheduled_arrival_s": leg["arrival_s"],
                      "deadline_s": deadline - started, "expected_travel_s": travel_s}
            report["leg_results"].append(timing)
            if deadline - clock() < travel_s + POLL:
                raise TimeoutError("控制通信未跟上摆动节奏，剩余时间不足；停止后不再追加动作")
            report.update(motion_attempted=True, leg=index + 1)
            safety.save_json(REPORT, report)
            if deadline - clock() < travel_s + POLL:
                raise TimeoutError("运行记录写入耗时过长，已取消后续动作")
            timing["command_sent_s"] = clock() - started
            # Mark intent before sending; an uncertain ACK must never replay.
            robot.call("moveL", relFlag=0, cartPosition=leg["pose"],
                       speed=leg["speed"], accel=LINEAR_ACCEL, tol=0,
                       ori_vel=ORIENTATION_SPEED, ori_acc=ORIENTATION_ACCEL,
                       executing_line_id=time.time_ns() % 100000000)
            timing["ack_s"] = clock() - started
            while True:
                read_started = clock()
                state = robot.motion_snapshot()
                now = clock()
                timing.update(last_snapshot=state, snapshot_elapsed_s=now - started,
                              snapshot_read_s=now - read_started)
                check_envelope(state, plan, leg)
                idle = state["inpos"] and not state.get("queue") and not state.get("active_queue")
                position_error, rotation_error = safety.pose_error(state["actual_position"], leg["pose"])
                joint_error = max(abs(a - b) for a, b in zip(
                    state["joint_actual_position"], leg["joints"]))
                timing.update(idle=bool(idle), max_joint_error_deg=joint_error,
                              position_error_mm=position_error, rotation_error_deg=rotation_error)
                reached = (idle and joint_error <= 0.08
                    and position_error <= 0.75 and rotation_error <= 0.5)
                remaining = deadline - now
                # A read started around the deadline may return just after it.
                # Accept only a fully reached endpoint within a bounded window.
                if reached and now <= deadline + FEEDBACK_GRACE:
                    timing["reached_s"] = now - started
                    timing["feedback_lateness_s"] = max(0.0, -remaining)
                    longest_leg_s = max(longest_leg_s, now - started - timing["command_sent_s"])
                    completed_legs += 1
                    report["completed_legs"] = completed_legs
                    break
                if remaining <= 0:
                    raise TimeoutError(
                        f"第 {index + 1} 段未按时确认到达摆动端点，已停止；"
                        f"耗时 {now - started:.3f}s，空闲={bool(idle)}，"
                        f"关节误差 {joint_error:.4f}°，位置误差 {position_error:.4f}mm，"
                        f"姿态误差 {rotation_error:.4f}°")
                sleep(min(POLL, remaining))
        final = robot.motion_snapshot()
        safety.check_motion_state(final)
        errors = [abs(a - b) for a, b in zip(
            final["joint_actual_position"], plan["origin_joints"])]
        report.update(returned_joints=list(final["joint_actual_position"]),
                      return_joint_errors_deg=errors, returned_at_s=clock() - started)
        if len(errors) != 6 or max(errors) > 0.08:
            raise RuntimeError("结束关节角与启动时 J1–J6 不一致，已停止")
        if report["returned_at_s"] > plan["duration_s"] + TOTAL_GRACE:
            raise TimeoutError("未在预留返回时间内确认 J1–J6 到位")
        # Keep the advertised window even when the controller finishes early.
        while clock() < started + plan["duration_s"]:
            state = robot.motion_snapshot()
            check_envelope(state, plan, plan["legs"][completed_legs - 1])
            safety.check_motion_state(state)
            remaining = started + plan["duration_s"] - clock()
            if remaining > 0:
                sleep(min(POLL, remaining))
        report.update(completed=True, elapsed_s=clock() - started)
    except BaseException as exc:
        report["error"] = str(exc)
        if report["motion_attempted"]:
            report["stop_attempted"] = True
            try:
                safety.stop_and_verify(robot)
            except Exception as stopping:
                report["stop_error"] = str(stopping)
                raise RuntimeError(f"{exc}; STOP UNVERIFIED: {stopping}") from exc
        raise
    finally:
        report["elapsed_s"] = clock() - started
        safety.save_json(REPORT, report)


def main():
    parser = argparse.ArgumentParser(description="TCP 等高左右舞动，J4/J5/J6 大幅摆腕，58 秒返回启动点")
    parser.add_argument("--amplitude-mm", type=float, default=200.0, help="左右单侧幅度，默认 200 mm")
    parser.add_argument("--cycles", type=int,
                        help="手动指定 58 秒内往返次数；默认在最多 9 次中自动选择最快通过校核的节奏")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="仅离线检查参数，不连接机器人")
    mode.add_argument("--plan", action="store_true", help="只读检查当前位置及整段轨迹，不执行动作")
    args = parser.parse_args()
    validate(args.amplitude_mm, args.cycles if args.cycles is not None else DEFAULT_CYCLES)
    if args.check:
        print(json.dumps({"duration_s": DURATION, "amplitude_mm": args.amplitude_mm,
                          "cycles": args.cycles if args.cycles is not None else DEFAULT_CYCLES,
                          "tempo_mode": "auto" if args.cycles is None else "explicit",
                          "motion": "cartesian_level_wrist_sway",
                          "wrist_amplitudes_deg": WRIST_AMPLITUDES,
                          "height": "captured_at_launch", "gripper_commands": False}, indent=2))
        return 0
    with controller_lease(timeout=8):
        safety.ensure_exclusive_client()
        robot = safety.Robot()
        plan = make_plan(robot, args.amplitude_mm, args.cycles)
        safety.save_json(ROOT / "logs" / "horizontal_sway_plan.json", plan)
        if args.plan:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        if plan["rejected_tempos"]:
            print(f"根据当前位置自动调整：目标最多 {plan['requested_cycles']} 次，"
                  f"校核通过 {plan['cycles']} 次；时长和摆幅保持不变。", flush=True)
        print("已记录启动 J1–J6 (°)：" + ", ".join(
            f"J{i + 1}={q:.6f}" for i, q in enumerate(plan["origin_joints"])), flush=True)
        status = read_controller_status(arm.ROBOT_IP, timeout=5)
        safety.check_state(status)
        print(f"当前高度 {plan['height_mm']:.2f} mm，左右各 {args.amplitude_mm:g} mm，"
              f"等高大幅摆腕最多往返 {plan['cycles']} 次，约 {DURATION:g} 秒回到启动点。Ctrl+C 停止。", flush=True)
        print(f"J4/J5/J6 配合幅度 {plan['wrist_amplitudes_deg']}°，"
              f"规划高度误差不超过 {plan['max_planned_height_error_mm']:.3f} mm。", flush=True)
        report = {"completed": False, "plan": plan, "created_unix": time.time()}
        # A failure stops before restoring the user's speed override.
        with full_speed(robot):
            execute(robot, plan, report)
        print(f"摆动结束，已回到启动点，完成 {report['completed_legs'] / 4:g} 次往返，"
              f"用时 {report['elapsed_s']:.2f} 秒。", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("摆动已中断。", flush=True)
        raise SystemExit(130)
    except Exception as exc:
        print(f"摆动未完成：{exc}", flush=True)
        raise SystemExit(2)
