"""Move to saved joints after the configured lift, without operating the gripper."""
from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import contextmanager

import arm_control as arm
import calibration_motion as safety
from controller_lease import controller_lease

REPORT = arm.POSES_FILE.parent / "logs" / "one_click_loading.json"
RAPID_RATE = 1.0
JOINT_TOLERANCE_DEG = 0.5
POSITION_TOLERANCE_MM = 3.0
ROTATION_TOLERANCE_DEG = 1.0
SETTLE_SECONDS = 0.1
POLL_SECONDS = 0.05


def at_target(state, target, target_pose):
    # Literal joint angles preserve the taught turn; TCP checks bound the
    # effect of accepting the controller's small residual joint error.
    if max(abs(a - b) for a, b in zip(state["joint_actual_position"], target)) > JOINT_TOLERANCE_DEG:
        return False
    position, rotation = arm._pose_error(state["actual_position"], target_pose)
    return position <= POSITION_TOLERANCE_MM and rotation <= ROTATION_TOLERANCE_DEG


@contextmanager
def full_speed(robot):
    previous = float(robot.call("get_rapid_rate")["value"])
    if not math.isfinite(previous) or not 0 <= previous <= 1:
        raise RuntimeError("控制器速度倍率无效")
    changed = abs(previous - RAPID_RATE) > 1e-6
    try:
        if changed:
            robot.call("rapid_rate", rate_value=RAPID_RATE)
        yield
    finally:
        if changed:
            robot.call("rapid_rate", rate_value=previous)


def settings(pose_name="loading"):
    data = json.loads(arm.POSES_FILE.read_text(encoding="utf-8-sig"))
    loading = data[pose_name]
    target = [float(value) for value in loading["joint_position"]]
    if len(target) != 6 or not all(math.isfinite(value) for value in target):
        raise ValueError("目标点必须配置六个有限关节角（度）")
    speed = float(loading.get("joint_speed", 30.0))
    accel = float(loading.get("joint_accel", 60.0))
    if not (math.isfinite(speed) and 0 < speed <= 180 and
            math.isfinite(accel) and 0 < accel <= 720):
        raise ValueError("目标点运动速度或加速度无效")
    return target, speed, accel, arm.load_joint3_lift_config()


def move_joint(robot, target, speed, accel, stage, report, *, min_height_mm=None):
    before = robot.motion_snapshot()
    safety.check_motion_state(before)
    if min_height_mm is not None and before["actual_position"][2] < min_height_mm - 2.0:
        raise RuntimeError("高处对位起点低于配置高度，已停止")
    target_pose = robot.forward(target)
    print(f"{stage}：开始，目标关节 {list(target)}", flush=True)
    if at_target(before, target, target_pose):
        print(f"{stage}：已到位，跳过运动", flush=True)
        return
    report.update(stage=stage, target_joints=list(target), motion_attempted=True)
    safety.save_json(REPORT, report)
    # Record intent before sending: a lost acknowledgement must never cause
    # an automatic second movement command.
    robot.call("joint_move", relFlag=0, jointPosition=list(target), speed=speed,
               accel=accel, tol=0, executing_line_id=time.time_ns() % 100000000)
    deadline = time.monotonic() + 120.0
    settled = idle_since = None
    while time.monotonic() < deadline:
        state = robot.motion_snapshot()
        safety.check_motion_state(state, stationary=False)
        if min_height_mm is not None and state["actual_position"][2] < min_height_mm - 2.0:
            raise RuntimeError("高处对位时高度下降超过范围，已停止")
        error = max(abs(a - b) for a, b in zip(state["joint_actual_position"], target))
        position_error, rotation_error = arm._pose_error(state["actual_position"], target_pose)
        report.update(last_actual_joints=list(state["joint_actual_position"]),
                      last_joint_error_deg=error, last_position_error_mm=position_error,
                      last_rotation_error_deg=rotation_error)
        # Compare absolute angles, never modulo 360: the exact turn matters.
        idle = state.get("inpos") and not state.get("queue") and not state.get("active_queue")
        if idle and at_target(state, target, target_pose):
            settled = time.monotonic() if settled is None else settled
            if time.monotonic() - settled >= SETTLE_SECONDS:
                print(f"{stage}: OK，关节误差 {error:.3f}°，位置误差 {position_error:.3f} mm", flush=True)
                return
        else:
            settled = None
        if idle and not at_target(state, target, target_pose):
            idle_since = time.monotonic() if idle_since is None else idle_since
            if time.monotonic() - idle_since >= 3.0:
                raise RuntimeError(
                    f"{stage}：控制器已停止但未达到目标；关节误差 {error:.3f}°，"
                    f"位置误差 {position_error:.3f} mm，姿态误差 {rotation_error:.3f}°")
        else:
            idle_since = None
        time.sleep(POLL_SECONDS)
    raise TimeoutError(f"{stage}未到位；最后关节误差 {error:.3f}°")


def sequence_settings(pose_name):
    data = json.loads(arm.POSES_FILE.read_text(encoding="utf-8-sig"))
    keys = data[pose_name].get("steps", [pose_name])
    if not isinstance(keys, list) or not keys or not all(isinstance(key, str) for key in keys):
        raise ValueError("目标点的 steps 必须为非空点名列表")
    return [(key, str(data[key].get("name", key)), *settings(key)) for key in keys]


def execute_sequence(robot, steps, report, *, lift_first=False, descend_last=False):
    # Resolve every target before starting the first movement.
    for _key, _label, target, _speed, _accel, _lift in steps:
        robot.forward(target)
    for index, (key, label, target, speed, accel, lift) in enumerate(steps):
        report.update(step_index=index + 1, step_count=len(steps), pose=key)
        stage = f"第 {index + 1}/{len(steps)} 步：{label}"
        print(stage, flush=True)
        if index == 0:
            execute(robot, target, speed, accel, lift, report, label=stage,
                    lift_first=lift_first, descend_last=descend_last)
        else:
            move_joint(robot, target, speed, accel, stage, report)
        report.setdefault("completed_steps", []).append(
            {"workflow": report.get("workflow"), "pose": key, "step_index": index + 1})
        safety.save_json(REPORT, report)


def execute_workflows(robot, workflows, report, *, lift_first=False, descend_last=False):
    # Check the complete combined route before allowing its first movement.
    for _key, steps in workflows:
        for _pose, _name, target, _speed, _accel, _lift in steps:
            robot.forward(target)
    for index, (key, steps) in enumerate(workflows):
        report.update(workflow=key, workflow_index=index + 1,
                      workflow_count=len(workflows))
        execute_sequence(robot, steps, report, lift_first=lift_first and index == 0,
                         descend_last=descend_last and index == 0)


def plan_descend_last(robot, current, target, lift_config):
    lift_z, step, maximum_delta = lift_config
    current_pose = safety.vector(robot.forward(current))
    target_pose = safety.vector(robot.forward(target))
    clearance = max(lift_z, float(current_pose[2]), float(target_pose[2]))
    lift, _ = arm.plan_joint3_lift(current, clearance, step, maximum_delta)
    approach, _ = arm.plan_joint3_lift(target, clearance, step, maximum_delta)
    lift, approach = safety.vector(lift).tolist(), safety.vector(approach).tolist()
    for raised, base in ((lift, current), (approach, target)):
        if any(abs(raised[i] - base[i]) > 1e-7 for i in (0, 1, 3, 4, 5)):
            raise RuntimeError("抬升或最终下降规划改变了 J3 以外的关节")
    # The alignment still uses MoveJ. Check its actual joint interpolation,
    # rather than assuming two high endpoints imply a high intervening path.
    samples = max(1, math.ceil(max(abs(a - b) for a, b in zip(lift, approach)) / 1.0))
    minimum = math.inf
    for index in range(samples + 1):
        q = [a + (b - a) * index / samples for a, b in zip(lift, approach)]
        height = float(safety.vector(robot.forward(q))[2])
        minimum = min(minimum, height)
        if height < clearance - 0.1:
            raise RuntimeError("高处对位的关节路径会提前下降，已取消全部动作")
    return {"lift_joints": lift, "approach_joints": approach,
            "target_joints": list(target), "clearance_mm": clearance,
            "minimum_alignment_height_mm": minimum, "alignment_samples": samples + 1}


def execute(robot, target, speed, accel, lift_config, report, label="上料处", *,
            lift_first=False, descend_last=False):
    state = robot.motion_snapshot()
    safety.check_motion_state(state)
    current = state["joint_actual_position"]
    if descend_last:
        plan = plan_descend_last(robot, current, target, lift_config)
        fresh = robot.motion_snapshot()
        safety.check_motion_state(fresh)
        if max(abs(a - b) for a, b in zip(fresh["joint_actual_position"], current)) > 0.2:
            raise RuntimeError("规划期间起点发生变化，请重新启动回位")
        report["descent_last_plan"] = plan
        move_joint(robot, plan["lift_joints"], speed, accel, "先仅转动 J3 提高", report)
        raised = robot.motion_snapshot()
        safety.check_motion_state(raised)
        if raised["actual_position"][2] < plan["clearance_mm"] - 2.0:
            raise RuntimeError("提高后未达到配置高度，取消对位和下降")
        move_joint(robot, plan["approach_joints"], speed, accel, f"在高处对准{label}", report,
                   min_height_mm=plan["clearance_mm"])
        move_joint(robot, target, speed, accel, f"最后仅转动 J3 下降到{label}", report)
        return
    if not lift_first and max(abs(a - b) for a, b in zip(current, target)) <= 0.2:
        print(f"已在{label}。", flush=True)
        return
    # Resolve both FK and the complete lift target before issuing any motion.
    robot.forward(target)
    lift_z, step, maximum_delta = lift_config
    if state["actual_position"][2] < lift_z:
        lift, _pose = arm.plan_joint3_lift(current, lift_z, step, maximum_delta)
        # FK planning can take time; reject a start changed by another client.
        fresh = robot.motion_snapshot()
        safety.check_motion_state(fresh)
        if max(abs(a - b) for a, b in zip(fresh["joint_actual_position"], current)) > 0.2:
            raise RuntimeError("规划期间起点发生变化，请重新启动回位")
        move_joint(robot, lift, speed, accel, "先仅转动 J3 提高", report)
        raised = robot.motion_snapshot()
        safety.check_motion_state(raised)
        if raised["actual_position"][2] < lift_z - 2.0:
            raise RuntimeError(f"提高后未达到配置高度，停止转向{label}")
    move_joint(robot, target, speed, accel, f"转到{label}", report)


def main():
    global REPORT
    parser = argparse.ArgumentParser(description="回到保存的关节姿态，保持夹爪原状态")
    parser.add_argument("--pose", default="loading", help="robot_poses.json 中的目标点名称")
    parser.add_argument("--then", dest="following", action="append", default=[],
                        help="完成后继续执行的点或流程；可重复指定")
    parser.add_argument("--check", action="store_true", help="仅离线检查配置，不连接机器人")
    parser.add_argument("--lift-first", action="store_true",
                        help="先检查并按需仅动 J3 抬升，即使已在第一个目标点也不跳过此检查")
    parser.add_argument("--descend-last", action="store_true",
                        help="首个目标先仅动 J3 抬升，再高处对位，最后仅动 J3 回落到保存点")
    args = parser.parse_args()
    workflow_keys = [args.pose, *args.following]
    workflows = [(key, sequence_settings(key)) for key in workflow_keys]
    steps = [step for _key, route in workflows for step in route]
    _key, _step_label, target, speed, accel, _last_lift = steps[-1]
    lift_config = steps[0][-1]
    data = json.loads(arm.POSES_FILE.read_text(encoding="utf-8-sig"))
    label = " → ".join(str(data[key].get("name", key)) for key in workflow_keys)
    if args.pose != "loading" or args.following:
        # Point keys are JSON data, not paths supplied to the filesystem.
        safe_key = "".join(char if char.isascii() and (char.isalnum() or char in "_-") else "_"
                           for char in "_then_".join(workflow_keys))
        REPORT = arm.POSES_FILE.parent / "logs" / f"return_pose_{safe_key}.json"
    if args.check:
        print(json.dumps({"pose": args.pose, "name": label,
                          "workflows": workflow_keys,
                          "target_joints_deg": target, "speed_deg_s": speed,
                          "accel_deg_s2": accel, "rapid_rate": RAPID_RATE,
                          "first_lift_z_mm": lift_config[0],
                          "lift_before_target_check": args.lift_first or args.descend_last,
                          "descend_after_alignment": args.descend_last,
                          "steps": [{"pose": key, "name": name, "joint_position": q,
                                     "joint_speed": v, "joint_accel": acc}
                                    for key, name, q, v, acc, _lift in steps]},
                         ensure_ascii=False, indent=2))
        return 0
    report = {"created_unix": time.time(), "completed": False, "motion_attempted": False}
    with controller_lease(8.0):
        safety.ensure_exclusive_client()
        robot = safety.Robot()
        try:
            enabled = robot.call("get_robot_state")
            if not enabled.get("enable") or enabled.get("power") != 1:
                raise RuntimeError("机械臂未上电或未使能")
            if str(enabled.get("errcode", "0")).lower() not in ("0", "0x0"):
                raise RuntimeError(f"控制器故障：{enabled}")
            with full_speed(robot):
                try:
                    execute_workflows(robot, workflows, report, lift_first=args.lift_first,
                                      descend_last=args.descend_last)
                    final = robot.motion_snapshot()
                    safety.check_motion_state(final)
                    if not at_target(final, target, robot.forward(target)):
                        raise RuntimeError("最终关节角与保存目标不符，拒绝报告到位")
                except BaseException:
                    # Stop before restoring the speed override on a failed move.
                    if report["motion_attempted"]:
                        report["stop_attempted"] = True
                        try:
                            safety.stop_and_verify(robot)
                        except Exception as stopping:
                            report["stop_error"] = str(stopping)
                    raise
            report.update(completed=True, stage=f"已到{label}", actual=final)
            safety.save_json(REPORT, report)
            print(f"已到{label}：", final["joint_actual_position"], flush=True)
            return 0
        except BaseException as exc:
            report["error"] = str(exc)
            if report["motion_attempted"] and not report.get("stop_attempted"):
                try:
                    safety.stop_and_verify(robot)
                except Exception as stopping:
                    report["stop_error"] = str(stopping)
            safety.save_json(REPORT, report)
            raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"回位未完成：{exc}", flush=True)
        raise SystemExit(2)
