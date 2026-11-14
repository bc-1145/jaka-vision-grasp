"""One-click: raise, observe, recognize, side-pick the shaft, and place at Home."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np

import arm_control
import calibration_motion as safety
from gripper_transport import close_connections
import pick_place_3d as vision
import side_pick_execute as side
from controller_lease import controller_lease

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "pick_place_config.json"
POSES = ROOT / "robot_poses.json"
RESULTS = ROOT / "pick_results"
REPORT = RESULTS / "one_click_run.json"

# Free-space profile is exactly the same as 一键回Home.  Only motion close to
# the loose parts or the release point is reduced.
RAPID_LINEAR_SPEED = 1500.0
RAPID_LINEAR_ACCEL = 8000.0
RAPID_JOINT_SPEED = 180.0
RAPID_JOINT_ACCEL = 720.0
SAFE_RETREAT_JOINT_SPEED = 30.0
SAFE_RETREAT_JOINT_ACCEL = 60.0
SHAFT_PREPICK_SPEED = 60.0
SHAFT_PICK_SPEED = 5.0
SHAFT_RETREAT_SPEED = 10.0
SIDE_RETREAT_SPEED = 60.0
HOME_DESCEND_SPEED = 60.0


class FastRobot(side.PickRobot):
    pass


def save(report):
    safety.save_json(REPORT, report)


def wait_joint(robot, target, speed, stage, report, accel=None):
    start = robot.motion_snapshot(); safety.check_motion_state(start)
    q0 = np.asarray(start["joint_actual_position"], dtype=float)
    q1 = np.asarray(target, dtype=float)
    report.update(stage=stage, target_joints=q1.tolist()); save(report)
    robot.line_id += 1
    try:
        acceleration = (min(240.0, max(20.0, float(speed) * 4))
                        if accel is None else float(accel))
        robot.call("joint_move", relFlag=0, jointPosition=q1.tolist(), speed=float(speed),
                   accel=acceleration, tol=0,
                   executing_line_id=robot.line_id)
        deadline = time.monotonic() + max(15.0, np.max(np.abs(q1 - q0)) / max(speed, 0.1) * 4 + 8)
        stable = None
        while time.monotonic() < deadline:
            state = robot.motion_snapshot(); safety.check_motion_state(state, stationary=False)
            error = float(np.max(np.abs(np.asarray(state["joint_actual_position"]) - q1)))
            if state.get("inpos") and not state.get("queue") and error < 0.1:
                stable = stable or time.monotonic()
                if time.monotonic() - stable > 0.3:
                    print(f"{stage}: OK", flush=True)
                    return
            else:
                stable = None
            time.sleep(0.1)
        raise TimeoutError(f"{stage} 未到位，控制器可能被安全区域阻止")
    except BaseException:
        safety.stop_and_verify(robot)
        raise


def wait_linear(robot, target, speed, stage, report, accel=None):
    state = robot.motion_snapshot(); safety.check_motion_state(state)
    report.update(stage=stage, target_pose=list(target)); save(report)
    robot.line_id += 1
    try:
        acceleration = (min(1000.0, max(100.0, float(speed) * 3))
                        if accel is None else float(accel))
        robot.call("moveL", relFlag=0, cartPosition=list(target), speed=float(speed),
                   accel=acceleration, tol=0, ori_vel=180.0, ori_acc=720.0,
                   executing_line_id=robot.line_id)
        deadline = time.monotonic() + 30
        stable = None
        while time.monotonic() < deadline:
            current = robot.motion_snapshot(); safety.check_motion_state(current, stationary=False)
            position, angle = safety.pose_error(current["actual_position"], target)
            if current.get("inpos") and position < 0.5 and angle < 0.2:
                stable = stable or time.monotonic()
                if time.monotonic() - stable > 0.3:
                    print(f"{stage}: OK", flush=True)
                    return
            else:
                stable = None
            time.sleep(0.1)
        raise TimeoutError(f"{stage} 未到位")
    except BaseException:
        safety.stop_and_verify(robot)
        raise


def load_photo_pose():
    data = json.loads(POSES.read_text(encoding="utf-8-sig"))["photo_observation"]
    approach = data.get("approach_position", data.get("approach_z"))
    if approach is None:
        raise RuntimeError("photo_observation 缺少安全接近点")
    return data["position"], approach, data["approach_joint"]


def go_to_photo(robot, report):
    photo, approach, photo_approach_joint = load_photo_pose()
    state = robot.motion_snapshot(); safety.check_motion_state(state)
    if max(safety.pose_error(state["actual_position"], photo)) < 0.5:
        return
    current_joints = np.asarray(state["joint_actual_position"], dtype=float)
    home_approach_joints = np.asarray(arm_control.load_home_joint(), dtype=float)
    near_home_approach = float(np.max(np.abs(current_joints - home_approach_joints))) < 10.0
    near_saved_approach = False
    if not np.isscalar(approach):
        approach_position_error, approach_angle_error = safety.pose_error(
            state["actual_position"], approach)
        near_saved_approach = approach_position_error < 150.0 and approach_angle_error < 10.0
    if np.isscalar(approach):
        # Backward compatibility for older pose files and tests.
        retreat = list(state["actual_position"])
        retreat[2] = max(float(retreat[2]), float(approach))
        retreat_stage = "先垂直提高到净空高度"
    elif near_saved_approach:
        retreat = list(state["actual_position"])
        retreat_stage = "已在新拍照接近点附近"
    elif not near_home_approach:
        config = vision.load_config(CONFIG)
        base_T_camera = (vision.pose_to_matrix(state["actual_position"]) @
                         np.asarray(config["camera"]["tcp_T_camera"], dtype=float))
        camera_axis = base_T_camera[:3, 2] / np.linalg.norm(base_T_camera[:3, 2])
        retreat = list(state["actual_position"])
        retreat[:3] = (np.asarray(retreat[:3]) - camera_axis * 100.0).tolist()
        retreat_stage = "先沿当前相机视线反向退开 100mm"
    else:
        # Home approach is already the verified free-space staging point.
        # Retiring along the Home camera axis previously triggered collision
        # protection before the robot ever headed toward the photo pose.
        retreat = list(state["actual_position"])
        retreat_stage = "Home 上方已是安全净空位"
    if safety.pose_error(state["actual_position"], retreat)[0] > 0.5:
        wait_linear(robot, retreat, 50.0, retreat_stage, report, 100.0)
    if not np.isscalar(approach) and not near_home_approach:
        wait_linear(robot, approach, 20.0, "低速直线到新拍照接近点", report, 50.0)
    else:
        wait_joint(robot, photo_approach_joint, 10.0, "低速转到新拍照接近点", report, 20.0)
    wait_linear(robot, photo, 50.0, "下降到已标定拍照点", report, 100.0)


def fresh_scan_and_plan(config):
    print("正在拍照、计算深度并识别螺栓…", flush=True)
    mesh, model_cloud, scene, objects, support, clusters, matches, plane = vision.scan_scene(config)
    vision.save_scan(RESULTS, scene, objects, model_cloud, clusters, matches)
    if not matches or matches[0].score < float(config["matching"]["minimum_score"]):
        raise RuntimeError("没有识别到达到阈值的螺栓")
    plan = vision.plan_saved_results(RESULTS, config)
    print(f"识别完成：{len(matches)} 个候选，最佳分数 {matches[0].score:.3f}", flush=True)
    return plan


def execute(robot, config, saved, report, manual_verify=False, before_verification=None):
    plan = side.build_plan(robot, config, saved)
    report["side_plan"] = plan; save(report)
    gripper = side.robot_gripper()
    gripper.connect(); gripper.set_opening_mm(config["gripper"]["open_command_mm"])
    close_connections()
    wait_joint(robot, plan["joints"]["lift"], SAFE_RETREAT_JOINT_SPEED,
               "抓取前沿相机反方向安全退开", report, SAFE_RETREAT_JOINT_ACCEL)
    wait_joint(robot, plan["joints"]["approach"], RAPID_JOINT_SPEED,
               "快速转到抓取接近点", report, RAPID_JOINT_ACCEL)
    wait_joint(robot, plan["joints"]["prepick"], SHAFT_PREPICK_SPEED, "慢速接近杆身 35mm", report)
    wait_joint(robot, plan["joints"]["pick"], SHAFT_PICK_SPEED, "慢速到达杆身抓点", report)
    gripper.set_opening_mm(config["gripper"]["close_command_mm"], settle_time=1.0)
    close_connections(); report.update(stage="已夹杆身", holding=True); save(report)
    wait_joint(robot, plan["joints"]["prepick"], SHAFT_RETREAT_SPEED, "夹紧后退出 35mm", report)
    wait_joint(robot, plan["joints"]["approach"], SIDE_RETREAT_SPEED, "退回抓取接近点", report)
    wait_joint(robot, plan["joints"]["lift"], SAFE_RETREAT_JOINT_SPEED,
               "抓后沿安全路线退出料框", report, SAFE_RETREAT_JOINT_ACCEL)
    if manual_verify:
        from grasp_verification import capture_verification, compare_verification
        if before_verification is None:
            raise RuntimeError("缺少抓取前视觉图，不能执行人工确认")
        report.update(stage="等待人工视觉确认", holding=True); save(report)
        photo_pose = load_photo_pose()[0]
        wait_linear(robot, photo_pose, 100.0, "回到同一拍照位复核抓取", report, 300.0)
        verification = capture_verification(RESULTS / "after_grasp", config["camera"].get("sdk"))
        comparison = compare_verification(
            before_verification, verification, RESULTS / "grasp_before_after.png")
        lift_pose = robot.forward(plan["joints"]["lift"]).tolist()
        wait_linear(robot, lift_pose, 300.0, "复核后重新提高到安全位", report, 900.0)
        report["grasp_verification"] = verification; save(report)
        report["grasp_comparison_path"] = comparison; save(report)
        print(f"抓取前后对比图：{comparison}", flush=True)
        answer = input("看图确认夹具中有工件后输入 YES；其他输入将松开并停止：").strip().upper()
        if answer != "YES":
            gripper.set_opening_mm(config["gripper"]["open_command_mm"], settle_time=1.0)
            close_connections(); report.update(stage="视觉确认失败，已在安全位松开", holding=False); save(report)
            raise RuntimeError("人工视觉确认未通过，未执行 Home 放置")
        report.update(stage="人工视觉确认成功", visual_grasp_confirmed=True); save(report)
    wait_joint(robot, plan["joints"]["home_approach"], RAPID_JOINT_SPEED,
               "快速转到 Home 上方", report, RAPID_JOINT_ACCEL)
    wait_joint(robot, plan["joints"]["home"], HOME_DESCEND_SPEED, "下降到 Home", report)
    gripper.set_opening_mm(config["gripper"]["open_command_mm"], settle_time=1.0)
    close_connections(); report.update(stage="已在 Home 松开", holding=False); save(report)
    wait_joint(robot, plan["joints"]["home_approach"], RAPID_JOINT_SPEED, "放置后提高", report,
               RAPID_JOINT_ACCEL)


def main():
    parser = argparse.ArgumentParser(description="一键视觉识别抓取到 Home")
    parser.add_argument("--check", action="store_true", help="只检查现有扫描和路线，不移动")
    parser.add_argument("--goto-photo-only", action="store_true",
                        help="只移动到保存的新拍照点并验证，不拍照抓取")
    parser.add_argument("--manual-verify", action="store_true",
                        help="抓起后暂停并保存彩色/深度图，确认后才放到 Home")
    args = parser.parse_args()
    config = vision.load_config(CONFIG)
    robot = FastRobot()
    report = {"created_unix": time.time(), "completed": False, "holding": False,
              "mode": "check" if args.check else "one_click_fast_staged"}
    try:
        if args.goto_photo_only:
            with controller_lease(timeout=8.0):
                safety.ensure_exclusive_client()
                go_to_photo(robot, report)
                final_state = robot.motion_snapshot(); safety.check_motion_state(final_state)
                position_error, angle_error = safety.pose_error(
                    final_state["actual_position"], load_photo_pose()[0])
                if position_error > 0.5 or angle_error > 0.2:
                    raise RuntimeError(
                        f"新拍照点验证失败：位置误差 {position_error:.3f} mm，"
                        f"姿态误差 {angle_error:.3f}°")
                report.update(completed=True, stage="已到新拍照点", holding=False)
                save(report)
                print(json.dumps({"arrived": True,
                                  "actual_position": final_state["actual_position"],
                                  "position_error_mm": position_error,
                                  "angle_error_deg": angle_error},
                                 ensure_ascii=False, indent=2), flush=True)
            return 0
        if args.check:
            with controller_lease(timeout=8.0):
                safety.ensure_exclusive_client()
                if (RESULTS / "plan.json").is_file():
                    saved = json.loads((RESULTS / "plan.json").read_text(encoding="utf-8"))
                else:
                    saved = vision.plan_saved_results(RESULTS, config)
                plan = side.build_plan(robot, config, saved)
            print(json.dumps({"ready": True, "photo_pose": load_photo_pose()[0],
                              "side_plan": plan}, ensure_ascii=False, indent=2))
            return 0
        # Hold ownership for the complete sequence.  The gripper GUI releases
        # its socket between operations and cannot poll during robot motion.
        with controller_lease(timeout=8.0):
            safety.ensure_exclusive_client()
            previous_rapid_rate = arm_control.get_rapid_rate()
            if abs(previous_rapid_rate - 1.0) > 1e-6:
                arm_control.set_rapid_rate(1.0)
            try:
                try:
                    go_to_photo(robot, report)
                    saved = fresh_scan_and_plan(config)
                    before_verification = None
                    if args.manual_verify:
                        from grasp_verification import capture_verification
                        before_verification = capture_verification(
                            RESULTS / "before_grasp", config["camera"].get("sdk"))
                        report["before_grasp_verification"] = before_verification; save(report)
                    execute(robot, config, saved, report, manual_verify=args.manual_verify,
                            before_verification=before_verification)
                except BaseException:
                    try:
                        safety.stop_and_verify(robot)
                    except Exception as stopping:
                        report["stop_error"] = str(stopping); save(report)
                    raise
            finally:
                if abs(previous_rapid_rate - 1.0) > 1e-6:
                    try:
                        arm_control.set_rapid_rate(previous_rapid_rate)
                    except Exception as restoring:
                        report["rapid_rate_restore_error"] = str(restoring); save(report)
                        print(f"警告：未能恢复原速度倍率：{restoring}", flush=True)
        report.update(completed=True, stage="完成"); save(report)
        print("一键抓取完成。", flush=True)
        return 0
    except BaseException as exc:
        report["error"] = str(exc); save(report)
        print(f"一键抓取未完成：{exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
