"""Home -> loading/bin -> vision -> open -> shaft grasp -> return Home holding."""
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
import one_click_loading as loading
from controller_lease import controller_lease

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "pick_place_config.json"
POSES = ROOT / "robot_poses.json"
RESULTS = ROOT / "pick_results"
REPORT = RESULTS / "one_click_run.json"

# Free-space profile is exactly the same as 一键回Home. Motion close to the
# loose parts is reduced.
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
    print("正在拍照、计算深度并识别零件…", flush=True)
    mesh, model_cloud, scene, objects, support, clusters, matches, plane = vision.scan_scene(config)
    vision.save_scan(RESULTS, scene, objects, model_cloud, clusters, matches)
    if not matches or matches[0].score < float(config["matching"]["minimum_score"]):
        raise RuntimeError("没有识别到达到阈值的零件")
    plan = vision.plan_saved_results(RESULTS, config)
    print(f"识别完成：{len(matches)} 个候选，最佳分数 {matches[0].score:.3f}", flush=True)
    return plan


def go_to_loading(robot, report):
    state = robot.motion_snapshot(); safety.check_motion_state(state)
    position_error, angle_error = safety.pose_error(
        state["actual_position"], arm_control.load_pose("home"))
    if position_error > 0.5 or angle_error > 0.5:
        raise RuntimeError("视觉抓取流程必须从 Home 出发，请先运行一键回Home")
    target, speed, accel, lift_config = loading.settings()
    loading.execute(robot, target, speed, accel, lift_config, report)
    final = robot.motion_snapshot(); safety.check_motion_state(final)
    if np.max(np.abs(np.asarray(final["joint_actual_position"]) - target)) > 0.2:
        raise RuntimeError("尚未到达上料处，不能开始视觉选件")
    report.update(stage="已到上料处，准备视觉选件", observation_state=final)
    save(report)


def verify_pick_alignment(robot, config, plan, report):
    """Check the computed contact center and jaw direction before closing."""
    state = robot.motion_snapshot(); safety.check_motion_state(state)
    tcp = vision.pose_to_matrix(state["actual_position"])
    center = vision.transform_points([config["gripper"]["tcp_to_grasp_center_mm"]], tcp)[0]
    error = float(np.linalg.norm(center - np.asarray(plan["grasp_base"])))
    orientation_error = safety.pose_error(state["actual_position"], plan["poses"]["pick"])[1]
    shaft = np.asarray(plan["shaft_axis_base"], dtype=float)
    shaft /= np.linalg.norm(shaft)
    jaw = tcp[:3, :3] @ np.asarray(config["gripper"]["jaw_axis_tcp"], dtype=float)
    jaw /= np.linalg.norm(jaw)
    shaft_z = float(plan["grasp_model_z_mm"])
    lower, upper = config["model"]["shaft_z_range_mm"]
    if (not np.isfinite([error, orientation_error, shaft_z, *shaft, *jaw]).all()
            or not lower < shaft_z < upper
            or shaft_z <= config["model"]["head_z_range_mm"][1]
            or error > 1.0 or orientation_error > 0.5 or abs(float(jaw @ shaft)) > 0.05):
        raise RuntimeError("细管抓点的位置、高度或夹爪方向未对准，未发送闭合指令")
    report.update(stage="细管抓点已对准，准备闭合", actual_grasp_center_base_mm=center.tolist(),
                  grasp_center_error_mm=error, pick_orientation_error_deg=orientation_error)
    save(report)


def return_home_holding(robot, report):
    close_connections()
    speed, accel, joint_speed, joint_accel, _rate = arm_control.load_home_motion()
    report.update(stage="保持夹持，执行一键回Home路线"); save(report)
    # Same staged implementation used by 一键回Home.bat; it does not open jaws.
    arm_control.move_home_staged(speed, accel, joint_speed, joint_accel)
    final = robot.motion_snapshot(); safety.check_motion_state(final)
    position_error, angle_error = safety.pose_error(
        final["actual_position"], arm_control.load_pose("home"))
    if position_error > 0.5 or angle_error > 0.5:
        raise RuntimeError("回 Home 未到位，夹爪保持当前状态")
    report.update(stage="已回到Home，夹爪保持闭合", actual=final,
                  home_position_error_mm=position_error, home_rotation_error_deg=angle_error)
    save(report)


def execute(robot, config, saved, report, manual_verify=False, before_verification=None):
    if manual_verify and before_verification is None:
        raise RuntimeError("缺少抓取前视觉图，不能执行人工确认")
    vision.verify_observation_unchanged(saved["observation"], vision.read_observation_pose())
    plan = side.build_plan(robot, config, saved)
    report["side_plan"] = plan; save(report)
    # Controller IK/FK planning can take time. Recheck after it completes and
    # before opening jaws or sending the first movement.
    vision.verify_observation_unchanged(saved["observation"], vision.read_observation_pose())
    gripper = side.robot_gripper()
    gripper.connect(); gripper.set_opening_mm(config["gripper"]["open_command_mm"])
    close_connections()
    wait_joint(robot, plan["joints"]["lift"], SAFE_RETREAT_JOINT_SPEED,
               "抓取前沿相机反方向安全退开", report, SAFE_RETREAT_JOINT_ACCEL)
    wait_joint(robot, plan["joints"]["approach"], RAPID_JOINT_SPEED,
               "快速转到抓取接近点", report, RAPID_JOINT_ACCEL)
    wait_joint(robot, plan["joints"]["prepick"], SHAFT_PREPICK_SPEED, "慢速接近杆身 35mm", report)
    wait_joint(robot, plan["joints"]["pick"], SHAFT_PICK_SPEED, "慢速到达杆身抓点", report)
    verify_pick_alignment(robot, config, plan, report)
    gripper.set_opening_mm(config["gripper"]["close_command_mm"], settle_time=1.0)
    close_connections()
    report.update(stage="夹爪已收至细管开度，准备提起", holding=True,
                  gripper_closed=True, close_command_mm=config["gripper"]["close_command_mm"],
                  visual_grasp_confirmed=False)
    save(report)
    wait_joint(robot, plan["joints"]["prepick"], SHAFT_RETREAT_SPEED, "夹紧后退出 35mm", report)
    wait_joint(robot, plan["joints"]["approach"], SIDE_RETREAT_SPEED, "退回抓取接近点", report)
    wait_joint(robot, plan["joints"]["lift"], SAFE_RETREAT_JOINT_SPEED,
               "抓后沿安全路线退出料框", report, SAFE_RETREAT_JOINT_ACCEL)
    from grasp_verification import capture_verification
    report["grasp_verification"] = capture_verification(
        RESULTS / "after_grasp", config["camera"].get("sdk"))
    save(report)
    if manual_verify:
        print(f"提起后的复核图：{report['grasp_verification']['view_path']}", flush=True)
        answer = input("看图确认只有一个工件被夹住后输入 YES；其他输入将保持夹爪并停止：").strip().upper()
        if answer != "YES":
            report.update(stage="视觉确认未通过，保持夹爪等待处理"); save(report)
            raise RuntimeError("人工视觉确认未通过，未执行回 Home")
        report.update(stage="人工视觉确认成功", visual_grasp_confirmed=True); save(report)
    return_home_holding(robot, report)


def run_cycle(robot, config, report, manual_verify=False):
    go_to_loading(robot, report)
    from grasp_verification import capture_verification
    before = capture_verification(RESULTS / "before_grasp", config["camera"].get("sdk"))
    report["before_grasp_verification"] = before; save(report)
    saved = fresh_scan_and_plan(config)
    execute(robot, config, saved, report, manual_verify=manual_verify, before_verification=before)


def main():
    parser = argparse.ArgumentParser(description="一键视觉识别抓取到 Home")
    parser.add_argument("--check", action="store_true", help="只检查现有扫描和路线，不移动")
    parser.add_argument("--goto-photo-only", action="store_true",
                        help="只移动到保存的新拍照点并验证，不拍照抓取")
    parser.add_argument("--manual-verify", action="store_true",
                        help="提起后保存彩色/深度图并暂停，确认后保持夹持回 Home")
    args = parser.parse_args()
    config = vision.load_config(CONFIG)
    robot = FastRobot()
    report = {"created_unix": time.time(), "completed": False, "holding": False,
              "mode": "check" if args.check else "home_loading_vision_shaft_home"}
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
            print(json.dumps({"ready": True, "home_pose": arm_control.load_pose("home"),
                              "loading_joints": loading.settings()[0],
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
                    run_cycle(robot, config, report, manual_verify=args.manual_verify)
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
        report.update(completed=True, sequence_completed=True, stage="流程完成，已回Home并保持夹持"); save(report)
        print("流程已回到 Home，夹爪保持闭合。提起后的复核图已保存。", flush=True)
        return 0
    except BaseException as exc:
        report["error"] = str(exc); save(report)
        print(f"一键抓取未完成：{exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
