"""Execute the reviewed side approach for the current calibrated scan."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time
import argparse

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import arm_control
import calibration_motion as motion
from gripper_backend import GripperClient, GripperProfile
from gripper_transport import close_connections
import pick_place_3d as vision

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "pick_results"
CONFIG = ROOT / "pick_place_config.json"
REPORT = RESULTS / "side_pick_run.json"
GRASP_MODEL_Z = 20.26


class PickRobot(motion.Robot):
    def __init__(self):
        self.line_id = int(time.time() * 1000) % 100000000

    def move(self, joints):
        self.line_id += 1
        return self.call("joint_move", relFlag=0, jointPosition=list(joints),
                         speed=2.0, accel=4.0, tol=0,
                         executing_line_id=self.line_id)


def robot_gripper():
    data = json.loads((ROOT / "gripper_devices.json").read_text(encoding="utf-8-sig"))
    item = data["devices"][0]
    return GripperClient(host=data["host"], profile=GripperProfile(**item))


def move_joint_path(robot, target_joints, report, stage, max_step=2.0):
    state = robot.motion_snapshot()
    motion.check_motion_state(state)
    start = np.asarray(state["joint_actual_position"], dtype=float)
    target = np.asarray(target_joints, dtype=float)
    count = max(1, int(math.ceil(np.max(np.abs(target - start)) / max_step)))
    print(f"{stage}: 0/{count}", flush=True)
    cursor = {"joints": start.tolist(), "pose": state["actual_position"]}
    for index in range(1, count + 1):
        q = start + (target - start) * index / count
        pose = robot.forward(q).tolist()
        next_point = {"joints": q.tolist(), "pose": pose}
        motion.move_step(robot, cursor, next_point)
        cursor = next_point
        report.update(stage=stage, stage_step=index, stage_steps=count)
        motion.save_json(REPORT, report)
        if index == count or index % 10 == 0:
            print(f"{stage}: {index}/{count}", flush=True)
    return cursor


def select_highest_score_match(matches, *, highest_graspable=False):
    """Lock the highest score before safety filtering; never substitute a part."""
    if not matches:
        raise RuntimeError("本次扫描没有可供抓取的零件")
    if any(not np.isfinite(item.score) for item in matches):
        raise RuntimeError("匹配分数无效，需要重新扫描")
    if highest_graspable:
        matches = [item for item in matches if item.score >= 0.78
                   and np.isfinite(item.clearance_mm) and item.clearance_mm >= 10.0]
        if not matches:
            raise RuntimeError("没有同时满足抓取匹配分和下爪净空的候选零件")
    selected = max(matches, key=lambda item: item.score)
    if (selected.score < 0.78 or not np.isfinite(selected.clearance_mm)
            or selected.clearance_mm < 10.0):
        raise RuntimeError(
            f"最高分零件 {selected.cluster_index} 未通过抓取门槛："
            f"匹配分 {selected.score:.3f}，外部净空 {selected.clearance_mm:.1f} mm；未改选其他零件")
    return selected


def shaft_middle_model_z(config):
    lower, upper = np.asarray(config["model"]["shaft_z_range_mm"], dtype=float)
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError("管身轴向范围无效")
    middle = float((lower + upper) / 2.0)
    if middle <= float(config["model"]["head_z_range_mm"][1]):
        raise ValueError("管身中点不能位于大头区域")
    return middle


def grasp_rotation(tool_axis, jaw_axis, jaw_axis_tcp):
    """Map the configured physical closing axis onto the desired world axis."""
    tool = np.asarray(tool_axis, float)
    jaw = np.asarray(jaw_axis, float)
    local = np.asarray(jaw_axis_tcp, float)
    if any(a.shape != (3,) or not np.isfinite(a).all() for a in (tool, jaw, local)):
        raise ValueError("Invalid tool or jaw axis")
    if min(np.linalg.norm(a) for a in (tool, jaw, local)) < 1e-8:
        raise ValueError("Tool and jaw axes must be nonzero")
    tool = tool / np.linalg.norm(tool)
    jaw = jaw / np.linalg.norm(jaw)
    local = local / np.linalg.norm(local)
    if abs(jaw @ tool) > 1e-6 or abs(local[2]) > 1e-6:
        raise ValueError("Jaw closing axis must be perpendicular to tool Z")
    world_basis = np.column_stack((jaw, np.cross(tool, jaw), tool))
    local_basis = np.column_stack((local, np.cross([0, 0, 1], local), [0, 0, 1]))
    return world_basis @ local_basis.T


def flange_position(grasp_center, rotation, tcp_to_grasp):
    return np.asarray(grasp_center, float) - np.asarray(rotation, float) @ np.asarray(tcp_to_grasp, float)


def build_plan(robot, config, saved, *, highest_graspable=False):
    matches = [vision.MatchResult(**item) for item in json.loads(
        (RESULTS / "matches.json").read_text(encoding="utf-8"))]
    preferred = select_highest_score_match(matches, highest_graspable=highest_graspable)
    eligible = (sorted([item for item in matches if item.score >= 0.78
                        and np.isfinite(item.clearance_mm) and item.clearance_mm >= 10.0],
                       key=lambda item: -item.score)
                if highest_graspable else [preferred])
    if any(item.observation != saved["observation"] for item in eligible):
        raise RuntimeError("识别结果与抓取方案不是同一次扫描")
    base_T_camera = vision.camera_transform(config, eligible[0].observation)
    tcp_to_grasp = np.asarray(config["gripper"]["tcp_to_grasp_center_mm"], dtype=float)
    tool_offset_z = float(tcp_to_grasp[2])
    tool_length = abs(tool_offset_z)
    if (tcp_to_grasp.shape != (3,) or not np.isfinite(tcp_to_grasp).all()
            or not 50 <= tool_length <= 400 or np.linalg.norm(tcp_to_grasp[:2]) > 100):
        raise RuntimeError("夹持中心偏移超出已支持的末端尺寸范围")
    tool_z_sign = 1.0 if tool_offset_z > 0 else -1.0
    state = robot.motion_snapshot()
    current_joints = np.asarray(state["joint_actual_position"], dtype=float)
    # The bin lies along the calibrated camera +Z viewing direction.  Retreat
    # opposite that ray before changing wrist orientation. Use the normalized
    # optical frame, whose direction was cross-checked against live depth.
    camera_axis = base_T_camera[:3, 2] / np.linalg.norm(base_T_camera[:3, 2])
    lift_pose = list(state["actual_position"])
    retreat_mm = float(config["camera"].get("retreat_mm", 100.0))
    if not 30.0 <= retreat_mm <= 150.0:
        raise ValueError("camera.retreat_mm must be between 30 and 150 mm")
    lift_pose[:3] = (np.asarray(lift_pose[:3]) - camera_axis * retreat_mm).tolist()
    joints = {"lift": robot.inverse(lift_pose, current_joints)}
    home_approach = np.asarray(arm_control.load_home_joint(), dtype=float)
    home_pose = arm_control.load_pose("home")
    home_joints = robot.inverse(home_pose, home_approach)

    cloud = o3d.io.read_point_cloud(str(RESULTS / "scene.ply"))
    scene_camera = np.asarray(cloud.points)
    scene_base = vision.transform_points(scene_camera, base_T_camera)
    o3d.utility.random.seed(7)
    mesh = vision.load_fcstd_mesh(ROOT / config["model"]["path"])
    model_cloud = mesh.sample_points_uniformly(int(config["matching"]["model_sample_points"]))
    model_points = np.asarray(model_cloud.points)
    candidates = []
    # Search approach orientations only for the locked highest-scoring part.
    for selected in eligible:
        camera_T_model = vision.matrix(selected.camera_T_model, "camera_T_model")
        base_T_model = base_T_camera @ camera_T_model
        shaft = base_T_model[:3, 2] / np.linalg.norm(base_T_model[:3, 2])
        aligned_model_camera = vision.transform_points(model_points, camera_T_model)
        distance_to_target, _ = cKDTree(aligned_model_camera).query(scene_camera)
        target_exclusion_mm = max(
            6.0, float(config["matching"]["ransac_distance_mm"]),
            2.0 * float(config["matching"]["icp_distance_mm"]))
        foreign_points = scene_base[distance_to_target > target_exclusion_mm]
        if len(foreign_points) < 100:
            continue
        foreign_tree = cKDTree(foreign_points)
        # The operator requests the middle of the thin tube. Keep that point
        # fixed while searching approach orientations; never drift to an end
        # merely because a different axial point has more clearance.
        grasp_z_values = np.asarray([shaft_middle_model_z(config)])
        grasp_options = vision.transform_points(
            np.column_stack((np.zeros(len(grasp_z_values)),
                             np.zeros(len(grasp_z_values)), grasp_z_values)),
            base_T_model)
        grasp_clearances, _ = foreign_tree.query(grasp_options)
        best_grasp_index = int(np.argmax(grasp_clearances))
        grasp = grasp_options[best_grasp_index]
        grasp_z = float(grasp_z_values[best_grasp_index])
        grasp_clearance = float(grasp_clearances[best_grasp_index])
        if grasp_clearance < 10.0:
            continue
        camera_ray = grasp - base_T_camera[:3, 3]
        camera_ray /= np.linalg.norm(camera_ray)
        basis_u = np.cross(camera_ray, [0.0, 0.0, 1.0])
        if np.linalg.norm(basis_u) < 0.1:
            basis_u = np.cross(camera_ray, [0.0, 1.0, 0.0])
        basis_u /= np.linalg.norm(basis_u)
        basis_v = np.cross(camera_ray, basis_u)
        for cone_deg in (0, 10, 20, 30, 40, 50, 60, 70):
            azimuths = ((0.0,) if cone_deg == 0 else
                        np.linspace(0, 2 * np.pi, 12, endpoint=False))
            cone = math.radians(cone_deg)
            for azimuth in azimuths:
                approach_axis = (camera_ray * math.cos(cone) +
                                 (basis_u * math.cos(azimuth) +
                                  basis_v * math.sin(azimuth)) * math.sin(cone))
                tool_axis = approach_axis * tool_z_sign
                jaw = np.cross(shaft, approach_axis)
                if np.linalg.norm(jaw) < 0.1:
                    continue
                jaw /= np.linalg.norm(jaw)
                for sign in (1.0, -1.0):
                    jaw_axis = jaw * sign
                    rotation = grasp_rotation(tool_axis, jaw_axis, config["gripper"]["jaw_axis_tcp"])
                    orientation = Rotation.from_matrix(rotation).as_euler("xyz", degrees=True)
                    poses, candidate_joints = {}, {}
                    previous = joints["lift"]
                    valid = True
                    for name, extra in (("approach", 70.0), ("prepick", 35.0), ("pick", 0.0)):
                        center = grasp - approach_axis * extra
                        pose = [*flange_position(center, rotation, tcp_to_grasp), *orientation]
                        try:
                            q = robot.inverse(pose, previous)
                        except Exception:
                            valid = False
                            break
                        poses[name] = pose; candidate_joints[name] = q; previous = q
                    if not valid:
                        continue
                    route_joints = np.vstack(tuple(candidate_joints.values()))
                    if (np.max(np.abs(route_joints)) > 179.0 or
                            np.max(np.abs(route_joints[:, 4])) > 119.9):
                        continue
                    jump = float(np.max(np.abs(
                        candidate_joints["approach"] - joints["lift"])))
                    joint_margin = 180.0 - float(np.max(np.abs(route_joints)))
                    margin_penalty = max(0.0, 15.0 - joint_margin) * 5.0
                    rank = (jump + 0.25 * cone_deg + margin_penalty -
                            0.5 * min(grasp_clearance, 30.0) - 5.0 * selected.score)
                    candidates.append({
                        "rank": rank, "selected": selected, "grasp": grasp,
                        "grasp_z": grasp_z, "grasp_clearance": grasp_clearance,
                        "cone_deg": cone_deg, "approach_axis": approach_axis,
                        "orientation": orientation, "poses": poses,
                        "candidate_joints": candidate_joints, "tree": foreign_tree,
                    })
    if not candidates:
        raise RuntimeError("最高分零件没有通过关节范围检查的抓取姿态；未改选其他零件")
    candidates.sort(key=lambda value: (-value["selected"].score, value["rank"]))
    selected_route = None
    best_rejected = None
    examined = {}
    for candidate in candidates:
        index = candidate["selected"].cluster_index
        examined[index] = examined.get(index, 0) + 1
        if examined[index] > 60:
            continue
        minimum_flange = float("inf"); minimum_axis = float("inf")
        approach_joints = candidate["candidate_joints"]["approach"]
        tree = candidate["tree"]
        route = [(joints["lift"], approach_joints, 31),
                 (approach_joints, candidate["candidate_joints"]["prepick"], 17),
                 (candidate["candidate_joints"]["prepick"], candidate["candidate_joints"]["pick"], 17)]
        for q0, q1, samples in route:
            for fraction in np.linspace(0, 1, samples):
                q = q0 * (1 - fraction) + q1 * fraction
                tcp = vision.pose_to_matrix(robot.forward(q))
                flange = tcp[:3, 3]
                center = vision.transform_points([tcp_to_grasp], tcp)[0]
                axis_samples = flange + np.linspace(0, 1, 91)[:, None] * (center - flange)
                minimum_flange = min(minimum_flange, float(tree.query(flange)[0]))
                minimum_axis = min(minimum_axis, float(np.min(tree.query(axis_samples)[0])))
                if minimum_flange < 50 or minimum_axis < 10:
                    break
            if minimum_flange < 50 or minimum_axis < 10:
                break
        if (best_rejected is None or minimum_axis > best_rejected[1]):
            best_rejected = (minimum_flange, minimum_axis)
        if minimum_flange >= 50 and minimum_axis >= 10:
            selected_route = (candidate, minimum_flange, minimum_axis)
            break
    if selected_route is None:
        detail = ("" if best_rejected is None else
                  f"（最佳法兰/工具轴净空 {best_rejected[0]:.1f}/{best_rejected[1]:.1f} mm）")
        raise RuntimeError(f"所有控制器可达抓取路线的点云净空都不足{detail}")
    candidate, minimum_flange, minimum_axis = selected_route
    selected = candidate["selected"]
    grasp = candidate["grasp"]
    grasp_z = candidate["grasp_z"]
    grasp_clearance = candidate["grasp_clearance"]
    cone_deg = candidate["cone_deg"]
    approach_axis = candidate["approach_axis"]
    orientation = candidate["orientation"]
    poses = candidate["poses"]
    candidate_joints = candidate["candidate_joints"]
    joints.update(candidate_joints, home_approach=home_approach, home=home_joints)
    return {"poses": poses, "joints": {key: value.tolist() for key, value in joints.items()},
            "selected_cluster_index": selected.cluster_index,
            "selection": "highest_score_with_required_clearance" if highest_graspable else "highest_score",
            "grasp_base": grasp.tolist(), "approach_axis": approach_axis.tolist(),
            "tool_axis": (approach_axis * tool_z_sign).tolist(),
            "strategy": "stage1_top_or_oblique_minimum_wrist_change",
            "grasp_model_z_mm": grasp_z,
            "shaft_axis_base": (base_T_camera[:3, :3] @
                                np.asarray(selected.camera_T_model)[:3, :3] @
                                np.asarray(config["model"]["axis_model"])).tolist(),
            "grasp_rpy_deg": orientation.tolist(), "camera_ray_cone_deg": cone_deg,
            "approach_distance_mm": 70.0, "prepick_distance_mm": 35.0,
            "match_score": selected.score, "foreign_clearance_mm": grasp_clearance,
            "observed_flange_clearance_mm": minimum_flange,
            "observed_tool_axis_clearance_mm": minimum_axis}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = vision.load_config(CONFIG)
    saved = json.loads((RESULTS / "plan.json").read_text(encoding="utf-8"))
    if not args.resume and time.time() - float(saved["created_unix"]) > 1200:
        raise RuntimeError("识别结果已过期，请重新扫描")
    if config["camera"].get("calibration_tool_id") != 0:
        raise RuntimeError("手眼标定未启用")
    motion.ensure_exclusive_client()
    robot = PickRobot()
    if args.resume:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        if report.get("completed") or report.get("holding") or report.get("stage") != "swing_to_approach":
            raise RuntimeError("当前执行记录不能从换姿态阶段续跑")
        plan = report["plan"]
        current = np.asarray(robot.motion_snapshot()["joint_actual_position"], dtype=float)
        start = np.asarray(plan["joints"]["lift"], dtype=float)
        end = np.asarray(plan["joints"]["approach"], dtype=float)
        fraction = float(np.dot(current - start, end - start) / np.dot(end - start, end - start))
        nearest = start + np.clip(fraction, 0, 1) * (end - start)
        if np.max(np.abs(current - nearest)) > 0.2:
            raise RuntimeError("机械臂不在已验证的侧向换姿态路径上")
        report.pop("error", None)
        report["resumed_unix"] = time.time()
    else:
        vision.verify_observation_unchanged(saved["observation"], vision.read_observation_pose())
        plan = build_plan(robot, config, saved)
        report = {"created_unix": time.time(), "completed": False, "stage": "preflight",
                  "plan": plan, "tool_center_is_operator_estimate": True,
                  "gripper_open_mm": config["gripper"]["open_command_mm"],
                  "gripper_close_mm": config["gripper"]["close_command_mm"]}
    motion.save_json(REPORT, report)
    gripper = robot_gripper()
    holding = False
    try:
        connection_error = None
        for _ in range(5):
            try:
                gripper.connect()
                break
            except (OSError, TimeoutError) as exc:
                connection_error = exc
                time.sleep(0.2)
        else:
            raise RuntimeError(f"夹爪映射状态读取失败：{connection_error}")
        gripper.set_opening_mm(config["gripper"]["open_command_mm"])
        close_connections()
        report["stage"] = "open"; motion.save_json(REPORT, report)
        if not args.resume:
            move_joint_path(robot, plan["joints"]["lift"], report, "lift_from_observation")
        move_joint_path(robot, plan["joints"]["approach"], report, "swing_to_approach")
        move_joint_path(robot, plan["joints"]["prepick"], report, "slow_approach_25mm", 1.0)
        move_joint_path(robot, plan["joints"]["pick"], report, "slow_approach_final", 0.5)
        gripper.set_opening_mm(config["gripper"]["close_command_mm"], settle_time=1.0)
        close_connections()
        holding = True; report["stage"] = "closed_on_shaft"; motion.save_json(REPORT, report)
        move_joint_path(robot, plan["joints"]["prepick"], report, "retreat_25mm", 0.5)
        move_joint_path(robot, plan["joints"]["approach"], report, "retreat_50mm", 1.0)
        move_joint_path(robot, plan["joints"]["lift"], report, "return_from_side")
        move_joint_path(robot, plan["joints"]["home_approach"], report, "transfer_above_home")
        move_joint_path(robot, plan["joints"]["home"], report, "descend_home", 1.0)
        gripper.set_opening_mm(config["gripper"]["open_command_mm"], settle_time=1.0)
        close_connections()
        holding = False; report["stage"] = "released_at_home"; motion.save_json(REPORT, report)
        move_joint_path(robot, plan["joints"]["home_approach"], report, "retreat_home", 1.0)
        report.update(completed=True, stage="complete", holding=False)
        motion.save_json(REPORT, report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    except BaseException as exc:
        report.update(error=str(exc), holding=holding)
        motion.save_json(REPORT, report)
        try:
            motion.stop_and_verify(robot)
        except Exception as stopping:
            report["stop_error"] = str(stopping); motion.save_json(REPORT, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
