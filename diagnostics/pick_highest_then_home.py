"""Guarded stages: highest-score pick, existing batch Home route, then release.

Each invocation executes only its named stage. Never retries actuator writes.
The Home point/configuration is read only. Camera/tool dimensions remain estimates.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

import arm_control
import calibration_motion as safety
from controller_lease import controller_lease
from grasp_verification import capture_verification
from gripper_transport import close_connections
import one_click_pick as pick
import pick_place_3d as vision
from pose_overlay import PoseOverlay
import side_pick_execute as side

OUTPUT = ROOT / "pick_results" / "selected_target"
REPORT = OUTPUT / "execution.json"


def save(report):
    safety.save_json(REPORT, report)


def assert_joint_start(robot, expected):
    state = robot.motion_snapshot()
    safety.check_motion_state(state)
    if np.max(np.abs(np.asarray(state["joint_actual_position"]) - expected)) > 0.2:
        raise RuntimeError("Robot joints changed since the preceding stage; no movement sent")


def snapshot_target(robot, config, target, folder):
    observation = vision.read_observation_pose()
    observation["camera_frame"] = target["observation"]["camera_frame"]
    result = capture_verification(OUTPUT / folder, config["camera"].get("sdk"))
    vision.verify_observation_unchanged(observation, vision.read_observation_pose())
    base_T_camera = vision.camera_transform(config, observation)
    match = dict(target["best"])
    match["camera_T_model"] = (np.linalg.inv(base_T_camera) @
                               np.asarray(target["base_T_model"])).tolist()
    overlay = PoseOverlay(OUTPUT, ROOT / "pick_place_config.json")
    overlay._load_assets()
    overlay.matches = [match]
    image = cv2.imread(result["color_path"])
    overlay._axes(image)
    destination = OUTPUT / folder / "target_overlay.png"
    cv2.imwrite(str(destination), image)
    result.update(overlay_path=str(destination), observation=observation)
    return result


def audit_pick_route(robot, config, target):
    """Sample every outbound joint segment; does not model the full robot links."""
    plan = target["side_plan"]
    scene = np.asarray(o3d.io.read_point_cloud(str(OUTPUT / "scene.ply")).points)
    mesh = vision.load_fcstd_mesh(ROOT / config["model"]["path"])
    o3d.utility.random.seed(7)
    model = np.asarray(mesh.sample_points_uniformly(
        int(config["matching"]["model_sample_points"])).points)
    if not len(model) or not len(scene):
        raise RuntimeError("Missing scan/model points")
    aligned = vision.transform_points(model, np.asarray(target["best"]["camera_T_model"]))
    distance, _ = cKDTree(aligned).query(scene)
    exclusion = max(6.0, float(config["matching"]["ransac_distance_mm"]),
                    2.0 * float(config["matching"]["icp_distance_mm"]))
    foreign = vision.transform_points(scene[distance > exclusion],
                                       np.asarray(target["base_T_camera"]))
    if len(foreign) < 100:
        raise RuntimeError("Insufficient surrounding depth for route audit")
    tree = cKDTree(foreign)
    state = robot.motion_snapshot()
    safety.check_motion_state(state)
    q0 = np.asarray(state["joint_actual_position"], dtype=float)
    contact = np.asarray(config["gripper"]["tcp_to_grasp_center_mm"])
    checks = []
    for name in ("lift", "approach", "prepick", "pick"):
        q1 = np.asarray(plan["joints"][name], dtype=float)
        change = float(np.max(np.abs(q1 - q0)))
        if not np.isfinite(q1).all() or change > float(config["robot"]["max_direct_rotation_deg"]):
            raise RuntimeError(f"Unsafe joint branch jump before {name}: {change:.1f} degrees")
        flange_min, axis_min = float("inf"), float("inf")
        samples = max(16, int(np.ceil(change)) + 1)
        for fraction in np.linspace(0, 1, samples):
            pose = vision.pose_to_matrix(robot.forward(q0 * (1 - fraction) + q1 * fraction))
            flange = pose[:3, 3]
            center = vision.transform_points([contact], pose)[0]
            axis = flange + np.linspace(0, 1, 91)[:, None] * (center - flange)
            flange_min = min(flange_min, float(tree.query(flange)[0]))
            axis_min = min(axis_min, float(np.min(tree.query(axis)[0])))
        checks.append(dict(stage=name, flange_clearance_mm=flange_min,
                           tool_axis_clearance_mm=axis_min, max_joint_change_deg=change))
        print(f"Read-only route audit {name}: flange {flange_min:.1f}, axis {axis_min:.1f} mm", flush=True)
        if flange_min < 50 or axis_min < 10:
            raise RuntimeError(f"Insufficient point-cloud clearance before {name}")
        q0 = q1
    return checks


def verify_locked_target_depth(config, target, folder="staging"):
    """Require new measured support at the same world pose before closure."""
    from clutter_matching import DepthVisibility, validate_shaft_curvature

    scene = vision.crop_cloud(vision.capture_point_cloud(config), config["camera"]["roi"])
    observation = config["_observation"]
    current_T_model = (np.linalg.inv(vision.camera_transform(config, observation)) @
                       np.asarray(target["base_T_model"]))
    mesh = vision.load_fcstd_mesh(ROOT / config["model"]["path"])
    o3d.utility.random.seed(7)
    model = mesh.sample_points_uniformly(int(config["matching"]["model_sample_points"]))
    aligned = vision.transform_points(np.asarray(model.points), current_T_model)
    points = np.asarray(scene.points)
    tolerance = float(config["matching"]["icp_distance_mm"])
    distance, _ = cKDTree(aligned).query(points)
    supported = points[distance <= tolerance]
    validation = {"created_unix":time.time(), "supported_points":len(supported),
                  "observation":observation, "world_pose_changed":False}
    instance = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(supported))
    validation.update(vision.validate_model_support(
        model, instance, current_T_model, config, visibility=DepthVisibility(scene, config)))
    if validation["accepted"]:
        validation.update(validate_shaft_curvature(instance, current_T_model, config))
        validation["accepted"] = bool(validation["curvature_accepted"])
    if len(supported) < int(config["matching"]["min_cluster_points"]):
        validation.update(accepted=False, reason="Too few current target points")
    (OUTPUT / folder).mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(OUTPUT / folder / "current_scene.ply"), scene)
    safety.save_json(OUTPUT / folder / "depth_validation.json", validation)
    print(json.dumps({"current_target_validation":validation}, ensure_ascii=False, indent=2), flush=True)
    if not validation["accepted"]:
        raise RuntimeError("Fresh depth does not verify the locked world target; no closure or movement sent")
    vision.verify_observation_unchanged(observation, vision.read_observation_pose())
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("approach", "pick", "return-release"))
    parser.add_argument("--direct-middle", action="store_true",
                        help="Operator-requested middle grasp using the fresh scan and checked route")
    args = parser.parse_args()
    config = vision.load_config(ROOT / "pick_place_config.json")
    report = {"created_unix":time.time(), "motion_sent":False, "completed":False,
              "physical_grasp_confirmed":False, "return_batch":"一键回Home.bat",
              "create_home":False, "release_at_existing_home":True}
    if args.stage != "approach":
        report = json.loads(REPORT.read_text(encoding="utf-8"))
    target = json.loads((OUTPUT / "world_target.json").read_text(encoding="utf-8"))
    robot = pick.FastRobot()
    pick.RESULTS = side.RESULTS = OUTPUT
    pick.REPORT = side.REPORT = REPORT
    motion_attempted = False
    try:
        with controller_lease(8):
            safety.ensure_exclusive_client()
            state = robot.motion_snapshot()
            safety.check_motion_state(state)
            status = robot.snapshot()
            safety.check_state(status)
            if max(abs(float(v)) for v in robot.call("get_curr_tool_offset")["tool_offset"]) > 1e-6:
                raise RuntimeError("Active TCP offset changed")
            if robot.call("get_clsn_sensitivity")["sensitivityLevel"] not in (1, 2, 3, 4, 5):
                raise RuntimeError("Collision detection is not enabled")
            plan = target["side_plan"]
            if args.stage == "approach":
                if time.time() - target["observation"]["created_unix"] > 1200:
                    raise RuntimeError("Scan is over twenty minutes old; acquire a new scan before movement")
                if target.get("error"):
                    raise RuntimeError(target["error"])
                vision.verify_observation_unchanged(target["observation"], vision.read_observation_pose())
                candidates = [vision.MatchResult(**item) for item in json.loads(
                    (OUTPUT / "matches.json").read_text(encoding="utf-8"))]
                if target.get("selection") == "highest_score_with_required_clearance":
                    candidates = [item for item in candidates
                                  if item.cluster_index == plan["selected_cluster_index"]]
                selected = side.select_highest_score_match(candidates)
                if (selected.cluster_index != plan["selected_cluster_index"] or
                        selected.camera_T_model != target["best"]["camera_T_model"]):
                    raise RuntimeError("Plan is not for the highest-score target in this scan")
                report.update(side_plan=plan, target_observation=target["observation"],
                              initial_state=state, home_pose=arm_control.load_pose("home"),
                              tool_geometry_independently_measured=False)
                report["route_audit"] = audit_pick_route(robot, config, target)
                report["before"] = snapshot_target(robot, config, target, "before_motion")
                vision.verify_observation_unchanged(target["observation"], vision.read_observation_pose())
                assert_joint_start(robot, np.asarray(state["joint_actual_position"]))
                save(report)
                gripper = side.robot_gripper()
                gripper.connect()
                gripper.set_opening_mm(config["gripper"]["open_command_mm"])
                close_connections()
                report.update(gripper_open_command_mm=config["gripper"]["open_command_mm"])
                motion_attempted = True
                report["motion_sent"] = True
                pick.wait_joint(robot, plan["joints"]["lift"], 5, "Slow retreat from scan pose", report, 10)
                pick.wait_joint(robot, plan["joints"]["approach"], 5, "Slow swing to target approach", report, 10)
                pick.wait_joint(robot, plan["joints"]["prepick"], 3, "Stop 35 mm before shaft", report, 6)
                report["staging_view"] = snapshot_target(robot, config, target, "staging")
                report.update(stage="at_prepick", actual=robot.motion_snapshot())
            elif args.stage == "pick":
                if report.get("stage") != "at_prepick":
                    raise RuntimeError("Prepick stage has not completed")
                if report["side_plan"] != plan:
                    raise RuntimeError("Scan or plan replaced after approach")
                assert_joint_start(robot, np.asarray(plan["joints"]["prepick"]))
                if args.direct_middle:
                    if abs(float(plan["grasp_model_z_mm"]) - side.shaft_middle_model_z(config)) > 0.01:
                        raise RuntimeError("Direct grasp must use the middle of the thin tube")
                    if time.time() - target["observation"]["created_unix"] > 1200:
                        raise RuntimeError("Direct grasp requires a scan less than twenty minutes old")
                    report.update(grasp_mode="operator_requested_direct_middle",
                                  close_range_depth_verified=False)
                else:
                    report["current_target_validation"] = verify_locked_target_depth(config, target)
                assert_joint_start(robot, np.asarray(plan["joints"]["prepick"]))
                save(report)
                motion_attempted = True
                pick.wait_joint(robot, plan["joints"]["pick"], 2, "Slow final approach to shaft", report, 4)
                pick.verify_pick_alignment(robot, config, plan, report)
                gripper = side.robot_gripper()
                gripper.connect()
                gripper.set_opening_mm(config["gripper"]["close_command_mm"], settle_time=1.0)
                close_connections()
                report.update(stage="closure_command_sent", gripper_closed=True,
                              close_command_mm=config["gripper"]["close_command_mm"])
                save(report)
                pick.wait_joint(robot, plan["joints"]["prepick"], 2, "Withdraw 35 mm", report, 4)
                pick.wait_joint(robot, plan["joints"]["approach"], 3, "Withdraw to approach", report, 6)
                pick.wait_joint(robot, plan["joints"]["lift"], 5, "Withdraw from bin", report, 10)
                report["after_pick"] = capture_verification(OUTPUT / "after_pick", config["camera"].get("sdk"))
                report.update(stage="withdrawn_awaiting_visual_check", actual=robot.motion_snapshot())
            else:
                if report.get("stage") != "withdrawn_awaiting_visual_check":
                    raise RuntimeError("Pickup/withdrawal stage has not completed")
                if not report.get("physical_grasp_confirmed"):
                    raise RuntimeError("Held part has not been visually confirmed")
                if report["home_pose"] != arm_control.load_pose("home"):
                    raise RuntimeError("Existing Home configuration changed during this pickup")
                assert_joint_start(robot, np.asarray(report["actual"]["joint_actual_position"]))
                close_connections()
                report.update(stage="returning_via_existing_batch_implementation")
                save(report)
                motion_attempted = True
                # Exact staged implementation called by 一键回Home.bat.
                # Conservative runtime speeds; robot_poses.json remains untouched.
                arm_control.move_home_staged(50, 100, 10, 20)
                final = robot.motion_snapshot()
                safety.check_motion_state(final)
                position_error, angle_error = safety.pose_error(final["actual_position"], report["home_pose"])
                if position_error > 0.5 or angle_error > 0.5:
                    raise RuntimeError("Existing Home target not reached; jaws remain closed")
                report.update(stage="at_existing_home", actual=final,
                              home_position_error_mm=position_error, home_angle_error_deg=angle_error)
                save(report)
                gripper = side.robot_gripper()
                gripper.connect()
                gripper.set_opening_mm(config["gripper"]["open_command_mm"], settle_time=1.0)
                close_connections()
                report.update(stage="released_at_existing_home", gripper_closed=False,
                              completed=True)
                report["after_release"] = capture_verification(OUTPUT / "after_release", config["camera"].get("sdk"))
            save(report)
            print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    except BaseException as exc:
        report.update(error=str(exc), failed_stage=args.stage)
        if motion_attempted:
            try:
                safety.stop_and_verify(robot)
            except Exception as stop_error:
                report["stop_error"] = str(stop_error)
        save(report)
        raise
    finally:
        close_connections()


if __name__ == "__main__":
    raise SystemExit(main())
