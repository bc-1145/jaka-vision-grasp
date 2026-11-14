"""Fresh eye-in-hand target coordinates and optional IK plan. No motion writes."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import calibration_motion as safety
from controller_lease import controller_lease
from gripper_transport import close_connections
import pick_place_3d as vision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "pick_results" / "selected_target")
    parser.add_argument("--plan", action="store_true", help="Read-only controller IK/FK route checks")
    parser.add_argument("--highest-graspable", action="store_true",
                        help="Choose the highest score among candidates meeting grasp clearance")
    args = parser.parse_args()
    config = vision.load_config(ROOT / "pick_place_config.json")
    report = {"created_unix":time.time(),"motion_sent":False,"selection":"highest_score",
              "return_batch":"一键回Home.bat", "create_home":False}
    args.output.mkdir(parents=True,exist_ok=True)
    try:
        with controller_lease(8):
            safety.ensure_exclusive_client()
            robot = safety.Robot()
            state = robot.motion_snapshot()
            safety.check_motion_state(state)
            report["start_state"] = state
            report["tool_offset"] = robot.call("get_curr_tool_offset")["tool_offset"]
            if np.max(np.abs(report["tool_offset"])) > 1e-6:
                raise RuntimeError("Active tool offset does not match calibrated flange frame")
            print("Capturing fresh native depth and synchronized robot observation...",flush=True)
            mesh,model,scene,objects,support,clusters,matches,plane = vision.scan_scene(config)
            vision.save_scan(args.output,scene,objects,model,clusters,matches)
            observation = config["_observation"]
            vision.verify_observation_unchanged(observation,vision.read_observation_pose())
            report.update(observation=observation,candidate_count=len(matches),
                          diagnostics=config.get("_matching_diagnostics",[]),
                          statistics=config.get("_matching_statistics",{}))
            if not matches:
                raise RuntimeError("No accepted target in fresh scan")
            raw_best = max(matches,key=lambda item:item.score)
            if args.highest_graspable:
                import side_pick_execute as side
                best = side.select_highest_score_match(matches, highest_graspable=True)
                report.update(selection="highest_score_with_required_clearance", raw_highest_score_target=asdict(raw_best))
            else:
                best = raw_best
            base_T_camera = vision.camera_transform(config,observation)
            base_T_model = base_T_camera @ np.asarray(best.camera_T_model)
            grasp = vision.transform_points([config["model"]["grasp_point_mm"]],base_T_model)[0]
            report.update(best=asdict(best),base_T_camera=base_T_camera.tolist(),
                          base_T_model=base_T_model.tolist(),model_origin_base_mm=base_T_model[:3,3].tolist(),
                          shaft_grasp_center_base_mm=grasp.tolist(),
                          shaft_axis_base=(base_T_model[:3,:3] @ np.asarray(config["model"]["axis_model"])).tolist(),
                          configured_grasp_offset_mm=config["gripper"]["tcp_to_grasp_center_mm"],
                          tool_geometry_independently_measured=False)
            print(json.dumps({k:report[k] for k in ("candidate_count","best","model_origin_base_mm",
                             "shaft_grasp_center_base_mm","shaft_axis_base")},ensure_ascii=False,indent=2),flush=True)
            safety.save_json(args.output / "world_target.json", report)
            if args.plan:
                import side_pick_execute as side
                side.RESULTS = args.output
                print("Checking highest-score target route with read-only controller IK/FK...", flush=True)
                report["side_plan"] = side.build_plan(robot, config, {"observation":observation},
                                                     highest_graspable=args.highest_graspable)
                selected_index = report["side_plan"]["selected_cluster_index"]
                if selected_index != best.cluster_index:
                    best = next(item for item in matches if item.cluster_index == selected_index)
                    base_T_model = base_T_camera @ np.asarray(best.camera_T_model)
                    report.update(best=asdict(best), base_T_model=base_T_model.tolist(),
                                  model_origin_base_mm=base_T_model[:3,3].tolist(),
                                  shaft_grasp_center_base_mm=vision.transform_points(
                                      [[0,0,side.shaft_middle_model_z(config)]],base_T_model)[0].tolist(),
                                  shaft_axis_base=base_T_model[:3,2].tolist())
                vision.verify_observation_unchanged(observation, vision.read_observation_pose())
                print(json.dumps({"motion_sent":False,"side_plan":report["side_plan"]},
                                 ensure_ascii=False,indent=2),flush=True)
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        close_connections()
        safety.save_json(args.output / "world_target.json",report)


if __name__ == "__main__":
    main()
