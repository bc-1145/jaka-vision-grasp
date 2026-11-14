"""Stand the currently observed table part on its large end; no Home edits."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation, Slerp
import calibration_motion as safety
from controller_lease import controller_lease
from grasp_verification import capture_verification
from gripper_transport import close_connections
import one_click_pick as motion
import pick_place_3d as vision
import side_pick_execute as side

OUT = ROOT / "pick_results" / "table_upright"
PLAN = OUT / "stand_plan.json"
REPORT = OUT / "stand_run.json"


def normalized_plane(plane):
    """Return a metric base-frame plane with an upward unit normal."""
    value=np.asarray(plane["plane_base"],dtype=float)
    if value.shape!=(4,) or not np.isfinite(value).all():
        raise ValueError("Table plane must contain four finite coefficients")
    length=float(np.linalg.norm(value[:3]))
    if length<1e-8:
        raise ValueError("Table plane has no normal")
    value=value/length
    if value[2]<0:
        value=-value
    if value[2]<.5:
        raise ValueError("Table normal is not sufficiently vertical")
    return value


def table_height_at(plane,xy):
    value=normalized_plane(plane)
    xy=np.asarray(xy,dtype=float)
    if xy.shape!=(2,) or not np.isfinite(xy).all():
        raise ValueError("Table lookup requires two finite XY coordinates")
    return float(-(value[:2]@xy+value[3])/value[2])


def _unit(value,name):
    value=np.asarray(value,dtype=float)
    if value.shape!=(3,) or not np.isfinite(value).all() or np.linalg.norm(value)<1e-8:
        raise ValueError("Degenerate "+name)
    return value/np.linalg.norm(value)


def planning_binding(config):
    """Bind the geometry used by this route, without changing Home."""
    geometry={"camera":{key:config["camera"].get(key) for key in
                        ("mount","tcp_T_camera","calibration_tool_id","point_cloud_frame","color_projection")},
              "gripper":{key:config["gripper"].get(key) for key in
                         ("tcp_to_grasp_center_mm","jaw_axis_tcp")},
              "model":config["model"],
              "table_observation_pose":config["robot"]["table_observation_pose"],
              "table_placement_xy_mm":config["robot"].get("table_placement_xy_mm")}
    return hashlib.sha256(json.dumps(geometry,sort_keys=True,allow_nan=False).encode()).hexdigest()


def verify_plan_binding(saved,config):
    if saved.get("planning_binding")!=planning_binding(config):
        raise RuntimeError("Camera, tool, model, placement or overview geometry changed; regenerate the table route")


def make_poses(target, plane, config, tilt, yaw, sign):
    model = vision.matrix(target["base_T_model"],"target.base_T_model").copy()
    if not np.isfinite([tilt,yaw,sign]).all() or not 0<=tilt<80 or sign not in (-1,1):
        raise ValueError("Invalid approach tilt, yaw or jaw sign")
    shaft = _unit(model[:3,2],"shaft axis")
    normal = normalized_plane(plane)[:3]
    middle = side.shaft_middle_model_z(config)
    grasp = vision.transform_points([[0,0,middle]], model)[0]
    horizontal = _unit(shaft-normal*(shaft@normal),"horizontal shaft; part may already be upright")
    radial = _unit(grasp-normal*(grasp@normal),"radial approach")
    toward = _unit(radial-.5*horizontal,"approach direction")
    approach = toward * np.sin(np.deg2rad(tilt)) - normal * np.cos(np.deg2rad(tilt))
    jaw = _unit(np.cross(shaft,approach),"jaw direction")*sign
    pick_rotation = side.grasp_rotation(approach, jaw, config["gripper"]["jaw_axis_tcp"])
    offset = np.asarray(config["gripper"]["tcp_to_grasp_center_mm"],dtype=float)
    if offset.shape!=(3,) or not np.isfinite(offset).all():
        raise ValueError("Invalid TCP-to-grasp offset")
    placement=config["robot"].get("table_placement_xy_mm")
    placement_xy=grasp[:2] if placement is None else np.asarray(placement,dtype=float)
    if placement_xy.shape!=(2,) or not np.isfinite(placement_xy).all():
        raise ValueError("Table placement requires two finite base XY coordinates")
    place_origin = np.r_[placement_xy,table_height_at(plane,placement_xy)]+normal*3
    place_center = place_origin + normal * middle
    turn = Rotation.from_rotvec(normal*np.deg2rad(yaw)).as_matrix() @ vision._rotation_between(shaft,normal)
    final_rotation = turn @ pick_rotation

    def pose(center, rotation):
        transform = np.eye(4); transform[:3,:3] = rotation
        transform[:3,3] = center - rotation @ offset
        return vision.matrix_to_pose(transform)

    poses = dict(approach=pose(grasp-approach*40,pick_rotation),
                 prepick=pose(grasp-approach*20,pick_rotation),
                 pick=pose(grasp,pick_rotation), lift=pose(grasp+normal*90,pick_rotation))
    turn_center=grasp+normal*90
    if placement is not None:
        # Move the held part into the taught table working area before
        # rotation. This is an explicit Cartesian transfer, not a Home edit.
        turn_center=np.r_[placement_xy,grasp[2]+90]
        poses["transfer"]=pose(turn_center,pick_rotation)
    rotation_path=Slerp([0,1],Rotation.from_matrix(np.stack((pick_rotation,final_rotation))))
    for index,fraction in enumerate(np.linspace(0,1,7)[1:],1):
        name="turned" if index==6 else f"turn_{index}"
        poses[name]=pose(turn_center,rotation_path(fraction).as_matrix())
    poses.update(above_place=pose(place_center+normal*40,final_rotation),
                 place=pose(place_center,final_rotation),
                 retreat=pose(place_center-final_rotation[:,2]*70,final_rotation))
    held = np.linalg.inv(vision.pose_to_matrix(poses["pick"])) @ model
    placed = vision.pose_to_matrix(poses["place"]) @ held
    if np.dot(placed[:3,2],normal) < .9999:
        raise RuntimeError("Computed final shaft is not upright")
    return poses, held, placed


def validate_route(robot,candidate,state,plane,config,vertices):
    """Sample the actual MoveJ/MoveL modes; this is not full collision checking."""
    coefficients=normalized_plane(plane)
    normal,d=coefficients[:3],coefficients[3]
    offset=np.asarray(config["gripper"]["tcp_to_grasp_center_mm"],dtype=float)
    q0=np.asarray(state["joint_actual_position"],dtype=float)
    p0=np.asarray(state["actual_position"],dtype=float)
    checks={"full_collision_checked":False,"continuous_sweep_proved":False,
            "method":"sampled controller FK/IK and CAD/table clearances",
            "segments":[],"return_to_overview_checked":False}
    steps=[(name,candidate["poses"][name],q) for name,q in candidate["joints"].items()]
    steps.append(("return_overview",state["actual_position"],state["joint_actual_position"]))
    linear_names={"prepick","pick","lift","transfer","above_place","place","retreat"}
    for name,endpoint,joints in steps:
        q1=np.asarray(joints,dtype=float)
        p1=np.asarray(endpoint,dtype=float)
        if q1.shape!=(6,) or not np.isfinite(q1).all() or np.max(np.abs(q1-q0))>100:
            raise RuntimeError("Invalid or discontinuous joints at "+name)
        linear=name in linear_names
        if linear:
            rotation_path=Slerp([0,1],Rotation.from_euler("xyz",np.stack((p0[3:],p1[3:])),degrees=True))
            angle=float(np.degrees((Rotation.from_euler("xyz",p0[3:],degrees=True).inv()*
                                   Rotation.from_euler("xyz",p1[3:],degrees=True)).magnitude()))
            count=max(12,int(np.linalg.norm(p1[:3]-p0[:3])/2)+1,int(angle/2)+1)
        else:
            count=max(12,int(np.max(np.abs(q1-q0))/2)+1)
        segment={"name":name,"mode":"MoveL" if linear else "MoveJ","samples":count,
                 "minimum_flange_table_mm":float("inf"),"minimum_contact_table_mm":float("inf"),
                 "minimum_held_cad_table_mm":None}
        previous=q0
        for fraction in np.linspace(0,1,count):
            if linear:
                pose=np.r_[p0[:3]*(1-fraction)+p1[:3]*fraction,
                           rotation_path(fraction).as_euler("xyz",degrees=True)]
                q=np.asarray(robot.inverse(pose,previous),dtype=float)
                if q.shape!=(6,) or not np.isfinite(q).all() or np.max(np.abs(q-previous))>10:
                    raise RuntimeError("Cartesian IK branch discontinuity at "+name)
                tcp_pose=np.asarray(robot.forward(q),dtype=float)
                position_error,angle_error=safety.pose_error(tcp_pose,pose)
                if position_error>.25 or angle_error>.2:
                    raise RuntimeError("Cartesian IK/FK disagreement at "+name)
            else:
                q=q0*(1-fraction)+q1*fraction
                tcp_pose=np.asarray(robot.forward(q),dtype=float)
            if abs(q[4])>119.9 or abs(q[2])>144:
                raise RuntimeError("Sampled joint range exceeded at "+name)
            tcp=vision.pose_to_matrix(tcp_pose)
            center=vision.transform_points([offset],tcp)[0]
            flange_clearance=float(tcp[:3,3]@normal+d)
            contact_clearance=float(center@normal+d)
            segment["minimum_flange_table_mm"]=min(segment["minimum_flange_table_mm"],flange_clearance)
            segment["minimum_contact_table_mm"]=min(segment["minimum_contact_table_mm"],contact_clearance)
            if flange_clearance<50 or contact_clearance<(50 if name=="return_overview" else 10):
                raise RuntimeError(f"Table clearance at {name}: flange {flange_clearance:.2f}, contact {contact_clearance:.2f}")
            if name in ("lift","transfer","turned","above_place","place") or name.startswith("turn_"):
                points=vision.transform_points(vertices,tcp@np.asarray(candidate["held_model"]))
                clearance=float(np.min(points@normal+d))
                old=segment["minimum_held_cad_table_mm"]
                segment["minimum_held_cad_table_mm"]=clearance if old is None else min(old,clearance)
                if clearance<0:
                    raise RuntimeError(f"Held CAD below table at {name}: {clearance:.2f}")
            previous=q
        if linear and np.max(np.abs(previous-q1))>.25:
            raise RuntimeError("Cartesian route reaches a different endpoint IK branch at "+name)
        checks["segments"].append(segment)
        q0,p0=q1,p1
    checks["return_to_overview_checked"]=True
    return checks


def plan(robot, target, plane, config):
    vision.verify_observation_unchanged(config["robot"]["table_observation_pose"],vision.read_observation_pose())
    vision.verify_observation_unchanged(target["observation"],vision.read_observation_pose())
    state=robot.motion_snapshot(); safety.check_motion_state(state)
    coefficients=normalized_plane(plane)
    normal,d=coefficients[:3],coefficients[3]
    tilt=float(np.degrees(np.arccos(np.clip(normal[2],-1,1))))
    if not np.isfinite(plane["support_points"]) or plane["support_points"]<500 or tilt>5:
        raise RuntimeError("Table plane is not established")
    plane=copy.deepcopy(plane)
    plane.update(plane_base=coefficients.tolist(),normal_tilt_deg=tilt)
    model=vision.matrix(target["base_T_model"],"target.base_T_model")
    grasp=vision.transform_points([[0,0,side.shaft_middle_model_z(config)]],model)[0]
    plane["height_at_grasp_xy_mm"]=table_height_at(plane,grasp[:2])
    placement=config["robot"].get("table_placement_xy_mm")
    placement_xy=grasp[:2] if placement is None else np.asarray(placement,dtype=float)
    plane["height_at_placement_xy_mm"]=table_height_at(plane,placement_xy)
    candidates=[]; rejected={}
    for tilt in (15,20,25,30,35,45,50,55,60):
        for yaw in (0,45,-45,90,-90,180):
            for sign in (1,-1):
                try:
                    poses, held, placed=make_poses(target,plane,config,tilt,yaw,sign)
                except ValueError:
                    rejected["degenerate_geometry"]=rejected.get("degenerate_geometry",0)+1
                    continue
                if any(np.dot(normal,np.asarray(p[:3]))+d < 55 for p in poses.values()):
                    rejected["flange_endpoint_height"]=rejected.get("flange_endpoint_height",0)+1
                    continue
                joints={}; previous=np.asarray(state["joint_actual_position"]); rank=0
                for name,p in poses.items():
                    try:
                        q=robot.inverse(p,previous)
                    except Exception:
                        key=f"ik_{name}"; rejected[key]=rejected.get(key,0)+1
                        break
                    jump=float(np.max(np.abs(q-previous)))
                    if jump>100 or abs(q[4])>119.9 or abs(q[2])>144:
                        key=f"joint_range_{name}"; rejected[key]=rejected.get(key,0)+1
                        break
                    rank+=jump; joints[name]=q.tolist(); previous=q
                if len(joints)==len(poses):
                    candidates.append(dict(rank=rank,poses=poses,joints=joints,
                                           held_model=held.tolist(),placed_model=placed.tolist(),tilt_deg=tilt))
    print(json.dumps({"controller_valid_routes":len(candidates),"rejected":rejected}),flush=True)
    mesh=vision.load_fcstd_mesh(ROOT/config["model"]["path"])
    vertices=np.asarray(mesh.vertices)[::max(1,len(mesh.vertices)//1500)]
    for candidate in sorted(candidates,key=lambda x:x["rank"]):
        try:
            checks=validate_route(robot,candidate,state,plane,config,vertices)
        except (RuntimeError,ValueError) as exc:
            print("Route rejected: "+str(exc),flush=True)
            continue
        else:
            vision.verify_observation_unchanged(target["observation"],vision.read_observation_pose())
            candidate.update(created_unix=time.time(),observation=target["observation"],
                             start_state=state,table_plane=plane,grasp_model_z_mm=side.shaft_middle_model_z(config),
                             placement_xy_mm=placement_xy.tolist(),
                             create_home=False,motion_sent=False,
                             route_checks=checks,planning_binding=planning_binding(config),
                             target_binding={key:copy.deepcopy(target.get(key)) for key in
                                             ("base_T_model","observation","best","shaft_grasp_center_base_mm")},
                             tool_geometry={k:config["gripper"][k] for k in
                                            ("tcp_to_grasp_center_mm","jaw_axis_tcp")})
            return candidate
    raise RuntimeError("No controller-valid upright route with table clearance")


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("stage",choices=("overview","plan","approach","pick","place"))
    parser.add_argument("--target-dir",type=Path,default=OUT)
    args=parser.parse_args(); config=vision.load_config(ROOT/"pick_place_config.json")
    robot=motion.FastRobot(); motion.REPORT=REPORT
    report={"completed":False,"motion_sent":False,"create_home":False}
    attempted=False
    try:
        with controller_lease(8):
            safety.ensure_exclusive_client()
            safety.check_motion_state(robot.motion_snapshot())
            if np.max(np.abs(robot.call("get_curr_tool_offset")["tool_offset"]))>1e-6:
                raise RuntimeError("TCP offset changed")
            if robot.call("get_clsn_sensitivity")["sensitivityLevel"] not in (1,2,3,4,5):
                raise RuntimeError("Collision protection is not enabled")
            if args.stage=="plan":
                target=json.loads((args.target_dir/"world_target.json").read_text(encoding="utf-8"))
                plane=json.loads((OUT/"table_plane.json").read_text(encoding="utf-8"))
                result=plan(robot,target,plane,config); safety.save_json(PLAN,result)
                print(json.dumps(result,ensure_ascii=False,indent=2),flush=True); return 0
            saved=json.loads(PLAN.read_text(encoding="utf-8"))
            if args.stage=="overview":
                report=json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else report
                current=robot.motion_snapshot(); safety.check_motion_state(current)
                q0=np.asarray(current["joint_actual_position"])
                q1=np.asarray(saved["start_state"]["joint_actual_position"])
                if np.max(np.abs(q1-q0))>100:
                    raise RuntimeError("Overview requires a separate transfer route from this pose")
                coefficients=normalized_plane(saved["table_plane"])
                normal,d=coefficients[:3],coefficients[3]
                for fraction in np.linspace(0,1,max(12,int(np.max(np.abs(q1-q0))/2)+1)):
                    tcp=vision.pose_to_matrix(robot.forward(q0*(1-fraction)+q1*fraction))
                    center=vision.transform_points([config["gripper"]["tcp_to_grasp_center_mm"]],tcp)[0]
                    if tcp[:3,3]@normal+d<50 or center@normal+d<50:
                        raise RuntimeError("Overview transfer sweeps too close to tabletop")
                attempted=True
                report.update(motion_sent=True,stage="returning_to_table_overview")
                motion.wait_joint(robot,q1.tolist(),15,"Return to elevated table overview",report,30)
                report.update(stage="at_table_overview",actual=robot.motion_snapshot())
                report["view"]=capture_verification(OUT/"returned_overview")
                safety.save_json(REPORT,report)
                print(json.dumps({"stage":report["stage"],"actual":report["actual"],"view":report["view"]},ensure_ascii=False,indent=2),flush=True)
                return 0
            verify_plan_binding(saved,config)
            if args.stage in ("approach","pick"):
                previous=json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {}
                staged=(args.stage=="pick" and previous.get("stage")=="prepick_awaiting_visual_check")
                if (args.stage=="pick" and not staged and
                        config["gripper"].get("geometry_validation",{}).get("pregrasp_visual_check_required")):
                    raise RuntimeError("Approach and inspect the actual jaw/shaft alignment before pickup")
                if staged:
                    if not previous.get("visual_alignment_confirmed") or previous.get("plan")!=saved:
                        raise RuntimeError("Pre-grasp visual alignment is not confirmed for this plan")
                    current=robot.motion_snapshot(); safety.check_motion_state(current)
                    if np.max(np.abs(np.asarray(current["joint_actual_position"])-previous["actual"]["joint_actual_position"]))>.2:
                        raise RuntimeError("Robot moved since pre-grasp verification")
                    report=previous
                else:
                    vision.verify_observation_unchanged(saved["observation"],vision.read_observation_pose())
                latest=saved.get("revalidated_observation",saved["observation"])
                if time.time()-latest["created_unix"]>1200:
                    raise RuntimeError("Target scan expired")
                report.update(plan=saved,stage="picking_middle",motion_sent=True)
                gripper=side.robot_gripper()
                attempted=True
                if not staged:
                    gripper.connect(); gripper.set_opening_mm(20.0); close_connections()
                    motion.wait_joint(robot,saved["joints"]["approach"],15,"Align over tube middle",report,30)
                    motion.wait_linear(robot,saved["poses"]["prepick"],20,"Approach tube middle",report,60)
                if args.stage=="approach":
                    report.update(stage="prepick_awaiting_visual_check",actual=robot.motion_snapshot(),physical_grasp_confirmed=False)
                    report["view"]=capture_verification(OUT/"prepick_check")
                    safety.save_json(REPORT,report)
                    print(json.dumps({"stage":report["stage"],"view":report["view"]},ensure_ascii=False,indent=2),flush=True)
                    return 0
                motion.wait_linear(robot,saved["poses"]["pick"],5,"Grip position at middle",report,15)
                gripper.connect(); gripper.set_opening_mm(8.65,settle_time=1); close_connections()
                report.update(gripper_closed=True,close_command_mm=8.65,physical_grasp_confirmed=False)
                safety.save_json(REPORT,report)
                motion.wait_linear(robot,saved["poses"]["lift"],25,"Lift clear of table",report,75)
                report.update(stage="lifted_for_verification",actual=robot.motion_snapshot())
                report["view"]=capture_verification(OUT/"after_lift")
            else:
                report=json.loads(REPORT.read_text(encoding="utf-8"))
                if report["stage"]!="lifted_for_verification" or not report.get("physical_grasp_confirmed"):
                    raise RuntimeError("Held table part has not been verified")
                current=robot.motion_snapshot(); safety.check_motion_state(current)
                if np.max(np.abs(np.asarray(current["joint_actual_position"])-report["actual"]["joint_actual_position"]))>.2:
                    raise RuntimeError("Robot moved since lift verification")
                attempted=True
                if "transfer" in saved["poses"]:
                    motion.wait_linear(robot,saved["poses"]["transfer"],25,"Transfer held part to table placement area",report,75)
                for name in saved["poses"]:
                    if name=="turned" or name.startswith("turn_"):
                        motion.wait_joint(robot,saved["joints"][name],15,"Rotate about grasp center: "+name,report,30)
                motion.wait_linear(robot,saved["poses"]["above_place"],30,"Align upright above table",report,90)
                motion.wait_linear(robot,saved["poses"]["place"],5,"Set upright on table",report,15)
                gripper=side.robot_gripper(); gripper.connect(); gripper.set_opening_mm(20.0,settle_time=1); close_connections()
                report.update(gripper_closed=False,stage="released_upright")
                safety.save_json(REPORT,report)
                motion.wait_linear(robot,saved["poses"]["retreat"],20,"Clear standing part",report,60)
                motion.wait_joint(robot,saved["start_state"]["joint_actual_position"],15,"Return to overview",report,30)
                report.update(stage="released_awaiting_visual_check",actual=robot.motion_snapshot())
                report["view"]=capture_verification(OUT/"after_stand")
            safety.save_json(REPORT,report)
            print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
            return 0
    except BaseException as exc:
        report["error"]=str(exc)
        if attempted:
            try: safety.stop_and_verify(robot)
            except Exception as stopping: report["stop_error"]=str(stopping)
        safety.save_json(REPORT,report); raise
    finally: close_connections()


if __name__=="__main__":
    raise SystemExit(main())
