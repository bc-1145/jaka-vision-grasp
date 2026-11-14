"""Read-only bin grasp planning with explicit 30/45/60 degree flange tilts.

The supplied robot is used only for status and controller IK/FK. This module
does not own a camera, open a controller connection, or send actuator writes.
Clearance checks sample a flange-to-contact segment, not the complete robot.
"""
from __future__ import annotations

import copy
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

import calibration_motion as safety
import pick_place_3d as vision
import side_pick_execute as side

ROOT=Path(__file__).resolve().parent
ALLOWED_TILTS=(30,45,60)
MODES={"retreat":"MoveL","approach":"MoveJ","prepick":"MoveL",
       "pick":"MoveL","lift_first":"MoveL","lift":"MoveL"}


def _get(value,name,default=None):
    return value.get(name,default) if isinstance(value,dict) else getattr(value,name,default)


def _vector(value,size,name):
    value=np.asarray(value,dtype=float)
    if value.shape!=(size,) or not np.isfinite(value).all():
        raise ValueError(name+" must contain finite values")
    return value


def _joint_check(q,previous,max_jump=100):
    q=_vector(q,6,"Controller joints")
    if abs(q[4])>119.9 or abs(q[2])>144 or np.max(np.abs(q-previous))>max_jump:
        raise RuntimeError("Controller joint range or literal branch jump exceeded")
    return q


@lru_cache(maxsize=4)
def _model_points(path,mtime_ns,count):
    import open3d as o3d
    mesh=vision.load_fcstd_mesh(Path(path))
    o3d.utility.random.seed(7)
    points=np.asarray(mesh.sample_points_uniformly(count).points).copy()
    return points


def _pose(center,rotation,offset):
    transform=np.eye(4)
    transform[:3,:3]=rotation
    transform[:3,3]=side.flange_position(center,rotation,offset)
    return vision.matrix_to_pose(transform)


def _clearance(tcp,offset,foreign_tree):
    flange=tcp[:3,3]
    center=vision.transform_points([offset],tcp)[0]
    length=float(np.linalg.norm(center-flange))
    axis=flange+np.linspace(0,1,max(2,int(math.ceil(length/2))+1))[:,None]*(center-flange)
    return float(foreign_tree.query(flange)[0]),float(np.min(foreign_tree.query(axis)[0]))


def _finger_geometry(config):
    """Optional measured rectangular envelopes; stroke estimates are not sizes."""
    supplied=config["gripper"].get("rectangular_fingers")
    if not supplied or supplied.get("measured") is not True or not supplied.get("measurement_source"):
        return None,dict(available=False,checked=False,shape="two rectangular fingers",
                         reason="No independently measured open/closed finger envelopes; empty-stroke images only establish provisional contact geometry",
                         complete_gripper_collision_checked=False)
    if supplied.get("open_command_mm")!=20 or supplied.get("closed_command_mm")!=8.65:
        raise ValueError("Measured finger boxes must correspond to the actual 20/8.65 commands")
    result={}
    for state in ("open","closed"):
        boxes=supplied.get(state+"_boxes_tcp_mm",[])
        if len(boxes)!=2:
            raise ValueError("Exactly two measured rectangular finger boxes are required")
        parsed=[]
        for box in boxes:
            low=_vector(box["min_mm"],3,"Finger box minimum")
            high=_vector(box["max_mm"],3,"Finger box maximum")
            if not np.all(low<high):
                raise ValueError("Measured finger box has no volume")
            parsed.append((low,high))
        result[state]=parsed
    # The axis-aligned union also bounds the intermediate closing translation.
    result["closing"]=[(np.minimum(a[0],b[0]),np.maximum(a[1],b[1]))
                       for a,b in zip(result["open"],result["closed"])]
    return result,dict(available=True,checked=True,shape="two measured rectangular TCP envelopes",
                        measurement_source=supplied["measurement_source"],minimum_foreign_clearance_mm=2,
                        closing_sweep_envelope_checked=True,complete_gripper_collision_checked=False)


def _finger_clearance(tcp,foreign_points,boxes):
    local=vision.transform_points(foreign_points,np.linalg.inv(tcp))
    closest=float("inf")
    for low,high in boxes:
        outside=np.maximum(np.maximum(low-local,local-high),0)
        closest=min(closest,float(np.min(np.linalg.norm(outside,axis=1))))
    return closest


def _orientation_candidates(shaft,reference_rotation,preferred_azimuth,config):
    candidates=[]
    for tilt in ALLOWED_TILTS:
        sine,cosine=np.sin(np.deg2rad(tilt)),np.cos(np.deg2rad(tilt))
        for delta in (0,30,-30,60,-60,90,-90,120,-120,150,-150,180):
            azimuth=preferred_azimuth+np.deg2rad(delta)
            tool=np.array([sine*np.cos(azimuth),sine*np.sin(azimuth),-cosine])
            jaw=np.cross(shaft,tool)
            if np.linalg.norm(jaw)<.1:
                continue
            jaw/=np.linalg.norm(jaw)
            for sign in (1,-1):
                rotation=side.grasp_rotation(tool,jaw*sign,config["gripper"]["jaw_axis_tcp"])
                difference=float((Rotation.from_matrix(reference_rotation).inv()*
                                  Rotation.from_matrix(rotation)).magnitude())
                candidates.append((difference,tilt,float(np.degrees(azimuth)%360),tool,rotation))
    return sorted(candidates,key=lambda value:value[0])


def _validate_route(robot,poses,joints,state,offset,foreign_tree,statistics,*,finger_boxes=None,orientation_min_z=None):
    q0=np.asarray(state["joint_actual_position"],float)
    p0=np.asarray(state["actual_position"],float)
    segments=[]
    for name,p1 in poses.items():
        p1=np.asarray(p1,float)
        q1=_joint_check(joints[name],q0)
        linear=MODES[name]=="MoveL"
        if linear:
            slerp=Slerp([0,1],Rotation.from_euler("xyz",np.stack((p0[3:],p1[3:])),degrees=True))
            angle=float(np.degrees((Rotation.from_euler("xyz",p0[3:],degrees=True).inv()*
                                   Rotation.from_euler("xyz",p1[3:],degrees=True)).magnitude()))
            count=max(8,int(math.ceil(np.linalg.norm(p1[:3]-p0[:3])/3))+1,int(math.ceil(angle/2))+1)
        else:
            count=max(8,int(math.ceil(np.max(np.abs(q1-q0))/2))+1)
        record=dict(name=name,mode=MODES[name],samples=count,
                    minimum_flange_foreign_mm=float("inf"),minimum_tool_axis_foreign_mm=float("inf"),
                    minimum_contact_base_z_mm=float("inf"),minimum_finger_foreign_mm=None)
        previous=q0
        for fraction in np.linspace(0,1,count):
            if linear:
                pose=np.r_[p0[:3]*(1-fraction)+p1[:3]*fraction,
                           slerp(fraction).as_euler("xyz",degrees=True)]
                statistics["inverse_calls"]+=1
                q=_joint_check(robot.inverse(pose,previous),previous,max_jump=10)
            else:
                q=_joint_check(q0*(1-fraction)+q1*fraction,previous,max_jump=10)
                pose=None
            statistics["forward_calls"]+=1
            actual=_vector(robot.forward(q),6,"Controller FK pose")
            if linear:
                translation,rotation=safety.pose_error(actual,pose)
                if translation>.25 or rotation>.2:
                    raise RuntimeError("Cartesian IK/FK mismatch at "+name)
            tcp=vision.pose_to_matrix(actual)
            flange,axis=_clearance(tcp,offset,foreign_tree)
            center=vision.transform_points([offset],tcp)[0]
            record["minimum_contact_base_z_mm"]=min(record["minimum_contact_base_z_mm"],float(center[2]))
            statistics["clearance_samples"]+=1
            record["minimum_flange_foreign_mm"]=min(record["minimum_flange_foreign_mm"],flange)
            record["minimum_tool_axis_foreign_mm"]=min(record["minimum_tool_axis_foreign_mm"],axis)
            if flange<50 or axis<10:
                raise RuntimeError(f"Foreign-point clearance at {name}: flange {flange:.2f}, tool axis {axis:.2f}")
            if name=="approach" and orientation_min_z is not None and center[2]<orientation_min_z-.25:
                raise RuntimeError("Orientation adjustment lowers the grasp center below its checked high level")
            if finger_boxes is not None:
                finger_state="closed" if name in ("lift_first","lift") else "closing" if name=="pick" else "open"
                clearance=_finger_clearance(tcp,foreign_tree.data,finger_boxes[finger_state])
                old=record["minimum_finger_foreign_mm"]
                record["minimum_finger_foreign_mm"]=clearance if old is None else min(old,clearance)
                if clearance<2:
                    raise RuntimeError(f"Measured rectangular finger envelope approaches foreign geometry at {name}: {clearance:.2f}")
            previous=q
        if linear and np.max(np.abs(previous-q1))>.25:
            raise RuntimeError("Cartesian endpoint uses a different IK branch at "+name)
        segments.append(record)
        q0,p0=q1,p1
    return dict(segments=segments,full_collision_checked=False,continuous_sweep_proved=False,
                model="sampled flange and flange-to-contact axis against measured non-target points",
                linear_spacing_mm=3,joint_spacing_deg=2,
                minimum_flange_foreign_mm=min(s["minimum_flange_foreign_mm"] for s in segments),
                minimum_tool_axis_foreign_mm=min(s["minimum_tool_axis_foreign_mm"] for s in segments))


def plan_bin_grasp(robot,config,observation,matches,scene):
    """Choose the highest-scoring reachable tube and return checked poses.

    ``scene`` is the current optical-camera point cloud (or Nx3 array).
    The caller owns controller/camera exclusivity and must recheck acquisition
    age and unchanged robot state before executing the returned movement modes.
    """
    started=time.time()
    state=robot.motion_snapshot()
    safety.check_motion_state(state)
    position,angle=safety.pose_error(state["actual_position"],observation["tcp_pose"])
    if position>.5 or angle>.5:
        raise RuntimeError("Robot moved since bin observation")
    base_camera=vision.camera_transform(config,observation)
    offset=_vector(config["gripper"]["tcp_to_grasp_center_mm"],3,"Grasp-center offset")
    finger_boxes,finger_report=_finger_geometry(config)
    middle=side.shaft_middle_model_z(config)
    values=np.asarray(scene.points if hasattr(scene,"points") else scene,dtype=float)
    if values.ndim!=2 or values.shape[1]!=3:
        raise ValueError("Scene must contain optical-camera XYZ points")
    values=values[np.isfinite(values).all(axis=1)&(values[:,2]>0)]
    if len(values)<100:
        raise RuntimeError("Insufficient measured scene for bin route checking")
    eligible=[]
    for item in matches:
        score,clearance=float(_get(item,"score",float("nan"))),float(_get(item,"clearance_mm",float("nan")))
        if np.isfinite([score,clearance]).all() and score>=.78 and clearance>=10:
            if _get(item,"observation")!=observation:
                raise RuntimeError("Bin matches do not belong to this observation")
            eligible.append(item)
    if not eligible:
        raise RuntimeError("No bin candidate meets score 0.78 and clearance 10mm")
    eligible.sort(key=lambda value:-float(_get(value,"score")))
    path=(ROOT/config["model"]["path"]).resolve()
    model=_model_points(str(path),path.stat().st_mtime_ns,int(config["matching"]["model_sample_points"]))
    world_scene=vision.transform_points(values,base_camera)
    camera_axis=base_camera[:3,2]/np.linalg.norm(base_camera[:3,2])
    retreat=np.asarray(state["actual_position"],float).copy()
    # A verified unchanged-jaw inspection already performed this lift. Do not
    # stack another fixed retreat until the wrist reaches its joint limit.
    retreat_distance=0 if config.get("_bin_inspection_retreat_completed") is True else 40
    retreat[:3]-=camera_axis*retreat_distance
    initial_tcp=vision.pose_to_matrix(state["actual_position"])
    initial_center=vision.transform_points([offset],initial_tcp)[0]
    retreat_center=vision.transform_points([offset],vision.pose_to_matrix(retreat))[0]
    statistics=dict(inverse_calls=0,forward_calls=0,clearance_samples=0,orientation_candidates=0,
                    candidates_tested=0,targets_tested=0,endpoint_rejected=0,ik_rejected=0,route_rejected=0)
    rejections=[]
    current_joints=np.asarray(state["joint_actual_position"],float)
    statistics["inverse_calls"]+=1
    retreat_joints=(current_joints.copy() if retreat_distance==0 else
                    _joint_check(robot.inverse(retreat,current_joints),current_joints))
    for match in eligible:
        statistics["targets_tested"]+=1
        camera_model=vision.matrix(_get(match,"camera_T_model"),"match.camera_T_model")
        base_model=base_camera@camera_model
        shaft=base_model[:3,2]/np.linalg.norm(base_model[:3,2])
        grasp=vision.transform_points([[0,0,middle]],base_model)[0]
        target_world=vision.transform_points(model,base_model)
        exclusion=max(6,float(config["matching"]["ransac_distance_mm"]),
                      2*float(config["matching"]["icp_distance_mm"]))
        distances=cKDTree(target_world).query(world_scene)[0]
        foreign=world_scene[distances>exclusion]
        if len(foreign)<100:
            rejections.append(dict(cluster_index=_get(match,"cluster_index"),reason="Too little non-target depth geometry"))
            continue
        foreign_tree=cKDTree(foreign)
        preferred=camera_axis[:2]
        if np.linalg.norm(preferred)<.05:
            preferred=grasp[:2]-np.asarray(state["actual_position"][:2])
        azimuth=float(np.arctan2(preferred[1],preferred[0]))
        orientations=_orientation_candidates(shaft,vision.pose_to_matrix(state["actual_position"])[:3,:3],azimuth,config)
        statistics["orientation_candidates"]+=len(orientations)
        for _,tilt,azimuth_deg,tool,rotation in orientations:
            statistics["candidates_tested"]+=1
            high_z=max(float(initial_center[2]),float(retreat_center[2]),float(grasp[2]+70))
            approach_center=np.r_[retreat_center[:2],high_z]
            prepick_center=np.r_[grasp[:2],high_z]
            centers=dict(approach=approach_center,prepick=prepick_center,pick=grasp,
                         lift_first=grasp+[0,0,40],lift=grasp+[0,0,70])
            poses={"retreat":retreat.tolist()}
            poses.update({name:_pose(center,rotation,offset) for name,center in centers.items()})
            if any(minimum<limit for pose in poses.values()
                   for minimum,limit in zip(_clearance(vision.pose_to_matrix(pose),offset,foreign_tree),(50,10))):
                statistics["endpoint_rejected"]+=1
                continue
            joints={"retreat":retreat_joints.tolist()}
            previous=retreat_joints
            try:
                for name,pose in list(poses.items())[1:]:
                    statistics["inverse_calls"]+=1
                    q=_joint_check(robot.inverse(pose,previous),previous)
                    joints[name]=q.tolist()
                    previous=q
            except (OSError,TimeoutError,ConnectionError):
                raise
            except (RuntimeError,ValueError) as exc:
                statistics["ik_rejected"]+=1
                rejections.append(dict(cluster_index=_get(match,"cluster_index"),tilt_deg=tilt,reason=str(exc)))
                continue
            try:
                checks=_validate_route(robot,poses,joints,state,offset,foreign_tree,statistics,
                                       finger_boxes=finger_boxes,
                                       orientation_min_z=max(float(initial_center[2]),float(grasp[2]+70)))
            except (OSError,TimeoutError,ConnectionError):
                raise
            except (RuntimeError,ValueError) as exc:
                statistics["route_rejected"]+=1
                rejections.append(dict(cluster_index=_get(match,"cluster_index"),tilt_deg=tilt,reason=str(exc)))
                continue
            end=robot.motion_snapshot()
            safety.check_motion_state(end)
            translation,rotation_error=safety.pose_error(end["actual_position"],state["actual_position"])
            if translation>.1 or rotation_error>.1 or np.max(np.abs(np.asarray(end["joint_actual_position"])-current_joints))>.1:
                raise RuntimeError("Robot moved during read-only bin planning")
            held=np.linalg.inv(vision.pose_to_matrix(poses["pick"]))@base_model
            large_z=float(config["model"].get("large_end_z_mm",0))
            small_z=float(config["model"]["shaft_z_range_mm"][1])
            grasp_tcp=vision.transform_points([grasp],np.linalg.inv(initial_tcp))[0]
            current_tcp_model=np.linalg.inv(initial_tcp)@base_model
            return dict(created_unix=time.time(),planning_seconds=time.time()-started,motion_sent=False,
                        observation=copy.deepcopy(observation),start_state=copy.deepcopy(state),
                        poses=poses,joints=joints,motion_modes=dict(MODES),
                        selected_cluster_index=_get(match,"cluster_index"),match_score=float(_get(match,"score")),
                        foreign_clearance_mm=float(_get(match,"clearance_mm")),selection="highest_score_reachable",
                        tilt_deg=tilt,allowed_tilts_deg=list(ALLOWED_TILTS),azimuth_deg=azimuth_deg,
                        tilt_definition="Angle of TCP/flange +Z relative to base vertically downward (-Z)",
                        camera_T_model=camera_model.tolist(),base_T_model=base_model.tolist(),
                        base_T_camera=base_camera.tolist(),held_model=held.tolist(),
                        shaft_axis_base=shaft.tolist(),shaft_axis_camera=camera_model[:3,2].tolist(),
                        large_end_base_mm=vision.transform_points([[0,0,large_z]],base_model)[0].tolist(),
                        small_end_base_mm=vision.transform_points([[0,0,small_z]],base_model)[0].tolist(),
                        direction_source="Original accepted CAD model axis; no direction flip during motion planning",
                        grasp_base=grasp.tolist(),grasp_model_z_mm=middle,
                        model_in_observation_tcp=current_tcp_model.tolist(),
                        grasp_in_observation_tcp_mm=grasp_tcp.tolist(),
                        grasp_relative_to_observation_center_tcp_mm=(grasp_tcp-offset).tolist(),
                        shaft_axis_in_observation_tcp=(initial_tcp[:3,:3].T@shaft).tolist(),
                        large_end_in_observation_tcp_mm=vision.transform_points([[0,0,large_z]],current_tcp_model)[0].tolist(),
                        small_end_in_observation_tcp_mm=vision.transform_points([[0,0,small_z]],current_tcp_model)[0].tolist(),
                        movement_strategy="high_orientation_then_horizontal_alignment_then_base_vertical_grasp",
                        grasp_centers_base_mm={name:center.tolist() for name,center in centers.items()},
                        initial_grasp_center_base_mm=initial_center.tolist(),
                        orientation_minimum_contact_z_mm=max(float(initial_center[2]),float(grasp[2]+70)),
                        horizontal_alignment_z_mm=high_z,
                        horizontal_alignment_delta_mm=(prepick_center-approach_center).tolist(),
                        descent_axis_base=[0,0,-1],lift_axis_base=[0,0,1],
                        approach_axis=[0,0,-1],tool_axis=tool.tolist(),grasp_rpy_deg=poses["pick"][3:],
                        approach_distance_mm=high_z-float(grasp[2]),prepick_distance_mm=high_z-float(grasp[2]),
                        lift_first_distance_mm=40,
                        initial_retreat_distance_mm=retreat_distance,
                        inspection_retreat_reused=retreat_distance==0,
                        lift_distance_mm=70,lift_verification_separation_mm=30,
                        tool_geometry={key:copy.deepcopy(config["gripper"][key]) for key in
                                       ("tcp_to_grasp_center_mm","jaw_axis_tcp")},
                        finger_geometry=finger_report,
                        model_path=str(path),model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        target_exclusion_mm=exclusion,foreign_scene_points=len(foreign),
                        route_checks=checks,statistics=statistics,rejected_candidates=rejections[-30:])
    detail=rejections[-1]["reason"] if rejections else "All pose endpoints violate measured clearance"
    raise RuntimeError(f"No checked bin grasp at flange tilt 30/45/60 degrees: {detail}; statistics={statistics}")
