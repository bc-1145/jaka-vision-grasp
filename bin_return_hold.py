"""Return a successful OR failed grasp through the existing Home BAT.

The gripper command is never changed. There is no release mode. Default is
read-only; --execute runs only the existing return route after measured checks.
The geometry audit covers the known table, flange/tool envelope and a possible
held part. It is not full robot, cable, fixture or bin collision checking.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import arm_control
import calibration_motion as safety
from controller_lease import controller_lease
from gripper_feedback import GripperFeedback
from gripper_transport import close_connections
import pick_place_3d as vision
from bin_return_release import BATCH, ROOT, file_hash, read_json, require_same_state, run_home_batch


def measured_source(source, max_age_seconds=180, *, now=None):
    """A failure is not a reason to discard a fresh measured stationary pose."""
    state_name = "final_state" if isinstance(source.get("final_state"), dict) else "actual"
    state = source.get(state_name)
    if not isinstance(state, dict):
        raise RuntimeError("Return requires a final measured robot state, including failed grasps")
    stamp = state.get("created_unix") or source.get(state_name+"_created_unix")
    if not isinstance(stamp, (int, float)) or not np.isfinite(stamp):
        raise RuntimeError("Final robot state has no explicit measured timestamp")
    if not 0 <= (time.time() if now is None else now)-stamp <= max_age_seconds:
        raise RuntimeError("Final measured robot state is stale or future-dated")
    safety.check_motion_state(state)
    return state


def predict_joint3_lift(robot, current, target_z, step, maximum_delta):
    """Read-only equivalent of arm_control.plan_joint3_lift's exact search."""
    current = safety.vector(current)
    start = safety.vector(robot.forward(current))
    if start[2] >= target_z:
        return current.copy(), start
    candidates = []
    for direction in (-1., 1.):
        previous_z = float(start[2])
        for index in range(1, int(np.floor(maximum_delta/step))+1):
            target = current.copy()
            target[2] += direction*index*step
            if not -144 <= target[2] <= 144:
                break
            try:
                pose = safety.vector(robot.forward(target))
            except Exception:
                break
            if pose[2] < previous_z-.5:
                break
            previous_z = float(pose[2])
            if pose[2] >= target_z:
                candidates.append((index*step, target, pose))
                break
    if not candidates:
        raise RuntimeError("Existing BAT's J3-only lift has no matching checked solution")
    _, target, pose = min(candidates, key=lambda value: value[0])
    return target, pose


def audit_existing_home_route(robot, state, source, config, settings, home_settings, *, vertices=None):
    """Expand the unchanged BAT's staged route and inspect table clearance.

    CAD bounding-box corners conservatively enclose a verified held object.
    For failed/unknown grasps, an object-sized sphere about the pinch center
    represents a possibly held tube. The gripper's unchanged command is never
    treated as evidence that the jaws are empty.
    """
    safety.check_motion_state(state)
    home = safety.vector(home_settings["position"])
    approach = safety.vector(home_settings["approach_joint"])
    approach_pose = np.r_[home[:2], home_settings["approach_z"], home[3:]]
    plane = np.asarray(settings["reference_table_plane"]["plane_base"], float)
    if plane.shape != (4,) or not np.isfinite(plane).all() or np.linalg.norm(plane[:3]) < 1e-8:
        raise RuntimeError("Known table plane is invalid")
    plane /= np.linalg.norm(plane[:3])
    if plane[2] < np.cos(np.deg2rad(5)):
        raise RuntimeError("Known table is not a verified upward horizontal plane")
    bounds = np.asarray(settings["table_xy_bounds_mm"], float)
    if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(bounds[0] >= bounds[1]):
        raise RuntimeError("Known table footprint is invalid")
    offset = safety.vector(config["gripper"]["tcp_to_grasp_center_mm"], size=3)
    if vertices is None:
        mesh = vision.load_fcstd_mesh(ROOT / config["model"]["path"])
        vertices = np.asarray(mesh.vertices)
    vertices = np.asarray(vertices, float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 8 or not np.isfinite(vertices).all():
        raise RuntimeError("Possible held part geometry is missing or invalid")
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    corners = np.asarray([[x, y, z] for x in (low[0], high[0])
                          for y in (low[1], high[1]) for z in (low[2], high[2])])
    grasp_model = np.asarray(config["model"]["grasp_point_mm"], float)
    possible_part_radius = float(np.max(np.linalg.norm(corners-grasp_model, axis=1)))+5
    held_value = source.get("plan", {}).get("held_model")
    held = vision.matrix(held_value, "source.plan.held_model") if held_value is not None else None
    verified_shape = source.get("physical_grasp_confirmed") is True and held is not None
    samples = []
    minima = dict(flange=None, tool_axis_envelope=None, possible_part=None)

    def overlap(points, margin):
        lower, upper = np.min(points[:, :2], axis=0), np.max(points[:, :2], axis=0)
        return bool(np.all(upper+margin >= bounds[0]) and np.all(lower-margin <= bounds[1]))

    def remember(name, value):
        minima[name] = float(value) if minima[name] is None else min(minima[name], float(value))
        if value < 5:
            raise RuntimeError(f"Existing Home BAT has obvious table interference: {name} clearance {value:.2f} mm")

    def check_pose(pose, label):
        tcp = vision.pose_to_matrix(pose)
        flange = tcp[:3, 3]
        center = vision.transform_points([offset], tcp)[0]
        axis = flange + np.linspace(0, 1, 25)[:, None]*(center-flange)
        if overlap(flange[None, :], 35):
            remember("flange", flange@plane[:3]+plane[3]-35)
        axis_near_table = np.all(axis[:, :2] >= bounds[0]-10, axis=1) & np.all(axis[:, :2] <= bounds[1]+10, axis=1)
        if axis_near_table.any():
            remember("tool_axis_envelope", np.min(axis[axis_near_table]@plane[:3]+plane[3])-10)
        if verified_shape:
            part = vision.transform_points(corners, tcp@held)
            if overlap(part, 5):
                remember("possible_part", np.min(part@plane[:3]+plane[3])-5)
        elif overlap(center[None, :], possible_part_radius):
            remember("possible_part", center@plane[:3]+plane[3]-possible_part_radius)
        samples.append(dict(stage=label, pose=list(map(float, pose))))

    def check_joint(q, label):
        q = safety.vector(q)
        if abs(q[2]) > 144 or abs(q[4]) > 119.9:
            raise RuntimeError("Existing Home BAT exceeds checked J3/J5 range")
        pose = safety.vector(robot.forward(q))
        check_pose(pose, label)
        return pose

    q = safety.vector(state["joint_actual_position"])
    actual = safety.vector(state["actual_position"])
    delta, angle = safety.pose_error(robot.forward(q), actual)
    if delta > .5 or angle > .5:
        raise RuntimeError("Controller FK disagrees with actual initial pose")
    check_pose(actual, "start")
    home_position_error, home_angle_error = arm_control._pose_error(actual.tolist(), home.tolist())
    if home_position_error <= .5 and home_angle_error <= .5:
        return dict(accepted=True, already_home=True, samples=samples, minima_mm=minima,
                    full_collision_checked=False, gripper_command_unchanged=True,
                    held_shape="CAD_box" if verified_shape else "unknown_object_sphere")
    if actual[2] < home_settings["first_lift_z"]:
        target, _ = predict_joint3_lift(robot, q, home_settings["first_lift_z"],
                                       home_settings["joint3_search_step"], home_settings["joint3_max_delta"])
        for fraction in np.linspace(0, 1, max(3, int(np.ceil(np.max(np.abs(target-q))))+1)):
            check_joint(q*(1-fraction)+target*fraction, "existing_bat_joint3_lift")
        q = target
    wrapped_error = float(np.max(np.abs((q-approach+180)%360-180)))
    if wrapped_error > .5:
        for fraction in np.linspace(0, 1, max(3, int(np.ceil(np.max(np.abs(approach-q))/2))+1)):
            check_joint(q*(1-fraction)+approach*fraction, "existing_bat_joint_align")
        q = approach
    actual = safety.vector(robot.forward(q))
    translation, rotation = arm_control._pose_error(actual.tolist(), approach_pose.tolist())
    if translation > 5 or rotation > 2:
        raise RuntimeError("Existing Home approach joints do not match existing approach pose")
    previous = q
    from scipy.spatial.transform import Rotation, Slerp
    orientations = Slerp([0, 1], Rotation.from_euler("xyz", [actual[3:], home[3:]], degrees=True))
    for fraction in np.linspace(0, 1, max(15, int(np.ceil(np.linalg.norm(home[:3]-actual[:3])/4))+1)):
        # Euler -180/+180 representations must not create an artificial full
        # turn while checking the BAT's Cartesian descent.
        pose = np.r_[actual[:3]*(1-fraction)+home[:3]*fraction,
                     orientations(fraction).as_euler("xyz", degrees=True)]
        joints = safety.vector(robot.inverse(pose, previous))
        if np.max(np.abs(joints-previous)) > 10:
            raise RuntimeError("Home descent IK branch changed")
        fk = check_joint(joints, "existing_bat_linear_descend")
        position_error, angle_error = safety.pose_error(fk, pose)
        if position_error > .5 or angle_error > .5:
            raise RuntimeError("Home descent IK/FK disagree")
        previous = joints
    require_same_state(robot.motion_snapshot(), state)
    return dict(accepted=True, samples=samples, minima_mm=minima,
                full_collision_checked=False, gripper_command_unchanged=True,
                checked_geometry="known table footprint, flange sphere, tool-axis capsule, possible held part",
                held_shape="CAD_box" if verified_shape else "unknown_object_sphere",
                unknown_object_radius_mm=possible_part_radius,
                excluded_checks=["full robot links", "cables", "unmeasured fixtures", "bin walls"])


class ReturnHold:
    def __init__(self, source_path, output, *, execute=False, max_age_seconds=180,
                 robot=None, feedback=None, batch_runner=None, route_auditor=None):
        self.source_path, self.output = Path(source_path), Path(output)
        self.execute, self.max_age_seconds = execute, max_age_seconds
        self.robot = robot if robot is not None else safety.Robot()
        self.feedback = feedback if feedback is not None else GripperFeedback()
        self.batch_runner, self.route_auditor = batch_runner or run_home_batch, route_auditor or audit_existing_home_route
        self.claim_path = self.source_path.with_name("return_hold.claim.json")
        if self.output.exists():
            raise RuntimeError("Use a new return-hold audit directory")
        self.output.mkdir(parents=True)
        self.report = dict(completed=False, motion_sent=False, create_home=False,
                           mode="return_hold", source_run=str(self.source_path),
                           created_unix=time.time(), gripper_command_unchanged=True,
                           gripper_write_attempted=False, released=False, holding=None,
                           holding_status="unknown", full_collision_checked=False, events=[])

    def event(self, stage, **details):
        self.report["stage"] = stage
        self.report["events"].append(dict(stage=stage, created_unix=time.time(), **details))
        safety.save_json(self.output/"run.json", self.report)
        print(stage, flush=True)

    def checked_state(self):
        safety.ensure_exclusive_client()
        safety.check_state(self.robot.snapshot())
        if max(abs(x) for x in safety.vector(self.robot.call("get_curr_tool_offset")["tool_offset"])) > 1e-6:
            raise RuntimeError("Tool offset changed from zero flange offset")
        if self.robot.call("get_clsn_sensitivity")["sensitivityLevel"] not in (1, 2, 3, 4, 5):
            raise RuntimeError("Collision protection is disabled")
        current = self.robot.motion_snapshot()
        safety.check_motion_state(current)
        return current

    def record_feedback(self, label):
        try:
            feedback = self.feedback.snapshot()
        except Exception as exc:
            feedback = dict(available=False, holding=None, dropped=None, reason=str(exc))
        holding = feedback.get("holding") if feedback.get("available") is True else None
        self.report.update(holding=holding,
                           holding_status="held" if holding is True else "not_held" if holding is False else "unknown")
        self.event(label, feedback=feedback)
        return feedback

    def run(self):
        try:
            source = read_json(self.source_path)
            expected = measured_source(source, self.max_age_seconds)
            config, settings = read_json(ROOT/"pick_place_config.json"), read_json(ROOT/"table_automation.json")
            home_file = arm_control.POSES_FILE
            home_settings = read_json(home_file)["home"]
            hashes = {str(path): file_hash(path) for path in
                      (self.source_path, Path(home_file), BATCH, ROOT/"arm_control.py",
                       ROOT/"pick_place_config.json", ROOT/"table_automation.json")}
            home = arm_control.load_pose("home")
            self.report.update(source_grasp_confirmed=source.get("physical_grasp_confirmed") is True,
                               source_error=source.get("error"), existing_home_pose=home, bound_hashes=hashes)
            with controller_lease(8):
                if self.claim_path.exists():
                    raise RuntimeError("This run already has a return-hold attempt; no automatic replay")
                current = self.checked_state()
                require_same_state(current, expected)
                self.record_feedback("返回前记录真实夹持状态，不改变夹爪")
                audit = self.route_auditor(self.robot, current, source, config, settings, home_settings)
                if audit.get("accepted") is not True:
                    raise RuntimeError("Existing Home BAT route is not checked for the possible held geometry")
                self.event("原有 Home BAT 桌面与可能夹持外形检查", route_audit=audit)
                if not self.execute:
                    self.report["ready"] = True
                    self.event("只读检查完成，未回位且未写夹爪")
                    return self.report
                measured_source(source, self.max_age_seconds)
                require_same_state(self.checked_state(), expected)
                if any(file_hash(path) != value for path, value in hashes.items()):
                    raise RuntimeError("Source, original Home BAT or installation changed during route checks")
                with self.claim_path.open("x", encoding="utf-8") as handle:
                    json.dump(dict(created_unix=time.time(), output=str(self.output),
                                   state="return_started", gripper_command_unchanged=True), handle)
                close_connections()
                self.report["motion_sent"] = True
                self.event("调用现有一键回 Home BAT，全程保持夹爪原状")
            code = self.batch_runner(self.output/"home_batch.log")
            self.event("原有 BAT 已结束，核对实测 Home", batch_exit_code=code,
                       footer_exit_is_not_motion_evidence=True)
            with controller_lease(8):
                final = self.checked_state()
                if any(file_hash(path) != value for path, value in hashes.items()):
                    raise RuntimeError("Bound source/Home/installation changed during return")
                position, angle = safety.pose_error(final["actual_position"], home)
                if position > .5 or angle > .5:
                    raise RuntimeError("Original Home not actually reached; gripper remains unchanged")
                self.record_feedback("Home 到位记录夹持状态，绝不松爪")
                self.report.update(completed=True, final_state=final, final_state_created_unix=time.time(),
                                   home_position_error_mm=position, home_angle_error_deg=angle,
                                   needs_fresh_observation_before_retry=True)
                safety.save_json(self.claim_path, dict(state="at_existing_home_hold", completed_unix=time.time(),
                                 output=str(self.output), gripper_command_unchanged=True))
                self.event("已回原有 Home，夹爪保持原状；后续重试须重新拍照定位")
            return self.report
        except BaseException as exc:
            self.report.update(completed=False, error=str(exc))
            if self.report["motion_sent"]:
                try:
                    with controller_lease(8):
                        safety.ensure_exclusive_client()
                        safety.stop_and_verify(self.robot)
                except BaseException as stop_error:
                    self.report["stop_error"] = str(stop_error)
            self.event("返回停止，不改夹爪、不自动重试 BAT", error=str(exc))
            raise
        finally:
            close_connections()
            safety.save_json(self.output/"run.json", self.report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-age-seconds", type=float, default=180)
    args = parser.parse_args()
    if not 0 < args.max_age_seconds <= 300:
        parser.error("Freshness limit must be greater than zero and at most 300 seconds")
    output = args.output or args.run.parent/time.strftime("return_hold_%Y%m%d_%H%M%S")
    report = ReturnHold(args.run, output, execute=args.execute, max_age_seconds=args.max_age_seconds).run()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("保持夹爪回 Home 未完成："+str(exc), file=sys.stderr, flush=True)
        raise SystemExit(1)
