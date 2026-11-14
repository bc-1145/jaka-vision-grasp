"""Supervised automatic table pickup and large-flange-down placement.

--check is offline; --observe and --plan-only never send actuator writes.
--execute runs one measured cycle. Uncertain writes and uncertain holding
are never retried. Home and persistent calibration are not modified.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def combine_grasp_evidence(feedback, appearances, follow_motion):
    """A setpoint, a projection, or one appearance cannot prove holding."""
    if feedback.get("dropped") is True:
        return {"confirmed": False, "reason": "Gripper reports dropped workpiece"}
    if not appearances or not all(item.get("accepted") is True for item in appearances):
        return {"confirmed": False, "reason": "Physical part appearance not established"}
    if feedback.get("holding") is False:
        return {"confirmed": False, "reason": "Gripper feedback contradicts held-object appearance"}
    if feedback.get("available") is True and feedback.get("holding") is True:
        return {"confirmed": True, "source": "hardware_and_appearance", "reason": "Measured grip and visible part agree"}
    if len(appearances) >= 2 and follow_motion.get("accepted") is True:
        return {"confirmed": True, "source": "appearance_and_motion", "reason": "Visible part follows two distinct measured lift poses"}
    return {"confirmed": False, "reason": "No measured grip or verified object-following motion"}


def inside_table(xy, settings):
    import numpy as np
    bounds = np.asarray(settings["table_xy_bounds_mm"], float)
    xy = np.asarray(xy, float)
    return bool(bounds.shape == (2, 2) and xy.shape == (2,) and np.isfinite(xy).all()
                and np.all(xy >= bounds[0]) and np.all(xy <= bounds[1]))


def require_fresh(observation, max_age_seconds=180, now=None):
    age = (time.time() if now is None else now) - float(observation["created_unix"])
    if not 0 <= age <= max_age_seconds:
        raise RuntimeError("Target measurement is stale or its timestamp is in the future")


def reference_binding(config):
    geometry = {"camera": config["camera"], "model": config["model"],
                "gripper": {key: config["gripper"][key] for key in
                            ("tcp_to_grasp_center_mm", "jaw_axis_tcp")}}
    return hashlib.sha256(json.dumps(geometry, sort_keys=True, allow_nan=False).encode()).hexdigest()


class AutomaticTable:
    def __init__(self, config, settings, output, *, execute=False, require_feedback=False):
        import numpy as np
        import open3d as o3d
        import calibration_motion as safety
        import one_click_pick as motion
        import pick_place_3d as vision
        import side_pick_execute as side
        import table_vision as evidence
        from diagnostics import stand_table_part as route
        from gripper_feedback import GripperFeedback
        self.np, self.o3d = np, o3d
        self.safety, self.motion, self.vision, self.side = safety, motion, vision, side
        self.evidence, self.route = evidence, route
        self.config, self.settings = copy.deepcopy(config), settings
        self.config["robot"]["table_placement_xy_mm"] = list(settings["placement_xy_mm"])
        self.output = Path(output)
        self.execute, self.require_feedback = execute, require_feedback
        self.output.mkdir(parents=True, exist_ok=True)
        self.robot = motion.FastRobot()
        self.feedback = GripperFeedback()
        self.report = dict(completed=False, motion_sent=False, actuator_write_attempted=False,
                           stage="created", run_created_unix=time.time(), create_home=False,
                           mode="execute" if execute else "read_only", events=[],
                           output=str(self.output), independent_tool_calibration=False)
        motion.REPORT = self.output / "run.json"
        self.plan = None
        self.last_view = None
        self.expected_state = None

    def event(self, stage, **details):
        item = dict(stage=stage, created_unix=time.time(), **details)
        self.report.update(stage=stage)
        self.report["events"].append(item)
        self.safety.save_json(self.output / "run.json", self.report)
        with (self.output / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
        print(stage, flush=True)

    def snapshot(self):
        state = self.robot.motion_snapshot()
        self.safety.check_motion_state(state)
        return state

    def check_installation(self):
        self.expected_state = self.snapshot()
        observation = self.vision.read_observation_pose()
        expected = self.config["robot"]["table_observation_pose"]
        if (observation.get("tool_id"), observation.get("user_id")) != (expected["tool_id"], expected["user_id"]):
            raise RuntimeError("Tool or user frame differs from the taught table overview")
        if self.np.max(self.np.abs(self.robot.call("get_curr_tool_offset")["tool_offset"])) > 1e-6:
            raise RuntimeError("Active TCP offset differs from calibrated flange frame")
        if self.robot.call("get_clsn_sensitivity")["sensitivityLevel"] not in (1, 2, 3, 4, 5):
            raise RuntimeError("Controller collision protection is disabled")
        feedback = self.feedback.snapshot()
        self.event("控制器与夹爪反馈检查", feedback=feedback)
        if self.require_feedback and not feedback.get("available"):
            raise RuntimeError("Real gripper feedback is required but not mapped: " + feedback.get("reason", ""))
        if feedback.get("holding") is True:
            raise RuntimeError("Gripper already holds a workpiece; do not open or begin a new pickup")

    def check_start(self):
        current = self.snapshot()
        if self.expected_state is not None:
            position, angle = self.safety.pose_error(current["actual_position"], self.expected_state["actual_position"])
            jump = self.np.max(self.np.abs(self.np.asarray(current["joint_actual_position"])
                                         - self.expected_state["joint_actual_position"]))
            if position > .5 or angle > .25 or jump > .25:
                raise RuntimeError("Arm moved outside the planned sequence; regenerate the route")
        observation = self.vision.read_observation_pose()
        taught = self.config["robot"]["table_observation_pose"]
        if (observation["tool_id"], observation["user_id"]) != (taught["tool_id"], taught["user_id"]):
            raise RuntimeError("Tool or user frame changed during execution")
        if self.np.max(self.np.abs(self.robot.call("get_curr_tool_offset")["tool_offset"])) > 1e-6:
            raise RuntimeError("TCP offset changed during execution")
        return current

    def holding_feedback(self, stage):
        feedback = self.feedback.snapshot()
        self.event(stage, feedback=feedback)
        if feedback.get("dropped") is True or feedback.get("holding") is False:
            raise RuntimeError("Gripper reports no held workpiece: " + stage)
        if self.require_feedback and not (feedback.get("available") and feedback.get("holding") is True):
            raise RuntimeError("Required physical holding feedback is unavailable: " + stage)
        return feedback

    def move_joint(self, joints, label, speed=15):
        if not self.execute:
            raise RuntimeError("Read-only mode forbids movement")
        self.check_start()
        self.report.update(motion_sent=True, actuator_write_attempted=True)
        self.event(label)
        self.motion.wait_joint(self.robot, joints, speed, label, self.report, speed * 2)
        self.expected_state = self.snapshot()

    def move_linear(self, pose, label, speed=10):
        if not self.execute:
            raise RuntimeError("Read-only mode forbids movement")
        self.check_start()
        self.report.update(motion_sent=True, actuator_write_attempted=True)
        self.event(label)
        self.motion.wait_linear(self.robot, pose, speed, label, self.report, speed * 3)
        self.expected_state = self.snapshot()

    def opening(self, value, label):
        from gripper_transport import close_connections
        from gripper_command import robot_profile
        from gripper_backend import GripperClient
        if not self.execute:
            raise RuntimeError("Read-only mode forbids gripper writes")
        self.check_start()
        self.report.update(actuator_write_attempted=True)
        self.event(label, opening_command_mm=value, command_is_measured_gap=False)
        host, profile = robot_profile()
        gripper = GripperClient(host, profile=profile)
        try:
            gripper.connect()
            gripper.set_opening_mm(value, settle_time=1)
        finally:
            close_connections()

    def overview(self):
        target = self.config["robot"]["table_observation_pose"]
        current = self.snapshot()
        position, angle = self.safety.pose_error(current["actual_position"], target["tcp_pose"])
        if position <= .5 and angle <= .5:
            self.event("已在高处总览位")
            return
        if not self.execute:
            raise RuntimeError("Read-only scan requires the arm to already be at the table overview")
        # A fixed workcell plane is used only for a return transfer. Every
        # actual pickup/place height is remeasured from the current scan.
        plane = self.settings["reference_table_plane"]
        normal = self.route.normalized_plane(plane)
        q0 = self.np.asarray(current["joint_actual_position"])
        q1 = self.np.asarray(target["joint_position_deg"])
        if self.np.max(self.np.abs(q1-q0)) > 100:
            raise RuntimeError("Current pose requires a separately taught transfer to the overview")
        checks = []
        for fraction in self.np.linspace(0, 1, max(15, int(self.np.max(self.np.abs(q1-q0)))+1)):
            q = q0*(1-fraction)+q1*fraction
            if abs(q[4]) > 119.9 or abs(q[2]) > 144:
                raise RuntimeError("Overview transfer exceeds checked joint range")
            tcp = self.vision.pose_to_matrix(self.robot.forward(q))
            center = self.vision.transform_points([self.config["gripper"]["tcp_to_grasp_center_mm"]], tcp)[0]
            clearances = [float(tcp[:3, 3]@normal[:3]+normal[3]), float(center@normal[:3]+normal[3])]
            if min(clearances) < 70:
                raise RuntimeError("Current pose has no checked high-clearance transfer to overview")
            checks.append(clearances)
        self.event("总览返回路线检查通过", sampled_clearances_mm=checks, full_collision_checked=False)
        self.move_joint(q1.tolist(), "返回高处总览位")

    def capture(self, name):
        from grasp_verification import capture_verification_isolated
        before = self.vision.read_observation_pose()
        result = capture_verification_isolated(self.output / name, self.config["camera"].get("sdk"))
        self.vision.verify_observation_unchanged(before, self.vision.read_observation_pose())
        # RGB verification uses the same calibrated optical coordinates as
        # native XYZ, although read_observation_pose only returns robot frames.
        result["observation"] = dict(before, camera_frame=self.config["camera"]["point_cloud_frame"])
        self.last_view = result
        self.event("保存复核画面：" + name, view=result)
        return result

    def cloud(self, name):
        cloud = self.vision.capture_point_cloud(self.config)
        observation = self.config["_observation"]
        transform = self.vision.camera_transform(self.config, observation)
        self.o3d.io.write_point_cloud(str(self.output / (name + ".ply")), cloud)
        return cloud, observation, transform

    def current_plane(self, cloud, transform, grasp_xy=None):
        plane = self.evidence.estimate_table_plane(cloud, transform, grasp_xy)
        if not plane.get("accepted"):
            raise RuntimeError("Current tabletop not established: " + plane["reason"])
        reference = self.settings["reference_table_plane"]
        anchor = grasp_xy if grasp_xy is not None else self.settings["table_anchor_xy_mm"]
        change = abs(self.route.table_height_at(plane, anchor)-self.route.table_height_at(reference, anchor))
        if change > self.settings["max_table_height_change_mm"]:
            raise RuntimeError("Table height changed beyond the validated workcell tolerance")
        return plane

    def inspect(self):
        """Fresh scan, pose, directional evidence; no saved target coordinates."""
        for attempt in range(self.settings["max_rescans"]):
            self.event("高位重新扫描零件", scan_attempt=attempt+1)
            mesh, model, scene, objects, support, clusters, matches, _ = self.vision.scan_scene(self.config)
            folder = self.output / f"scan_{attempt+1}"
            folder.mkdir()
            self.vision.save_scan(folder, scene, objects, model, clusters, matches)
            observation = self.config["_observation"]
            transform = self.vision.camera_transform(self.config, observation)
            points = self.vision.transform_points(self.np.asarray(scene.points), transform)
            plane = self.current_plane(scene, transform, self.settings["table_anchor_xy_mm"])
            eligible = []
            for match in matches:
                base_model = transform @ self.np.asarray(match.camera_T_model)
                center = self.vision.transform_points([[0, 0, self.side.shaft_middle_model_z(self.config)]], base_model)[0]
                if inside_table(center[:2], self.settings) and self.np.isfinite(match.score):
                    eligible.append((match, base_model, center))
            if not eligible:
                self.event("本次没有工作区内有效候选", candidate_count=len(matches))
                continue
            match, base_model, center = max(eligible, key=lambda item: item[0].score)
            target = dict(best=asdict(match), observation=observation, base_T_model=base_model.tolist(),
                          base_T_camera=transform.tolist(), shaft_grasp_center_base_mm=center.tolist(),
                          source_scan=str(folder), created_unix=time.time())
            normal = self.route.normalized_plane(plane)[:3]
            alignment = float(base_model[:3, 2]@normal)
            if alignment > self.np.cos(self.np.deg2rad(8)):
                upright = self.evidence.verify_upright(points, base_model, plane, self.config)
                self.event("检查零件是否已竖立", upright=upright)
                if upright.get("accepted"):
                    self.report.update(already_upright=True, upright_verification=upright, target=target)
                    return target, plane, True
            if match.score < self.settings["minimum_match_score"] or match.clearance_mm < 10:
                self.event("最高分候选未通过抓取门槛", score=match.score, clearance_mm=match.clearance_mm)
                continue
            direction = self.evidence.flange_direction_evidence(points, base_model, plane, self.config)
            if direction.get("status") == "reversed":
                # A reversed result is corrected only when the complete CAD
                # support and independent wide-end evidence both agree.
                vertices = self.np.asarray(mesh.vertices)
                length = float(vertices[:, 2].max())
                flip = self.np.eye(4)
                flip[:3, :3] = self.np.diag([1., -1., -1.])
                flip[2, 3] = length
                corrected = base_model @ flip
                camera_model = self.np.linalg.inv(transform) @ corrected
                corrected_direction = self.evidence.flange_direction_evidence(points, corrected, plane, self.config)
                local = self.vision.transform_points(points, self.np.linalg.inv(corrected))
                radius = float(self.np.max(self.np.linalg.norm(vertices[:, :2], axis=1)))
                nearby = ((self.np.linalg.norm(local[:, :2], axis=1) < radius+3)
                          & (local[:, 2] > -3) & (local[:, 2] < length+3)
                          & (points@normal+self.route.normalized_plane(plane)[3] > 1.5))
                instance = self.o3d.geometry.PointCloud(self.o3d.utility.Vector3dVector(
                    self.vision.transform_points(points[nearby], self.np.linalg.inv(transform))))
                fit = self.vision.validate_model_support(model, instance, camera_model, self.config)
                if corrected_direction.get("accepted") and fit.get("accepted"):
                    base_model = corrected
                    center = self.vision.transform_points([[0, 0, self.side.shaft_middle_model_z(self.config)]], base_model)[0]
                    target.update(base_T_model=base_model.tolist(), shaft_grasp_center_base_mm=center.tolist(), direction_corrected=True)
                    direction = corrected_direction
                    alignment = float(base_model[:3, 2]@normal)
            self.event("核对大法兰方向", direction=direction)
            if not direction.get("accepted") or abs(alignment) > self.np.sin(self.np.deg2rad(20)):
                continue
            # The selected target is removed before placement; every other
            # measured object in the placement footprint must remain clear.
            placement = self.np.asarray(self.settings["placement_xy_mm"])
            local = self.vision.transform_points(points, self.np.linalg.inv(base_model))
            dimensions = self.np.asarray(mesh.vertices)
            radius = float(self.np.max(self.np.linalg.norm(dimensions[:, :2], axis=1)))
            length = float(dimensions[:, 2].max())
            target_volume = ((self.np.linalg.norm(local[:, :2], axis=1) < radius+4)
                             & (local[:, 2] > -4) & (local[:, 2] < length+4))
            elevated = points@normal+self.route.normalized_plane(plane)[3] > 5
            footprint = self.np.linalg.norm(points[:, :2]-placement, axis=1) < radius+12
            foreign = int(self.np.count_nonzero(elevated & footprint & ~target_volume))
            self.event("检查台面放置区域", placement_xy_mm=placement.tolist(), foreign_points=foreign)
            if foreign > 20:
                raise RuntimeError("Measured objects occupy the placement footprint")
            self.safety.save_json(self.output / "target.json", target)
            self.safety.save_json(self.output / "table_plane.json", plane)
            self.report.update(target=target, table_plane=plane)
            return target, plane, False
        raise RuntimeError("No fresh, direction-verified lying tube after bounded rescans")

    def appearance(self, view, held=False):
        camera = self.vision.camera_transform(self.config, view["observation"])
        if held:
            tcp = self.vision.pose_to_matrix(view["observation"]["tcp_pose"])
            model = tcp @ self.np.asarray(self.plan["held_model"])
            reference = self.settings["empty_closed_reference"]
        else:
            model = self.np.asarray(self.report["target"]["base_T_model"])
            reference = self.settings["empty_open_reference"]
        camera_model = self.np.linalg.inv(camera) @ model
        result = self.evidence.verify_rgb_presence(view["color_path"], camera_model, self.config,
                                                  empty_image=ROOT / reference)
        self.event("检查夹持外观" if held else "检查下爪对齐", appearance=result)
        return result, camera_model

    def execute_cycle(self, target, plane):
        require_fresh(target["observation"], self.settings["max_target_age_seconds"])
        self.plan = self.route.plan(self.robot, target, plane, self.config)
        self.safety.save_json(self.output / "plan.json", self.plan)
        self.event("新路线规划完成", route_checks=self.plan["route_checks"])
        if not self.execute:
            self.report.update(planned=True)
            return
        require_fresh(target["observation"], self.settings["max_target_age_seconds"])
        self.route.verify_plan_binding(self.plan, self.config)
        self.vision.verify_observation_unchanged(target["observation"], self.vision.read_observation_pose())
        self.report["plan"] = self.plan
        self.opening(20, "张开夹爪")
        self.move_joint(self.plan["joints"]["approach"], "对准管身中部")
        self.move_linear(self.plan["poses"]["prepick"], "接近管身中部", 20)
        pre = self.capture("prepick")
        appearance, _ = self.appearance(pre)
        if not appearance.get("accepted"):
            raise RuntimeError("Actual tube is not aligned between the jaw tips; no closure sent")
        self.move_linear(self.plan["poses"]["pick"], "进入夹持位置", 5)
        self.opening(8.65, "闭合夹爪")
        self.report.update(gripper_closed=True, physical_grasp_confirmed=False)
        initial_feedback = self.holding_feedback("闭合后检查真实夹持反馈")
        if initial_feedback.get("available"):
            stable = self.feedback.wait_for_hold(timeout=2, stable_samples=3)
            self.event("闭合后连续反馈", stable_feedback=stable)
            if not stable.get("confirmed"):
                raise RuntimeError("Physical grip feedback is not stable before first lift")
        lift = list(self.plan["poses"]["lift"])
        first_lift = list(lift)
        first_lift[:3] = (self.np.asarray(lift[:3])-self.route.normalized_plane(plane)[:3]*30).tolist()
        self.move_linear(first_lift, "首次抬升复核", 25)
        first = self.capture("lift_60mm")
        appearance1, camera_model = self.appearance(first, held=True)
        feedback1 = self.holding_feedback("首次抬升反馈")
        if feedback1.get("dropped") or feedback1.get("holding") is False or not appearance1.get("accepted"):
            self.event("首次夹持证据不足", feedback=feedback1, appearance=appearance1)
            raise RuntimeError("First lift does not establish holding; retain jaws and stop")
        self.move_linear(lift, "继续抬升核对随动", 25)
        second = self.capture("lift_90mm")
        appearance2, _ = self.appearance(second, held=True)
        feedback2 = self.holding_feedback("再次抬升反馈")
        if feedback2.get("available") and feedback2.get("holding") is True:
            stable = self.feedback.wait_for_hold(timeout=2, stable_samples=3)
            self.event("核对连续夹持反馈", stable_feedback=stable)
            if not stable.get("confirmed"):
                raise RuntimeError("Physical grip feedback is not stable")
            feedback2 = stable["samples"][-1]
        follow = self.evidence.verify_follow_motion(first["color_path"], second["color_path"], camera_model,
                      self.config, empty_image=ROOT / self.settings["empty_closed_reference"])
        decision = combine_grasp_evidence(feedback2, [appearance1, appearance2], follow)
        self.event("确认实物夹持", feedback=feedback2, follow_motion=follow, decision=decision)
        self.report["grasp_verification"] = decision
        if not decision["confirmed"]:
            raise RuntimeError(decision["reason"])
        self.report.update(physical_grasp_confirmed=True)
        self.route.verify_plan_binding(self.plan, self.config)
        if "transfer" in self.plan["poses"]:
            self.move_linear(self.plan["poses"]["transfer"], "持件移到台面中部上方", 20)
        for name in self.plan["poses"]:
            if name.startswith("turn_") or name == "turned":
                self.move_joint(self.plan["joints"][name], "绕夹持中心翻正："+name)
        rotated = self.capture("turned")
        rotated_appearance, _ = self.appearance(rotated, held=True)
        feedback = self.holding_feedback("翻正后反馈")
        if feedback.get("dropped") or feedback.get("holding") is False or not rotated_appearance.get("accepted"):
            raise RuntimeError("Held object lost or uncertain after rotation; placement stopped")
        self.move_linear(self.plan["poses"]["above_place"], "移动到竖放上方", 30)
        self.holding_feedback("下降前检查夹持")
        self.move_linear(self.plan["poses"]["place"], "大法兰朝下放到台面", 5)
        self.holding_feedback("松爪前检查夹持")
        self.opening(20, "松开零件")
        self.report.update(gripper_closed=False, released=True)
        self.move_linear(self.plan["poses"]["retreat"], "退离零件", 20)
        self.move_joint(self.plan["start_state"]["joint_actual_position"], "返回高处总览位")
        self.capture("after_place")
        for attempt in range(2):
            cloud, _, transform = self.cloud(f"after_place_{attempt+1}")
            current_plane = self.current_plane(cloud, transform, target["shaft_grasp_center_base_mm"][:2])
            points = self.vision.transform_points(self.np.asarray(cloud.points), transform)
            result = self.evidence.verify_upright(points, self.plan["placed_model"], current_plane, self.config)
            self.event("检查竖放结果", verification=result, attempt=attempt+1)
            self.report["upright_verification"] = result
            if result.get("accepted"):
                self.report.update(completed=True)
                return
        raise RuntimeError("Released workpiece is not verified upright; no blind regrasp attempted")

    def run(self, plan_only=False):
        from controller_lease import controller_lease
        from gripper_transport import close_connections
        try:
            with controller_lease(8):
                self.safety.ensure_exclusive_client()
                self.check_installation()
                initial = self.capture("startup")
                empty = self.evidence.verify_empty_jaws(initial["color_path"], self.config,
                    open_reference=ROOT / self.settings["empty_open_reference"],
                    closed_reference=ROOT / self.settings["empty_closed_reference"])
                self.event("启动前确认空爪", empty_jaws=empty)
                if self.execute and not empty.get("accepted"):
                    raise RuntimeError("Initial empty jaws are not established; no opening or transfer sent")
                self.overview()
                self.capture("overview")
                target, plane, upright = self.inspect()
                if upright:
                    self.capture("already_upright")
                    self.report.update(completed=True)
                    self.event("零件已竖立，本次无需抓取")
                elif plan_only or self.execute:
                    self.execute_cycle(target, plane)
                    self.event("自动流程完成" if self.report["completed"] else "只读规划完成")
                else:
                    self.event("只读识别完成")
                self.report["final_state"] = self.snapshot()
        except BaseException as exc:
            self.report.update(completed=False, error=str(exc))
            if self.report["actuator_write_attempted"]:
                try:
                    # Stop only; never reset controller faults or release an
                    # object whose holding state is uncertain.
                    with controller_lease(8):
                        self.safety.stop_and_verify(self.robot)
                except BaseException as stop_error:
                    self.report["stop_error"] = str(stop_error)
            self.event("流程停止，保留现场状态", error=str(exc))
            raise
        finally:
            close_connections()
            self.safety.save_json(self.output / "run.json", self.report)
            print("记录：" + str(self.output / "run.json"), flush=True)
        return self.report


def check_settings(config, settings):
    import numpy as np
    if settings.get("schema_version") != 1:
        raise ValueError("Unsupported automation settings")
    bounds = np.asarray(settings["table_xy_bounds_mm"], float)
    if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or not np.all(bounds[0] < bounds[1]):
        raise ValueError("Invalid table work area")
    for name in ("empty_open_reference", "empty_closed_reference"):
        path = ROOT / settings[name]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != settings[name+"_sha256"]:
            raise ValueError("Empty-jaw reference missing or changed: " + name)
    if settings["reference_geometry_binding"] != reference_binding(config):
        raise ValueError("Reference images no longer match camera/tool/model installation")
    if not inside_table(settings["placement_xy_mm"], settings):
        raise ValueError("Placement center lies outside the measured work area")
    if not 1 <= settings["max_rescans"] <= 3:
        raise ValueError("Rescan count out of bounds")
    if not .78 <= settings["minimum_match_score"] <= 1:
        raise ValueError("Match score cannot weaken the validated threshold")
    return {"offline_check": "passed", "home_modified": False,
            "geometry_independently_measured": config["gripper"].get("geometry_validation", {}).get("independent_metric_calibration", False)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Offline configuration check (default)")
    mode.add_argument("--observe", action="store_true", help="Fresh read-only scene inspection")
    mode.add_argument("--plan-only", action="store_true", help="Fresh scan and read-only controller IK/FK")
    mode.add_argument("--execute", action="store_true", help="Run one automatic physical cycle")
    parser.add_argument("--require-hardware-feedback", action="store_true")
    parser.add_argument("--settings", type=Path, default=ROOT / "table_automation.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = read_json(ROOT / "pick_place_config.json")
    settings = read_json(args.settings)
    print(json.dumps(check_settings(config, settings), ensure_ascii=False), flush=True)
    if not (args.execute or args.observe or args.plan_only):
        return 0
    output = args.output or ROOT / "pick_results" / "automatic_table" / time.strftime("run_%Y%m%d_%H%M%S")
    # A run directory is an audit record, never an executable resume token.
    if output.exists():
        raise RuntimeError("Use a new output directory; automatic runs cannot resume old motion")
    runner = AutomaticTable(config, settings, output, execute=args.execute,
                            require_feedback=args.require_hardware_feedback)
    runner.run(plan_only=args.plan_only)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("自动流程未完成：" + str(exc), file=sys.stderr, flush=True)
        raise SystemExit(1)
