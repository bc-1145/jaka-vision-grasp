"""BAT-first bin photo, native-depth planning, and supervised grasp attempt.

Flange tilt is measured from base vertically downward and is restricted to
30,45,60 degrees. No automatic release, Home return or table placement.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time

from auto_table_upright import AutomaticTable, ROOT, check_settings, combine_grasp_evidence, read_json, require_fresh


def empty_inspection_retreat(robot, config, observation, state):
    """Read-only IK/FK validation of a short upward, unchanged-jaw inspection."""
    import numpy as np
    import calibration_motion as safety
    import pick_place_3d as vision
    safety.check_motion_state(state)
    error, angle = safety.pose_error(state["actual_position"], observation["tcp_pose"])
    if error > .5 or angle > .25:
        raise RuntimeError("Inspection observation no longer matches robot")
    start = np.asarray(state["actual_position"], float)
    # RGB capture records the wrist pose but has no point-cloud frame tag.
    # This transform only defines the calibrated optical viewing direction.
    camera = vision.camera_transform(config, dict(observation,
                                      camera_frame=config["camera"]["point_cloud_frame"]))
    delta = -camera[:3, 2]*40
    if delta[2] < 35 or np.linalg.norm(delta[:2]) > 10:
        raise RuntimeError("Empty inspection retreat is not a bounded upward movement")
    previous = np.asarray(state["joint_actual_position"], float)
    checks = []
    for fraction in np.linspace(0, 1, 15):
        pose = start.copy()
        pose[:3] += delta*fraction
        joints = np.asarray(robot.inverse(pose, previous), float)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise RuntimeError("Invalid inspection IK")
        if abs(joints[4]) > 119.9 or abs(joints[2]) > 144 or np.max(np.abs(joints-previous)) > 10:
            raise RuntimeError("Inspection IK branch or joint range exceeded")
        translation, rotation = safety.pose_error(robot.forward(joints), pose)
        if translation > .25 or rotation > .2:
            raise RuntimeError("Inspection IK/FK mismatch")
        checks.append(dict(pose=pose.tolist(), joints=joints.tolist()))
        previous = joints
    return dict(pose=checks[-1]["pose"], samples=checks, delta_mm=delta.tolist(),
                gripper_unchanged=True, full_collision_checked=False)


class AutomaticBin(AutomaticTable):
    def compare_verified_empty_view(self, view, source_path):
        """Reobserve a measured empty bin view at the same physical wrist pose."""
        import cv2
        source = read_json(source_path)
        events = source["events"]
        proofs = [i for i, item in enumerate(events)
                  if item.get("empty_jaws", {}).get("status") == "visually_empty_from_motion"
                  and item["empty_jaws"].get("accepted") is True]
        if not proofs:
            raise RuntimeError("Empty-view source has no accepted motion evidence")
        proof_index = proofs[-1]
        if any("opening_command_mm" in item for item in events[proof_index+1:]):
            raise RuntimeError("Source operated the jaws after the empty proof")
        reference = next(item["view"] for item in reversed(events[:proof_index]) if "view" in item)
        self.vision.verify_observation_unchanged(reference["observation"], view["observation"])
        if source.get("plan", {}).get("tool_geometry") != {
            key: self.config["gripper"][key] for key in ("tcp_to_grasp_center_mm", "jaw_axis_tcp")}:
            raise RuntimeError("Empty-view source tool geometry differs")
        source_camera = self.np.asarray(source["plan"]["base_T_camera"])
        current_camera = self.vision.camera_transform(self.config, view["observation"])
        if self.np.max(self.np.abs(source_camera-current_camera)) > .5:
            raise RuntimeError("Empty-view hand-eye frame or physical pose differs")
        previous = cv2.imread(reference["color_path"], 0)
        current = cv2.imread(view["color_path"], 0)
        opened = cv2.imread(str(ROOT / self.settings["empty_open_reference"]), 0)
        closed = cv2.imread(str(ROOT / self.settings["empty_closed_reference"]), 0)
        if any(item is None for item in (previous, current, opened, closed)) or previous.shape != current.shape:
            raise RuntimeError("Empty-view image unavailable")
        visible, fixed = self.evidence._empty_motion_masks(self.config, opened, closed)
        difference = self.np.abs(previous.astype(float)-current.astype(float))
        novel = ((difference > 40) & visible).astype(self.np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(novel, 8)
        component = int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0
        metrics = dict(source_run=str(source_path), current_view=view["color_path"],
                       change_25_fraction=float(self.np.mean(difference[visible] > 25)),
                       difference_95_px=float(self.np.percentile(difference[visible], 95)),
                       fixed_tool_error=float(self.np.median(difference[fixed])),
                       largest_novel_component_pixels=component, hidden_volume_observed=False)
        accepted = (metrics["change_25_fraction"] <= .02 and metrics["difference_95_px"] <= 20
                    and metrics["fixed_tool_error"] <= 10 and component < 120)
        return dict(accepted=accepted, status="reobserved_verified_empty" if accepted else "unknown",
                    reason="Fresh same-pose image compared with measured empty-view evidence", **metrics)

    def assert_at_bin(self):
        import arm_control
        state = self.snapshot()
        target = arm_control.load_pose("loading")
        position, angle = self.safety.pose_error(state["actual_position"], target)
        if position > 3 or angle > 1:
            # Only the two exact optical-axis inspection retreats are allowed;
            # this is not a general workspace region or a table observation.
            camera = self.vision.pose_to_matrix(target) @ self.np.asarray(self.config["camera"]["tcp_T_camera"])
            allowed = []
            for distance in (40, 80):
                raised = self.np.asarray(target, float).copy()
                raised[:3] -= camera[:3, 2]*distance
                shift, turn = self.safety.pose_error(state["actual_position"], raised)
                allowed.append(shift <= 3 and turn <= 1)
            if not any(allowed):
                raise RuntimeError("Not at the saved BAT loading/photo position or its checked inspection retreat")
            self.config["_bin_inspection_retreat_completed"] = True
        self.event("确认已在零件框 BAT 拍照位置或其向上观察位", actual=state)

    def run_batch_first(self):
        # BAT owns the controller lease; do not hold our lease while it runs.
        start = time.time()
        result = subprocess.run(["cmd", "/d", "/c", str(ROOT / "启动入口" / "一键取.bat")],
                                cwd=str(ROOT), stdin=subprocess.DEVNULL, timeout=180)
        log = read_json(ROOT / "logs" / "one_click_loading.json")
        if not log.get("completed") or log.get("created_unix", 0) < start-1:
            raise RuntimeError("Existing loading BAT did not complete this transfer")
        # Windows timeout /t returns failure with redirected stdin even after
        # LOADING OK. Verify controller pose and the fresh motion report.
        self.event("优先调用现有一键取 BAT", batch_exit_code=result.returncode,
                   batch_report=log, footer_exit_is_not_motion_evidence=True)

    def scan_and_plan(self):
        from bin_grasp_plan import plan_bin_grasp
        from bin_support import prepare_bin_reference, bin_flange_direction_evidence
        from bin_target_report import build_target_report
        for attempt in range(2):
            self.event("在零件框上方采集原生深度", attempt=attempt+1)
            captured = getattr(self, "pre_captured", None)
            if captured is None:
                mesh, model, scene, objects, support, clusters, matches, camera_plane = self.vision.scan_scene(self.config)
            else:
                self.pre_captured = None
                cloud_path, observation_path = captured
                observation = read_json(observation_path)
                require_fresh(observation, 180)
                self.vision.verify_observation_unchanged(observation, self.vision.read_observation_pose())
                self.config["_observation"] = observation
                mesh = self.vision.load_fcstd_mesh(ROOT / self.config["model"]["path"])
                self.o3d.utility.random.seed(7)
                model = mesh.sample_points_uniformly(int(self.config["matching"]["model_sample_points"]))
                scene = self.vision.crop_cloud(self.o3d.io.read_point_cloud(str(cloud_path)), self.config["camera"]["roi"])
                objects, camera_plane, support = self.vision.remove_support_plane(scene, self.config)
                objects = self.vision.exclude_camera_geometry(objects, self.config)
                clusters = self.vision.cluster_objects(objects, self.config)
                matches = self.vision.find_matches(model, scene, clusters, self.config)
            observation = self.config["_observation"]
            transform = self.vision.camera_transform(self.config, observation)
            output = self.output / f"scan_{attempt+1}"
            output.mkdir()
            self.vision.save_scan(output, scene, objects, model, clusters, matches)
            reference = prepare_bin_reference(scene, transform, self.config, matches)
            qualified = []
            evidence = []
            for match in matches:
                base_model = transform @ self.np.asarray(match.camera_T_model)
                direction = bin_flange_direction_evidence(reference, base_model, self.config)
                evidence.append(dict(cluster_index=match.cluster_index, match_score=match.score, direction=direction))
                if direction.get("accepted"):
                    qualified.append(match)
            self.event("核对零件大头与轴向", candidates=evidence, bin_reference=reference["report"],
                       depth_units="mm", point_cloud_frame=self.config["camera"]["point_cloud_frame"])
            if not qualified:
                continue
            preferred = max(qualified, key=lambda item: item.score)
            direction = next(item["direction"] for item in evidence if item["cluster_index"] == preferred.cluster_index)
            geometry = build_target_report(self.config, observation, preferred, direction_evidence=direction,
                                          image_path=(self.last_view or {}).get("color_path"))
            self.safety.save_json(self.output / "target.json", geometry)
            print(json.dumps(geometry, ensure_ascii=False, allow_nan=False), flush=True)
            try:
                self.plan = plan_bin_grasp(self.robot, self.config, observation, qualified, scene)
            except Exception as exc:
                geometry["planning_error"] = str(exc)
                self.safety.save_json(self.output / "target.json", geometry)
                raise
            match = next(item for item in matches if item.cluster_index == self.plan["selected_cluster_index"])
            self.report["target"] = dict(base_T_model=self.plan["base_T_model"], observation=observation,
                                          best=asdict(match), depth_units="mm")
            rotation = self.vision.pose_to_matrix(observation["tcp_pose"])[:3, :3]
            current_tilt = float(self.np.degrees(self.np.arccos(self.np.clip(-rotation[2, 2], -1, 1))))
            self.report.update(plan=self.plan, selected_tilt_deg=self.plan["tilt_deg"],
                               current_tilt_deg=current_tilt,
                               flange_adjustment_required=abs(current_tilt-self.plan["tilt_deg"]) > 1,
                               depth_units="mm")
            self.safety.save_json(self.output / "plan.json", self.plan)
            direction = next(item["direction"] for item in evidence if item["cluster_index"] == match.cluster_index)
            geometry = build_target_report(self.config, observation, match, self.plan,
                                          direction_evidence=direction, image_path=(self.last_view or {}).get("color_path"))
            geometry["selected_flange_tilt_deg"] = self.plan["tilt_deg"]
            self.report["target_json"] = geometry
            self.safety.save_json(self.output / "target.json", geometry)
            print(json.dumps(geometry, ensure_ascii=False, allow_nan=False), flush=True)
            self.event("已选定方便抓取的目标与法兰角度", selected_cluster=match.cluster_index,
                       score=match.score, tilt_deg=self.plan["tilt_deg"], current_tilt_deg=current_tilt,
                       large_end_base_mm=self.plan["large_end_base_mm"], grasp_base_mm=self.plan["grasp_base"],
                       shaft_axis_base=self.plan["shaft_axis_base"], route_checks=self.plan["route_checks"])
            self.annotate(match)
            return
        raise RuntimeError("No bin target has independent wide-end direction evidence")

    def annotate(self, match):
        import cv2
        from pose_overlay import PoseOverlay
        view = self.last_view
        if view is None:
            raise RuntimeError("No acquisition RGB image to annotate")
        self.vision.verify_observation_unchanged(view["observation"], self.plan["observation"])
        overlay = PoseOverlay(self.output, ROOT / "pick_place_config.json")
        overlay._load_assets()
        overlay.matches = [asdict(match)]
        image = cv2.imread(view["color_path"])
        overlay._axes(image)
        caption = f"FLANGE TILT {self.plan['tilt_deg']} deg | native XYZ mm | score {match.score:.3f}"
        cv2.putText(image, caption, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)
        cv2.imwrite(str(self.output / "selected_target.png"), image)

    def attempt(self):
        require_fresh(self.plan["observation"], 180)
        self.vision.verify_observation_unchanged(self.plan["observation"], self.vision.read_observation_pose())
        self.opening(20, "张开夹爪")
        distance, _ = self.safety.pose_error(self.plan["poses"]["retreat"], self.plan["start_state"]["actual_position"])
        if distance > .5:
            self.move_linear(self.plan["poses"]["retreat"], "先从拍照位退出 40 mm", 20)
        else:
            self.event("已在实测向上观察位，无需重复抬升")
        self.move_joint(self.plan["joints"]["approach"], "在高位调整法兰到选定角度", 15)
        self.move_linear(self.plan["poses"]["prepick"], "保持高度水平对准零件中部", 15)
        view = self.capture("prepick")
        appearance, _ = self.appearance(view)
        if not appearance.get("accepted"):
            raise RuntimeError("Actual target alignment is not established; no closure sent")
        self.move_linear(self.plan["poses"]["pick"], "按原生深度沿基座竖直下降到管身中部", 5)
        self.opening(8.65, "闭合到用户指定的 8.65 指令")
        self.report.update(gripper_closed=True, physical_grasp_confirmed=False)
        feedback = self.holding_feedback("闭合后夹持反馈")
        if feedback.get("available"):
            stable = self.feedback.wait_for_hold(timeout=2, stable_samples=3)
            self.event("连续夹持反馈", stable_feedback=stable)
            if not stable.get("confirmed"):
                raise RuntimeError("Physical grip feedback is not stable before withdrawal")
        self.move_linear(self.plan["poses"]["lift_first"], "竖直抬起 40 mm", 10)
        first = self.capture("withdraw_40mm")
        appearance1, camera_model = self.appearance(first, held=True)
        self.holding_feedback("第一次退出夹持检查")
        if not appearance1.get("accepted"):
            raise RuntimeError("No visible held workpiece after first withdrawal")
        self.move_linear(self.plan["poses"]["lift"], "继续竖直抬起至 70 mm 核对随动", 10)
        second = self.capture("withdraw_70mm")
        appearance2, _ = self.appearance(second, held=True)
        feedback = self.holding_feedback("第二次退出夹持检查")
        follow = self.evidence.verify_follow_motion(first["color_path"], second["color_path"], camera_model,
                    self.config, empty_image=ROOT / self.settings["empty_closed_reference"])
        decision = combine_grasp_evidence(feedback, [appearance1, appearance2], follow)
        self.event("判定零件是否确实随夹爪退出", follow_motion=follow, decision=decision)
        if not decision["confirmed"]:
            raise RuntimeError(decision["reason"])
        self.report.update(completed=True, physical_grasp_confirmed=True, grasp_verification=decision)
        self.event("已确认从零件框夹起，保持夹持")

    def run_bin(self, already_at_bin=False, empty_reference_run=None):
        from controller_lease import controller_lease
        from gripper_transport import close_connections
        try:
            if self.execute and not already_at_bin:
                self.report.update(actuator_write_attempted=True)
                self.run_batch_first()
            with controller_lease(8):
                self.safety.ensure_exclusive_client()
                self.check_installation()
                self.assert_at_bin()
                view = self.capture("bin_overview")
                empty = self.evidence.verify_empty_jaws(view["color_path"], self.config,
                         open_reference=ROOT / self.settings["empty_open_reference"],
                         closed_reference=ROOT / self.settings["empty_closed_reference"])
                self.event("下爪前空爪检查", empty_jaws=empty)
                if self.execute and not empty.get("accepted") and empty_reference_run is not None:
                    empty = self.compare_verified_empty_view(view, empty_reference_run)
                    self.event("新拍画面复核已实测空爪状态", empty_jaws=empty)
                if self.execute and not empty.get("accepted"):
                    inspection = empty_inspection_retreat(self.robot, self.config,
                                                        view["observation"], self.check_start())
                    self.event("保持夹爪原状，检查向上拍照路线", inspection=inspection)
                    self.move_linear(inspection["pose"], "向上退 40 mm 观察料框与空爪", 15)
                    after = self.capture("bin_overview_retreat")
                    empty = self.evidence.verify_empty_from_motion(view["color_path"], after["color_path"],
                             self.config, open_reference=ROOT / self.settings["empty_open_reference"],
                             closed_reference=ROOT / self.settings["empty_closed_reference"])
                    self.event("通过实测位移核对空爪", empty_jaws=empty)
                    if not empty.get("accepted"):
                        raise RuntimeError("Initial empty jaws remain uncertain after upward inspection; no opening sent")
                    self.config["_bin_inspection_retreat_completed"] = True
                self.scan_and_plan()
                if self.execute:
                    self.attempt()
                else:
                    self.event("零件框拍照和只读规划完成")
                self.report["final_state"] = self.snapshot()
                self.report["final_state_created_unix"] = time.time()
        except BaseException as exc:
            self.report.update(completed=False, error=str(exc))
            if self.report["actuator_write_attempted"]:
                try:
                    with controller_lease(8):
                        self.safety.stop_and_verify(self.robot)
                except BaseException as stopping:
                    self.report["stop_error"] = str(stopping)
            self.event("零件框尝试停止，保留夹爪状态", error=str(exc))
            try:
                with controller_lease(8):
                    self.report["final_state"] = self.snapshot()
                    self.report["final_state_created_unix"] = time.time()
            except Exception as snapshot_error:
                self.report["final_state_error"] = str(snapshot_error)
            raise
        finally:
            close_connections()
            self.safety.save_json(self.output / "run.json", self.report)
            print("记录：" + str(self.output / "run.json"), flush=True)
        return self.report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Run BAT and attempt physical grasp")
    parser.add_argument("--already-at-bin", action="store_true", help="Existing BAT was already run; verify its target before continuing")
    parser.add_argument("--empty-reference-run", type=Path, help="Reobserve an accepted same-pose empty-motion proof")
    parser.add_argument("--captured-cloud", type=Path, help="Fresh native optical XYZ PLY; requires its actual acquisition observation")
    parser.add_argument("--captured-observation", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config, settings = read_json(ROOT / "pick_place_config.json"), read_json(ROOT / "table_automation.json")
    print(json.dumps(check_settings(config, settings), ensure_ascii=False), flush=True)
    output = args.output or ROOT / "pick_results" / "bin_grasp" / time.strftime("run_%Y%m%d_%H%M%S")
    if output.exists():
        raise RuntimeError("Use a new run directory")
    if bool(args.captured_cloud) != bool(args.captured_observation):
        parser.error("Captured cloud and observation must be supplied together")
    workflow = AutomaticBin(config, settings, output, execute=args.execute)
    if args.captured_cloud:
        workflow.pre_captured = (args.captured_cloud, args.captured_observation)
    workflow.run_bin(args.already_at_bin, args.empty_reference_run)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("零件框流程未完成：" + str(exc), file=sys.stderr, flush=True)
        raise SystemExit(1)
