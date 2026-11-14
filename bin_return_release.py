"""Return one freshly verified bin grasp through the existing Home BAT, then release.

Default mode checks readiness without actuator writes. --execute is required.
No Home point is created or changed. A consumed/uncertain return token cannot
be replayed, and neither the BAT nor the gripper write is automatically retried.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import arm_control
import calibration_motion as safety
from controller_lease import controller_lease
from gripper_feedback import GripperFeedback
from gripper_transport import close_connections
import side_pick_execute as side

ROOT = Path(__file__).resolve().parent
BATCH = ROOT / "启动入口" / "一键回Home.bat"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def confirmed_source(source, max_age_seconds=180, *, now=None):
    if source.get("physical_grasp_confirmed") is not True:
        raise RuntimeError("Source run has no confirmed physical grasp")
    if source.get("released") or source.get("gripper_closed") is False or source.get("error"):
        raise RuntimeError("Source run was released, failed, or has uncertain holding")
    state = source.get("final_state") or source.get("actual")
    if not isinstance(state, dict):
        raise RuntimeError("Source run has no final measured robot state")
    stamp = (state.get("created_unix") or source.get("final_state_created_unix")
             or source.get("actual_created_unix"))
    if not isinstance(stamp, (int, float)) or not np.isfinite(stamp):
        raise RuntimeError("Final measured robot state has no explicit sample timestamp")
    age = (time.time() if now is None else now) - stamp
    if not 0 <= age <= max_age_seconds:
        raise RuntimeError("Final grasp measurement is stale or future-dated")
    safety.vector(state.get("actual_position"))
    safety.vector(state.get("joint_actual_position"))
    return state, stamp


def require_same_state(current, expected):
    safety.check_motion_state(current)
    position, angle = safety.pose_error(current["actual_position"], expected["actual_position"])
    joint_error = float(np.max(np.abs(safety.vector(current["joint_actual_position"])
                                     - safety.vector(expected["joint_actual_position"]))))
    if position > .5 or angle > .5 or joint_error > .2:
        raise RuntimeError("Robot moved since the source run's final grasp measurement")


def run_home_batch(log_path, timeout=180):
    """Run exactly once, without any parent controller lease held."""
    if not BATCH.is_file():
        raise RuntimeError("Existing Home BAT is missing")
    with Path(log_path).open("wb") as log:
        process = subprocess.Popen(["cmd.exe", "/d", "/c", str(BATCH)], cwd=str(ROOT),
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # End the whole launcher tree before reacquiring command ownership.
            # This does not establish robot stop; the caller explicitly stops
            # and verifies the controller, retaining the gripper command.
            subprocess.run(["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                           timeout=10, check=False)
            process.wait(timeout=10)
            raise RuntimeError("Home BAT timed out; movement outcome is uncertain, no release")


def verify_held_image(source, output):
    """Physical retention check when no real grip-state registers are mapped."""
    import pick_place_3d as vision
    import table_vision
    from grasp_verification import capture_verification
    config = vision.load_config(ROOT / "pick_place_config.json")
    settings = read_json(ROOT / "table_automation.json")
    before = vision.read_observation_pose()
    # This is an RGB-only observation; identify the existing calibrated
    # reference frame explicitly rather than pretending to capture native XYZ.
    before["camera_frame"] = config["camera"]["point_cloud_frame"]
    view = capture_verification(Path(output) / "home_held", config["camera"].get("sdk"))
    vision.verify_observation_unchanged(before, vision.read_observation_pose())
    held = vision.matrix(source["plan"]["held_model"], "source.plan.held_model")
    tcp = vision.pose_to_matrix(before["tcp_pose"])
    camera = vision.camera_transform(config, before)
    result = table_vision.verify_rgb_presence(
        view["color_path"], np.linalg.inv(camera) @ tcp @ held, config,
        empty_image=ROOT / settings["empty_closed_reference"])
    return {"accepted": result.get("accepted") is True, "evidence": result, "view": view,
            "observation": before, "source": "fresh_home_rgb_vs_empty_jaws"}


class ReturnRelease:
    def __init__(self, source_path, output, *, execute=False, max_age_seconds=180,
                 robot=None, feedback=None, batch_runner=None, held_verifier=None):
        self.source_path, self.output = Path(source_path), Path(output)
        self.execute, self.max_age_seconds = execute, max_age_seconds
        self.robot = robot if robot is not None else safety.Robot()
        self.feedback = feedback if feedback is not None else GripperFeedback()
        self.batch_runner = batch_runner or run_home_batch
        self.held_verifier = held_verifier or verify_held_image
        self.claim_path = self.source_path.with_name("return_release.claim.json")
        if self.output.exists():
            raise RuntimeError("Use a new return/release audit directory")
        self.output.mkdir(parents=True)
        self.report = dict(completed=False, released=False, motion_sent=False,
                           release_write_attempted=False, create_home=False,
                           existing_home_batch=str(BATCH), source_run=str(self.source_path),
                           created_unix=time.time(), execute=execute, events=[])

    def event(self, stage, **details):
        self.report["stage"] = stage
        self.report["events"].append(dict(stage=stage, created_unix=time.time(), **details))
        safety.save_json(self.output / "run.json", self.report)
        print(stage, flush=True)

    def checked_state(self):
        safety.ensure_exclusive_client()
        # check_state additionally verifies tool 0, user 0, enabled and powered.
        status = self.robot.snapshot()
        safety.check_state(status)
        if max(abs(x) for x in safety.vector(self.robot.call("get_curr_tool_offset")["tool_offset"])) > 1e-6:
            raise RuntimeError("Tool offset changed from calibrated zero flange offset")
        if self.robot.call("get_clsn_sensitivity")["sensitivityLevel"] not in (1, 2, 3, 4, 5):
            raise RuntimeError("Collision protection is disabled")
        current = self.robot.motion_snapshot()
        safety.check_motion_state(current)
        return current

    def check_feedback(self, label):
        sample = self.feedback.snapshot()
        self.event(label, feedback=sample)
        if sample.get("dropped") is True or sample.get("holding") is False:
            raise RuntimeError("Real grip feedback contradicts continued holding")
        if sample.get("available") is True:
            if sample.get("holding") is not True:
                raise RuntimeError("Physical holding feedback is not established")
            stable = self.feedback.wait_for_hold(timeout=2, stable_samples=3)
            self.event(label + "：连续反馈", verification=stable)
            if stable.get("confirmed") is not True:
                raise RuntimeError("Physical holding feedback is unstable")
        return sample

    def run(self):
        try:
            source = read_json(self.source_path)
            expected, _ = confirmed_source(source, self.max_age_seconds)
            original_source_hash = file_hash(self.source_path)
            original_home_hash = file_hash(arm_control.POSES_FILE)
            existing_home = arm_control.load_pose("home")
            with controller_lease(8):
                if self.claim_path.exists():
                    raise RuntimeError("This confirmed grasp already has a return/release attempt; no replay")
                state = self.checked_state()
                require_same_state(state, expected)
                self.check_feedback("返回前夹持反馈")
                self.report.update(initial_state=state, source_sha256=original_source_hash,
                                   existing_home_pose=existing_home, home_config_sha256=original_home_hash)
                self.event("已核对新鲜抓取记录与当前姿态")
                if not self.execute:
                    self.report["ready"] = True
                    self.event("只读核对完成，未调用 BAT 或夹爪")
                    return self.report
                confirmed_source(source, self.max_age_seconds)
                require_same_state(self.checked_state(), expected)
                if file_hash(self.source_path) != original_source_hash or file_hash(arm_control.POSES_FILE) != original_home_hash:
                    raise RuntimeError("Source grasp or existing Home changed before return")
                claim = dict(source_sha256=original_source_hash, output=str(self.output),
                             created_unix=time.time(), state="attempt_started", release_write_attempted=False)
                with self.claim_path.open("x", encoding="utf-8") as handle:
                    json.dump(claim, handle, ensure_ascii=False, indent=2)
                close_connections()
                self.report["motion_sent"] = True
                self.event("准备调用原有一键回 Home BAT，夹爪保持原状")
            # The existing BAT controls the arm itself. Never keep the parent
            # command lease/socket while waiting for its Python child.
            returncode = self.batch_runner(self.output / "home_batch.log")
            self.event("原有 Home BAT 已退出，等待实际到位核对", batch_exit_code=returncode,
                       footer_exit_is_not_motion_evidence=True)
            with controller_lease(8):
                final = self.checked_state()
                if file_hash(arm_control.POSES_FILE) != original_home_hash or arm_control.load_pose("home") != existing_home:
                    raise RuntimeError("Existing Home changed during return; no release")
                position, angle = safety.pose_error(final["actual_position"], existing_home)
                self.report.update(home_state=final, home_position_error_mm=position,
                                   home_orientation_error_deg=angle)
                if position > .5 or angle > .5:
                    raise RuntimeError("Actual robot is not at the existing Home; jaws retained")
                feedback = self.check_feedback("Home 到位后夹持反馈")
                if feedback.get("available") is not True:
                    visual = self.held_verifier(source, self.output)
                    self.event("未映射硬件反馈，核对 Home 实物夹持", verification=visual)
                    if visual.get("accepted") is not True:
                        raise RuntimeError("Held part at Home is not physically verified; no release")
                    require_same_state(self.checked_state(), final)
                # Re-read immediately before the only release write. Missing
                # feedback remains explicitly unknown, never converted to true.
                self.check_feedback("松爪前最后夹持反馈")
                require_same_state(self.checked_state(), final)
                self.report["release_write_attempted"] = True
                claim.update(state="release_write_started", release_write_attempted=True)
                safety.save_json(self.claim_path, claim)
                self.event("已核实原有 Home，到位释放一次", opening_command_mm=20,
                           command_is_measured_gap=False)
                gripper = side.robot_gripper()
                try:
                    gripper.connect()
                    gripper.set_opening_mm(20, settle_time=1)
                finally:
                    close_connections()
                self.report.update(released=True, completed=True, gripper_closed=False,
                                   final_state=self.robot.motion_snapshot(), final_state_created_unix=time.time())
                claim.update(state="released", completed_unix=time.time())
                safety.save_json(self.claim_path, claim)
                self.event("已在原有 Home 发送张开指令，流程完成")
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
            self.event("返回放件停止；不重试，不额外松爪", error=str(exc))
            raise
        finally:
            close_connections()
            safety.save_json(self.output / "run.json", self.report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="Fresh physically verified grasp run.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-age-seconds", type=float, default=180)
    args = parser.parse_args()
    if not 0 < args.max_age_seconds <= 300:
        parser.error("Freshness limit must be greater than zero and at most 300 seconds")
    output = args.output or args.run.parent / time.strftime("return_release_%Y%m%d_%H%M%S")
    result = ReturnRelease(args.run, output, execute=args.execute, max_age_seconds=args.max_age_seconds).run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("返回放件未完成：" + str(exc), file=sys.stderr, flush=True)
        raise SystemExit(1)
