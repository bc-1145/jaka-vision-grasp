"""Fresh bin attempts followed by the original Home BAT, without releasing.

Default is read-only. Failed attempts can repeat only after a fresh Home image
proves empty jaws; communication uncertainty and controller faults never retry.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
from auto_bin_grasp import AutomaticBin
from auto_table_upright import ROOT, read_json, check_settings, require_fresh
from bin_return_hold import ReturnHold


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def compare_coordinates(previous, current):
    """Audit newly measured coordinates, never change camera calibration."""
    result = dict(global_calibration_updated=False, same_target_candidate=False,
                  identity_is_proven=False, relative_coordinate_correction_mm=None)
    if not previous:
        return result
    old, new = previous.get("plan"), current
    if not isinstance(old, dict):
        return result
    require_fresh(new["observation"])
    if float(new["observation"]["created_unix"]) <= float(old["observation"]["created_unix"]):
        raise RuntimeError("Retry must use a newer target observation")
    a, b = np.asarray(old["grasp_base"], float), np.asarray(new["grasp_base"], float)
    u, v = np.asarray(old["shaft_axis_base"], float), np.asarray(new["shaft_axis_base"], float)
    if any(x.shape != (3,) or not np.isfinite(x).all() for x in (a,b,u,v)):
        raise RuntimeError("Invalid measured target geometry")
    if min(np.linalg.norm(u), np.linalg.norm(v)) < .99:
        raise RuntimeError("Invalid measured target axis")
    distance, dot = float(np.linalg.norm(b-a)), float(np.dot(u/np.linalg.norm(u),v/np.linalg.norm(v)))
    same = distance < 25 and dot > .95
    result.update(same_target_candidate=same, grasp_center_displacement_mm=distance, directed_axis_dot=dot)
    if same:
        result["relative_coordinate_correction_mm"] = (b-a).tolist()
    return result


class CycleBin(AutomaticBin):
    def scan_and_plan(self):
        super().scan_and_plan()
        audit = compare_coordinates(getattr(self, "previous_report", None), self.plan)
        self.report["coordinate_comparison"] = audit
        save(self.output/"coordinate_comparison.json", audit)
        self.event("记录新测坐标差，不修改标定", coordinate_comparison=audit)


def retry_blocker(report, error=None):
    if isinstance(error, (OSError, ConnectionError, TimeoutError)):
        return "Communication/write outcome is uncertain"
    if error is not None and type(error).__name__ in ("WriteUncertainError", "CommunicationError"):
        return "Communication/write outcome is uncertain"
    if report.get("stop_error") or report.get("final_state_error"):
        return "Stop or final-state verification failed"
    text = (str(error or "") + " " + str(report.get("error", ""))).lower()
    if any(word in text for word in ("writeuncertain", "uncertain write", "write outcome", "communicationerror")):
        return "Communication/write outcome is uncertain"
    state = report.get("final_state")
    if not isinstance(state, dict) or not state:
        return "No measured final state"
    for key in ("estop", "collision", "soft_limit", "paused", "queue", "active_queue",
                "emergency_stop", "protective_stop", "on_soft_limit"):
        if state.get(key):
            return "Controller is faulted or active: " + key
    return None


def verify_fresh_home_empty(runner, return_runner, output):
    """Read-only proof at the measured original Home, never open the jaws."""
    import arm_control
    from controller_lease import controller_lease
    from grasp_verification import capture_verification_isolated
    from gripper_transport import close_connections
    try:
        with controller_lease(8):
            state = return_runner.checked_state()
            distance, angle = runner.safety.pose_error(state["actual_position"], arm_control.load_pose("home"))
            if distance > .5 or angle > .5:
                raise RuntimeError("Fresh empty proof requires actual original Home")
            before = runner.vision.read_observation_pose()
            if (before.get("tool_id"), before.get("user_id")) != (0, 0):
                raise RuntimeError("Fresh empty proof requires original flange and base frames")
            started = time.time()
            view = capture_verification_isolated(Path(output), runner.config["camera"].get("sdk"))
            runner.vision.verify_observation_unchanged(before, runner.vision.read_observation_pose())
            # Recheck actual Home after camera acquisition, not just camera pose.
            after = return_runner.checked_state()
            from bin_return_release import require_same_state
            require_same_state(after, state)
            feedback = runner.feedback.snapshot()
            empty = runner.evidence.verify_empty_jaws(view["color_path"], runner.config,
                    open_reference=ROOT/runner.settings["empty_open_reference"],
                    closed_reference=ROOT/runner.settings["empty_closed_reference"])
            contradiction = feedback.get("holding") is True or (feedback.get("available") is True and feedback.get("holding") is not False)
            proof = dict(accepted=empty.get("accepted") is True and not contradiction,
                         capture_started_unix=started, created_unix=time.time(), view=view,
                         empty_jaws=empty, feedback=feedback, final_state=after,
                         gripper_write_attempted=False)
            save(Path(output)/"empty_proof.json", proof)
            return proof
    finally:
        close_connections()


class AutomaticBinCycle:
    def __init__(self, config, settings, output, *, execute=False, max_attempts=2,
                 runner_factory=CycleBin, return_factory=ReturnHold, empty_verifier=verify_fresh_home_empty):
        if type(max_attempts) is not int or not 1 <= max_attempts <= 3:
            raise ValueError("max_attempts must be between 1 and 3")
        self.config, self.settings = copy.deepcopy(config), copy.deepcopy(settings)
        self.output, self.execute, self.max_attempts = Path(output), execute, max_attempts
        self.runner_factory, self.return_factory, self.empty_verifier = runner_factory, return_factory, empty_verifier
        self.report = dict(completed=False, execute=execute, attempts=[], released=False,
                           global_calibration_updated=False, gripper_kept_unchanged_at_home=True)

    def run(self, already_at_bin=False, empty_reference_run=None):
        previous = None
        self.output.mkdir(parents=True, exist_ok=False)
        try:
            for index in range(self.max_attempts if self.execute else 1):
                folder = self.output/f"attempt_{index+1:02d}"
                runner = self.runner_factory(copy.deepcopy(self.config), copy.deepcopy(self.settings),
                                             folder/"grasp", execute=self.execute)
                runner.previous_report = previous
                error = None
                try:
                    result = runner.run_bin(already_at_bin if index == 0 else False,
                                            empty_reference_run if index == 0 else None)
                except Exception as exc:
                    error, result = exc, runner.report
                entry = dict(grasp=result, exception_type=type(error).__name__ if error else None)
                self.report["attempts"].append(entry)
                if not self.execute:
                    self.report.update(ready=error is None, reason=str(error) if error else "Read-only scan and plan complete")
                    break
                actual = any(result.get(key) for key in ("actuator_write_attempted", "motion_sent", "gripper_closed"))
                if not actual:
                    self.report["reason"] = "No physical attempt was made; no return or retry"
                    break
                blocker = retry_blocker(result, error)
                # A faulted or unverified arm must remain stopped. In particular,
                # do not invoke even the original BAT with an asserted E-stop.
                if result.get("stop_error") or result.get("final_state_error") or not isinstance(result.get("final_state"), dict) or not result.get("final_state") or (
                        blocker and blocker.startswith("Controller is faulted")):
                    entry["return_blocked"] = True
                    self.report["reason"] = blocker or "Final measured state is unavailable; no return or retry"
                    break
                returning = self.return_factory(folder/"grasp"/"run.json", folder/"return_home", execute=True)
                try:
                    home = returning.run()
                    entry["return_home"] = home
                except Exception as exc:
                    entry["return_error"] = str(exc)
                    self.report["reason"] = "Home return failed; no automatic retry"
                    break
                if home.get("completed") is not True:
                    self.report["reason"] = "Original Home is not verified; no retry"
                    break
                self.report.update(holding=home.get("holding"), holding_status=home.get("holding_status", "unknown"))
                if result.get("physical_grasp_confirmed") is True and result.get("completed") is True and error is None:
                    self.report.update(completed=True, source_grasp_confirmed=True, reason="Confirmed grasp returned to original Home; jaws unchanged")
                    break
                blocker = retry_blocker(result, error)
                if blocker or index+1 >= self.max_attempts:
                    self.report["reason"] = blocker or "Maximum attempts reached"
                    break
                try:
                    proof = self.empty_verifier(runner, returning, folder/"retry_empty_home")
                except Exception as exc:
                    entry["retry_empty_error"] = str(exc)
                    self.report["reason"] = "Fresh empty-jaw proof failed; keep gripper unchanged"
                    break
                entry["retry_empty_proof"] = proof
                if proof.get("accepted") is not True:
                    self.report["reason"] = "Jaws may hold a part; no opening or retry"
                    break
                previous = copy.deepcopy(result)
            return self.report
        finally:
            save(self.output/"run.json", self.report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--already-at-bin", action="store_true")
    parser.add_argument("--empty-reference-run", type=Path)
    parser.add_argument("--max-attempts", type=int, choices=range(1,4), default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config, settings = read_json(ROOT/"pick_place_config.json"), read_json(ROOT/"table_automation.json")
    check_settings(config, settings)
    output = args.output or ROOT/"pick_results"/"bin_cycle"/time.strftime("run_%Y%m%d_%H%M%S")
    report = AutomaticBinCycle(config, settings, output, execute=args.execute, max_attempts=args.max_attempts).run(
        args.already_at_bin, args.empty_reference_run)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.execute and not report.get("completed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
