"""Resume a manually verified pick after a transient safe-retreat stop."""
from __future__ import annotations

import json
import time

import numpy as np

import arm_control
import calibration_motion as safety
from controller_lease import controller_lease
import one_click_pick as one
import pick_place_3d as vision
import side_pick_execute as side


def main():
    config = vision.load_config(one.CONFIG)
    report = json.loads(one.REPORT.read_text(encoding="utf-8"))
    plan = report.get("side_plan")
    before = report.get("before_grasp_verification")
    if not plan or not before or report.get("holding"):
        raise RuntimeError("当前记录不能从安全退避点续跑")
    robot = one.FastRobot()
    with controller_lease(8.0):
        safety.ensure_exclusive_client()
        state = robot.motion_snapshot(); safety.check_motion_state(state)
        error = float(np.max(np.abs(
            np.asarray(state["joint_actual_position"]) - np.asarray(plan["joints"]["lift"]))))
        if error > 1.0:
            raise RuntimeError(f"机器人离已验证退避点 {error:.2f}°，拒绝续跑")
        old_rate = arm_control.get_rapid_rate()
        if abs(old_rate - 1.0) > 1e-6:
            arm_control.set_rapid_rate(1.0)
        original_builder = side.build_plan
        side.build_plan = lambda *_args, **_kwargs: plan
        report.pop("error", None); report.pop("stop_error", None)
        report["resumed_unix"] = time.time(); one.save(report)
        try:
            one.execute(robot, config, None, report, manual_verify=True,
                        before_verification=before)
            report.update(completed=True, holding=False, stage="完成")
            one.save(report)
            print("人工确认抓取并放置完成。", flush=True)
            return 0
        except BaseException:
            try:
                safety.stop_and_verify(robot)
            except Exception as stopping:
                report["stop_error"] = str(stopping); one.save(report)
            raise
        finally:
            side.build_plan = original_builder
            if abs(old_rate - 1.0) > 1e-6:
                arm_control.set_rapid_rate(old_rate)


if __name__ == "__main__":
    raise SystemExit(main())
