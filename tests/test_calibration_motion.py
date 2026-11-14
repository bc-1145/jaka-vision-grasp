import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

import calibration_motion as motion
import pick_place_3d as vision


def state(pose=None, joints=None):
    return {"actual_position": pose or [192, -60, 280, 179, 0.4, -179],
            "joint_actual_position": joints or [160, -1, 79, 1, 102, 159],
            "emergency_stop": 0, "protective_stop": 0, "on_soft_limit": 0,
            "paused": False, "drag_status": False, "errcode": "0x0",
            "enabled": True, "powered_on": 1, "inpos": True,
            "interp_state": 0, "instVel": [0] * 6, "current_tool_id": 0,
            "current_user_id": 0, "rapidrate": 1}


def valid_plan():
    s = state()
    return {"created_unix": time.time(), "origin_pose": s["actual_position"],
            "origin_joints": s["joint_actual_position"], "legs": [],
            "checks": {"joint_min_deg": [150, -10, 70, -10, 90, 150],
                       "joint_max_deg": [170, 10, 90, 10, 110, 170]}}


def valid_site(plan):
    return {"measurement_source": "test fixture", "attachment_radius_mm": 200,
            "attachment_clearance_mm": 60, "arm_and_cables_clear": True,
            "joint_limits_deg": [[-180, 180]] * 6,
            "reviewed_plan_sha256": hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
            "verified_payload": {"mass": 0.8, "centroid": [0, 0, 90]},
            "use_existing_controller_payload_for_slow_calibration": False}


class TrajectoryTests(unittest.TestCase):
    def test_nine_views_excitation_and_fixed_flange(self):
        origin = state()["actual_position"]
        views = motion.make_views(origin)
        self.assertEqual(len(views), 9)
        inverse = np.linalg.inv(vision.pose_to_matrix(origin))
        rotations = []
        for view in views[1:]:
            np.testing.assert_allclose(view["pose"][:3], origin[:3])
            relative = inverse @ vision.pose_to_matrix(view["pose"])
            rotations.append(Rotation.from_matrix(relative[:3, :3]).as_rotvec())
            self.assertAlmostEqual(motion.pose_error(view["pose"], origin)[1], motion.TILT_DEG)
        singular = np.linalg.svd(rotations, compute_uv=False)
        self.assertGreater(singular[1], np.deg2rad(8))
        # Both fitted and held-out sets must permit the same AX=XB solver.
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler("xyz", [10, 15, 20], degrees=True).as_matrix()
        transform[:3, 3] = [30, 40, 100]
        a = [inverse @ vision.pose_to_matrix(v["pose"]) for v in views[1:]]
        b = [np.linalg.inv(transform) @ item @ transform for item in a]
        result = motion.handeye.solve_motion_pairs(a, b)
        self.assertTrue(result["accepted"])
        np.testing.assert_allclose(result["candidate_tcp_T_camera"], transform, atol=1e-6)

    def test_site_requires_clearance_limits_payload_and_specific_path(self):
        plan = valid_plan()
        site = valid_site(plan)
        payload = site["verified_payload"]
        result = motion.validate_site(plan, site, payload)
        self.assertGreater(result["required_clearance_mm"], 30)
        for change in ({"attachment_clearance_mm": 5}, {"attachment_radius_mm": float("nan")},
                       {"joint_limits_deg": None}, {"joint_limits_deg": [[-10, 10]] * 6},
                       {"arm_and_cables_clear": False}, {"reviewed_plan_sha256": "old"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                motion.validate_site(plan, {**site, **change}, payload)
        with self.assertRaisesRegex(ValueError, "负载"):
            motion.validate_site(plan, site, {"mass": 0, "centroid": [0, 0, 0]})

    def test_invalid_site_blocks_before_capture_or_movement(self):
        robot = Mock()
        robot.call.return_value = {"mass": 0, "centroid": [0, 0, 0]}
        capture = Mock()
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "new"
            with self.assertRaises(ValueError):
                motion.run_collection(robot, valid_plan(), {}, Path("config"), target,
                                      capture=capture, exclusive=Mock())
            self.assertFalse(target.exists())
        robot.move.assert_not_called()
        capture.assert_not_called()

    def test_reject_fault_missing_state_and_frame_changes(self):
        for key, value in (("current_tool_id", 1), ("current_user_id", 2),
                           ("protective_stop", 1), ("paused", True), ("inpos", False),
                           ("enabled", False), ("errcode", "0x42"), ("emergency_stop", None)):
            s = state()
            if value is None:
                del s[key]
            else:
                s[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                motion.check_state(s)

    def test_motion_ack_loss_stops_and_never_retries(self):
        robot = Mock()
        s = state()
        robot.motion_snapshot.return_value = {**s, "queue": 0, "active_queue": 0,
                                             "estop": 0, "collision": 0, "soft_limit": 0}
        robot.move.side_effect = ConnectionError("lost ack")
        start = {"pose": s["actual_position"], "joints": s["joint_actual_position"]}
        target = copy.deepcopy(start)
        target["joints"][5] += 1
        with self.assertRaisesRegex(ConnectionError, "lost ack"):
            motion.move_step(robot, start, target)
        robot.move.assert_called_once()
        robot.stop.assert_called_once()

    def test_stale_inpos_is_not_completion(self):
        robot = Mock()
        s = state()
        robot.motion_snapshot.return_value = {**s, "queue": 0, "active_queue": 0,
                                             "estop": 0, "collision": 0, "soft_limit": 0}
        start = {"pose": s["actual_position"], "joints": s["joint_actual_position"]}
        target = copy.deepcopy(start)
        target["joints"][5] += 1
        ticks = iter(range(1000))
        with patch.object(motion.time, "monotonic", side_effect=lambda: next(ticks)), \
                patch.object(motion.time, "sleep"), self.assertRaises(TimeoutError):
            motion.move_step(robot, start, target)
        robot.move.assert_called_once()
        robot.stop.assert_called_once()

    def test_mid_motion_cancel_stops_without_return_move(self):
        robot = Mock()
        s = state()
        robot.motion_snapshot.return_value = {**s, "queue": 0, "active_queue": 0,
                                             "estop": 0, "collision": 0, "soft_limit": 0}
        start = {"pose": s["actual_position"], "joints": s["joint_actual_position"]}
        target = copy.deepcopy(start)
        target["joints"][5] += 1
        with self.assertRaisesRegex(RuntimeError, "取消"):
            motion.move_step(robot, start, target, Mock(side_effect=[False, True]))
        robot.move.assert_called_once()
        robot.stop.assert_called_once()

    def test_stop_failure_is_reported_not_hidden(self):
        robot = Mock()
        s = state()
        robot.motion_snapshot.return_value = {**s, "queue": 0, "active_queue": 0,
                                             "estop": 0, "collision": 0, "soft_limit": 0}
        robot.move.side_effect = ConnectionError("lost ack")
        robot.stop.side_effect = ConnectionError("stop unreachable")
        start = {"pose": s["actual_position"], "joints": s["joint_actual_position"]}
        with self.assertRaisesRegex(RuntimeError, "STOP UNVERIFIED"):
            motion.move_step(robot, start, start)

    def test_full_turn_not_hidden_by_wrapping(self):
        robot = Mock()
        s = state()
        robot.motion_snapshot.return_value = {**s, "queue": 0, "active_queue": 0,
                                             "estop": 0, "collision": 0, "soft_limit": 0}
        start = {"pose": s["actual_position"], "joints": s["joint_actual_position"]}
        target = copy.deepcopy(start)
        target["joints"][0] += 360
        with self.assertRaisesRegex(RuntimeError, "increment"):
            motion.move_step(robot, start, target)
        robot.move.assert_not_called()

    def test_nine_captures_and_return_only_after_success(self):
        for fail_second_capture in (False, True):
            with self.subTest(failure=fail_second_capture), tempfile.TemporaryDirectory() as temporary:
                plan = valid_plan()
                origin = {"pose": plan["origin_pose"], "joints": plan["origin_joints"]}
                for i in range(8):
                    target = copy.deepcopy(origin)
                    target["pose"][3] += i + 1
                    target["joints"][3] += 1
                    plan["legs"].append({"view": str(i), "outward": [target], "return": [origin]})
                site = valid_site(plan)
                robot = Mock()
                robot.snapshot.return_value = state()
                robot.call.side_effect = lambda name: site["verified_payload"] if name == "get_payload" else {"sensitivityLevel": 2}
                cursor = copy.deepcopy(origin)
                samples, moves = [], []

                def move(_robot, _start, end, _cancelled):
                    cursor.update(copy.deepcopy(end))
                    moves.append(copy.deepcopy(end))

                def capture(_config, directory, _cancelled):
                    if fail_second_capture and len(samples) == 1:
                        raise RuntimeError("camera failed")
                    samples.append({"observation": {"tcp_pose": cursor["pose"]}})
                    motion.save_json(directory / "manifest.json", {"schema_version": 1, "samples": samples})

                dataset = Path(temporary) / "samples"
                with patch.object(motion, "move_step", side_effect=move):
                    if fail_second_capture:
                        with self.assertRaisesRegex(RuntimeError, "camera failed"):
                            motion.run_collection(robot, plan, site, Path("config"), dataset,
                                                  capture=capture, exclusive=Mock())
                        self.assertEqual(len(moves), 1)
                    else:
                        report = motion.run_collection(robot, plan, site, Path("config"), dataset,
                                                       capture=capture, exclusive=Mock())
                        self.assertTrue(report["completed"])
                        self.assertEqual(report["samples"], 9)
                        self.assertEqual(len(moves), 16)
                        self.assertEqual(cursor, origin)
                report = json.loads((dataset / "run_report.json").read_text(encoding="utf-8"))
                self.assertEqual(report["completed"], not fail_second_capture)


if __name__ == "__main__":
    unittest.main()
