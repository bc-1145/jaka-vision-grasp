"""Automatic cycle decisions and failure paths; no hardware access."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
import numpy as np

import auto_table_upright as auto


class EvidenceDecisionTests(unittest.TestCase):
    def test_setpoint_does_not_establish_holding(self):
        self.assertFalse(auto.combine_grasp_evidence({"opening_mm": 8.65}, [{"accepted": True}], {})["confirmed"])

    def test_two_images_need_following_motion(self):
        self.assertFalse(auto.combine_grasp_evidence({}, [{"accepted": True}]*2, {"accepted": False})["confirmed"])
        self.assertTrue(auto.combine_grasp_evidence({}, [{"accepted": True}]*2, {"accepted": True})["confirmed"])

    def test_explicit_nonholding_or_drop_overrules_appearance(self):
        for feedback in ({"holding": False}, {"dropped": True, "holding": True, "available": True}):
            self.assertFalse(auto.combine_grasp_evidence(feedback, [{"accepted": True}]*2, {"accepted": True})["confirmed"])

    def test_hardware_still_requires_actual_visible_part(self):
        feedback = {"available": True, "holding": True}
        self.assertFalse(auto.combine_grasp_evidence(feedback, [{"accepted": False}], {})["confirmed"])
        self.assertTrue(auto.combine_grasp_evidence(feedback, [{"accepted": True}], {})["confirmed"])

    def test_stale_and_future_observations_rejected(self):
        for stamp in (100, 501):
            with self.assertRaises(RuntimeError):
                auto.require_fresh({"created_unix": stamp}, now=500)
        auto.require_fresh({"created_unix": 450}, now=500)

    def test_nonfinite_or_outside_target_rejected(self):
        config = {"table_xy_bounds_mm": [[-150, -740], [140, -400]]}
        self.assertTrue(auto.inside_table([43, -537], config))
        self.assertFalse(auto.inside_table([150, -537], config))
        self.assertFalse(auto.inside_table([float("nan"), -537], config))


class CycleSupervisionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.runner = object.__new__(auto.AutomaticTable)
        r = self.runner
        r.robot = Mock()
        r.np = np
        r.output = Path(self.temporary.name)
        r.execute = True
        r.require_feedback = False
        r.config = {}
        r.settings = {"max_target_age_seconds": 180, "empty_closed_reference": "empty.png"}
        self.target = {"observation": {"created_unix": time.time()}, "shaft_grasp_center_base_mm": [43, -537, 84]}
        r.report = {"completed": False, "target": self.target}
        poses = {name: [0, -400, z, 180, 0, 90] for name, z in
                 (("approach", 240), ("prepick", 220), ("pick", 200), ("lift", 290),
                  ("turn_1", 290), ("turned", 290), ("above_place", 250), ("place", 210), ("retreat", 260))}
        self.plan = {"poses": poses, "joints": {name: [1]*6 for name in poses},
                     "route_checks": {}, "held_model": np.eye(4).tolist(), "placed_model": np.eye(4).tolist(),
                     "start_state": {"joint_actual_position": [0]*6}}
        r.route = SimpleNamespace(plan=Mock(return_value=self.plan), verify_plan_binding=Mock(),
                                  normalized_plane=lambda p: np.array([0, 0, 1, -67]))
        r.vision = SimpleNamespace(verify_observation_unchanged=Mock(), read_observation_pose=Mock(),
                                   transform_points=lambda pts, t: pts)
        r.safety = SimpleNamespace(save_json=Mock())
        r.event = Mock()
        self.actions = []
        r.move_joint = lambda q, label, *a: self.actions.append(("joint", label))
        r.move_linear = lambda p, label, *a: self.actions.append(("linear", label))
        r.opening = lambda value, label: self.actions.append(("opening", value))
        r.capture = Mock(return_value={"color_path": "image.png"})
        r.appearance = Mock(return_value=({"accepted": True}, np.eye(4)))
        r.feedback = Mock()
        r.feedback.snapshot.return_value = {"available": False, "holding": None, "dropped": None}
        r.evidence = SimpleNamespace(verify_follow_motion=Mock(return_value={"accepted": True}),
                                     verify_upright=Mock(return_value={"accepted": True}))
        r.cloud = Mock(return_value=(SimpleNamespace(points=[[0, 0, 0]]), {}, np.eye(4)))
        r.current_plane = Mock(return_value={"plane_base": [0, 0, 1, -67]})

    def run_cycle(self):
        self.runner.execute_cycle(self.target, {"plane_base": [0, 0, 1, -67]})

    def test_complete_cycle_has_confirmed_holding_before_turn_and_verifies_release(self):
        self.run_cycle()
        self.assertTrue(self.runner.report["completed"])
        self.assertTrue(self.runner.report["physical_grasp_confirmed"])
        self.assertEqual([a[1] for a in self.actions if a[0] == "opening"], [20, 8.65, 20])
        self.assertEqual(self.actions[-1], ("joint", "返回高处总览位"))

    def test_failed_alignment_never_closes(self):
        self.runner.appearance.return_value = ({"accepted": False}, np.eye(4))
        with self.assertRaisesRegex(RuntimeError, "aligned"):
            self.run_cycle()
        self.assertEqual([a[1] for a in self.actions if a[0] == "opening"], [20])

    def test_empty_first_lift_stops_without_flip_or_release(self):
        self.runner.appearance.side_effect = [({"accepted": True}, np.eye(4)), ({"accepted": False}, np.eye(4))]
        with self.assertRaisesRegex(RuntimeError, "First lift"):
            self.run_cycle()
        self.assertFalse(any("翻正" in str(a) for a in self.actions))
        self.assertEqual([a[1] for a in self.actions if a[0] == "opening"], [20, 8.65])

    def test_hardware_drop_stops_even_with_positive_images(self):
        self.runner.feedback.snapshot.return_value = {"available": True, "holding": False, "dropped": True}
        with self.assertRaises(RuntimeError):
            self.run_cycle()
        self.assertFalse(any("翻正" in str(a) for a in self.actions))

    def test_unproven_follow_motion_never_turns(self):
        self.runner.evidence.verify_follow_motion.return_value = {"accepted": False}
        with self.assertRaisesRegex(RuntimeError, "motion"):
            self.run_cycle()
        self.assertFalse(any("翻正" in str(a) for a in self.actions))

    def test_unstable_hardware_feedback_does_not_fall_back_to_vision(self):
        self.runner.feedback.snapshot.return_value = {"available": True, "holding": True}
        self.runner.feedback.wait_for_hold.return_value = {"confirmed": False, "samples": []}
        with self.assertRaisesRegex(RuntimeError, "stable"):
            self.run_cycle()
        self.assertFalse(any("翻正" in str(a) for a in self.actions))

    def test_lost_part_after_turn_does_not_descend_or_open(self):
        self.runner.appearance.side_effect = [({"accepted": True}, np.eye(4))]*3+[({"accepted": False}, np.eye(4))]
        with self.assertRaisesRegex(RuntimeError, "after rotation"):
            self.run_cycle()
        self.assertFalse(any("放到台面" in str(a) for a in self.actions))
        self.assertEqual([a[1] for a in self.actions if a[0] == "opening"], [20, 8.65])

    def test_failed_final_verification_never_marks_completed_or_regrasps(self):
        self.runner.evidence.verify_upright.return_value = {"accepted": False}
        with self.assertRaisesRegex(RuntimeError, "upright"):
            self.run_cycle()
        self.assertFalse(self.runner.report["completed"])
        self.assertEqual(self.runner.cloud.call_count, 2)
        self.assertEqual([a[1] for a in self.actions if a[0] == "opening"].count(8.65), 1)

    def test_ambiguous_gripper_write_is_not_retried(self):
        def uncertain(value, label):
            self.actions.append(("opening", value))
            if value == 8.65:
                raise RuntimeError("write uncertain")
        self.runner.opening = uncertain
        with self.assertRaisesRegex(RuntimeError, "uncertain"):
            self.run_cycle()
        self.assertEqual(self.actions[-1], ("opening", 8.65))

    def test_read_only_planning_has_no_actuator_actions(self):
        self.runner.execute = False
        self.run_cycle()
        self.assertEqual(self.actions, [])
        self.assertTrue(self.runner.report["planned"])

    def test_required_feedback_cannot_disappear_and_fall_back_to_vision(self):
        self.runner.require_feedback = True
        self.runner.feedback.snapshot.side_effect = [
            {"available": True, "holding": True}, {"available": False, "holding": None}]
        self.runner.feedback.wait_for_hold.return_value = {"confirmed": True, "samples": []}
        with self.assertRaisesRegex(RuntimeError, "Required physical"):
            self.run_cycle()
        self.assertFalse(any("翻正" in str(a) for a in self.actions))

    def test_drop_immediately_before_release_prevents_opening(self):
        unknown = {"available": False, "holding": None}
        self.runner.feedback.snapshot.side_effect = [unknown]*5+[
            {"available": True, "holding": False, "dropped": True}]
        with self.assertRaisesRegex(RuntimeError, "no held"):
            self.run_cycle()
        self.assertEqual([a[1] for a in self.actions if a[0] == "opening"], [20, 8.65])

    def test_transfer_is_executed_before_rotation(self):
        self.plan["poses"]["transfer"] = [43, -400, 290, 180, 0, 90]
        self.run_cycle()
        transfer = next(i for i, action in enumerate(self.actions) if "持件移到" in str(action))
        turning = next(i for i, action in enumerate(self.actions) if "翻正" in str(action))
        self.assertLess(transfer, turning)

    def test_unverified_initial_empty_jaws_prevent_all_actuators(self):
        r = self.runner
        r.report["actuator_write_attempted"] = False
        r.safety.ensure_exclusive_client = Mock()
        r.check_installation = Mock()
        r.evidence.verify_empty_jaws = Mock(return_value={"accepted": False})
        r.settings.update(empty_open_reference="open.png", empty_closed_reference="closed.png")
        r.overview = Mock()
        with patch("controller_lease.controller_lease", return_value=nullcontext()), patch("gripper_transport.close_connections"):
            with self.assertRaisesRegex(RuntimeError, "Initial empty"):
                r.run()
        r.overview.assert_not_called()
        self.assertEqual(self.actions, [])

    def test_manual_pose_change_rejects_next_planned_move(self):
        r = self.runner
        r.expected_state = {"actual_position": [0]*6, "joint_actual_position": [0]*6}
        r.snapshot = Mock(return_value={"actual_position": [5, 0, 0, 0, 0, 0], "joint_actual_position": [0]*6})
        r.safety.pose_error = Mock(return_value=(5, 0))
        with self.assertRaisesRegex(RuntimeError, "outside the planned"):
            r.check_start()


if __name__ == "__main__":
    unittest.main()
