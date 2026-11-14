from contextlib import contextmanager
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
import arm_control
import bin_return_hold as hold


HOME = [6, -473, 222, 180, 45, 90]
PICK = [200, 10, 350, 180, 30, 180]


def state(pose=PICK):
    return dict(actual_position=list(pose), joint_actual_position=[10,20,30,40,50,60],
                inpos=True, queue=0, active_queue=0, estop=0, collision=0, soft_limit=0, paused=False)


class ReturnHoldTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.source = self.folder/"grasp.json"
        self.source_data = dict(physical_grasp_confirmed=True, gripper_closed=True,
                               final_state=state(), final_state_created_unix=time.time())
        self.source.write_text(json.dumps(self.source_data), encoding="utf-8")
        self.phase, self.depth, self.batch_calls = "pick", 0, 0
        self.robot, self.feedback = Mock(), Mock()
        self.robot.motion_snapshot.side_effect = lambda: state(HOME if self.phase=="home" else PICK)
        self.robot.snapshot.return_value = dict(emergency_stop=0, protective_stop=0, on_soft_limit=0,
                paused=False, drag_status=0, errcode=0, enabled=True, powered_on=1,
                current_tool_id=0, current_user_id=0, inpos=True, interp_state=0,
                instVel=[0]*6, rapidrate=.5, actual_position=PICK,
                joint_actual_position=[10,20,30,40,50,60])
        self.robot.call.side_effect = lambda name: ({"tool_offset": [0]*6}
                            if name=="get_curr_tool_offset" else {"sensitivityLevel":3})
        self.feedback.snapshot.return_value = dict(available=False, holding=None, dropped=None, reason="unmapped")
        self.audit = Mock(return_value=dict(accepted=True, full_collision_checked=False))
        self.gripper = Mock()
        self.patches = [patch.object(hold.safety, "ensure_exclusive_client"),
                        patch.object(hold.safety, "stop_and_verify"),
                        patch.object(hold, "close_connections"),
                        patch.object(hold, "controller_lease", side_effect=self.lease),
                        patch.object(hold.arm_control, "load_pose", return_value=HOME),
                        patch("side_pick_execute.robot_gripper", return_value=self.gripper)]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    @contextmanager
    def lease(self, timeout):
        self.depth += 1
        try:
            yield
        finally:
            self.depth -= 1

    def batch(self, path):
        self.assertEqual(self.depth, 0)
        self.batch_calls += 1
        self.phase = "home"
        return 1  # Existing BAT footer may fail after successful physical return.

    def runner(self, *, execute=True, output="result"):
        return hold.ReturnHold(self.source, self.folder/output, execute=execute,
                 robot=self.robot, feedback=self.feedback, batch_runner=self.batch,
                 route_auditor=self.audit)

    def rewrite(self, **details):
        self.source_data.update(details)
        self.source.write_text(json.dumps(self.source_data), encoding="utf-8")

    def test_default_read_only_does_not_invoke_bat_or_gripper(self):
        result = self.runner(execute=False).run()
        self.assertTrue(result["ready"])
        self.assertEqual(self.batch_calls, 0)
        self.gripper.set_opening_mm.assert_not_called()

    def test_confirmed_grasp_returns_without_any_gripper_write(self):
        result = self.runner().run()
        self.assertTrue(result["completed"])
        self.assertTrue(result["gripper_command_unchanged"])
        self.assertFalse(result["gripper_write_attempted"])
        self.assertIsNone(result["holding"])
        self.assertEqual(result["holding_status"], "unknown")
        self.assertFalse(result["released"])
        self.assertEqual(self.batch_calls, 1)
        self.gripper.connect.assert_not_called()
        self.gripper.set_opening_mm.assert_not_called()

    def test_failed_run_also_returns_preserving_gripper(self):
        self.rewrite(physical_grasp_confirmed=False, completed=False, error="grasp uncertain")
        result = self.runner().run()
        self.assertTrue(result["completed"])
        self.assertFalse(result["source_grasp_confirmed"])
        self.assertEqual(result["source_error"], "grasp uncertain")
        self.assertEqual(self.batch_calls, 1)
        self.gripper.set_opening_mm.assert_not_called()

    def test_real_not_held_state_still_returns_failed_grasp_without_touching_gripper(self):
        self.feedback.snapshot.return_value = dict(available=True, holding=False, dropped=True)
        self.rewrite(physical_grasp_confirmed=False)
        result = self.runner().run()
        self.assertTrue(result["completed"])
        self.assertIs(result["holding"], False)
        self.assertEqual(result["holding_status"], "not_held")
        self.gripper.set_opening_mm.assert_not_called()

    def test_stale_failed_run_cannot_trigger_return(self):
        self.rewrite(physical_grasp_confirmed=False, final_state_created_unix=time.time()-600)
        with self.assertRaisesRegex(RuntimeError, "stale"):
            self.runner().run()
        self.assertEqual(self.batch_calls, 0)

    def test_final_timestamp_cannot_refresh_a_different_actual_state(self):
        self.source_data["actual"] = self.source_data.pop("final_state")
        self.source.write_text(json.dumps(self.source_data), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "explicit measured timestamp"):
            self.runner().run()
        self.assertEqual(self.batch_calls, 0)

    def test_active_motion_prevents_return(self):
        current = state()
        current["queue"] = 1
        self.robot.motion_snapshot.side_effect = None
        self.robot.motion_snapshot.return_value = current
        with self.assertRaisesRegex(RuntimeError, "not stationary"):
            self.runner().run()
        self.assertEqual(self.batch_calls, 0)

    def test_actual_home_required_even_when_bat_exit_zero(self):
        runner = self.runner()
        runner.batch_runner = Mock(return_value=0)
        with self.assertRaisesRegex(RuntimeError, "not actually reached"):
            runner.run()
        runner.batch_runner.assert_called_once()
        self.gripper.set_opening_mm.assert_not_called()

    def test_route_rejection_prevents_bat(self):
        self.audit.side_effect = RuntimeError("obvious table interference")
        with self.assertRaisesRegex(RuntimeError, "table interference"):
            self.runner().run()
        self.assertEqual(self.batch_calls, 0)

    def test_uncertain_batch_never_retried_and_never_changes_gripper(self):
        runner = self.runner()
        runner.batch_runner = Mock(side_effect=RuntimeError("uncertain BAT outcome"))
        with self.assertRaisesRegex(RuntimeError, "uncertain BAT"):
            runner.run()
        runner.batch_runner.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "no automatic replay"):
            self.runner(output="second").run()
        self.gripper.set_opening_mm.assert_not_called()


class RouteAuditTests(unittest.TestCase):
    def test_joint3_prediction_matches_original_bat_search(self):
        robot = Mock()
        robot.forward.side_effect = lambda q: [0, 0, 200-q[2], 180, 0, 0]
        joints = [0,0,0,0,0,0]
        actual = hold.predict_joint3_lift(robot, joints, 220, 1, 100)
        with patch.object(arm_control, "forward_kinematics", side_effect=robot.forward):
            expected = arm_control.plan_joint3_lift(joints, 220, 1, 100)
        np.testing.assert_allclose(actual[0], expected[0])
        np.testing.assert_allclose(actual[1], expected[1])

    def test_unknown_grasp_uses_possible_object_envelope_at_home(self):
        config = {"gripper":{"tcp_to_grasp_center_mm":[0,0,100]},
                  "model":{"grasp_point_mm":[0,0,25]}}
        settings = {"reference_table_plane":{"plane_base":[0,0,1,-65]},
                    "table_xy_bounds_mm":[[-100,-600],[100,-400]]}
        home = [0,-500,230,180,0,0]
        current = state(home)
        current["joint_actual_position"]=[0]*6
        robot = Mock()
        robot.forward.return_value=home
        vertices = np.array([[x,y,z] for x in (-17,17) for y in (-17,17) for z in (0,50)])
        result = hold.audit_existing_home_route(robot,current,{},config,settings,
                    {"position":home,"approach_joint":[0]*6,"approach_z":280},vertices=vertices)
        self.assertTrue(result["accepted"])
        self.assertFalse(result["full_collision_checked"])
        self.assertEqual(result["held_shape"],"unknown_object_sphere")
        self.assertGreater(result["minima_mm"]["possible_part"],5)

    def test_unknown_possible_held_part_intersecting_table_is_rejected(self):
        config = {"gripper":{"tcp_to_grasp_center_mm":[0,0,100]},
                  "model":{"grasp_point_mm":[0,0,25]}}
        settings = {"reference_table_plane":{"plane_base":[0,0,1,-65]},
                    "table_xy_bounds_mm":[[-100,-600],[100,-400]]}
        home = [0,-500,190,180,0,0]
        current=state(home)
        current["joint_actual_position"]=[0]*6
        robot=Mock()
        robot.forward.return_value=home
        vertices=np.array([[x,y,z] for x in (-17,17) for y in (-17,17) for z in (0,50)])
        with self.assertRaisesRegex(RuntimeError,"table interference"):
            hold.audit_existing_home_route(robot,current,{},config,settings,
                    {"position":home,"approach_joint":[0]*6,"approach_z":280},vertices=vertices)


if __name__ == "__main__":
    unittest.main()
