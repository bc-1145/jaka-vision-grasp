from contextlib import contextmanager
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import bin_return_release as release


HOME = [6, -473, 222, 180, 45, 90]
PICK = [200, 10, 250, 180, 30, 180]


def state(pose=PICK):
    return dict(actual_position=list(pose), joint_actual_position=[10, 20, 30, 40, 50, 60],
                inpos=True, queue=0, active_queue=0, estop=0, collision=0,
                soft_limit=0, paused=False)


class ReturnReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.source = self.folder / "grasp.json"
        self.source_data = dict(physical_grasp_confirmed=True, gripper_closed=True,
                                final_state=state(), final_state_created_unix=time.time())
        self.source.write_text(json.dumps(self.source_data), encoding="utf-8")
        self.phase = "pick"
        self.batch_code = 0
        self.lease_depth = 0
        self.batch_calls = 0
        self.robot = Mock()
        self.robot.motion_snapshot.side_effect = lambda: state(HOME if self.phase == "home" else PICK)
        self.robot.snapshot.return_value = dict(emergency_stop=0, protective_stop=0,
                on_soft_limit=0, paused=False, drag_status=0, errcode=0, enabled=True,
                powered_on=1, current_tool_id=0, current_user_id=0, inpos=True,
                interp_state=0, instVel=[0]*6, rapidrate=.5,
                actual_position=PICK, joint_actual_position=[10,20,30,40,50,60])
        self.robot.call.side_effect = lambda name: ({"tool_offset": [0]*6}
                        if name == "get_curr_tool_offset" else {"sensitivityLevel": 3})
        self.feedback = Mock()
        self.feedback.snapshot.return_value = dict(available=False, holding=None,
                                                   dropped=None, reason="unmapped")
        self.visual = Mock(return_value={"accepted": True})
        self.gripper = Mock()
        self.patches = [patch.object(release.safety, "ensure_exclusive_client"),
                        patch.object(release.safety, "stop_and_verify"),
                        patch.object(release, "close_connections"),
                        patch.object(release.arm_control, "load_pose", return_value=HOME),
                        patch.object(release.side, "robot_gripper", return_value=self.gripper),
                        patch.object(release, "controller_lease", side_effect=self.lease)]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    @contextmanager
    def lease(self, timeout):
        self.lease_depth += 1
        try:
            yield
        finally:
            self.lease_depth -= 1

    def batch(self, log_path):
        self.assertEqual(self.lease_depth, 0, "Parent lease must not be held across BAT")
        self.batch_calls += 1
        self.phase = "home"
        return self.batch_code

    def runner(self, execute=True, output="result"):
        return release.ReturnRelease(self.source, self.folder/output, execute=execute,
                 robot=self.robot, feedback=self.feedback, batch_runner=self.batch,
                 held_verifier=self.visual)

    def rewrite(self, **updates):
        self.source_data.update(updates)
        self.source.write_text(json.dumps(self.source_data), encoding="utf-8")

    def test_read_only_never_runs_batch_or_gripper(self):
        result = self.runner(execute=False).run()
        self.assertTrue(result["ready"])
        self.assertEqual(self.batch_calls, 0)
        self.gripper.set_opening_mm.assert_not_called()
        self.assertFalse((self.folder/"return_release.claim.json").exists())

    def test_success_releases_once_after_checked_home(self):
        result = self.runner().run()
        self.assertTrue(result["completed"])
        self.assertTrue(result["released"])
        self.assertEqual(self.batch_calls, 1)
        self.gripper.set_opening_mm.assert_called_once_with(20, settle_time=1)
        self.visual.assert_called_once()
        self.assertTrue(all(event["feedback"]["holding"] is None
                            for event in result["events"] if "feedback" in event))

    def test_footer_exit_one_is_not_motion_failure_when_actual_home_matches(self):
        self.batch_code = 1
        result = self.runner().run()
        self.assertTrue(result["completed"])
        self.assertEqual(self.batch_calls, 1)
        self.gripper.set_opening_mm.assert_called_once()

    def test_exit_zero_but_not_at_home_does_not_release(self):
        def wrong_home(log_path):
            self.batch_calls += 1
            return 0
        runner = self.runner()
        runner.batch_runner = wrong_home
        with self.assertRaisesRegex(RuntimeError, "not at the existing Home"):
            runner.run()
        self.gripper.set_opening_mm.assert_not_called()

    def test_stale_source_never_moves_or_opens(self):
        self.rewrite(final_state_created_unix=time.time()-600)
        with self.assertRaisesRegex(RuntimeError, "stale"):
            self.runner().run()
        self.assertEqual(self.batch_calls, 0)
        self.gripper.set_opening_mm.assert_not_called()

    def test_missing_source_sample_time_not_substituted_with_file_mtime(self):
        self.source_data.pop("final_state_created_unix")
        self.source.write_text(json.dumps(self.source_data), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "explicit sample timestamp"):
            self.runner().run()
        self.assertEqual(self.batch_calls, 0)

    def test_unconfirmed_source_cannot_return(self):
        self.rewrite(physical_grasp_confirmed=False)
        with self.assertRaisesRegex(RuntimeError, "no confirmed physical grasp"):
            self.runner().run()
        self.assertEqual(self.batch_calls, 0)

    def test_changed_joint_state_prevents_batch(self):
        current = state()
        current["joint_actual_position"][2] += 1
        self.robot.motion_snapshot.return_value = current
        self.robot.motion_snapshot.side_effect = None
        with self.assertRaisesRegex(RuntimeError, "Robot moved"):
            self.runner().run()
        self.assertEqual(self.batch_calls, 0)

    def test_wrong_tool_or_user_frame_prevents_batch(self):
        for key in ("current_tool_id", "current_user_id"):
            with self.subTest(key=key):
                self.robot.snapshot.return_value[key] = 1
                with self.assertRaisesRegex(RuntimeError, "tool 0 and base frame 0"):
                    self.runner(output=key).run()
                self.robot.snapshot.return_value[key] = 0
        self.assertEqual(self.batch_calls, 0)

    def test_missing_feedback_requires_fresh_actual_held_image_at_home(self):
        self.visual.return_value = {"accepted": False}
        with self.assertRaisesRegex(RuntimeError, "not physically verified"):
            self.runner().run()
        self.gripper.set_opening_mm.assert_not_called()

    def test_drop_feedback_after_home_prevents_release(self):
        self.feedback.snapshot.side_effect = [dict(available=False, holding=None, dropped=None),
                                             dict(available=True, holding=False, dropped=True)]
        with self.assertRaisesRegex(RuntimeError, "contradicts"):
            self.runner().run()
        self.gripper.set_opening_mm.assert_not_called()

    def test_uncertain_release_write_never_retries(self):
        self.gripper.set_opening_mm.side_effect = RuntimeError("ACK lost; physical outcome uncertain")
        runner = self.runner()
        with self.assertRaisesRegex(RuntimeError, "ACK lost"):
            runner.run()
        self.gripper.set_opening_mm.assert_called_once_with(20, settle_time=1)
        self.assertFalse(runner.report["released"])
        self.assertTrue(runner.report["release_write_attempted"])
        with self.assertRaisesRegex(RuntimeError, "no replay"):
            self.runner(output="second_attempt").run()
        self.assertEqual(self.batch_calls, 1)

    def test_batch_exception_is_not_retried_or_followed_by_release(self):
        runner = self.runner()
        runner.batch_runner = Mock(side_effect=RuntimeError("batch outcome uncertain"))
        with self.assertRaisesRegex(RuntimeError, "batch outcome uncertain"):
            runner.run()
        runner.batch_runner.assert_called_once()
        self.gripper.set_opening_mm.assert_not_called()

    def test_home_rgb_observation_identifies_calibrated_camera_frame(self):
        import numpy as np
        import pick_place_3d as vision
        import table_vision
        import grasp_verification
        observation = dict(tcp_pose=PICK, tool_id=0, user_id=0, created_unix=time.time())
        config = dict(camera={"sdk": None, "point_cloud_frame": "opencv_optical"})
        with patch.object(vision, "load_config", return_value=config), \
             patch.object(vision, "read_observation_pose", side_effect=lambda: copy.deepcopy(observation)), \
             patch.object(vision, "camera_transform", return_value=np.eye(4)) as camera, \
             patch.object(grasp_verification, "capture_verification", return_value={"color_path": "mock.png"}), \
             patch.object(table_vision, "verify_rgb_presence", return_value={"accepted": True}), \
             patch.object(release, "read_json", return_value={"empty_closed_reference": "reference.png"}):
            result = release.verify_held_image({"plan": {"held_model": np.eye(4).tolist()}}, self.folder)
        self.assertTrue(result["accepted"])
        self.assertEqual(camera.call_args.args[1]["camera_frame"], "opencv_optical")


if __name__ == "__main__":
    unittest.main()
