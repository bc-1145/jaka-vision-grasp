import copy
import unittest

import numpy as np

from auto_bin_grasp import empty_inspection_retreat
from tests.test_bin_grasp_plan import FakeRobot


class BinInspectionTests(unittest.TestCase):
    def setUp(self):
        self.pose = [0, 0, 250, 180, 0, 0]
        self.robot = FakeRobot(self.pose)
        self.observation = dict(tcp_pose=self.pose, tool_id=0, user_id=0)
        self.config = {"camera": {"mount": "eye_in_hand", "tcp_T_camera": np.eye(4).tolist(),
                                   "calibration_tool_id": 0, "point_cloud_frame": "opencv_optical"}}

    def test_checked_retreat_increases_height_without_orientation_change(self):
        result = empty_inspection_retreat(self.robot, self.config, self.observation, self.robot.motion_snapshot())
        self.assertAlmostEqual(result["pose"][2], 290)
        self.assertEqual(result["pose"][3:], self.pose[3:])
        self.assertEqual(len(result["samples"]), 15)
        self.assertTrue(result["gripper_unchanged"])

    def test_sideways_view_rejected(self):
        self.robot = FakeRobot([0, 0, 250, 90, 0, 0])
        self.observation["tcp_pose"] = self.robot.pose.tolist()
        with self.assertRaisesRegex(RuntimeError, "upward"):
            empty_inspection_retreat(self.robot, self.config, self.observation, self.robot.motion_snapshot())

    def test_moved_robot_rejected(self):
        state = self.robot.motion_snapshot()
        state["actual_position"][0] += 2
        with self.assertRaisesRegex(RuntimeError, "no longer matches"):
            empty_inspection_retreat(self.robot, self.config, self.observation, state)

    def test_ik_mismatch_rejected(self):
        self.robot.forward = lambda q: [0, 0, 0, 0, 0, 0]
        with self.assertRaisesRegex(RuntimeError, "IK/FK"):
            empty_inspection_retreat(self.robot, self.config, self.observation, self.robot.motion_snapshot())


if __name__ == "__main__":
    unittest.main()
