import unittest
from unittest.mock import Mock, patch

import one_click_pick as pick


def state(pose):
    return {"actual_position": pose, "joint_actual_position": [0] * 6,
            "estop": 0, "collision": 0, "soft_limit": 0, "paused": False,
            "queue": 0, "active_queue": 0, "inpos": True}


class OneClickStagingTests(unittest.TestCase):
    def test_raise_is_first_motion_before_photo_pose(self):
        robot = Mock()
        robot.motion_snapshot.return_value = state([10, 20, 280, 170, 0, 90])
        report, calls = {}, []
        photo = [192, -60, 280, 179, 0, -179]
        with patch.object(pick, "load_photo_pose", return_value=(photo, 340, [1, 2, 3, 4, 5, 6])), \
                patch.object(pick, "wait_linear", side_effect=lambda *a: calls.append(("linear", a[1], a[3]))), \
                patch.object(pick, "wait_joint", side_effect=lambda *a: calls.append(("joint", a[1], a[3]))):
            pick.go_to_photo(robot, report)
        self.assertEqual(calls[0], ("linear", [10, 20, 340.0, 170, 0, 90], "先垂直提高到净空高度"))
        self.assertEqual(calls[1][0:2], ("joint", [1, 2, 3, 4, 5, 6]))
        self.assertEqual(calls[2], ("linear", photo, "下降到已标定拍照点"))

    def test_skip_navigation_when_already_at_photo_pose(self):
        photo = [192, -60, 280, 179, 0, -179]
        robot = Mock(); robot.motion_snapshot.return_value = state(photo)
        with patch.object(pick, "load_photo_pose", return_value=(photo, 340, [1] * 6)), \
                patch.object(pick, "wait_linear") as linear, patch.object(pick, "wait_joint") as joint:
            pick.go_to_photo(robot, {})
        linear.assert_not_called(); joint.assert_not_called()


if __name__ == "__main__":
    unittest.main()
