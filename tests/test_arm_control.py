import unittest
from unittest.mock import patch

import arm_control


class StagedHomeTests(unittest.TestCase):
    def test_home_motion_uses_maximum_profile(self):
        self.assertEqual(
            arm_control.load_home_motion(),
            (1500.0, 8000.0, 180.0, 720.0, 1.0),
        )

    def test_absolute_move_sets_fast_orientation_limits(self):
        with patch.object(arm_control, "call", return_value={"errorCode": "0"}) as command:
            arm_control.move_abs([1, 2, 3, 4, 5, 6], speed=1500, accel=8000)
        payload = command.call_args.args[0]
        self.assertEqual(payload["speed"], 1500)
        self.assertEqual(payload["accel"], 8000)
        self.assertEqual(payload["ori_vel"], 180.0)
        self.assertEqual(payload["ori_acc"], 720.0)

    def test_lifts_with_joint3_only_then_aligns_then_descends(self):
        current = [20.0, -40.0, 250.0, 170.0, 8.0, 70.0]
        home = [-90.0, -180.0, 280.0, -180.0, 0.0, 90.0]
        approach = [-90.0, -180.0, 340.0, -180.0, 0.0, 90.0]
        approach_joint = [61.726, 4.958, 55.272, -0.004, 119.771, 151.726]
        start_joint = [67.764, 46.485, 44.214, -0.443, 84.345, 170.955]
        lift_joint = list(start_joint)
        lift_joint[2] = 5.214
        lift_pose = [-100.0, -300.0, 342.0, -175.0, 4.0, 78.0]
        state = {"pose": current, "joints": start_joint}
        calls = []
        joint_calls = []

        def move(target, *_args, **_kwargs):
            calls.append(list(target))
            state["pose"] = list(target)
            return list(target)

        def move_joint(target, *_args, **_kwargs):
            joint_calls.append(list(target))
            state["joints"] = list(target)
            state["pose"] = list(lift_pose if list(target) == lift_joint else approach)
            return list(target)

        with patch.object(arm_control, "load_home", return_value=(home, 340.0)), \
                patch.object(arm_control, "load_home_joint", return_value=approach_joint), \
                patch.object(arm_control, "load_joint3_lift_config",
                             return_value=(340.0, 1.0, 100.0)), \
                patch.object(arm_control, "plan_joint3_lift",
                             return_value=(lift_joint, lift_pose)) as planner, \
                patch.object(arm_control, "get_motion_state", return_value={"inpos": True}), \
                patch.object(arm_control, "get_tcp", side_effect=lambda: list(state["pose"])), \
                patch.object(arm_control, "get_joints", side_effect=lambda: list(state["joints"])), \
                patch.object(arm_control, "move_verified", side_effect=move), \
                patch.object(arm_control, "move_joint_verified", side_effect=move_joint):
            result = arm_control.move_home_staged()

        planner.assert_called_once_with(start_joint, 340.0, 1.0, 100.0)
        changed_axes = [index for index, (before, after) in
                        enumerate(zip(start_joint, joint_calls[0])) if before != after]
        self.assertEqual(changed_axes, [2])
        self.assertLess(joint_calls[0][2], start_joint[2])
        self.assertEqual(calls, [home])
        self.assertEqual(joint_calls, [lift_joint, approach_joint])
        self.assertEqual(result, home)

    def test_skips_lift_when_already_above_clearance(self):
        current = [10.0, 20.0, 420.0, 180.0, 0.0, 80.0]
        home = [-90.0, -180.0, 280.0, -180.0, 0.0, 90.0]
        approach = [-90.0, -180.0, 340.0, -180.0, 0.0, 90.0]
        approach_joint = [61.726, 4.958, 55.272, -0.004, 119.771, 151.726]
        state = {"pose": current, "joints": [0.0] * 6}
        calls = []
        joint_calls = []

        def move(target, *_args, **_kwargs):
            calls.append(list(target))
            state["pose"] = list(target)
            return list(target)

        def move_joint(target, *_args, **_kwargs):
            joint_calls.append(list(target))
            state["joints"] = list(target)
            state["pose"] = list(approach)
            return list(target)

        with patch.object(arm_control, "load_home", return_value=(home, 340.0)), \
                patch.object(arm_control, "load_home_joint", return_value=approach_joint), \
                patch.object(arm_control, "load_joint3_lift_config",
                             return_value=(340.0, 1.0, 100.0)), \
                patch.object(arm_control, "plan_joint3_lift") as planner, \
                patch.object(arm_control, "get_motion_state", return_value={"inpos": True}), \
                patch.object(arm_control, "get_tcp", side_effect=lambda: list(state["pose"])), \
                patch.object(arm_control, "get_joints", side_effect=lambda: list(state["joints"])), \
                patch.object(arm_control, "move_verified", side_effect=move), \
                patch.object(arm_control, "move_joint_verified", side_effect=move_joint):
            arm_control.move_home_staged()

        planner.assert_not_called()
        self.assertEqual(calls, [home])
        self.assertEqual(joint_calls, [approach_joint])

    def test_already_home_sends_no_motion(self):
        home = [-90.0, -180.0, 280.0, -180.0, 0.0, 90.0]
        with patch.object(arm_control, "load_home", return_value=(home, 340.0)), \
                patch.object(arm_control, "load_home_joint", return_value=[0.0] * 6), \
                patch.object(arm_control, "load_joint3_lift_config",
                             return_value=(340.0, 1.0, 100.0)), \
                patch.object(arm_control, "get_motion_state", return_value={"inpos": True}), \
                patch.object(arm_control, "get_tcp", return_value=home), \
                patch.object(arm_control, "plan_joint3_lift") as planner, \
                patch.object(arm_control, "move_joint_verified") as move_joint, \
                patch.object(arm_control, "move_verified") as move:
            result = arm_control.move_home_staged()

        self.assertEqual(result, home)
        planner.assert_not_called()
        move_joint.assert_not_called()
        move.assert_not_called()

    def test_joint3_no_solution_stops_before_all_motion(self):
        home = [-90.0, -180.0, 280.0, -180.0, 0.0, 90.0]
        current = [20.0, -40.0, 250.0, 170.0, 8.0, 70.0]
        with patch.object(arm_control, "load_home", return_value=(home, 340.0)), \
                patch.object(arm_control, "load_home_joint", return_value=[0.0] * 6), \
                patch.object(arm_control, "load_joint3_lift_config",
                             return_value=(340.0, 1.0, 100.0)), \
                patch.object(arm_control, "get_motion_state", return_value={"inpos": True}), \
                patch.object(arm_control, "get_tcp", return_value=current), \
                patch.object(arm_control, "get_joints", return_value=[0.0] * 6), \
                patch.object(arm_control, "plan_joint3_lift",
                             side_effect=RuntimeError("joint 3 alone cannot raise")), \
                patch.object(arm_control, "move_joint_verified") as move_joint, \
                patch.object(arm_control, "move_verified") as move:
            with self.assertRaisesRegex(RuntimeError, "joint 3 alone cannot raise"):
                arm_control.move_home_staged()

        move_joint.assert_not_called()
        move.assert_not_called()

    def test_rejects_busy_robot_before_motion(self):
        with patch.object(arm_control, "load_home", return_value=([0] * 6, 340.0)), \
                patch.object(arm_control, "load_home_joint", return_value=[0] * 6), \
                patch.object(arm_control, "load_joint3_lift_config",
                             return_value=(340.0, 1.0, 100.0)), \
                patch.object(arm_control, "get_motion_state", return_value={
                    "inpos": False, "queue": 1, "active_queue": 1,
                }), patch.object(arm_control, "move_verified") as move:
            with self.assertRaisesRegex(RuntimeError, "active/faulted"):
                arm_control.move_home_staged()
        move.assert_not_called()


if __name__ == "__main__":
    unittest.main()
