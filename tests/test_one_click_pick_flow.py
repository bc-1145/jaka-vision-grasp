"""Check the requested cycle without issuing camera or robot commands."""
import copy
import unittest
from unittest.mock import Mock, patch

import one_click_pick as pick
import grasp_verification


def state(pose):
    return {"actual_position": pose, "joint_actual_position": [0] * 6,
            "estop": 0, "collision": 0, "soft_limit": 0, "paused": False,
            "queue": 0, "active_queue": 0, "inpos": True}


CONFIG = {"camera": {},
          "gripper": {"open_command_mm": 20.0, "close_command_mm": 8.65,
                      "tcp_to_grasp_center_mm": [0, 0, 180], "jaw_axis_tcp": [1, 0, 0]},
          "model": {"shaft_z_range_mm": [3.6, 51.2], "head_z_range_mm": [0, 3.2]}}
PLAN = {"joints": {"lift": [1] * 6, "approach": [2] * 6,
                   "prepick": [3] * 6, "pick": [4] * 6},
        "poses": {"pick": [10, 20, 230, 180, 0, 0]},
        "grasp_base": [10, 20, 50], "shaft_axis_base": [0, 1, 0], "grasp_model_z_mm": 27.4}
OBSERVATION = {"tcp_pose": [10, 20, 230, 180, 0, 0], "tool_id": 0, "user_id": 0}
RETURN_HOME = pick.return_home_holding


class PickFlowTests(unittest.TestCase):
    def setUp(self):
        self.report = {}
        self.robot = Mock()
        self.robot.motion_snapshot.return_value = state(PLAN["poses"]["pick"])
        self.gripper = Mock()
        self.events = []
        self.gripper.set_opening_mm.side_effect = lambda mm, **kw: self.events.append(("gripper", mm))
        self.patches = [patch.object(pick, "save"), patch.object(pick, "close_connections"),
                        patch.object(pick.vision, "read_observation_pose", return_value=OBSERVATION),
                        patch.object(pick.side, "build_plan", return_value=copy.deepcopy(PLAN)),
                        patch.object(pick.side, "robot_gripper", return_value=self.gripper),
                        patch.object(pick, "wait_joint", side_effect=lambda robot, q, *a:
                                     self.events.append(("joint", q[0]))),
                        patch.object(grasp_verification, "capture_verification",
                                     side_effect=lambda *a: self.events.append(("camera", "after")) or
                                     {"view_path": "after.png"}),
                        patch.object(pick, "return_home_holding",
                                     side_effect=lambda *a: self.events.append(("home", "closed")))]
        self.started = [p.start() for p in self.patches]
        self.addCleanup(lambda: [p.stop() for p in reversed(self.patches)])

    def execute(self, **kwargs):
        pick.execute(self.robot, CONFIG, {"observation": OBSERVATION}, self.report, **kwargs)

    def test_open_align_close_lift_then_home_without_releasing(self):
        self.execute()
        self.assertEqual(self.events, [("gripper", 20.0), ("joint", 1), ("joint", 2),
                                     ("joint", 3), ("joint", 4), ("gripper", 8.65),
                                     ("joint", 3), ("joint", 2), ("joint", 1),
                                     ("camera", "after"), ("home", "closed")])
        self.assertTrue(self.report["holding"])
        self.assertTrue(self.report["gripper_closed"])
        self.assertFalse(self.report["visual_grasp_confirmed"])
        self.assertLess(self.report["grasp_center_error_mm"], 0.001)

    def test_wrong_height_prevents_closure_and_home(self):
        self.robot.motion_snapshot.return_value = state([10, 20, 235, 180, 0, 0])
        with self.assertRaisesRegex(RuntimeError, "未对准"):
            self.execute()
        self.gripper.set_opening_mm.assert_called_once_with(20.0)
        self.assertNotIn(("home", "closed"), self.events)

    def test_head_grasp_prevents_closure(self):
        plan = copy.deepcopy(PLAN); plan["grasp_model_z_mm"] = 2.0
        with patch.object(pick.side, "build_plan", return_value=plan):
            with self.assertRaisesRegex(RuntimeError, "未对准"):
                self.execute()
        self.gripper.set_opening_mm.assert_called_once_with(20.0)

    def test_changed_observation_prevents_opening_and_movement(self):
        changed = copy.deepcopy(OBSERVATION); changed["tcp_pose"][0] += 5
        with patch.object(pick.vision, "read_observation_pose", return_value=changed):
            with self.assertRaisesRegex(RuntimeError, "robot moved"):
                self.execute()
        self.assertEqual(self.events, [])
        self.gripper.set_opening_mm.assert_not_called()

    def test_home_uses_batch_motion_implementation_and_keeps_jaws_closed(self):
        self.report["holding"] = True
        with patch.object(pick.arm_control, "load_home_motion", return_value=(10, 20, 30, 60, 1)), \
             patch.object(pick.arm_control, "load_pose", return_value=PLAN["poses"]["pick"]), \
             patch.object(pick.arm_control, "move_home_staged") as home:
            RETURN_HOME(self.robot, self.report)
        home.assert_called_once_with(10, 20, 30, 60)
        self.gripper.set_opening_mm.assert_not_called()
        self.assertTrue(self.report["holding"])
        self.assertEqual(self.report["home_position_error_mm"], 0)

    def test_motion_during_planning_prevents_opening_and_movement(self):
        changed = copy.deepcopy(OBSERVATION); changed["tcp_pose"][0] += 5
        with patch.object(pick.vision, "read_observation_pose", side_effect=[OBSERVATION, changed]):
            with self.assertRaisesRegex(RuntimeError, "robot moved"):
                self.execute()
        self.assertEqual(self.events, [])
        self.gripper.set_opening_mm.assert_not_called()

    def test_manual_rejection_keeps_closed_at_retreat(self):
        with patch("builtins.input", return_value="NO"):
            with self.assertRaisesRegex(RuntimeError, "未执行回 Home"):
                self.execute(manual_verify=True, before_verification={"view_path": "before.png"})
        self.assertEqual([c.args[0] for c in self.gripper.set_opening_mm.call_args_list], [20.0, 8.65])
        self.assertTrue(self.report["holding"])
        self.assertNotIn(("home", "closed"), self.events)

    def test_no_match_stops_after_loading_and_cameras(self):
        with patch.object(pick, "go_to_loading", side_effect=lambda *a: self.events.append("loading")), \
             patch.object(pick, "fresh_scan_and_plan", side_effect=RuntimeError("no valid match")), \
             patch.object(pick, "execute") as execute:
            with self.assertRaisesRegex(RuntimeError, "no valid match"):
                pick.run_cycle(self.robot, CONFIG, self.report)
        self.assertEqual(self.events, ["loading", ("camera", "after")])
        self.gripper.set_opening_mm.assert_not_called()
        execute.assert_not_called()

    def test_loading_requires_home_start(self):
        with patch.object(pick.arm_control, "load_pose", return_value=[0] * 6), \
             patch.object(pick.loading, "execute") as movement:
            with self.assertRaisesRegex(RuntimeError, "从 Home 出发"):
                pick.go_to_loading(self.robot, self.report)
        movement.assert_not_called()

    def test_cycle_selects_after_loading_images_and_before_execute(self):
        with patch.object(pick, "go_to_loading", side_effect=lambda *a: self.events.append("loading")), \
             patch.object(pick, "fresh_scan_and_plan", side_effect=lambda *a:
                          self.events.append("select") or {"observation": OBSERVATION}), \
             patch.object(pick, "execute", side_effect=lambda *a, **kw: self.events.append("execute")):
            pick.run_cycle(self.robot, CONFIG, self.report)
        self.assertEqual(self.events, ["loading", ("camera", "after"), "select", "execute"])


if __name__ == "__main__":
    unittest.main()
