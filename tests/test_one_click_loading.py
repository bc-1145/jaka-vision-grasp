import unittest
from itertools import count
from unittest.mock import Mock, patch

import one_click_loading as loading

TARGET = [0, 0, -90, 0, -90, 0]


def state(joints, z=250, **flags):
    return {"actual_position": [100, -100, z, 180, 0, 180],
            "joint_actual_position": joints, "inpos": True,
            "queue": 0, "active_queue": 0, "estop": 0,
            "collision": 0, "soft_limit": 0, "paused": False, **flags}


class LoadingTests(unittest.TestCase):
    def staged_robot(self, *, sag=False):
        import math

        start = [10, 20, 0, 0, 0, 0]
        target = [30, 20, -1, 0, 0, 0]

        def forward(q):
            height = 100 + 10 * q[2]
            if sag and q[2] > 10:
                height -= 80 * math.sin(math.pi * (q[0] - 10) / 20)
            return [q[0], q[1], height, 0, 0, q[5]]

        def lift(q, height, *_args):
            raised = list(q)
            raised[2] = (height - 100) / 10
            return raised, forward(raised)

        robot = Mock()
        robot.forward.side_effect = forward
        raised, _ = lift(start, 340)
        robot.motion_snapshot.side_effect = [state(start, 100), state(start, 100), state(raised, 340)]
        return robot, start, target, lift

    def test_descend_last_keeps_alignment_high_and_lowers_only_j3_at_end(self):
        robot, start, target, lift = self.staged_robot()
        report = {}
        with patch.object(loading.arm, "plan_joint3_lift", side_effect=lift), \
                patch.object(loading, "move_joint") as move:
            loading.execute(robot, target, 30, 60, (340, 1, 100), report, descend_last=True)
        raised, aligned, final = [call.args[1] for call in move.call_args_list]
        self.assertEqual([i for i in range(6) if raised[i] != start[i]], [2])
        self.assertEqual(aligned[0], target[0])
        self.assertEqual(final, target)
        self.assertEqual([i for i in range(6) if aligned[i] != final[i]], [2])
        self.assertGreaterEqual(report["descent_last_plan"]["minimum_alignment_height_mm"], 340)
        self.assertEqual(move.call_args_list[1].kwargs["min_height_mm"], 340)

    def test_alignment_path_sag_rejects_all_motion_before_first_lift(self):
        robot, _start, target, lift = self.staged_robot(sag=True)
        with patch.object(loading.arm, "plan_joint3_lift", side_effect=lift), \
                patch.object(loading, "move_joint") as move:
            with self.assertRaisesRegex(RuntimeError, "提前下降"):
                loading.execute(robot, target, 30, 60, (340, 1, 100), {}, descend_last=True)
        move.assert_not_called()
        robot.call.assert_not_called()

    def test_failed_alignment_never_starts_final_descent(self):
        robot, _start, target, lift = self.staged_robot()
        with patch.object(loading.arm, "plan_joint3_lift", side_effect=lift), \
                patch.object(loading, "move_joint", side_effect=[None, RuntimeError("alignment failed")]) as move:
            with self.assertRaisesRegex(RuntimeError, "alignment failed"):
                loading.execute(robot, target, 30, 60, (340, 1, 100), {}, descend_last=True)
        self.assertEqual(move.call_count, 2)

    def test_runtime_alignment_height_drop_is_rejected(self):
        robot = Mock()
        target = [20] * 6
        robot.forward.return_value = state(target, 340)["actual_position"]
        robot.motion_snapshot.side_effect = [state([0] * 6, 340), state([10] * 6, 330)]
        with patch.object(loading.safety, "save_json"):
            with self.assertRaisesRegex(RuntimeError, "高度下降"):
                loading.move_joint(robot, target, 30, 60, "alignment", {}, min_height_mm=340)
        robot.call.assert_called_once()

    def test_controller_residual_error_accepts_only_with_tcp_close(self):
        target = [88.03545761589484, 39.993392749564535, 35.113171650037984,
                  3.135691200493649, 105.68600048321854, 179.9839906135683]
        actual = [88.03546, 39.69127, 35.11317, 3.13569, 105.68531, 179.98399]
        target_pose = [2.89256, -336.91999, 241.60581, 176.96926, -0.76930, 88.89891]
        snapshot = state(actual)
        snapshot["actual_position"] = [2.90268, -336.62592, 243.38005, 176.97382, -0.46653, 88.89883]
        self.assertTrue(loading.at_target(snapshot, target, target_pose))
        snapshot["actual_position"][2] += 5
        self.assertFalse(loading.at_target(snapshot, target, target_pose))

    def test_complete_combined_route_visits_all_three_liumang_steps(self):
        workflows = [(key, loading.sequence_settings(key)) for key in ("loading", "liumang")]
        visited = []
        with patch.object(loading, "execute", side_effect=lambda *a, **k: visited.append(a[1])), \
                patch.object(loading, "move_joint", side_effect=lambda *a: visited.append(a[1])), \
                patch.object(loading.safety, "save_json"):
            loading.execute_workflows(Mock(), workflows, {})
        self.assertEqual(visited, [step[2] for _key, route in workflows for step in route])
        self.assertEqual(len(visited), 4)

    def test_residual_error_finishes_move_without_resending(self):
        robot = Mock()
        actual = list(TARGET)
        actual[1] += 0.302
        target_pose = state(TARGET)["actual_position"]
        robot.forward.return_value = target_pose
        robot.motion_snapshot.side_effect = [state([10] * 6), state(actual), state(actual)]
        with patch.object(loading.safety, "save_json"), \
                patch.object(loading.time, "sleep"), \
                patch.object(loading.time, "monotonic", side_effect=count(step=0.2)):
            loading.move_joint(robot, TARGET, 30, 60, "residual", {})
        robot.call.assert_called_once()

    def test_stopped_far_from_target_reports_error_without_retry(self):
        robot = Mock()
        robot.forward.return_value = state(TARGET)["actual_position"]
        robot.motion_snapshot.return_value = state([10] * 6)
        with patch.object(loading.safety, "save_json"), \
                patch.object(loading.time, "sleep"), \
                patch.object(loading.time, "monotonic", side_effect=count()):
            with self.assertRaisesRegex(RuntimeError, "控制器已停止但未达到目标"):
                loading.move_joint(robot, TARGET, 30, 60, "stopped", {})
        robot.call.assert_called_once()

    def test_combined_workflows_keep_order_and_separate_lift_boundaries(self):
        robot = Mock()
        first = [("loading", "上料处", TARGET, 30, 60, (340, 1, 100))]
        second = [("liumang", "流氓", [10] * 6, 30, 60, (340, 1, 100))]
        report = {}
        with patch.object(loading, "execute_sequence") as execute:
            loading.execute_workflows(robot, [("loading", first), ("liumang", second)], report)
        self.assertEqual([c.args[1] for c in execute.call_args_list], [first, second])
        self.assertEqual(report["workflow_index"], 2)
        self.assertEqual(robot.forward.call_count, 2)

    def test_loading_failure_prevents_liumang(self):
        robot = Mock()
        route = [("point", "point", TARGET, 30, 60, (340, 1, 100))]
        with patch.object(loading, "execute_sequence", side_effect=RuntimeError("loading failed")) as execute:
            with self.assertRaisesRegex(RuntimeError, "loading failed"):
                loading.execute_workflows(robot, [("loading", route), ("liumang", route)], {})
        execute.assert_called_once()

    def test_invalid_later_workflow_prevents_all_movement(self):
        robot = Mock()
        robot.forward.side_effect = [TARGET, RuntimeError("invalid later target")]
        route = [("point", "point", TARGET, 30, 60, (340, 1, 100))]
        with patch.object(loading, "execute_sequence") as execute:
            with self.assertRaisesRegex(RuntimeError, "invalid later target"):
                loading.execute_workflows(robot, [("loading", route), ("liumang", route)], {})
        execute.assert_not_called()

    def test_sequence_lifts_only_before_first_step(self):
        robot = Mock()
        first = [10, 20, 30, 40, 50, 60]
        second = [20, 30, 40, 50, 60, 70]
        steps = [("first", "first", first, 30, 60, (340, 1, 100)),
                 ("second", "second", second, 30, 60, (340, 1, 100))]
        events = []
        report = {}
        robot.forward.side_effect = lambda q: events.append(("fk", q))
        with patch.object(loading, "execute", side_effect=lambda *a, **k: events.append(("lift_and_first", a[1]))) as execute, \
                patch.object(loading, "move_joint", side_effect=lambda *a: events.append(("second", a[1]))), \
                patch.object(loading.safety, "save_json"):
            loading.execute_sequence(robot, steps, report)
        self.assertEqual(events, [("fk", first), ("fk", second),
                                  ("lift_and_first", first), ("second", second)])
        execute.assert_called_once()
        self.assertEqual(report["step_index"], 2)

    def test_invalid_second_target_prevents_first_movement(self):
        robot = Mock()
        robot.forward.side_effect = [TARGET, RuntimeError("invalid FK")]
        steps = [("first", "first", TARGET, 30, 60, (340, 1, 100)),
                 ("second", "second", TARGET, 30, 60, (340, 1, 100))]
        with patch.object(loading, "execute") as execute, patch.object(loading, "move_joint") as move:
            with self.assertRaisesRegex(RuntimeError, "invalid FK"):
                loading.execute_sequence(robot, steps, {})
        execute.assert_not_called()
        move.assert_not_called()

    def test_first_step_failure_prevents_second(self):
        robot = Mock()
        steps = [("first", "first", TARGET, 30, 60, (340, 1, 100)),
                 ("second", "second", TARGET, 30, 60, (340, 1, 100))]
        with patch.object(loading, "execute", side_effect=RuntimeError("first failed")), \
                patch.object(loading, "move_joint") as move:
            with self.assertRaisesRegex(RuntimeError, "first failed"):
                loading.execute_sequence(robot, steps, {})
        move.assert_not_called()

    def test_original_override_restored_after_failure(self):
        robot = Mock()
        robot.call.side_effect = [{"value": 0.25}, {}, {}]
        with self.assertRaisesRegex(RuntimeError, "movement failed"):
            with loading.full_speed(robot):
                raise RuntimeError("movement failed")
        self.assertEqual(robot.call.call_args_list[1].kwargs, {"rate_value": 1.0})
        self.assertEqual(robot.call.call_args_list[2].kwargs, {"rate_value": 0.25})

    def test_only_joint3_lift_precedes_exact_loading_target(self):
        start = [10, 20, 30, 40, 50, 60]
        lift = [10, 20, 5, 40, 50, 60]
        robot = Mock()
        robot.motion_snapshot.side_effect = [state(start), state(start), state(lift, 342)]
        with patch.object(loading.arm, "plan_joint3_lift", return_value=(lift, [0, 0, 342, 0, 0, 0])), \
                patch.object(loading, "move_joint") as move:
            loading.execute(robot, TARGET, 30, 60, (340, 1, 100), {})
        self.assertEqual([call.args[1] for call in move.call_args_list], [lift, TARGET])

    def test_lift_first_raises_even_when_already_at_low_target(self):
        lift = list(TARGET)
        lift[2] += 20
        robot = Mock()
        robot.motion_snapshot.side_effect = [state(TARGET, 71), state(TARGET, 71), state(lift, 342)]
        with patch.object(loading.arm, "plan_joint3_lift", return_value=(lift, [0, 0, 342, 0, 0, 0])), \
                patch.object(loading, "move_joint") as move:
            loading.execute(robot, TARGET, 30, 60, (340, 1, 100), {}, lift_first=True)
        self.assertEqual([call.args[1] for call in move.call_args_list], [lift, TARGET])
        self.assertEqual([i for i in range(6) if lift[i] != TARGET[i]], [2])

    def test_failed_height_confirmation_prevents_target_move(self):
        lift = list(TARGET)
        lift[2] += 20
        robot = Mock()
        robot.motion_snapshot.side_effect = [state(TARGET, 71), state(TARGET, 71), state(lift, 330)]
        with patch.object(loading.arm, "plan_joint3_lift", return_value=(lift, [0, 0, 342, 0, 0, 0])), \
                patch.object(loading, "move_joint") as move:
            with self.assertRaisesRegex(RuntimeError, "未达到配置高度"):
                loading.execute(robot, TARGET, 30, 60, (340, 1, 100), {}, lift_first=True)
        self.assertEqual([call.args[1] for call in move.call_args_list], [lift])

    def test_lift_first_does_not_raise_again_when_already_high(self):
        robot = Mock()
        robot.motion_snapshot.return_value = state([10] * 6, 350)
        with patch.object(loading.arm, "plan_joint3_lift") as plan, \
                patch.object(loading, "move_joint") as move:
            loading.execute(robot, TARGET, 30, 60, (340, 1, 100), {}, lift_first=True)
        plan.assert_not_called()
        self.assertEqual([call.args[1] for call in move.call_args_list], [TARGET])

    def test_default_behavior_still_skips_when_at_target(self):
        robot = Mock()
        robot.motion_snapshot.return_value = state(TARGET, 71)
        with patch.object(loading.arm, "plan_joint3_lift") as plan, \
                patch.object(loading, "move_joint") as move:
            loading.execute(robot, TARGET, 30, 60, (340, 1, 100), {})
        plan.assert_not_called()
        move.assert_not_called()

    def test_workflows_apply_mandatory_lift_only_to_first_workflow(self):
        route = [("point", "point", TARGET, 30, 60, (340, 1, 100))]
        with patch.object(loading, "execute_sequence") as execute:
            loading.execute_workflows(Mock(), [("first", route), ("second", route)], {}, lift_first=True)
        self.assertEqual([call.kwargs["lift_first"] for call in execute.call_args_list], [True, False])

    def test_collision_prevents_all_movement(self):
        robot = Mock()
        robot.motion_snapshot.return_value = state([0] * 6, collision=1)
        with patch.object(loading, "move_joint") as move:
            with self.assertRaisesRegex(RuntimeError, "collision"):
                loading.execute(robot, TARGET, 30, 60, (340, 1, 100), {})
        move.assert_not_called()
        robot.call.assert_not_called()

    def test_start_changed_during_planning_prevents_movement(self):
        robot = Mock()
        robot.motion_snapshot.side_effect = [state([0] * 6), state([1] * 6)]
        with patch.object(loading.arm, "plan_joint3_lift", return_value=([0] * 6, [0] * 6)), \
                patch.object(loading, "move_joint") as move:
            with self.assertRaisesRegex(RuntimeError, "起点发生变化"):
                loading.execute(robot, TARGET, 30, 60, (340, 1, 100), {})
        move.assert_not_called()

    def test_motion_ack_loss_is_never_retried(self):
        robot = Mock()
        robot.motion_snapshot.return_value = state([0] * 6)
        robot.call.side_effect = ConnectionError("ack lost")
        report = {}
        with patch.object(loading.safety, "save_json"):
            with self.assertRaises(ConnectionError):
                loading.move_joint(robot, TARGET, 30, 60, "上料", report)
        robot.call.assert_called_once()
        self.assertTrue(report["motion_attempted"])

    def test_full_turn_difference_is_not_reported_as_arrived(self):
        robot = Mock()
        actual = list(TARGET)
        actual[0] = 360
        robot.motion_snapshot.return_value = state(actual, 400)
        with patch.object(loading, "move_joint") as move:
            loading.execute(robot, TARGET, 30, 60, (340, 1, 100), {})
        move.assert_called_once()
        self.assertEqual(move.call_args.args[1], TARGET)


if __name__ == "__main__":
    unittest.main()
