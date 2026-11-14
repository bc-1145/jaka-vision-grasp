import copy
import math
import unittest
from unittest.mock import patch

import numpy as np

import horizontal_sway as sway


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeRobot:
    def __init__(self):
        self.joints = [0, 40, 40, -90, -5, 275]
        self.commands = []
        self.height_error = 0
        self.height_error_after_write = 0
        self.fail_write = None
        self.stuck = False

    def forward(self, joints):
        angle = math.radians(joints[0])
        return np.array([529 * math.sin(angle), -529 * math.cos(angle) + 10 * (joints[1] - 40),
                         334.69 + 10 * (joints[2] - 40) + self.wrist_height(joints[3], joints[4]),
                         joints[3], joints[4], joints[5] + joints[0]])

    def wrist_height(self, j4, j5):
        return 2 * (j5 + 5)

    def inverse(self, pose, reference):
        def unwrap(angle, ref):
            return angle + 360 * round((ref - angle) / 360)

        j1 = unwrap(math.degrees(math.asin(pose[0] / 529)), reference[0])
        j2 = 40 + (pose[1] + 529 * math.cos(math.radians(j1))) / 10
        j4, j5 = unwrap(pose[3], reference[3]), unwrap(pose[4], reference[4])
        j3 = 40 + (pose[2] - 334.69 - self.wrist_height(j4, j5)) / 10
        j6 = unwrap(pose[5] - j1, reference[5])
        return np.array([j1, j2, j3, j4, j5, j6])

    def motion_snapshot(self):
        pose = self.forward(self.joints).tolist()
        pose[2] += self.height_error
        return {"actual_position": pose, "joint_actual_position": list(self.joints),
                "inpos": True, "queue": 0, "active_queue": 0,
                "estop": 0, "collision": 0, "soft_limit": 0, "paused": False}

    def call(self, name, **parameters):
        self.commands.append((name, parameters))
        if not self.stuck:
            self.joints = self.inverse(parameters["cartPosition"], self.joints).tolist()
        self.height_error = self.height_error_after_write
        if self.fail_write:
            raise self.fail_write
        return {"errorCode": "0", "cmdName": name}


class CoupledRobot(FakeRobot):
    def wrist_height(self, j4, j5):
        return super().wrist_height(j4, j5) + 0.02 * (j4 + 90) ** 2


class DelayedRobot(FakeRobot):
    """Separate ACK, stale motion feedback and actual settling time."""
    def __init__(self, clock, settle_s=2.46, read_s=0.03):
        super().__init__()
        self.clock = clock
        self.settle_s = settle_s
        self.read_s = read_s
        self.ready_at = 0.0

    def call(self, name, **parameters):
        if self.clock.time() < self.ready_at:
            raise AssertionError("new motion sent before prior motion settled")
        self.ready_at = self.clock.time() + self.settle_s
        result = super().call(name, **parameters)
        self.clock.sleep(0.02)
        return result

    def motion_snapshot(self):
        state = super().motion_snapshot()
        state["inpos"] = self.clock.time() >= self.ready_at
        state["active_queue"] = int(not state["inpos"])
        self.clock.sleep(self.read_s)
        return state


class SwayTests(unittest.TestCase):
    def setUp(self):
        self.robot = FakeRobot()
        self.clock = FakeClock()
        self.addCleanup(patch.stopall)
        patch.object(sway.safety, "save_json").start()
        patch.object(sway.safety, "check_state").start()
        patch.object(sway, "read_controller_status", return_value={}).start()
        patch.object(sway, "DURATION", 20.0).start()
        patch.object(sway, "DEFAULT_CYCLES", 3).start()
        # Retain the original two-cycle timing regressions as well as testing
        # the faster default separately.
        self.plan = sway.make_plan(self.robot, cycles=2)
        self.report = {"completed": False}

    def run_sway(self):
        return sway.execute(self.robot, self.plan, self.report,
                            clock=self.clock.time, sleep=self.clock.sleep)

    def test_plan_covers_full_span_at_constant_height_without_moving(self):
        self.assertEqual(self.robot.commands, [])
        for name, dx in (("left", -200), ("right", 200)):
            self.assertAlmostEqual(self.plan["endpoint_poses"][name][0], dx)
            self.assertAlmostEqual(self.plan["endpoint_poses"][name][2], 334.69)
        self.assertLess(self.plan["max_planned_height_error_mm"], 1e-9)
        self.assertEqual(self.plan["legs"][-1]["joints"], self.plan["origin_joints"])

    def test_twenty_seconds_coordinates_wrist_and_returns_all_joints(self):
        self.run_sway()
        self.assertEqual(len(self.robot.commands), 8)
        self.assertAlmostEqual(self.clock.now, 20)
        self.assertTrue(self.report["completed"])
        np.testing.assert_allclose(self.robot.joints, self.plan["origin_joints"], atol=1e-8)
        for name, args in self.robot.commands:
            self.assertEqual(name, "moveL")
            self.assertEqual(args["cartPosition"][2], self.plan["height_mm"])
            self.assertEqual(args["tol"], 0)
            self.assertLessEqual(args["speed"], sway.MAX_LINEAR_SPEED)
        right = self.plan["legs"][0]["joints"]
        left = self.plan["legs"][2]["joints"]
        for index, delta in ((3, 20), (4, -15), (5, -30)):
            self.assertAlmostEqual(right[index] - self.plan["origin_joints"][index], delta)
        self.assertLess(left[3], self.plan["origin_joints"][3])
        self.assertGreater(left[5], self.plan["origin_joints"][5])

    def test_faster_default_completes_three_cycles_with_delayed_feedback(self):
        self.plan = sway.make_plan(self.robot)
        self.assertEqual(self.plan["cycles"], 3)
        self.assertEqual(len(self.plan["legs"]), 12)
        self.assertGreater(self.plan["legs"][0]["speed"], 190)
        self.robot = DelayedRobot(self.clock, settle_s=1.52)
        self.run_sway()
        self.assertTrue(self.report["completed"])
        self.assertEqual(len(self.robot.commands), 12)
        self.assertGreaterEqual(self.report["elapsed_s"], 20)
        self.assertLessEqual(self.report["elapsed_s"], 20 + sway.TOTAL_GRACE)
        np.testing.assert_allclose(self.robot.joints, self.plan["origin_joints"], atol=1e-8)

    def test_one_minute_keeps_fast_tempo_and_returns_after_nine_cycles(self):
        previous_plan = sway.make_plan(self.robot)
        with patch.object(sway, "DURATION", 60.0), patch.object(sway, "DEFAULT_CYCLES", 9):
            self.plan = sway.make_plan(self.robot)
        self.assertEqual(self.plan["duration_s"], 60)
        self.assertEqual(self.plan["cycles"], 9)
        self.assertEqual(len(self.plan["legs"]), 36)
        self.assertAlmostEqual(self.plan["legs"][-1]["arrival_s"], 60)
        for old, new in zip(previous_plan["legs"], self.plan["legs"]):
            self.assertAlmostEqual(old["speed"], new["speed"])
        self.robot = DelayedRobot(self.clock, settle_s=1.52)
        self.run_sway()
        self.assertTrue(self.report["completed"])
        self.assertEqual(len(self.robot.commands), 36)
        self.assertGreaterEqual(self.report["elapsed_s"], 60)
        self.assertLessEqual(self.report["elapsed_s"], 60 + sway.TOTAL_GRACE)
        np.testing.assert_allclose(self.robot.joints, self.plan["origin_joints"], atol=1e-8)

    def test_longer_paths_choose_feasible_tempo_without_increasing_speed_limit(self):
        self.robot.joints[0] = 15
        with patch.object(sway, "DURATION", 60.0), patch.object(sway, "DEFAULT_CYCLES", 9):
            with self.assertRaisesRegex(sway.TempoError, "TCP 速度"):
                sway.make_plan(self.robot, cycles=9)
            self.plan = sway.make_plan(self.robot)
        self.assertEqual(self.plan["duration_s"], 60)
        self.assertEqual(self.plan["requested_cycles"], 9)
        self.assertLess(self.plan["cycles"], 9)
        self.assertEqual(self.plan["rejected_tempos"][0]["cycles"], 9)
        self.assertEqual(self.plan["wrist_amplitudes_deg"], [20, 15, 30])
        self.assertEqual(self.plan["amplitude_mm"], 200)
        self.assertEqual(self.robot.commands, [])
        for leg in self.plan["legs"]:
            self.assertLessEqual(leg["speed"], sway.MAX_LINEAR_SPEED)
            self.assertLessEqual(max(leg["sampled_joint_speed_deg_s"]), sway.MAX_JOINT_SPEED)
            self.assertLessEqual(max(leg["sampled_joint_accel_deg_s2"]), sway.MAX_JOINT_ACCEL)
        self.run_sway()
        self.assertTrue(self.report["completed"])
        self.assertAlmostEqual(self.report["elapsed_s"], 60)
        np.testing.assert_allclose(self.robot.joints, self.plan["origin_joints"], atol=1e-8)

    def test_no_feasible_tempo_never_sends_motion(self):
        with patch.object(sway, "MAX_LINEAR_SPEED", 1):
            with self.assertRaisesRegex(sway.TempoError, "没有可用节奏"):
                sway.make_plan(self.robot)
        self.assertEqual(self.robot.commands, [])

    def test_geometric_branch_error_is_not_treated_as_tempo_problem(self):
        inverse = self.robot.inverse

        def flipped(pose, reference):
            q = inverse(pose, reference)
            q[5] += 360
            return q

        with patch.object(self.robot, "inverse", side_effect=flipped):
            with self.assertRaisesRegex(ValueError, "关节跳转") as caught:
                sway.make_plan(self.robot)
        self.assertNotIsInstance(caught.exception, sway.TempoError)
        self.assertEqual(self.robot.commands, [])

    def test_cartesian_height_preserves_large_wrist_despite_joint_interpolation_bow(self):
        robot = CoupledRobot()
        plan = sway.make_plan(robot)
        self.assertNotIn("wrist_scale", plan)
        self.assertEqual(plan["wrist_amplitudes_deg"], [20, 15, 30])
        self.assertLessEqual(plan["max_planned_height_error_mm"], 0.1)
        self.assertNotEqual(plan["j3_compensation_deg"]["right"], 0)
        self.assertEqual(robot.commands, [])
        for name in ("left", "right"):
            self.assertAlmostEqual(plan["endpoint_poses"][name][2], plan["height_mm"], places=6)
        midpoint = (np.asarray(plan["origin_joints"]) + plan["legs"][0]["joints"]) / 2
        self.assertGreater(abs(robot.forward(midpoint)[2] - plan["height_mm"]), 0.1)
        self.assertTrue(all(leg["pose"][2] == plan["height_mm"] for leg in plan["legs"]))

    def test_branch_jump_and_excessive_tempo_fail_before_motion(self):
        robot = FakeRobot()
        inverse = robot.inverse

        def flipped(pose, reference):
            q = inverse(pose, reference)
            q[5] += 360
            return q

        with patch.object(robot, "inverse", side_effect=flipped):
            with self.assertRaisesRegex(ValueError, "跳转"):
                sway.make_plan(robot)
        with self.assertRaises(ValueError):
            sway.make_plan(robot, cycles=4)
        self.assertEqual(robot.commands, [])

    def test_impossible_height_compensation_never_sends_motion(self):
        robot = FakeRobot()
        original = robot.forward

        def uncorrectable(joints):
            pose = original(joints)
            pose[2] = 334.69 + joints[3] + 90
            return pose

        with patch.object(robot, "forward", side_effect=uncorrectable):
            with self.assertRaisesRegex(ValueError, "J3 无法"):
                sway.make_plan(robot)
        self.assertEqual(robot.commands, [])

    def test_unreachable_span_and_joint_guard_fail_before_motion(self):
        with self.assertRaises(ValueError):
            sway.endpoint_joints([500, 10, 334, 0, 0, 0], [0] * 6, 200)
        with self.assertRaises(ValueError):
            sway.endpoint_joints([0, -529, 334, 0, 0, 0], [170, 0, 0, 0, 0, 0], 200)
        self.assertEqual(self.robot.commands, [])

    def test_height_violation_stops_without_returning_or_retrying(self):
        self.robot.height_error_after_write = 2
        with patch.object(sway.safety, "stop_and_verify") as stop:
            with self.assertRaisesRegex(RuntimeError, "高度"):
                self.run_sway()
        stop.assert_called_once_with(self.robot)
        self.assertEqual(len(self.robot.commands), 1)
        self.assertFalse(self.report["completed"])

    def test_lost_ack_and_keyboard_interrupt_stop_without_replay(self):
        for error in (ConnectionError("lost ACK"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                self.robot = FakeRobot()
                self.robot.fail_write = error
                with patch.object(sway.safety, "stop_and_verify") as stop:
                    with self.assertRaises(type(error)):
                        self.run_sway()
                stop.assert_called_once_with(self.robot)
                self.assertEqual(len(self.robot.commands), 1)

    def test_late_arrival_stops_at_first_deadline(self):
        self.robot.stuck = True
        with patch.object(sway.safety, "stop_and_verify") as stop:
            with self.assertRaisesRegex(TimeoutError, "未按时"):
                self.run_sway()
        self.assertAlmostEqual(self.clock.now, 2.5 + sway.ENDPOINT_GRACE)
        self.assertEqual(len(self.robot.commands), 1)
        stop.assert_called_once_with(self.robot)
        result = self.report["leg_results"][0]
        self.assertGreater(result["position_error_mm"], 100)
        self.assertIn("last_snapshot", result)

    def test_feedback_latency_and_delayed_idle_do_not_steal_next_leg_budget(self):
        self.robot = DelayedRobot(self.clock)
        self.run_sway()
        self.assertTrue(self.report["completed"])
        self.assertEqual(len(self.robot.commands), 8)
        self.assertGreater(self.report["leg_results"][0]["reached_s"], 2.5)
        self.assertLessEqual(self.report["elapsed_s"], 20 + sway.TOTAL_GRACE)
        for result, (_, command) in zip(self.report["leg_results"], self.robot.commands):
            self.assertLessEqual(result["reached_s"], result["deadline_s"])
            self.assertEqual(command["speed"], self.plan["legs"][0]["speed"])
            self.assertAlmostEqual(result["snapshot_read_s"], 0.03)

    def test_repeated_delays_finish_at_center_within_duration(self):
        self.robot = DelayedRobot(self.clock, settle_s=2.65)
        with patch.object(sway.safety, "stop_and_verify") as stop:
            self.run_sway()
        self.assertTrue(self.report["completed"])
        self.assertLess(len(self.robot.commands), 8)
        self.assertEqual(len(self.robot.commands) % 2, 0)
        self.assertLessEqual(self.report["elapsed_s"], 20 + sway.TOTAL_GRACE)
        np.testing.assert_allclose(self.robot.joints, self.plan["origin_joints"], atol=1e-8)
        stop.assert_not_called()

    def test_in_tolerance_idle_feedback_just_after_deadline_is_accepted(self):
        self.robot = DelayedRobot(self.clock, settle_s=2.84, read_s=0.06)
        self.run_sway()
        first = self.report["leg_results"][0]
        self.assertGreater(first["feedback_lateness_s"], 0)
        self.assertLessEqual(first["feedback_lateness_s"], sway.FEEDBACK_GRACE)
        self.assertTrue(self.report["completed"])
        np.testing.assert_allclose(self.robot.joints, self.plan["origin_joints"], atol=1e-8)

    def test_logged_settling_pattern_does_not_accumulate_into_false_timeout(self):
        with patch.object(sway, "DURATION", 60.0), patch.object(sway, "DEFAULT_CYCLES", 9):
            self.plan = sway.make_plan(self.robot)
        self.robot = DelayedRobot(self.clock)
        original_call = self.robot.call
        # Observed durations from the failed run, including its seventh leg.
        durations = [1.657, 1.828, 1.750, 1.625, 1.641, 1.797, 1.750]

        def variable_settling(name, **parameters):
            self.robot.settle_s = durations[len(self.robot.commands) % len(durations)]
            return original_call(name, **parameters)

        with patch.object(self.robot, "call", side_effect=variable_settling):
            self.run_sway()
        self.assertTrue(self.report["completed"])
        self.assertGreaterEqual(self.report["completed_legs"], 30)
        self.assertLess(self.report["completed_legs"], 36)
        self.assertEqual(self.report["completed_legs"] % 2, 0)
        self.assertLessEqual(self.report["elapsed_s"], 60 + sway.TOTAL_GRACE)
        np.testing.assert_allclose(self.robot.joints, self.plan["origin_joints"], atol=1e-8)

    def test_idle_reply_received_after_deadline_still_stops(self):
        self.robot = DelayedRobot(self.clock, settle_s=0, read_s=3)
        with patch.object(sway.safety, "stop_and_verify") as stop:
            with self.assertRaisesRegex(TimeoutError, "未按时"):
                self.run_sway()
        self.assertTrue(self.report["leg_results"][0]["idle"])
        self.assertEqual(len(self.robot.commands), 1)
        stop.assert_called_once_with(self.robot)

    def test_slow_log_write_cancels_command_before_dispatch(self):
        with patch.object(sway.safety, "save_json", side_effect=lambda *args: self.clock.sleep(1)), \
                patch.object(sway.safety, "stop_and_verify") as stop:
            with self.assertRaisesRegex(TimeoutError, "写入耗时"):
                self.run_sway()
        self.assertEqual(self.robot.commands, [])

    def test_changed_start_never_sends_motion_or_stop(self):
        self.robot.joints[1] += 1
        with patch.object(sway.safety, "stop_and_verify") as stop:
            with self.assertRaisesRegex(RuntimeError, "起点改变"):
                self.run_sway()
        self.assertEqual(self.robot.commands, [])
        stop.assert_not_called()

    def test_other_axes_and_inflight_faults_are_blocked(self):
        for change in ({"collision": 1}, {"joint_actual_position": [0, 40, 40, -90, -5, 999]}):
            state = copy.deepcopy(self.robot.motion_snapshot())
            state.update(change)
            with self.assertRaises(RuntimeError):
                sway.check_envelope(state, self.plan, self.plan["legs"][0])

    def test_arrival_waits_for_wrist_even_when_j1_has_arrived(self):
        original = self.robot.call

        def incomplete(name, **parameters):
            result = original(name, **parameters)
            self.robot.joints[3] = self.plan["origin_joints"][3]
            return result

        with patch.object(self.robot, "call", side_effect=incomplete), \
                patch.object(sway.safety, "stop_and_verify") as stop:
            with self.assertRaisesRegex(TimeoutError, "未按时"):
                self.run_sway()
        self.assertEqual(len(self.robot.commands), 1)
        stop.assert_called_once_with(self.robot)


if __name__ == "__main__":
    unittest.main()
