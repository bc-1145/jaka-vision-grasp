import copy
import math
import struct
import time
import unittest
from unittest.mock import patch

from gripper_backend import GripperProfile
from gripper_feedback import GripperFeedback, feedback_from_status, read_feedback


PROFILE = GripperProfile(device_id="robot", unit_id=1)


def words(value):
    return list(struct.unpack(">HH", struct.pack(">f", value)))


def mapped(state=2, initialization=5, position=8.65, current=.25):
    values = [initialization, state, *words(position), 0, 0, *words(current)]
    return {"extio": {"mode": 1, "num": 1,
            "setup": [[0, "robot", [115200, 8, 1, 78, 1],
                       [[0, 0], [0, 0], [64, 8], [0, 20]], "robot"]],
            "pinmap": [[0, 0, 0, 0]], "status": [[], [], values, [16800, 0]]}}


class FeedbackTests(unittest.TestCase):
    def test_real_holding_feedback_decodes_position_and_current(self):
        value = feedback_from_status(mapped(), PROFILE)
        self.assertTrue(value["available"])
        self.assertIs(value["holding"], True)
        self.assertEqual(value["grip_state"], 2)
        self.assertAlmostEqual(value["actual_position_mm"], 8.65, places=4)
        self.assertAlmostEqual(value["current_a"], .25)

    def test_no_input_mapping_does_not_use_ao_setpoint_as_grasp(self):
        status = mapped()
        status["extio"]["setup"][0][3][2] = [0, 0]
        status["extio"]["status"][2] = []
        value = feedback_from_status(status, PROFILE)
        self.assertFalse(value["available"])
        self.assertIsNone(value["holding"])
        self.assertIsNone(value["actual_position_mm"])
        self.assertIn("0x0041", value["reason"])

    def test_stopped_bus_never_confirms_cached_holding(self):
        status = mapped()
        status["extio"]["mode"] = 0
        value = feedback_from_status(status, PROFILE)
        self.assertFalse(value["available"])
        self.assertIsNone(value["holding"])

    def test_drop_and_moving_are_not_holding(self):
        for state in (0, 1, 3):
            with self.subTest(state=state):
                value = feedback_from_status(mapped(state=state), PROFILE)
                self.assertIs(value["holding"], False)
                self.assertEqual(value["dropped"], state == 3)

    def test_uninitialized_feedback_cannot_confirm_hold(self):
        value = feedback_from_status(mapped(initialization=0), PROFILE)
        self.assertFalse(value["available"])
        self.assertIsNone(value["holding"])

    def test_unknown_state_nan_position_and_invalid_words_are_rejected(self):
        fixtures = [mapped(state=4), mapped(position=float("nan")), mapped(current=float("inf")),
                    mapped(position=100)]
        badword = mapped()
        badword["extio"]["status"][2][1] = -1
        fixtures.append(badword)
        for status in fixtures:
            with self.subTest(status=status):
                value = feedback_from_status(status, PROFILE)
                self.assertFalse(value["available"])
                self.assertIsNone(value["holding"])

    def test_identity_and_unit_changes_are_rejected(self):
        for field, value in ((4, "other"),):
            status = mapped()
            status["extio"]["setup"][0][field] = value
            self.assertFalse(feedback_from_status(status, PROFILE)["available"])
        status = mapped()
        status["extio"]["setup"][0][2][2] = 2
        self.assertFalse(feedback_from_status(status, PROFILE)["available"])

    def test_second_device_uses_its_own_input_offset(self):
        status = mapped()
        other = copy.deepcopy(status["extio"]["setup"][0])
        other[4] = "other"
        other[2][2] = 2
        status["extio"]["setup"].insert(0, other)
        status["extio"]["pinmap"] = [[0, 0, 0, 0], [0, 0, 8, 20]]
        status["extio"]["status"][2] = [5, 3, *words(20), 0, 0, *words(0)] + status["extio"]["status"][2]
        status["extio"]["num"] = 2
        self.assertIs(feedback_from_status(status, PROFILE)["holding"], True)

    def test_position_and_current_without_state_never_confirm_grasp(self):
        status = mapped()
        status["extio"]["setup"][0][3][2] = [66, 6]
        status["extio"]["status"][2] = status["extio"]["status"][2][2:]
        value = feedback_from_status(status, PROFILE)
        self.assertIsNotNone(value["actual_position_mm"])
        self.assertIsNotNone(value["current_a"])
        self.assertFalse(value["available"])
        self.assertIsNone(value["holding"])

    def test_missing_input_payload_does_not_index_into_outputs(self):
        status = mapped()
        status["extio"]["status"][2] = []
        self.assertIsNone(feedback_from_status(status, PROFILE)["holding"])

    def test_read_uses_status_port_only(self):
        with patch("gripper_feedback.read_controller_status", return_value=mapped()) as reader:
            value = read_feedback("10.5.5.100", PROFILE)
        self.assertIs(value["holding"], True)
        reader.assert_called_once_with("10.5.5.100", 2.0)

    def test_unavailable_hold_verification_returns_without_retry(self):
        with patch("gripper_feedback.read_feedback") as reader:
            reader.return_value = dict(available=False, reason="not mapped")
            result = GripperFeedback("10.5.5.100", PROFILE).wait_for_hold()
        self.assertFalse(result["confirmed"])
        self.assertEqual(reader.call_count, 1)

    def test_hold_verification_requires_several_fresh_holding_samples(self):
        reader = lambda: feedback_from_status(mapped(), PROFILE)
        result = GripperFeedback(reader=reader).wait_for_hold(timeout=.5, stable_samples=3, interval=.01)
        self.assertTrue(result["confirmed"])
        self.assertEqual(len(result["samples"]), 3)

    def test_repeated_or_stale_feedback_is_not_confirmed(self):
        for stamp in (time.time() - 30, time.time()):
            sample = feedback_from_status(mapped(), PROFILE, created_unix=stamp)
            result = GripperFeedback(reader=lambda: sample).wait_for_hold(timeout=.5, interval=.01)
            self.assertFalse(result["confirmed"])
            self.assertIn("stale", result["reason"])


if __name__ == "__main__":
    unittest.main()
