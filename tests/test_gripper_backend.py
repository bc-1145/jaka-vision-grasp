import unittest
from unittest.mock import patch

from gripper_backend import (
    GripperClient, WriteUncertainError, validate_target, opening_to_words, words_to_opening,
)


class GripperTests(unittest.TestCase):
    def test_millimeter_conversion_and_range(self):
        for opening, words in ((0, (0, 0)), (5, (16544, 0)),
                               (10, (16672, 0)), (20, (16800, 0)),
                               (12.34, (16709, 28836))):
            self.assertEqual(opening_to_words(opening), words)
            self.assertAlmostEqual(words_to_opening(*words), opening, places=5)
        for invalid in (-0.01, 20.01, "NaN", "inf", "bad", None):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                opening_to_words(invalid)
        for words in ((65536, 0), (1.5, 0), (0x7f80, 0), (0xbf80, 0), (16800, 1)):
            with self.subTest(words=words), self.assertRaises(ValueError):
                words_to_opening(*words)

    def test_millimeters_write_both_float_words(self):
        calls = []

        def response(obj, **kwargs):
            calls.append(obj)
            return {"errorCode": 0, "value": 16709 if obj.get("index") == 3 else 28836}

        with patch("gripper_backend.send_command", side_effect=response):
            snapshot = GripperClient().set_opening_mm(12.34, settle_time=0)
        writes = [c for c in calls if c["cmdName"] == "set_analog_output"]
        self.assertEqual([(w["index"], w["value"]) for w in writes],
                         [(3, 16709.0), (4, 28836.0)])
        self.assertAlmostEqual(words_to_opening(snapshot.high, snapshot.low), 12.34, places=5)

    def test_live_write_skips_settle_delay_and_still_verifies_readback(self):
        with patch("gripper_backend.send_command", return_value={"errorCode": 0, "value": 0}), patch("gripper_backend.time.sleep") as sleep:
            result = GripperClient().set_opening_mm(0, settle_time=0)
        self.assertEqual(result.high, 0)
        sleep.assert_not_called()

    def test_rejects_invalid_targets_before_network(self):
        with patch("gripper_backend.send_command") as command:
            client = GripperClient()
            for value in (-1, 16801, "NaN", "inf", "bad", None):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    client.set_target(value)
            command.assert_not_called()
        self.assertEqual(validate_target("8400"), 8400)

    def test_simulation_never_calls_hardware(self):
        with patch("gripper_backend.send_command") as command:
            client = GripperClient(simulated=True)
            self.assertEqual(client.snapshot().high, 0)
            self.assertEqual(client.set_target(8400).high, 8400)
            self.assertEqual(client.set_target(16800).low, 0)
            command.assert_not_called()

    def test_connect_snapshot_is_read_only(self):
        calls = []

        def response(obj, **kwargs):
            calls.append(obj)
            self.assertEqual(kwargs, {"timeout": 3.0, "ip": "192.168.1.2"})
            return {"errorCode": 0, "value": 0}

        with patch("gripper_backend.send_command", side_effect=response):
            GripperClient("192.168.1.2").snapshot()
        self.assertEqual([c["cmdName"] for c in calls], [
            "get_robot_state", "get_analog_output", "get_analog_output"])
        self.assertEqual([c["index"] for c in calls[1:]], [3, 4])

    def test_write_order_and_payload_match_sdk(self):
        calls = []

        def response(obj, **kwargs):
            calls.append(obj)
            return {"errorCode": "0", "value": 8400 if obj.get("index") == 3 else 0}

        with patch("gripper_backend.send_command", side_effect=response), patch("gripper_backend.time.sleep"):
            result = GripperClient().set_target(8400)
        writes = [c for c in calls if c["cmdName"] == "set_analog_output"]
        self.assertEqual(calls[0]["cmdName"], "get_motion_state")
        self.assertEqual(writes, [
            {"cmdName": "set_analog_output", "type": 2, "index": 3, "value": 8400.0},
            {"cmdName": "set_analog_output", "type": 2, "index": 4, "value": 0.0}])
        self.assertEqual(result.high, 8400)

    def test_fault_prevents_writes(self):
        for fault in ("estop", "collision", "soft_limit"):
            with self.subTest(fault=fault), patch("gripper_backend.send_command", return_value={
                    "errorCode": 0, fault: True}) as command:
                with self.assertRaises(RuntimeError):
                    GripperClient().set_target(100)
                self.assertEqual(command.call_count, 1)

    def test_partial_write_does_not_retry(self):
        with patch("gripper_backend.send_command", side_effect=[
                {"errorCode": 0}, {"errorCode": 0}, TimeoutError("no reply")]) as command:
            with self.assertRaisesRegex(WriteUncertainError, "寄存器 4"):
                GripperClient().set_target(100)
            self.assertEqual(command.call_count, 3)

    def test_readback_mismatch_is_uncertain(self):
        with patch("gripper_backend.send_command", return_value={"errorCode": 0, "value": 0}), patch("gripper_backend.time.sleep"):
            with self.assertRaisesRegex(WriteUncertainError, "回读"):
                GripperClient().set_target(100)

    def test_error_code_or_missing_readback_fails_connection(self):
        for reply in ({"errorCode": 5}, {"errorCode": 0}, {"value": 0}):
            with self.subTest(reply=reply), patch("gripper_backend.send_command", return_value=reply):
                with self.assertRaises(RuntimeError):
                    GripperClient().snapshot()


if __name__ == "__main__":
    unittest.main()
