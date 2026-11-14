import copy
import threading
import unittest
from unittest.mock import patch

from gripper_backend import (
    GripperClient, GripperProfile, profiles_from_status, words_to_opening,
)


def status_fixture():
    return {"extio": {
        "num": 2,
        "setup": [
            [0, "机器人夹爪", [115200, 8, 1, 78, 1],
             [[0, 0], [0, 0], [0, 0], [0, 20]], "arm-gripper"],
            [0, "定位夹爪", [115200, 8, 2, 78, 1],
             [[0, 0], [0, 0], [0, 0], [0, 10]], "fixture-gripper"],
        ],
        "pinmap": [[0, 0, 0, 0], [0, 0, 0, 20]],
    }}


class DualGripperTests(unittest.TestCase):
    def test_derives_real_offsets_instead_of_adjacent_gripper_assumption(self):
        profiles = profiles_from_status(status_fixture())
        self.assertEqual([(p.name, p.unit_id, p.high_index, p.low_index) for p in profiles], [
            ("机器人夹爪", 1, 3, 4), ("定位夹爪", 2, 23, 24)])

    def test_rejects_overlapping_or_incomplete_mappings(self):
        overlapping = status_fixture()
        overlapping["extio"]["pinmap"][1][3] = 0
        with self.assertRaisesRegex(ValueError, "重叠"):
            profiles_from_status(overlapping)
        incomplete = status_fixture()
        incomplete["extio"]["num"] = 3
        with self.assertRaises(ValueError):
            profiles_from_status(incomplete)

    def test_second_gripper_reads_and_writes_only_its_own_pair(self):
        profile = profiles_from_status(status_fixture())[1]
        commands = []

        def response(command, **kwargs):
            commands.append(command)
            return {"errorCode": 0, "value": 16709 if command.get("index") == 23 else 28836}

        with patch("gripper_backend.read_controller_status", return_value=status_fixture()), patch(
                "gripper_backend.send_command", side_effect=response):
            client = GripperClient(profile=profile)
            client.connect()
            result = client.set_opening_mm(12.34, settle_time=0)
        writes = [item for item in commands if item["cmdName"] == "set_analog_output"]
        self.assertEqual([(item["index"], item["value"]) for item in writes],
                         [(23, 16709.0), (24, 28836.0)])
        self.assertTrue(all(item.get("index") in (None, 23, 24) for item in commands))
        self.assertAlmostEqual(words_to_opening(result.high, result.low), 12.34, places=5)

    def test_changed_mapping_blocks_writes(self):
        original = status_fixture()
        profile = profiles_from_status(original)[1]
        changed = copy.deepcopy(original)
        changed["extio"]["pinmap"][1][3] = 30
        with patch("gripper_backend.read_controller_status", return_value=changed), patch(
                "gripper_backend.send_command") as command:
            with self.assertRaisesRegex(RuntimeError, "映射已变化"):
                GripperClient(profile=profile).set_opening_mm(10)
            command.assert_not_called()

    def test_same_controller_clients_serialize_complete_transactions(self):
        first = GripperClient(simulated=True, profile=GripperProfile())
        second = GripperClient(simulated=True, profile=GripperProfile("第二爪", 23, 24, 2))
        entered = threading.Event()
        release = threading.Event()
        second_entered = threading.Event()
        original_first = first._write_words_locked
        original_second = second._write_words_locked
        results = []

        def hold(*args, **kwargs):
            entered.set()
            release.wait(2)
            return original_first(*args, **kwargs)

        def observe(*args, **kwargs):
            second_entered.set()
            return original_second(*args, **kwargs)

        with patch.object(first, "_write_words_locked", side_effect=hold), patch.object(
                second, "_write_words_locked", side_effect=observe):
            one = threading.Thread(target=lambda: results.append(first.set_opening_mm(5)))
            two = threading.Thread(target=lambda: results.append(second.set_opening_mm(15)))
            one.start()
            try:
                self.assertTrue(entered.wait(1))
                two.start()
                self.assertFalse(second_entered.wait(0.05))
            finally:
                release.set()
                one.join(2)
                if two.ident is not None:
                    two.join(2)
        self.assertTrue(second_entered.is_set())
        self.assertEqual(len(results), 2)
        self.assertEqual(words_to_opening(first.snapshot().high, first.snapshot().low), 5)
        self.assertEqual(words_to_opening(second.snapshot().high, second.snapshot().low), 15)
