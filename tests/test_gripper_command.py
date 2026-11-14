import io
import unittest
from contextlib import nullcontext, redirect_stdout, redirect_stderr
from types import SimpleNamespace
from unittest.mock import patch

import gripper_command as command


class GripperCommandTests(unittest.TestCase):
    def test_open_and_card_width_forward_wait_to_single_write(self):
        for action, opening in ((["--open"], 20.0), (["--mm", "5.96"], 5.96)):
            with self.subTest(opening=opening), \
                    patch("sys.argv", ["gripper_command.py", *action, "--settle-seconds", "1"]), \
                    patch.object(command, "controller_lease", return_value=nullcontext()), \
                    patch.object(command, "GripperClient") as client, \
                    patch.object(command, "close_connections") as close, redirect_stdout(io.StringIO()):
                high, low = command.opening_to_words(opening)
                client.return_value.set_opening_mm.return_value = SimpleNamespace(high=high, low=low)
                self.assertEqual(command.main(), 0)
                client.return_value.set_opening_mm.assert_called_once_with(opening, settle_time=1)
                close.assert_called_once()

    def test_check_with_wait_never_connects(self):
        with patch("sys.argv", ["gripper_command.py", "--mm", "5.96", "--settle-seconds", "1", "--check"]), \
                patch.object(command, "controller_lease") as lease, \
                patch.object(command, "GripperClient") as client, redirect_stdout(io.StringIO()):
            self.assertEqual(command.main(), 0)
        lease.assert_not_called()
        client.assert_not_called()

    def test_invalid_wait_rejected_before_io(self):
        for seconds in ("nan", "inf", "-1", "6"):
            with self.subTest(seconds=seconds), \
                    patch("sys.argv", ["gripper_command.py", "--open", "--settle-seconds", seconds]), \
                    patch.object(command, "controller_lease") as lease, \
                    patch.object(command, "robot_profile") as profile, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    command.main()
                lease.assert_not_called()
                profile.assert_not_called()

    def test_uncertain_write_closes_connection_without_retry(self):
        with patch("sys.argv", ["gripper_command.py", "--mm", "5.96", "--settle-seconds", "1"]), \
                patch.object(command, "controller_lease", return_value=nullcontext()), \
                patch.object(command, "GripperClient") as client, \
                patch.object(command, "close_connections") as close:
            client.return_value.set_opening_mm.side_effect = RuntimeError("uncertain write")
            with self.assertRaisesRegex(RuntimeError, "uncertain write"):
                command.main()
            client.return_value.set_opening_mm.assert_called_once()
            close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
