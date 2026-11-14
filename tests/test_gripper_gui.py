"""GUI checks run with Tkinter; all device communication is mocked."""

import time
import threading
import unittest
from unittest.mock import patch

try:
    from gripper_app import GripperApp
except ImportError:
    GripperApp = None

from gripper_backend import GripperProfile, WriteUncertainError, words_to_opening


@unittest.skipIf(GripperApp is None, "Tkinter is not installed")
class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.command = patch("gripper_backend.send_command").start()
        self.addCleanup(patch.stopall)
        self.window = GripperApp(profiles=[
            GripperProfile("机器人夹爪", 3, 4, 1),
            GripperProfile("定位夹爪", 23, 24, 2),
        ])
        self.window.withdraw()
        self.app = self.window.panels[0]
        self.app.auto_refresh.set(False)

    def tearDown(self):
        self.window.destroy()
        self.command.assert_not_called()

    def wait_for_result(self):
        deadline = time.monotonic() + 3
        while (self.app.busy or self.app.pending_target is not None
               or self.app.live_job is not None) and time.monotonic() < deadline:
            self.app.update()
            time.sleep(0.01)
        self.assertFalse(self.app.busy, "background operation did not complete")
        self.assertIsNone(self.app.pending_target)
        self.assertIsNone(self.app.live_job)

    def connect_simulator(self):
        self.app.mode.set("sim")
        self.app._mode_changed()
        self.app._connect()
        self.wait_for_result()
        self.assertTrue(self.app.connected)

    def test_startup_and_slider_do_not_send(self):
        self.assertEqual(self.app.mode.get(), "real")
        self.assertFalse(self.app.connected)
        self.app._slider_changed("10")
        self.assertEqual(self.app.millimeters.get(), "10.00")
        self.assertFalse(self.app.busy)

    def test_live_slider_sends_without_button(self):
        self.connect_simulator()
        self.app._slider_changed("12.5")
        self.wait_for_result()
        snapshot = self.app.client.snapshot()
        self.assertEqual(words_to_opening(snapshot.high, snapshot.low), 12.5)
        self.assertIn("12.50 mm", self.app.opening_text.get())

    def test_live_drag_coalesces_to_latest_while_request_runs(self):
        self.connect_simulator()
        original = self.app.client.set_opening_mm
        started = threading.Event()
        release = threading.Event()
        sent = []

        def delayed(target, **kwargs):
            sent.append(target)
            if len(sent) == 1:
                started.set()
                release.wait(2)
            return original(target, **kwargs)

        with patch.object(self.app.client, "set_opening_mm", side_effect=delayed):
            try:
                self.app._slider_changed("4")
                self.app.update()
                self.assertTrue(started.wait(1))
                for value in ("8", "12", "18"):
                    self.app._slider_changed(value)
                self.assertEqual(str(self.app.slider.cget("state")), "normal")
                self.assertEqual(self.app.millimeters.get(), "18.00")
            finally:
                release.set()
            self.wait_for_result()
        self.assertEqual(sent, [4, 18])
        snapshot = self.app.client.snapshot()
        self.assertEqual(words_to_opening(snapshot.high, snapshot.low), 18)

    def test_disabling_live_cancels_queued_drag(self):
        self.connect_simulator()
        self.app._slider_changed("10")
        self.app.live_enabled.set(False)
        self.app._live_changed()
        self.wait_for_result()
        self.assertEqual(self.app.client.snapshot().high, 0)

    def test_real_connection_requires_permission_for_live_drag(self):
        self.connect_simulator()
        self.app.mode.set("real")
        self.app.armed.set(False)
        self.app._controls()
        self.app._slider_changed("10")
        self.assertIsNone(self.app.pending_target)
        self.assertFalse(self.app.busy)
        self.assertEqual(self.app.client.snapshot().high, 0)

    def test_millimeter_entry_requires_explicit_commit(self):
        self.connect_simulator()
        self.app.millimeters.set("7.5")
        self.app._entry_changed()
        self.assertEqual(self.app.client.snapshot().high, 0)
        self.app._send()
        self.wait_for_result()
        snapshot = self.app.client.snapshot()
        self.assertEqual(words_to_opening(snapshot.high, snapshot.low), 7.5)
        self.app.millimeters.set("21")
        self.app._send()
        self.assertFalse(self.app.busy)
        snapshot = self.app.client.snapshot()
        self.assertEqual(words_to_opening(snapshot.high, snapshot.low), 7.5)

    def test_two_panels_keep_targets_and_permissions_independent(self):
        self.connect_simulator()
        second = self.window.panels[1]
        second.auto_refresh.set(False)
        second.mode.set("sim")
        second._mode_changed()
        second._connect()
        deadline = time.monotonic() + 3
        while second.busy and time.monotonic() < deadline:
            self.window.update()
            time.sleep(0.01)
        self.assertTrue(second.connected)
        self.app._send(5)
        self.wait_for_result()
        self.window.notebook.select(second)
        self.window.update()
        first = self.app
        self.app = second
        self.app._slider_changed("15")
        self.wait_for_result()
        self.assertEqual(words_to_opening(first.client.snapshot().high, first.client.snapshot().low), 5)
        self.assertEqual(words_to_opening(second.client.snapshot().high, second.client.snapshot().low), 15)
        self.assertIn("23", second.register_text.get())
        first.armed.set(True)
        second.armed.set(False)
        second._disconnect()
        self.assertTrue(first.connected)
        self.assertTrue(first.armed.get())

    def test_switching_tabs_cancels_unsent_drag(self):
        self.connect_simulator()
        self.app._slider_changed("10")
        self.window.notebook.select(self.window.panels[1])
        self.window._tab_changed()
        self.wait_for_result()
        self.assertEqual(self.app.client.snapshot().high, 0)

    def test_connect_set_open_close_disconnect_in_simulation(self):
        self.connect_simulator()
        for target in (10, 20, 0):
            self.app._send(target)
            self.wait_for_result()
            snapshot = self.app.client.snapshot()
            self.assertEqual(words_to_opening(snapshot.high, snapshot.low), target)
        self.app._disconnect()
        self.assertFalse(self.app.connected)
        self.assertFalse(self.app.armed.get())

    def test_write_failure_disarms_and_requires_reconnection(self):
        self.connect_simulator()
        self.app.armed.set(True)
        with patch.object(self.app.client, "set_opening_mm", side_effect=WriteUncertainError("结果尚未确认")):
            self.app._send(10)
            self.wait_for_result()
        self.assertFalse(self.app.connected)
        self.assertFalse(self.app.armed.get())
        self.assertIn("结果尚未确认", self.app.operation_text.get())
        self.app._send(10)
        self.assertFalse(self.app.busy)

    def test_read_failure_recovers_without_replaying_target_or_restoring_permission(self):
        self.connect_simulator()
        self.app.armed.set(True)
        with patch.object(self.app.client, "snapshot", side_effect=ConnectionError("temporary loss")):
            self.app._refresh()
            self.wait_for_result()
        self.assertFalse(self.app.connected)
        self.assertFalse(self.app.armed.get())
        self.assertIsNotNone(self.app.reconnect_job)
        with patch.object(self.app.client, "set_opening_mm") as write:
            self.app._cancel_reconnect()
            self.app._reconnect()
            self.wait_for_result()
            self.assertTrue(self.app.connected)
            self.assertFalse(self.app.armed.get())
            self.assertIsNone(self.app.pending_target)
            write.assert_not_called()

    def test_manual_disconnect_cancels_automatic_reconnection(self):
        self.connect_simulator()
        self.app.connected = False
        self.app._schedule_reconnect()
        self.assertIsNotNone(self.app.reconnect_job)
        self.app._disconnect()
        self.assertIsNone(self.app.reconnect_job)
        self.assertIsNone(self.app.client)

    def test_operation_rejection_does_not_drop_connection(self):
        self.connect_simulator()
        with patch.object(self.app.client, "set_opening_mm", side_effect=RuntimeError("estop=1")):
            self.app._send(10)
            self.wait_for_result()
        self.assertTrue(self.app.connected)
        self.assertFalse(self.app.armed.get())
        self.assertIsNone(self.app.reconnect_job)


if __name__ == "__main__":
    unittest.main()
