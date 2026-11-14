"""Failure-path tests; no camera SDK or hardware is used."""
import multiprocessing as mp
import unittest
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np

from camera_capture import CameraCapture, FrameMailbox, capture_worker


class MailboxTests(unittest.TestCase):
    def setUp(self):
        self.mailbox = FrameMailbox(mp.get_context("spawn"), capacity=128)
        self.color = np.full((2, 3, 3), 17, dtype=np.uint8)
        self.depth = np.full((3, 2), 1234, dtype=np.uint16)

    def test_images_are_copies_and_only_latest_frame_is_kept(self):
        self.assertIsNone(self.mailbox.read(0))
        self.mailbox.publish(self.color, self.depth)
        seq, color, depth = self.mailbox.read(0)
        np.testing.assert_array_equal(color, self.color)
        np.testing.assert_array_equal(depth, self.depth)
        self.assertIsNone(self.mailbox.read(seq))
        self.mailbox.publish(self.color * 2, self.depth + 3)
        self.mailbox.publish(self.color * 3, self.depth + 4)
        seq2, latest, latest_depth = self.mailbox.read(seq)
        self.assertEqual(seq2, 3)
        np.testing.assert_array_equal(latest, self.color * 3)
        np.testing.assert_array_equal(latest_depth, self.depth + 4)
        np.testing.assert_array_equal(color, self.color)
        np.testing.assert_array_equal(depth, self.depth)

    def test_busy_or_abandoned_worker_lock_never_blocks_ui(self):
        self.mailbox.publish(self.color, self.depth)
        self.mailbox.lock.acquire()
        try:
            self.assertIsNone(self.mailbox.read(0))
        finally:
            self.mailbox.lock.release()

    def test_oversize_frame_does_not_replace_last_valid_frame(self):
        self.mailbox.publish(self.color, self.depth)
        with self.assertRaises(ValueError):
            self.mailbox.publish(np.zeros((10, 10, 3), np.uint8), self.depth)
        self.assertEqual(self.mailbox.read(0)[0], 1)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.capture = CameraCapture("unused")
        self.worker = MagicMock()
        self.worker.is_alive.return_value = True
        self.worker.terminate.side_effect = lambda: setattr(self.worker.is_alive, "return_value", False)
        self.worker.exitcode = 9
        self.capture.process = self.worker
        self.capture.stop = MagicMock()
        self.capture.mailbox = MagicMock()
        self.capture.mailbox.read.return_value = None
        self.capture.started_at = 10

    def test_crash_reconnects_with_new_worker_and_no_ui_exit(self):
        self.worker.is_alive.return_value = False
        self.capture.poll(11)
        self.assertIsNone(self.capture.process)
        self.assertEqual(self.capture.retry_at, 12)
        self.assertIn("reconnecting", self.capture.status)
        with patch.object(self.capture, "_start") as start:
            self.capture.poll(11.5)
            start.assert_not_called()
            self.capture.poll(12)
            start.assert_called_once_with(12)

    def test_no_first_frame_has_timeout_but_does_not_exit(self):
        self.capture.poll(39)
        self.worker.terminate.assert_not_called()
        self.capture.poll(40)
        self.worker.terminate.assert_called_once()
        self.assertIn("reconnecting", self.capture.status)

    def test_stream_stall_has_shorter_timeout(self):
        self.capture.last_frame_at = 12
        self.capture.poll(14)
        self.assertIn("Waiting", self.capture.status)
        self.capture.poll(20)
        self.assertIn("reconnecting", self.capture.status)
        self.worker.terminate.assert_called_once()

    def test_close_never_schedules_a_restart(self):
        self.worker.is_alive.return_value = False
        self.capture.close()
        with patch.object(self.capture, "_start") as start:
            self.capture.poll(1000)
            start.assert_not_called()
        self.assertTrue(self.capture.closed)

    def test_launch_failure_is_retried_without_raising(self):
        self.capture.process = None
        with patch.object(self.capture, "_start", side_effect=OSError("spawn failed")):
            self.capture.poll(10)
        self.assertIn("reconnecting", self.capture.status)
        self.assertEqual(self.capture.retry_at, 11)

    def test_repeat_failures_back_off_and_remain_bounded(self):
        for index in range(7):
            with self.subTest(index=index):
                self.capture.process = None
                now = 100 + index * 20
                self.capture._retry("test failure", now)
                self.assertEqual(self.capture.retry_at - now, min(15, 2 ** index))


class AcquisitionTests(unittest.TestCase):
    def run_worker(self, color_responses, *, depth_init_error=None, with_points=False,
                   point_init_error=None, point_frame_error=None):
        sensor, color, depth, mailbox, stop = [MagicMock() for _ in range(5)]
        stop.value = False
        sensor.create_stream.side_effect = [color, depth]
        point_stream, point_mailbox = MagicMock(), MagicMock()
        if with_points:
            sensor.create_stream.side_effect = [color, depth, point_stream]
            point_stream.frame.data = np.ones((100, 3), dtype=np.float32)
            if point_frame_error is not None:
                type(point_stream).frame = PropertyMock(side_effect=point_frame_error)
            if point_init_error is not None:
                point_stream.init.side_effect = point_init_error
        type(color).frame = PropertyMock(side_effect=color_responses)
        depth.frame = object()
        if depth_init_error is not None:
            depth.init.side_effect = depth_init_error
        mailbox.publish.side_effect = lambda *_: setattr(stop, "value", True)
        modules = {name: MagicMock() for name in (
            "api.inu_sensor", "api.image_stream", "api.depth_stream", "api.shared")}
        modules["api.inu_sensor"].InuSensor.return_value = sensor
        self.resources = sensor, color, depth, mailbox
        self.point_resources = point_stream, point_mailbox
        with patch.dict("sys.modules", modules), \
             patch("camera_capture.configure_logging"), \
             patch("camera_capture.time.sleep"), \
             patch("color_depth_viewer.prepare_runtime"), \
             patch("color_depth_viewer.frame_to_color", return_value=np.zeros((2, 3, 3), np.uint8)), \
             patch("color_depth_viewer.frame_to_depth", return_value=np.ones((2, 3), np.uint16)):
            capture_worker("unused", mailbox, stop, point_mailbox if with_points else None)

    def test_native_xyz_shares_sensor_with_images_and_is_released(self):
        self.run_worker([object()], with_points=True)
        sensor, color, depth, mailbox = self.resources
        points, point_mailbox = self.point_resources
        sensor.init.assert_called_once()
        mailbox.publish.assert_called_once()
        point_mailbox.publish.assert_called_once()
        points.stop.assert_called_once()
        points.terminate.assert_called_once()

    def test_native_xyz_initialization_failure_releases_both_image_streams(self):
        with self.assertRaisesRegex(RuntimeError, "XYZ init failed"):
            self.run_worker([], with_points=True, point_init_error=RuntimeError("XYZ init failed"))
        sensor, color, depth, mailbox = self.resources
        points, point_mailbox = self.point_resources
        color.stop.assert_called_once()
        depth.stop.assert_called_once()
        points.stop.assert_not_called()
        points.terminate.assert_called_once()
        point_mailbox.publish.assert_not_called()

    def test_single_xyz_timeout_keeps_image_frame_and_does_not_fail_sensor(self):
        self.run_worker([object()], with_points=True,
                        point_frame_error=RuntimeError("InuService timed out (0x5000033)"))
        sensor, color, depth, mailbox = self.resources
        points, point_mailbox = self.point_resources
        sensor.init.assert_called_once()
        mailbox.publish.assert_called_once()
        point_mailbox.publish.assert_not_called()

    def test_one_service_timeout_recovers_without_reinitializing_sensor(self):
        self.run_worker([RuntimeError("Call to InuService timed out (0x5000033)"), object()])
        sensor, color, depth, mailbox = self.resources
        sensor.init.assert_called_once()
        mailbox.publish.assert_called_once()
        color.stop.assert_called_once()
        depth.stop.assert_called_once()
        sensor.terminate.assert_called_once()

    def test_repeated_timeouts_are_bounded_and_resources_are_closed(self):
        error = RuntimeError("Call to InuService timed out (0x5000033)")
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            self.run_worker([error, error, error])
        sensor, color, depth, mailbox = self.resources
        mailbox.publish.assert_not_called()
        sensor.stop.assert_called_once()
        color.terminate.assert_called_once()
        depth.terminate.assert_called_once()

    def test_partial_initialization_releases_already_started_stream(self):
        with self.assertRaisesRegex(RuntimeError, "depth init failed"):
            self.run_worker([], depth_init_error=RuntimeError("depth init failed"))
        sensor, color, depth, mailbox = self.resources
        color.stop.assert_called_once()
        depth.stop.assert_not_called()
        depth.terminate.assert_called_once()
        sensor.terminate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
