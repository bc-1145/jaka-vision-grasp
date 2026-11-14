"""Keep native camera acquisition isolated from the preview window."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

log = logging.getLogger(__name__)
BUFFER_BYTES = 32 * 1024 * 1024


class FrameMailbox:
    """One bounded frame slot. Readers never wait for a native worker's lock."""
    def __init__(self, context, capacity=BUFFER_BYTES):
        self.capacity = capacity
        self.data = context.RawArray("B", capacity)
        self.meta = context.RawArray("q", 7)
        self.lock = context.Lock()

    def publish(self, color, depth):
        color = np.ascontiguousarray(color, dtype=np.uint8)
        depth = np.ascontiguousarray(depth, dtype=np.uint16)
        if color.ndim != 3 or color.shape[2] != 3 or depth.ndim != 2:
            raise ValueError("Unexpected camera image dimensions")
        color_bytes, depth_bytes = color.nbytes, depth.nbytes
        if color_bytes + depth_bytes > self.capacity:
            raise ValueError("Camera frame exceeds the preview buffer capacity")
        with self.lock:
            data = np.frombuffer(self.data, dtype=np.uint8)
            data[:color_bytes] = color.reshape(-1)
            data[color_bytes:color_bytes + depth_bytes] = depth.view(np.uint8).reshape(-1)
            sequence = self.meta[0] + 1
            self.meta[:] = (sequence, *color.shape[:2], *depth.shape, color_bytes, depth_bytes)

    def read(self, last_sequence):
        if not self.lock.acquire(False):
            return None
        try:
            sequence, ch, cw, dh, dw, cb, db = self.meta[:]
            if sequence == 0 or sequence == last_sequence:
                return None
            data = np.frombuffer(self.data, dtype=np.uint8)
            color = data[:cb].copy().reshape(ch, cw, 3)
            depth = data[cb:cb + db].copy().view(np.uint16).reshape(dh, dw)
            return sequence, color, depth
        finally:
            self.lock.release()


class PointCloudMailbox:
    """Latest native XYZ frame; preview readers never block acquisition."""
    def __init__(self, context, capacity=BUFFER_BYTES):
        self.capacity = capacity
        self.data = context.RawArray("B", capacity)
        self.meta = context.RawArray("q", 3)
        self.lock = context.Lock()

    def publish(self, points):
        points = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 3)
        if points.nbytes > self.capacity:
            raise ValueError("Point cloud exceeds the preview buffer capacity")
        with self.lock:
            np.frombuffer(self.data, dtype=np.uint8)[:points.nbytes] = points.view(np.uint8).reshape(-1)
            self.meta[:] = (self.meta[0] + 1, len(points), time.time_ns())

    def read(self, last_sequence):
        if not self.lock.acquire(False):
            return None
        try:
            sequence, count, stamp = self.meta[:]
            if not sequence or sequence == last_sequence:
                return None
            points = np.frombuffer(self.data, dtype=np.float32, count=count * 3).copy().reshape(-1, 3)
            return sequence, stamp / 1e9, points
        finally:
            self.lock.release()


def configure_logging(filename):
    directory = Path(__file__).resolve().parent / "logs"
    directory.mkdir(exist_ok=True)
    handler = RotatingFileHandler(directory / filename, maxBytes=2_000_000,
                                 backupCount=3, encoding="utf-8")
    logging.basicConfig(level=logging.INFO, handlers=[handler],
                        format="%(asctime)s %(levelname)s %(process)d %(message)s", force=True)


def capture_worker(sdk_dir, mailbox, stop, point_mailbox=None):
    # A vendor DLL failure can terminate this child without closing the UI.
    configure_logging("camera_capture.log")
    from color_depth_viewer import prepare_runtime, frame_to_color, frame_to_depth
    prepare_runtime(Path(sdk_dir))
    sensor = None
    streams = []
    sensor_started = False
    started = []
    try:
        from api.inu_sensor import InuSensor
        from api.image_stream import ImageOutputFormat, ImagePostProcessing
        from api.depth_stream import DepthOutputFormat, DepthPostProcessing
        from api.shared import StreamType

        sensor = InuSensor()
        sensor.init()
        sensor.start()
        sensor_started = True
        color_stream = sensor.create_stream(StreamType.GENERAL_CAMERA)
        streams.append(color_stream)
        color_stream.init(ImageOutputFormat.BGRA, ImagePostProcessing.NONE_PROCESSING)
        color_stream.start()
        started.append(color_stream)
        depth_stream = sensor.create_stream(StreamType.DEPTH)
        streams.append(depth_stream)
        depth_stream.init(DepthOutputFormat.DEPTH, DepthPostProcessing.DEFAULT)
        depth_stream.start()
        started.append(depth_stream)
        point_stream = None
        if point_mailbox is not None:
            point_stream = sensor.create_stream(StreamType.POINT_CLOUD)
            streams.append(point_stream)
            point_stream.init()
            point_stream.start()
            started.append(point_stream)
        log.info("Color and depth acquisition started; native XYZ=%s", point_stream is not None)
        timeouts = 0
        point_timeouts = 0
        next_points = 0.0
        while not stop.value:
            try:
                color_frame = color_stream.frame
                # Copy before asking for another frame: native buffers may be reused.
                color = frame_to_color(color_frame) if color_frame is not None else None
                color = None if color is None else color.copy()
                depth_frame = depth_stream.frame
                depth = frame_to_depth(depth_frame) if depth_frame is not None else None
            except RuntimeError as exc:
                if "0x5000033" not in str(exc).lower() and "timed out" not in str(exc).lower():
                    raise
                timeouts += 1
                if timeouts >= 3:
                    raise
                log.warning("Temporary camera frame timeout (%s/3): %s", timeouts, exc)
                time.sleep(0.1)
                continue
            if color is not None and depth is not None:
                mailbox.publish(color, depth)
                if point_stream is not None and time.monotonic() >= next_points:
                    try:
                        point_frame = point_stream.frame
                        raw = np.asarray(point_frame.data)
                        if raw.size >= 300 and raw.size % 3 == 0:
                            point_mailbox.publish(raw.reshape(-1, 3))
                        point_timeouts = 0
                    except RuntimeError as exc:
                        if "0x5000033" not in str(exc).lower() and "timed out" not in str(exc).lower():
                            raise
                        point_timeouts += 1
                        log.warning("Temporary XYZ timeout (%s/3): %s", point_timeouts, exc)
                        if point_timeouts >= 3:
                            raise
                    next_points = time.monotonic() + (1.0 if point_timeouts else 0.3)
                if timeouts:
                    log.info("Frame acquisition recovered after a temporary timeout")
                timeouts = 0
            else:
                time.sleep(0.01)
    except Exception:
        log.exception("Camera acquisition failed; the preview will reconnect")
        raise
    finally:
        for stream in reversed(streams):
            if stream in started:
                try:
                    stream.stop()
                except Exception:
                    log.exception("Could not stop camera stream")
            try:
                stream.terminate()
            except Exception:
                log.exception("Could not terminate camera stream")
        if sensor is not None:
            if sensor_started:
                try:
                    sensor.stop()
                except Exception:
                    log.exception("Could not stop camera sensor")
            try:
                sensor.terminate()
            except Exception:
                log.exception("Could not terminate camera sensor")


class CameraCapture:
    START_TIMEOUT = 30.0
    STALL_TIMEOUT = 8.0

    def __init__(self, sdk_dir, *, context=None, worker=capture_worker, with_points=False):
        self.sdk_dir = str(sdk_dir)
        self.context = context or mp.get_context("spawn")
        self.worker = worker
        self.with_points = with_points
        self.point_mailbox = None
        self.point_sequence = 0
        self.process = None
        self.mailbox = None
        self.stop = None
        self.sequence = 0
        self.started_at = 0.0
        self.last_frame_at = None
        self.retry_at = 0.0
        self.failures = 0
        self.closed = False
        self.status = "Connecting to camera..."

    def _start(self, now):
        self.mailbox = FrameMailbox(self.context)
        self.stop = self.context.RawValue("b", False)
        self.point_mailbox = PointCloudMailbox(self.context) if self.with_points else None
        self.point_sequence = 0
        worker_args = (self.sdk_dir, self.mailbox, self.stop)
        if self.with_points:
            worker_args += (self.point_mailbox,)
        process = self.context.Process(target=self.worker,
            args=worker_args, name="camera-acquisition", daemon=True)
        process.start()
        self.process = process
        self.sequence = 0
        self.last_frame_at = None
        self.started_at = now
        self.status = "Connecting to camera..."
        log.info("Started camera acquisition worker PID %s", self.process.pid)

    def _stop(self):
        if self.process is None:
            return
        self.stop.value = True
        self.process.join(0.3)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(0.5)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(0.5)
        if self.process.is_alive():
            # Do not start a second owner while this one still owns the device.
            raise RuntimeError("Camera worker could not be stopped")
        self.process.close()
        self.process = None
        self.mailbox = None
        self.point_mailbox = None
        self.stop = None

    def _retry(self, reason, now):
        log.warning("%s; reconnecting", reason)
        self._stop()
        delay = min(15, 2 ** min(self.failures, 4))
        self.failures += 1
        self.retry_at = now + delay
        self.status = f"Camera disconnected - reconnecting in {delay}s"

    def poll(self, now=None):
        if self.closed:
            return None
        now = time.monotonic() if now is None else now
        if self.process is None:
            if now >= self.retry_at:
                try:
                    self._start(now)
                except Exception:
                    log.exception("Could not launch camera worker")
                    self._retry("Camera worker launch failed", now)
            return None
        if not self.process.is_alive():
            self._retry(f"Camera worker exited with code {self.process.exitcode}", now)
            return None
        result = self.mailbox.read(self.sequence)
        if result is not None:
            self.sequence, color, depth = result
            first = self.last_frame_at is None
            self.last_frame_at = now
            self.status = "Live"
            if first:
                log.info("Camera frames received: color=%s depth=%s", color.shape, depth.shape)
            if now - self.started_at > 30:
                self.failures = 0
            return color, depth
        timeout = self.START_TIMEOUT if self.last_frame_at is None else self.STALL_TIMEOUT
        since = self.started_at if self.last_frame_at is None else self.last_frame_at
        if now - since >= timeout:
            self._retry("Camera stopped delivering frames", now)
        elif self.last_frame_at is not None and now - self.last_frame_at > 1:
            self.status = "Waiting for camera frames..."
        return None

    def poll_points(self):
        if self.point_mailbox is None:
            return None
        result = self.point_mailbox.read(self.point_sequence)
        if result is not None:
            self.point_sequence = result[0]
        return result

    def close(self):
        self.closed = True
        self._stop()
