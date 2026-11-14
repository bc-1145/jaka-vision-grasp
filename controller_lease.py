"""Cross-process lease for the JAKA command port.

The controller accepts only one command client reliably.  The gripper GUI
leases the port for a single read/write, while robot programs hold the lease
for their complete motion sequence.
"""
from __future__ import annotations

import os
import time
from pathlib import Path


class ControllerBusyError(RuntimeError):
    pass


class ControllerLease:
    def __init__(self, timeout: float = 0.0):
        self.timeout = max(0.0, float(timeout))
        self.handle = None

    def __enter__(self):
        path = Path(__file__).with_name(".jaka-command-port.lock")
        self.handle = path.open("a+b")
        self.handle.seek(0, os.SEEK_END)
        if self.handle.tell() == 0:
            self.handle.write(b"\0")
            self.handle.flush()
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                self.handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except OSError as exc:
                if time.monotonic() >= deadline:
                    self.handle.close()
                    self.handle = None
                    raise ControllerBusyError("机械臂程序正在使用控制端口，请稍后再操作夹具") from exc
                time.sleep(0.05)

    def __exit__(self, exc_type, exc, traceback):
        if self.handle is None:
            return
        try:
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None


def controller_lease(timeout: float = 0.0) -> ControllerLease:
    return ControllerLease(timeout)
