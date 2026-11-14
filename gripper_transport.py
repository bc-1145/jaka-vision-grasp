"""Shared, serialized JAKA command connection; only reads may be retried."""

from __future__ import annotations

import atexit
import json
import logging
import socket
import threading
import time

import arm_control as sdk

log = logging.getLogger(__name__)
READ_COMMANDS = frozenset({"get_robot_state", "get_motion_state", "get_analog_output",
                           "get_actual_tcp_pos", "get_actual_joint_pos"})


class CommunicationError(ConnectionError):
    """No usable response was received. A write may already have executed."""


class ControllerConnection:
    def __init__(self, host: str):
        self.host = host
        self.lock = threading.RLock()
        self.socket = None
        self.buffer = b""

    def close(self):
        with self.lock:
            if self.socket is not None:
                self.socket.close()
            self.socket = None
            self.buffer = b""

    def _request(self, command: dict, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        if self.socket is None:
            self.socket = socket.create_connection((self.host, sdk.CMD_PORT), timeout=timeout)
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        self.socket.settimeout(max(0.001, deadline - time.monotonic()))
        self.socket.sendall(json.dumps(command).encode("utf-8") + b"\n")
        decoder = json.JSONDecoder()
        while True:
            if self.buffer:
                try:
                    text = self.buffer.decode("utf-8")
                except UnicodeDecodeError:
                    text = self.buffer.decode("gb18030", errors="replace")
                try:
                    reply, end = decoder.raw_decode(text.lstrip())
                except json.JSONDecodeError:
                    pass  # TCP may split a JSON object at any byte.
                else:
                    # Exactly one response is allowed per request. Keeping an
                    # unknown extra frame could associate stale data with a write.
                    if text.lstrip()[end:].strip():
                        raise CommunicationError("控制器返回了多余响应，已关闭连接。")
                    self.buffer = b""
                    if not isinstance(reply, dict) or reply.get("cmdName") != command["cmdName"]:
                        raise CommunicationError("控制器响应与请求不一致，已关闭连接。")
                    return reply
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("等待控制器响应超时。")
            self.socket.settimeout(remaining)
            chunk = self.socket.recv(65536)
            if not chunk:
                raise CommunicationError("控制器关闭了 TCP 连接。")
            self.buffer += chunk
            if len(self.buffer) > 1024 * 1024:
                raise CommunicationError("控制器响应超过允许长度。")

    def request(self, command: dict, timeout: float = 3.0) -> dict:
        with self.lock:
            attempts = 2 if command.get("cmdName") in READ_COMMANDS else 1
            for attempt in range(attempts):
                try:
                    return self._request(command, timeout)
                except (OSError, CommunicationError) as exc:
                    self.close()  # Never let a late response reach the next request.
                    log.warning("%s:%s %s attempt %s: %s", self.host, sdk.CMD_PORT,
                                command.get("cmdName"), attempt + 1, exc)
                    if attempt + 1 == attempts:
                        raise CommunicationError(f"{self.host}:{sdk.CMD_PORT} 通信失败：{exc}") from exc


_connections: dict[str, ControllerConnection] = {}
_registry_lock = threading.Lock()


def send_command(command: dict, *, timeout: float = 3.0, ip: str = sdk.ROBOT_IP) -> dict:
    with _registry_lock:
        connection = _connections.setdefault(ip, ControllerConnection(ip))
    return connection.request(command, timeout)


def close_connections():
    with _registry_lock:
        for connection in _connections.values():
            connection.close()
        _connections.clear()


atexit.register(close_connections)
