# -*- coding: utf-8 -*-
"""Chinese desktop control panel. Run: python gripper_app.py"""

from __future__ import annotations

import json
import argparse
import logging
from logging.handlers import RotatingFileHandler
import queue
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import scrolledtext, ttk

import arm_control as sdk
from gripper_backend import (
    GripperClient, GripperProfile, MAX_OPENING_MM, Snapshot, WriteUncertainError, validate_opening,
    opening_to_words, words_to_opening,
)
from gripper_transport import CommunicationError, close_connections
from controller_lease import ControllerBusyError, controller_lease

log = logging.getLogger(__name__)


class GripperPanel(ttk.Frame):
    LIVE_INTERVAL_MS = 120

    def __init__(self, master, profile: GripperProfile, host=sdk.ROBOT_IP):
        super().__init__(master)
        self.profile = profile
        self.client = None
        self.connected = False
        self.busy = False
        self.active_action = None
        self.closing = False
        self.pending_target = None
        self.live_job = None
        self.event_job = None
        self.poll_job = None
        self.reconnect_job = None
        self.reconnect_attempt = 0
        self.last_write = 0.0
        self.confirmed_target = None
        self.events = queue.Queue()
        self.last_poll = 0.0
        self.mode = tk.StringVar(self, value="real")
        self.host = tk.StringVar(self, value=host)
        self.armed = tk.BooleanVar(self, value=False)
        self.millimeters = tk.StringVar(self, value="0.00")
        self.opening_mm = tk.DoubleVar(self, value=0)
        self.live_enabled = tk.BooleanVar(self, value=True)
        self.opening_text = tk.StringVar(self, value="回读设定开度：— mm")
        self.auto_refresh = tk.BooleanVar(self, value=True)
        self.connection_text = tk.StringVar(self, value="未连接 · 真机模式")
        self.operation_text = tk.StringVar(self, value="等待连接，连接操作只读取状态。")
        self.register_text = tk.StringVar(self, value=self._register_caption())
        self.read_time = tk.StringVar(self, value="最近读取：—")
        self.preview_text = tk.StringVar(self)
        self._build_ui()
        self._preview(0)
        self._controls()
        self.event_job = self.after(20, self._drain_events)
        self.poll_job = self.after(500, self._poll)
        self._log("程序就绪；连接前请选择真机或模拟模式。")

    def _build_ui(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TLabel", font=("Microsoft YaHei UI", 10))
        style.configure("TButton", font=("Microsoft YaHei UI", 10), padding=7)
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 21, "bold"))
        style.configure("State.TLabel", font=("Microsoft YaHei UI", 12, "bold"))
        root = ttk.Frame(self, padding=22)
        root.pack(fill="both", expand=True)
        ttk.Label(root, text=self.profile.name, style="Title.TLabel").pack(anchor="w")
        ttk.Label(root, text=f"JAKA  /  从站 ID {self.profile.unit_id}  /  "
                  f"位置输出 {self.profile.high_index}、{self.profile.low_index}").pack(anchor="w", pady=(3, 13))

        connection = ttk.LabelFrame(root, text="设备连接", padding=12)
        connection.pack(fill="x")
        row = ttk.Frame(connection)
        row.pack(fill="x")
        self.real_radio = ttk.Radiobutton(row, text="真机", variable=self.mode,
                                         value="real", command=self._mode_changed)
        self.real_radio.pack(side="left")
        self.sim_radio = ttk.Radiobutton(row, text="模拟", variable=self.mode,
                                        value="sim", command=self._mode_changed)
        self.sim_radio.pack(side="left", padx=(8, 20))
        ttk.Label(row, text="机器人 IP").pack(side="left")
        self.host_entry = ttk.Entry(row, textvariable=self.host, width=19)
        self.host_entry.pack(side="left", padx=8)
        ttk.Label(row, text=f"TCP {sdk.CMD_PORT}").pack(side="left")
        self.connect_button = ttk.Button(row, text="连接", command=self._connect)
        self.connect_button.pack(side="right")
        self.disconnect_button = ttk.Button(row, text="断开", command=self._disconnect)
        self.disconnect_button.pack(side="right", padx=8)
        ttk.Label(connection, textvariable=self.connection_text,
                  style="State.TLabel").pack(anchor="w", pady=(10, 0))

        control = ttk.LabelFrame(root, text="开度控制", padding=12)
        control.pack(fill="x", pady=12)
        self.arm_check = ttk.Checkbutton(
            control, text="允许本次连接发送真机夹具指令", variable=self.armed,
            command=self._controls)
        self.arm_check.pack(anchor="w")
        self.live_check = ttk.Checkbutton(
            control, text="滑动实时发送（最多约 8 次/秒，只保留最新目标）",
            variable=self.live_enabled, command=self._live_changed)
        self.live_check.pack(anchor="w", pady=(5, 0))
        self.slider = ttk.Scale(control, from_=0, to=MAX_OPENING_MM, variable=self.opening_mm,
                                command=self._slider_changed)
        self.slider.pack(fill="x", pady=(12, 4))
        ttk.Label(control, textvariable=self.preview_text).pack(anchor="w")
        row = ttk.Frame(control)
        row.pack(fill="x", pady=12)
        ttk.Label(row, text="目标开合").pack(side="left")
        self.target_entry = ttk.Entry(row, textvariable=self.millimeters, width=12)
        self.target_entry.pack(side="left", padx=8)
        self.target_entry.bind("<KeyRelease>", self._entry_changed)
        self.target_entry.bind("<Return>", lambda event: self._send())
        ttk.Label(row, text=f"mm（0～{MAX_OPENING_MM:g}）").pack(side="left")
        self.apply_button = ttk.Button(row, text="发送目标开度", command=self._send)
        self.apply_button.pack(side="right")
        row = ttk.Frame(control)
        row.pack(fill="x")
        self.open_button = ttk.Button(row, text=f"全开  ·  {MAX_OPENING_MM:g} mm",
                                      command=lambda: self._send(MAX_OPENING_MM))
        self.open_button.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.close_button = ttk.Button(row, text="闭合  ·  0 mm",
                                       command=lambda: self._send(0))
        self.close_button.pack(side="left", fill="x", expand=True, padx=(6, 0))
        ttk.Label(control, text="允许真机指令后，拖动滑块即发送；输入毫米数后按回车或点发送。",
                  wraplength=730).pack(anchor="w", pady=(10, 0))

        status = ttk.LabelFrame(root, text="设备回读", padding=12)
        status.pack(fill="x")
        ttk.Label(status, textvariable=self.opening_text,
                  style="State.TLabel").pack(anchor="w", pady=(0, 5))
        row = ttk.Frame(status)
        row.pack(fill="x")
        ttk.Label(row, textvariable=self.register_text,
                  style="State.TLabel").pack(side="left")
        self.refresh_button = ttk.Button(row, text="刷新状态", command=self._refresh)
        self.refresh_button.pack(side="right")
        self.refresh_check = ttk.Checkbutton(status, text="每 2 秒读取一次状态（空闲时）",
                                             variable=self.auto_refresh)
        self.refresh_check.pack(anchor="w", pady=(6, 0))
        ttk.Label(status, textvariable=self.read_time).pack(anchor="w")
        ttk.Label(status, text="回读毫米数是设备输出设定，不是实测开度或夹持检测。",
                  wraplength=730).pack(anchor="w", pady=(5, 0))
        ttk.Label(root, textvariable=self.operation_text, wraplength=820,
                  foreground="#9b4200").pack(anchor="w", pady=(10, 5))
        self.log_box = scrolledtext.ScrolledText(root, height=7, state="disabled",
                                                font=("Microsoft YaHei UI", 9))
        self.log_box.pack(fill="both", expand=True)
        ttk.Label(root, text="本界面仅控制夹具；现场停止请使用实体急停。SDK 未提供夹具停止、力度或速度接口。",
                  wraplength=820).pack(anchor="w", pady=(8, 0))

    @staticmethod
    def _enabled(widget, enabled):
        widget.configure(state="normal" if enabled else "disabled")

    def _register_caption(self, high="—", low="—"):
        return f"寄存器 {self.profile.high_index}：{high}     寄存器 {self.profile.low_index}：{low}"

    def _controls(self):
        idle = not self.busy and not self.closing
        offline = idle and not self.connected
        for widget in (self.real_radio, self.sim_radio):
            self._enabled(widget, offline)
        self._enabled(self.host_entry, offline and self.mode.get() == "real")
        self._enabled(self.connect_button, offline)
        self._enabled(self.disconnect_button, idle and (self.connected or self.reconnect_job is not None))
        self._enabled(self.refresh_button, idle and self.connected)
        self._enabled(self.arm_check, not self.closing and self.connected and self.mode.get() == "real")
        self._enabled(self.live_check, not self.closing)
        allowed = idle and self.connected and (
            self.mode.get() == "sim" or self.armed.get())
        for widget in (self.open_button, self.close_button, self.apply_button):
            self._enabled(widget, allowed)
        # Dragging must remain responsive while one request is in flight.
        editable = not self.closing and (not self.busy or self.connected)
        for widget in (self.slider, self.target_entry):
            self._enabled(widget, editable)
        if not self._may_write():
            self._cancel_live()

    def _mode_changed(self):
        self._cancel_reconnect()
        self.armed.set(False)
        name = "模拟模式" if self.mode.get() == "sim" else "真机模式"
        self.connection_text.set(f"未连接 · {name}")
        self._controls()

    def _preview(self, target):
        percentage = 100 * target / MAX_OPENING_MM
        high, low = opening_to_words(target)
        self.preview_text.set(
            f"目标开合：{target:.2f} mm   |   {percentage:.1f}%   |   "
            f"寄存器 {self.profile.high_index}/{self.profile.low_index}：{high} / {low}")

    def _sync_target(self, target):
        self.millimeters.set(f"{target:.2f}")
        self.opening_mm.set(target)
        self._preview(target)

    def _slider_changed(self, value):
        target = round(validate_opening(value), 2)
        self._sync_target(target)
        if self.live_enabled.get() and self._may_write():
            self.pending_target = target
            self._schedule_live()

    def _entry_changed(self, event=None):
        try:
            target = validate_opening(self.millimeters.get())
        except ValueError:
            self.preview_text.set(f"请输入 0～{MAX_OPENING_MM:g} mm 的开合目标。")
            return
        # Typing never sends an incomplete number and supersedes queued drag
        # targets. Enter or the send button explicitly commits the value.
        self._cancel_live()
        self.opening_mm.set(target)
        self._preview(target)

    def _may_write(self):
        return self.connected and not self.closing and (
            self.mode.get() == "sim" or self.armed.get())

    def _cancel_live(self):
        self.pending_target = None
        if self.live_job is not None:
            self.after_cancel(self.live_job)
            self.live_job = None

    def _live_changed(self):
        if not self.live_enabled.get():
            self._cancel_live()
        self._controls()

    def _schedule_live(self):
        if (self.pending_target is None or self.live_job is not None
                or self.busy or not self.live_enabled.get() or not self._may_write()):
            return
        remaining = self.LIVE_INTERVAL_MS - (time.monotonic() - self.last_write) * 1000
        self.live_job = self.after(max(0, int(remaining + 1)), self._flush_live)

    def _flush_live(self):
        self.live_job = None
        if not self.live_enabled.get() or not self._may_write():
            self._cancel_live()
            return
        if self.busy:
            return  # Completion schedules the latest target; no FIFO backlog.
        target = self.pending_target
        self.pending_target = None
        if target is not None and (self.confirmed_target is None or
                opening_to_words(target) != opening_to_words(self.confirmed_target)):
            self._send(target, live=True)

    def _log(self, message):
        log.info("%s: %s", self.profile.name, message)
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"[{datetime.now():%H:%M:%S}] {message}\n")
        # Keep the panel bounded when running for long periods.
        if int(self.log_box.index("end-1c").split(".")[0]) > 600:
            self.log_box.delete("1.0", "101.0")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _start(self, action, function):
        if self.busy or self.closing:
            return
        self.busy = True
        self.active_action = action
        self._controls()

        client = self.client

        def work():
            try:
                if client is not None and not client.simulated:
                    with controller_lease(timeout=1.0):
                        try:
                            result = function()
                        finally:
                            # Close TCP before yielding the lease so a robot
                            # program can never observe an old GUI connection.
                            close_connections()
                else:
                    result = function()
                self.events.put((action, result, None))
            except Exception as exc:
                self.events.put((action, None, exc))

        threading.Thread(target=work, name="gripper-io", daemon=True).start()

    def _connect(self):
        if self.busy or self.connected or self.closing:
            return
        self._cancel_reconnect()
        self.reconnect_attempt = 0
        try:
            self.client = GripperClient(self.host.get(), self.mode.get() == "sim", profile=self.profile)
        except ValueError as exc:
            self.operation_text.set(str(exc))
            return
        self.armed.set(False)
        self._cancel_live()
        self.confirmed_target = None
        self.connection_text.set("正在连接并读取状态…")
        self.operation_text.set(f"正在核对 {self.profile.name} 的映射并读取状态。")
        client = self.client
        self._start("connect", client.connect)

    def _disconnect(self):
        if self.busy:
            return
        self._cancel_reconnect()
        self.connected = False
        self.client = None
        self.armed.set(False)
        self._cancel_live()
        self.confirmed_target = None
        self.connection_text.set("已断开")
        self.register_text.set(self._register_caption())
        self.opening_text.set("回读设定开度：— mm")
        self.read_time.set("最近读取：—")
        self.operation_text.set("已断开界面连接；夹具保持设备当前状态。")
        self._log("界面已断开。断开连接不会停止或松开夹具。")
        self._controls()

    def _cancel_reconnect(self):
        if self.reconnect_job is not None:
            self.after_cancel(self.reconnect_job)
            self.reconnect_job = None

    def _schedule_reconnect(self):
        if self.closing or self.client is None or self.reconnect_job is not None:
            return
        delay = min(15, 2 ** min(self.reconnect_attempt, 4))
        self.reconnect_attempt += 1
        self.connection_text.set(f"通信中断 · {delay} 秒后自动重连（只读）")
        self.reconnect_job = self.after(delay * 1000, self._reconnect)

    def _reconnect(self):
        self.reconnect_job = None
        if self.closing or self.client is None or self.connected:
            return
        if self.busy:
            self._schedule_reconnect()
            return
        self.connection_text.set("正在自动重连并核对夹爪映射…")
        self._start("reconnect", self.client.connect)

    def _refresh(self):
        if self.connected and not self.busy and not self.closing:
            self._start("refresh", self.client.snapshot)

    def _send(self, preset=None, *, live=False):
        if not self._may_write() or self.busy:
            return
        try:
            target = validate_opening(self.millimeters.get() if preset is None else preset)
        except ValueError as exc:
            self.operation_text.set(str(exc))
            self._log(str(exc))
            return
        if not live:
            self._cancel_live()
        self._sync_target(target)
        self.last_write = time.monotonic()
        self.operation_text.set(f"正在发送目标开合 {target:.2f} mm 并回读…")
        self._log(f"{'模拟' if self.client.simulated else '真机'}发送：{target:.2f} mm")
        client = self.client
        self._start("live" if live else "write", lambda: client.set_opening_mm(
            target, settle_time=0.0 if live else 0.3))

    def _display(self, snapshot: Snapshot):
        self.register_text.set(self._register_caption(f"{snapshot.high:g}", f"{snapshot.low:g}"))
        try:
            self.confirmed_target = words_to_opening(snapshot.high, snapshot.low)
        except ValueError:
            self.confirmed_target = None
            self.opening_text.set("回读设定开度：— mm（寄存器异常）")
        else:
            self.opening_text.set(f"回读设定开度：{self.confirmed_target:.2f} mm")
        self.read_time.set(f"最近读取：{datetime.now():%H:%M:%S}")

    def _drain_events(self):
        self.event_job = None
        try:
            while True:
                action, result, error = self.events.get_nowait()
                self.busy = False
                self.active_action = None
                if error is not None:
                    self._cancel_live()
                    self.operation_text.set(str(error))
                    self._log(f"{action} 失败：{error}")
                    if isinstance(error, ControllerBusyError):
                        # A lease refusal sends nothing and does not invalidate
                        # an established connection. Background polling must not
                        # silently revoke permission after every arm workflow.
                        self.connection_text.set("控制端口忙 · 等待空闲后恢复读取")
                        self.operation_text.set("其他程序正在使用控制端口；待发送目标已清除。")
                        if action in ("connect", "reconnect"):
                            self._schedule_reconnect()
                        elif action in ("write", "live"):
                            self.operation_text.set("控制端口忙，本次开度未发送；请空闲后重新操作。")
                    elif isinstance(error, (OSError, CommunicationError)):
                        self.armed.set(False)
                        self.connected = False
                        self.connection_text.set("通信中断")
                        self._schedule_reconnect()
                    elif action in ("write", "live") and self.connected:
                        self.armed.set(False)
                        # Preflight faults are operation errors, not TCP loss.
                        # An uncertain write remains a manual recovery case.
                        if isinstance(error, WriteUncertainError):
                            self.connected = False
                            self.connection_text.set("写入结果未确认 · 请重新连接回读")
                        else:
                            self.connection_text.set("已连接 · 操作被阻止")
                    else:
                        self.armed.set(False)
                        self.connected = False
                        self.connection_text.set("连接或配置异常 · 请重新连接")
                else:
                    self.connected = True
                    self.reconnect_attempt = 0
                    prefix = "模拟已连接" if self.client.simulated else f"真机已连接 · {self.client.host}:{sdk.CMD_PORT}"
                    self.connection_text.set(prefix)
                    self._display(result)
                    if action in ("connect", "reconnect"):
                        if self.confirmed_target is not None:
                            self._sync_target(self.confirmed_target)
                        self.operation_text.set("连接已恢复，请重新允许真机指令；旧目标已清除。" if action == "reconnect"
                                                else "连接成功。允许真机指令后，可滑动实时控制开合。")
                        self._log(f"{prefix}；状态：{json.dumps(result.state, ensure_ascii=False)}")
                    elif action in ("write", "live"):
                        self.operation_text.set(f"设定开合 {self.confirmed_target:.2f} mm，寄存器回读一致。")
                        self._log(f"写入回读一致：{self._register_caption(result.high, result.low)}")
                    if self.confirmed_target is None:
                        self.operation_text.set("寄存器值超出当前 SDK 约定，请检查设备配置后再操作。")
                        self.armed.set(False)
                        self._cancel_live()
                self.last_poll = time.monotonic()
                if self.closing:
                    self.destroy()
                    return
                self._controls()
                self._schedule_live()
        except queue.Empty:
            pass
        self.event_job = self.after(20, self._drain_events)

    def _poll(self):
        self.poll_job = None
        if (not self.closing and self.connected and not self.busy
                and self.pending_target is None
                and self.auto_refresh.get() and time.monotonic() - self.last_poll >= 2):
            self._refresh()
        if not self.closing:
            self.poll_job = self.after(500, self._poll)

    def _close(self):
        self._cancel_reconnect()
        self._cancel_live()
        if self.busy:
            self.closing = True
            self.operation_text.set("正在等待本次通信结束后退出…")
            self._controls()
        else:
            self.destroy()

    def destroy(self):
        # Each panel owns its timers; closing one must not stop the other.
        for name in ("event_job", "poll_job", "live_job", "reconnect_job"):
            callback = getattr(self, name, None)
            if callback is not None:
                self.after_cancel(callback)
                setattr(self, name, None)
        super().destroy()


def load_device_config():
    path = Path(__file__).with_name("gripper_devices.json")
    if not path.exists():
        return sdk.ROBOT_IP, [GripperProfile()]
    with path.open(encoding="utf-8") as file:
        config = json.load(file)
    profiles = [GripperProfile(**item) for item in config["devices"]]
    if not profiles:
        raise ValueError("夹爪配置中没有设备，请先运行只读识别工具。")
    used = set()
    for profile in profiles:
        pair = {profile.high_index, profile.low_index}
        if used.intersection(pair):
            raise ValueError("夹爪配置中的位置输出存在重叠。")
        used.update(pair)
    return config["host"], profiles


class GripperApp(tk.Tk):
    def __init__(self, profiles=None, host=None):
        super().__init__()
        if profiles is None:
            default_host, profiles = load_device_config()
            host = host or default_host
        self.title("JAKA · 双夹爪控制台" if len(profiles) > 1 else "JAKA · 夹爪控制台")
        self.geometry("920x850")
        self.minsize(800, 780)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)
        self.panels = []
        for profile in profiles:
            panel = GripperPanel(self.notebook, profile, host or sdk.ROBOT_IP)
            self.panels.append(panel)
            self.notebook.add(panel, text=f"  {profile.name} · ID {profile.unit_id}  ")
        self.notebook.bind("<<NotebookTabChanged>>", self._tab_changed)
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _tab_changed(self, event=None):
        selected = self.notebook.select()
        for panel in self.panels:
            if str(panel) != selected:
                panel._cancel_live()

    def connect_all(self):
        for panel in self.panels:
            panel._connect()

    def _close(self):
        for panel in self.panels:
            if panel.winfo_exists():
                panel._close()
        self._await_closed()

    def _await_closed(self):
        if any(panel.winfo_exists() for panel in self.panels):
            self.after(50, self._await_closed)
        else:
            close_connections()
            self.destroy()


if __name__ == "__main__":
    log_dir = Path(__file__).with_name("logs")
    log_dir.mkdir(exist_ok=True)
    handler = RotatingFileHandler(log_dir / "gripper.log", maxBytes=2_000_000,
                                 backupCount=3, encoding="utf-8")
    logging.basicConfig(level=logging.INFO, handlers=[handler],
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="JAKA Z-EFG-20P 夹具控制台")
    parser.add_argument("--connect", action="store_true", help="启动时只读连接真机")
    args = parser.parse_args()
    app = GripperApp()
    if args.connect:
        app.after(100, app.connect_all)
    app.mainloop()
