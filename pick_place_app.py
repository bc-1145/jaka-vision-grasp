# -*- coding: utf-8 -*-
"""Tk desktop front-end for the 3D CAD matching and pick/place workflow."""

from __future__ import annotations

import json
import os
import argparse
import queue
import subprocess
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, scrolledtext, simpledialog, ttk

import arm_control
import pick_place_3d


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "pick_place_config.json"
RESULTS_DIR = ROOT / "pick_results"
CAMERA_PYTHON = ROOT / ".camera-runtime" / "python.exe"
WORKER_SCRIPT = ROOT / "pick_place_3d.py"


class PickPlaceApp(ttk.Frame):
    def __init__(self, master: tk.Tk):
        super().__init__(master)
        self.master = master
        self.events: queue.Queue = queue.Queue()
        self.busy = False
        self.external_process = None
        self.preview_image = None
        self.matches: list[dict] = []
        self.markerless_dialog = None
        self.markerless_buttons = []
        self.markerless_dataset = ROOT / "calibration_data" / "markerless"
        self.auto_calibration_active = False
        self.auto_cancel_file = ROOT / "calibration_data" / "cancel_auto_capture.flag"
        self.auto_stop_button = None
        self.armed = tk.BooleanVar(value=False)
        self.config_state = tk.StringVar(value="正在读取配置…")
        self.robot_state = tk.StringVar(value="机器人状态：未读取")
        self.operation = tk.StringVar(value="就绪：默认只识别和规划，不发送真机动作。")
        self.match_summary = tk.StringVar(value="尚未扫描")
        self._build_ui()
        self._refresh_config()
        self._load_results(silent=True)
        self.after(50, self._drain_events)

    def _build_ui(self):
        self.master.title("3D 视觉抓取 · JAKA / NU4000")
        self.master.geometry("1180x820")
        self.master.minsize(1000, 700)
        self.master.protocol("WM_DELETE_WINDOW", self._close)
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TLabel", font=("Microsoft YaHei UI", 10))
        style.configure("TButton", font=("Microsoft YaHei UI", 10), padding=7)
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 21, "bold"))
        style.configure("Status.TLabel", font=("Microsoft YaHei UI", 11, "bold"))
        self.pack(fill="both", expand=True)

        root = ttk.Frame(self, padding=18)
        root.pack(fill="both", expand=True)
        header = ttk.Frame(root)
        header.pack(fill="x")
        ttk.Label(header, text="3D 视觉抓取", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="NU4000 点云  ·  FCStd 配准  ·  抓取放置",
                  foreground="#555").pack(side="left", padx=16, pady=(8, 0))
        self.refresh_button = ttk.Button(header, text="刷新状态", command=self._refresh_all)
        self.refresh_button.pack(side="right")

        status = ttk.Frame(root)
        status.pack(fill="x", pady=(10, 12))
        ttk.Label(status, textvariable=self.config_state, style="Status.TLabel").pack(side="left")
        ttk.Label(status, textvariable=self.robot_state, style="Status.TLabel").pack(side="right")

        body = ttk.Panedwindow(root, orient="horizontal")
        body.pack(fill="both", expand=True)
        controls = ttk.Frame(body, padding=(0, 0, 12, 0), width=310)
        visual = ttk.Frame(body)
        body.add(controls, weight=0)
        body.add(visual, weight=1)

        workflow = ttk.LabelFrame(controls, text="工作流程", padding=12)
        workflow.pack(fill="x")
        self.scan_button = ttk.Button(workflow, text="1  扫描并匹配 CAD", command=self._scan)
        self.scan_button.pack(fill="x")
        self.view_button = ttk.Button(workflow, text="打开三维结果", command=self._view_3d)
        self.view_button.pack(fill="x", pady=(7, 0))
        self.plan_button = ttk.Button(workflow, text="2  生成杆身抓取方案", command=self._plan)
        self.plan_button.pack(fill="x", pady=(7, 0))
        self.execute_button = ttk.Button(workflow, text="3  执行当前方案", command=self._execute)
        self.execute_button.pack(fill="x", pady=(7, 0))
        ttk.Label(workflow, textvariable=self.match_summary, wraplength=270,
                  foreground="#174f72").pack(anchor="w", pady=(10, 0))

        permission = ttk.LabelFrame(controls, text="真机许可", padding=12)
        permission.pack(fill="x", pady=10)
        self.arm_check = ttk.Checkbutton(
            permission, text="允许本次执行机械臂和夹爪动作", variable=self.armed,
            command=self._update_controls)
        self.arm_check.pack(anchor="w")
        ttk.Label(permission,
                  text="执行前仍需输入 EXECUTE；抓取抬升后还会要求确认工件已夹牢。",
                  wraplength=270).pack(anchor="w", pady=(6, 0))

        calibration = ttk.LabelFrame(controls, text="标定", padding=12)
        calibration.pack(fill="x")
        self.camera_cal_button = ttk.Button(
            calibration, text="无板手眼标定（不用标定板）", command=self._open_markerless_dialog)
        self.camera_cal_button.pack(fill="x")
        self.platform_cal_button = ttk.Button(
            calibration, text="识别并保存平台高度", command=lambda: self._launch_calibration("calibrate-platform"))
        self.platform_cal_button.pack(fill="x", pady=(7, 0))
        self.tool_button = ttk.Button(calibration, text="设置夹持中心与最大外径",
                                      command=self._configure_tool)
        self.tool_button.pack(fill="x", pady=(7, 0))
        self.config_button = ttk.Button(calibration, text="打开配置文件", command=self._open_config)
        self.config_button.pack(fill="x", pady=(7, 0))

        ttk.Label(controls,
                  text="定位夹爪中转需要在配置中示教放入、取出姿态；未完成时程序会阻止该路线。",
                  wraplength=280, foreground="#744000").pack(anchor="w", pady=10)

        preview_frame = ttk.LabelFrame(visual, text="俯视预览", padding=8)
        preview_frame.pack(fill="both", expand=True)
        self.preview_label = ttk.Label(preview_frame, text="扫描后显示点云投影与 CAD 对齐结果",
                                       anchor="center")
        self.preview_label.pack(fill="both", expand=True)

        candidate_frame = ttk.LabelFrame(visual, text="匹配候选", padding=8)
        candidate_frame.pack(fill="x", pady=(10, 0))
        columns = ("rank", "score", "fitness", "rmse", "clearance", "points")
        self.candidates = ttk.Treeview(candidate_frame, columns=columns, show="headings", height=5)
        headings = {"rank": "排名", "score": "综合分", "fitness": "覆盖率",
                    "rmse": "误差 mm", "clearance": "净空 mm", "points": "点数"}
        widths = {"rank": 54, "score": 82, "fitness": 82, "rmse": 90,
                  "clearance": 90, "points": 80}
        for name in columns:
            self.candidates.heading(name, text=headings[name])
            self.candidates.column(name, width=widths[name], anchor="center")
        self.candidates.pack(fill="x")

        operation = ttk.Label(root, textvariable=self.operation, wraplength=1120,
                              foreground="#8b4300")
        operation.pack(anchor="w", pady=(10, 5))
        self.progress = ttk.Progressbar(root, mode="indeterminate")
        self.progress.pack(fill="x")
        self.log = scrolledtext.ScrolledText(root, height=8, state="disabled",
                                             font=("Consolas", 9))
        self.log.pack(fill="x", pady=(7, 0))
        self._log("软件就绪；所有视觉操作均在独立 Python 3.10 后台进程运行。")

        self.action_widgets = [self.scan_button, self.view_button, self.plan_button,
                               self.execute_button, self.camera_cal_button,
                               self.platform_cal_button, self.tool_button,
                               self.config_button, self.refresh_button]
        self._update_controls()

    def _log(self, text: str):
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log.configure(state="normal")
        self.log.insert("end", f"[{stamp}] {text.rstrip()}\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _set_busy(self, busy: bool, text: str | None = None):
        self.busy = busy
        if text:
            self.operation.set(text)
        if busy:
            self.progress.start(12)
        else:
            self.progress.stop()
        self._update_controls()

    def _update_controls(self):
        for widget in getattr(self, "action_widgets", []):
            widget.configure(state="disabled" if self.busy else "normal")
        if not self.busy:
            executable = False
            try:
                plan = json.loads((RESULTS_DIR / "plan.json").read_text(encoding="utf-8"))
                executable = plan.get("executable", True) and not pick_place_3d.missing_execution_calibration(
                    pick_place_3d.load_config(CONFIG_PATH))
            except (OSError, ValueError, KeyError):
                pass
            self.execute_button.configure(state="normal" if self.armed.get() and executable else "disabled")
        self.arm_check.configure(state="disabled" if self.busy else "normal")
        for widget in self.markerless_buttons:
            if widget.winfo_exists():
                widget.configure(state="disabled" if self.busy else "normal")
        if self.auto_stop_button is not None and self.auto_stop_button.winfo_exists():
            self.auto_stop_button.configure(state="normal" if self.auto_calibration_active else "disabled")

    def _worker_command(self, *arguments: str) -> tuple[int, str]:
        command = [str(CAMERA_PYTHON), "-u", str(WORKER_SCRIPT),
                   "--config", str(CONFIG_PATH), *arguments]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
        output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
        return result.returncode, output

    def _start_task(self, label: str, work, done=None):
        if self.busy:
            return
        self._set_busy(True, label)

        def target():
            try:
                result = work()
                self.events.put(("done", label, result, done))
            except Exception as exc:
                self.events.put(("error", label, exc, done))

        threading.Thread(target=target, name="pick-place-worker", daemon=True).start()

    def _drain_events(self):
        try:
            while True:
                kind, label, value, callback = self.events.get_nowait()
                self._set_busy(False)
                if kind == "error":
                    self.armed.set(False)
                    self.operation.set(f"失败：{value}")
                    self._log(f"{label}失败：{value}")
                    messagebox.showerror("操作失败", str(value), parent=self.master)
                else:
                    if callback:
                        try:
                            callback(value)
                        except Exception as exc:
                            self.armed.set(False)
                            self.operation.set(f"失败：{exc}")
                            self._log(f"{label}结果处理失败：{exc}")
                            messagebox.showerror("操作失败", str(exc), parent=self.master)
        except queue.Empty:
            pass
        self.after(50, self._drain_events)

    def _refresh_config(self):
        try:
            config = pick_place_3d.load_config(CONFIG_PATH)
            missing = pick_place_3d.missing_execution_calibration(config)
            if missing:
                self.config_state.set(f"视觉：就绪  ·  真机：缺少 {len(missing)} 项标定")
            else:
                self.config_state.set("视觉：就绪  ·  真机标定：完整")
        except Exception as exc:
            self.config_state.set(f"配置错误：{exc}")

    def _refresh_all(self):
        self._refresh_config()

        def work():
            state = arm_control.get_status_snapshot()
            tcp = arm_control.get_tcp()
            motion = arm_control.get_motion_state()
            return state, tcp, motion

        def done(result):
            state, tcp, motion = result
            ready = (state.get("enabled") and state.get("powered_on") and
                     not state.get("protective_stop") and not state.get("emergency_stop") and
                     motion.get("inpos") and not motion.get("queue"))
            self.robot_state.set(
                f"机器人：{'就绪' if ready else '未就绪'}  ·  TCP {', '.join(f'{v:.1f}' for v in tcp[:3])}")
            self.operation.set("状态已刷新。")
            self._log(f"机器人状态：{motion}")

        self._start_task("正在读取机器人状态…", work, done)

    def _scan(self):
        def work():
            return self._worker_command("scan", "--output", str(RESULTS_DIR))

        def done(result):
            code, output = result
            if output:
                self._log(output)
            self._load_results()
            self.operation.set("扫描完成。" if code == 0 else "扫描完成，但没有候选通过阈值。")

        self._start_task("正在采集点云并匹配 CAD…", work, done)

    def _load_results(self, silent=False):
        match_path = RESULTS_DIR / "matches.json"
        try:
            self.matches = json.loads(match_path.read_text(encoding="utf-8")) if match_path.is_file() else []
        except Exception:
            self.matches = []
        for item in self.candidates.get_children():
            self.candidates.delete(item)
        for rank, match in enumerate(self.matches, 1):
            self.candidates.insert("", "end", values=(
                rank, f"{match['score']:.3f}", f"{match['fitness']:.3f}",
                f"{match['rmse_mm']:.2f}", f"{match['clearance_mm']:.1f}",
                match["cluster_points"],
            ))
        if self.matches:
            best = self.matches[0]
            self.match_summary.set(
                f"最佳候选：综合分 {best['score']:.3f}，误差 {best['rmse_mm']:.2f} mm")
        elif match_path.is_file():
            self.match_summary.set("本次扫描未找到达到阈值的模型实例")
        preview = RESULTS_DIR / "preview.png"
        if preview.is_file():
            try:
                self.preview_image = tk.PhotoImage(file=str(preview))
                self.preview_label.configure(image=self.preview_image, text="")
            except tk.TclError as exc:
                if not silent:
                    self._log(f"预览图加载失败：{exc}")
        plan_path = RESULTS_DIR / "plan.json"
        if plan_path.is_file():
            try:
                self._show_plan(json.loads(plan_path.read_text(encoding="utf-8")))
            except (OSError, ValueError, KeyError):
                pass

    def _show_plan(self, plan):
        image_path = RESULTS_DIR / "grasp_preview.png"
        if image_path.is_file():
            try:
                self.preview_image = tk.PhotoImage(file=str(image_path))
                self.preview_label.configure(image=self.preview_image, text="")
            except tk.TclError as exc:
                self._log(f"抓点预览加载失败：{exc}")
        if plan.get("executable") is False:
            self.armed.set(False)
            self.operation.set(
                f"杆身抓取方案已生成 · 夹持设定 {plan['close_command_mm']:g} mm · 待标定后计算真机轨迹。")
        else:
            self.operation.set(f"方案已生成：{plan['route']} 路线，旋转 {plan['direct_rotation_deg']:.1f}°。")
        if plan.get("selected_grasp"):
            selected = plan["selected_grasp"]
            xyz = ", ".join(f"{value:.2f}" for value in selected["grasp_center_camera_mm"])
            self.match_summary.set(
                f"抓杆身 · 候选簇 {selected['cluster_index']}\n相机抓取中心：{xyz} mm\n夹持设定：{plan['close_command_mm']:g} mm")
            self._log(f"杆身抓取方案：相机中心 [{xyz}] mm；home 释放位姿 {plan['place']}；方案文件 pick_results/grasp_plan.md")
        self._update_controls()

    def _view_3d(self):
        if not (RESULTS_DIR / "scene.ply").is_file():
            messagebox.showinfo("没有结果", "请先扫描。", parent=self.master)
            return
        subprocess.Popen([str(CAMERA_PYTHON), str(WORKER_SCRIPT), "--config", str(CONFIG_PATH),
                          "view-results", "--input", str(RESULTS_DIR)], cwd=ROOT)
        self._log("已打开独立三维结果窗口。")

    def _plan(self):
        if not self.matches:
            messagebox.showinfo("没有候选", "请先扫描并确认存在匹配候选。", parent=self.master)
            return

        def work():
            return self._worker_command("plan-results", "--input", str(RESULTS_DIR))

        def done(result):
            code, output = result
            if code:
                raise RuntimeError(output or "无法生成方案")
            self._log(output)
            plan = json.loads((RESULTS_DIR / "plan.json").read_text(encoding="utf-8"))
            self._show_plan(plan)

        self._start_task("正在生成抓取与放置方案…", work, done)

    def _ask_grasp_confirmation(self) -> bool:
        response: queue.Queue = queue.Queue(maxsize=1)

        def ask():
            result = messagebox.askyesno(
                "确认夹持",
                "机械臂已抬升。请现场确认工件已被可靠夹牢。\n\n选择“是”继续放置；选择“否”停止流程。",
                parent=self.master,
            )
            response.put(result)

        self.after(0, ask)
        return bool(response.get())

    def _execute(self):
        plan_path = RESULTS_DIR / "plan.json"
        if not self.armed.get():
            return
        if not plan_path.is_file():
            messagebox.showinfo("没有方案", "请先生成当前场景的动作方案。", parent=self.master)
            return
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("executable") is False:
            messagebox.showinfo("预览方案", "杆身抓取方案已生成，需补齐坐标标定后计算真机轨迹。", parent=self.master)
            self.armed.set(False)
            self._update_controls()
            return
        age = time.time() - float(plan.get("created_unix", 0))
        if age > 300:
            messagebox.showwarning("方案已过期", "方案超过 5 分钟，请重新扫描并生成方案。", parent=self.master)
            self.armed.set(False)
            return
        token = simpledialog.askstring("确认真机执行", "请输入 EXECUTE：", parent=self.master)
        if token != "EXECUTE":
            self.armed.set(False)
            self._update_controls()
            return
        if not messagebox.askyesno(
                "最后确认", "确认工作区无人、无障碍物，待抓物和平台位置未发生变化？",
                parent=self.master):
            self.armed.set(False)
            self._update_controls()
            return

        def work():
            config = pick_place_3d.load_config(CONFIG_PATH)
            pick_place_3d.execute_plan(plan, config, self._ask_grasp_confirmation)
            return True

        def done(_result):
            self.armed.set(False)
            self.operation.set("抓取和放置流程已完成。")
            self._log("真机流程完成。")
            self._update_controls()

        self._start_task("正在执行真机抓取流程…", work, done)

    def _markerless_progress(self):
        path = self.markerless_dataset / "manifest.json"
        try:
            count = len(json.loads(path.read_text(encoding="utf-8"))["samples"])
        except (OSError, ValueError, KeyError):
            count = 0
        if self.markerless_dialog is not None and self.markerless_dialog.winfo_exists():
            self.markerless_count.set(f"已采集 {count} 组 · 至少 8 组，建议 8～12 组不同角度")

    def _open_markerless_dialog(self):
        if self.markerless_dialog is not None and self.markerless_dialog.winfo_exists():
            self.markerless_dialog.lift()
            return
        dialog = tk.Toplevel(self.master)
        self.markerless_dialog = dialog
        dialog.protocol("WM_DELETE_WINDOW", lambda: self._cancel_automatic_calibration()
                        if self.auto_calibration_active else dialog.destroy())
        dialog.title("无板手眼标定 · 静止场景多角度采集")
        dialog.geometry("760x600")
        frame = ttk.Frame(dialog, padding=20)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="用现有料框标定，不用买标定板", style="Status.TLabel").pack(anchor="w")
        instructions = (
            "1. 料框和里面的工件保持原样，整个采集期间不要移动它们。\n\n"
            "2. 点击“生成自动轨迹”：围绕当前观察位生成 9 个角度，最多倾斜 8°。"
            "这一步只计算逆解和检查路径，不移动机械臂。\n\n"
            "3. 现场安装尺寸、机械臂及线缆净空、关节限位和负载核实后，"
            "“自动移动并采集”逐点低速移动，停稳采集，正常完成后回到本次起点并求解。"
            "执行期间请先断开其他夹具控制窗口，保持相机预览打开。\n\n"
            "4. 通信、运动或采集异常会中止流程，不会自动重发运动或继续回程。"
            "软件停止不代替现场急停。\n\n"
            "也可手动换角度后用下方按钮逐次采集。手眼结果先保存供复核，不自动用于抓取。")
        ttk.Label(frame, text=instructions, wraplength=660, justify="left").pack(anchor="w", pady=14)
        self.markerless_count = tk.StringVar(dialog)
        ttk.Label(frame, textvariable=self.markerless_count, style="Status.TLabel").pack(anchor="w", pady=8)
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=8)
        capture = ttk.Button(row, text="采集当前姿态", command=lambda: self._markerless_task("capture"))
        capture.pack(side="left", fill="x", expand=True, padx=(0, 6))
        solve = ttk.Button(row, text="求解并验证", command=lambda: self._markerless_task("solve"))
        solve.pack(side="left", fill="x", expand=True, padx=6)
        refresh = ttk.Button(row, text="刷新进度", command=self._markerless_progress)
        refresh.pack(side="left", padx=(6, 0))
        self.markerless_buttons = [capture, solve, refresh]
        automatic = ttk.Frame(frame)
        automatic.pack(fill="x", pady=8)
        plan_button = ttk.Button(automatic, text="生成自动轨迹", command=lambda: self._automatic_calibration("plan"))
        plan_button.pack(side="left", padx=(0, 6))
        preview_button = ttk.Button(automatic, text="查看轨迹图", command=self._view_calibration_trajectory)
        preview_button.pack(side="left", padx=6)
        run_button = ttk.Button(automatic, text="自动移动并采集", command=lambda: self._automatic_calibration("run"))
        run_button.pack(side="left", padx=6)
        self.auto_stop_button = ttk.Button(automatic, text="停止采集", command=self._cancel_automatic_calibration)
        self.auto_stop_button.pack(side="left", padx=6)
        self.markerless_buttons.extend([plan_button, preview_button, run_button])
        ttk.Label(frame, textvariable=self.operation, wraplength=660, foreground="#8b4300").pack(anchor="w", pady=8)
        self._markerless_progress()
        self._update_controls()

    def _markerless_task(self, action):
        if self.busy:
            return

        def work():
            environment = os.environ.copy()
            environment["PYTHONIOENCODING"] = "utf-8"
            process = subprocess.run(
                [str(CAMERA_PYTHON), "-u", str(ROOT / "markerless_handeye.py"), action,
                 "--config", str(CONFIG_PATH), "--dataset", str(self.markerless_dataset)], cwd=ROOT, capture_output=True,
                text=True, encoding="utf-8", errors="replace", env=environment)
            return process.returncode, (process.stdout + "\n" + process.stderr).strip()

        def done(result):
            code, output = result
            self._markerless_progress()
            self._log(output)
            if code:
                raise RuntimeError(output or "无板标定未完成")
            if action == "capture":
                self.operation.set("本组已采集。保持场景不动，手动换一个拍摄角度、停稳后再采集。")
            else:
                self.operation.set("无板手眼候选已求解并通过内部验证，结果已保存供复核，尚未写入真机配置。")

        self._start_task("正在采集当前点云和机器人位姿…" if action == "capture" else "正在配准多组场景并求解手眼关系…", work, done)

    def _view_calibration_trajectory(self):
        path = ROOT / "calibration_data" / "auto_trajectory.png"
        if path.is_file():
            os.startfile(path)
        else:
            messagebox.showinfo("还没有轨迹", "请先生成自动轨迹。", parent=self.master)

    def _cancel_automatic_calibration(self):
        if self.auto_calibration_active:
            self.auto_cancel_file.write_text("stop\n", encoding="utf-8")
            self.operation.set("已请求停止采集，正在等待停止确认。")

    def _automatic_calibration(self, action):
        if self.busy:
            return
        if action == "run":
            site = ROOT / "calibration_data" / "calibration_site.json"
            try:
                settings = json.loads(site.read_text(encoding="utf-8"))
                if not settings.get("reviewed_plan_sha256"):
                    raise ValueError("pending site review")
            except (OSError, ValueError):
                messagebox.showinfo("自动采集待补现场数据",
                                    "轨迹可先计算。真机移动还缺：附件尺寸、整条机械臂及线缆净空、实际关节限位，以及正确的末端负载配置。",
                                    parent=self.master)
                return
            self.auto_cancel_file.unlink(missing_ok=True)
            self.auto_calibration_active = True
            self.markerless_dataset = ROOT / "calibration_data" / time.strftime("auto_%Y%m%d_%H%M%S")

        def work():
            environment = os.environ.copy()
            environment["PYTHONIOENCODING"] = "utf-8"
            arguments = [str(CAMERA_PYTHON), "-u", str(ROOT / "calibration_motion.py"), action,
                         "--config", str(CONFIG_PATH)]
            if action == "run":
                arguments += ["--dataset", str(self.markerless_dataset), "--cancel-file", str(self.auto_cancel_file)]
            try:
                result = subprocess.run(arguments, cwd=ROOT, capture_output=True, text=True,
                                        encoding="utf-8", errors="replace", env=environment)
                return result.returncode, (result.stdout + "\n" + result.stderr).strip()
            except Exception as exc:
                return 2, str(exc)

        def done(result):
            self.auto_calibration_active = False
            self._update_controls()
            self._markerless_progress()
            code, output = result
            self._log(output)
            if code:
                raise RuntimeError(output or "自动采集未完成")
            self.operation.set("9 姿态自动轨迹已生成；运动学检查通过，待核实现场条件。" if action == "plan"
                               else "自动采集、返回起点及手眼求解完成，结果已保存供复核。")

        self._start_task("正在计算和核验轨迹（不移动）…" if action == "plan" else "正在逐点移动、停稳采集…", work, done)

    def _launch_calibration(self, command: str):
        if self.busy:
            return
        creation_flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        process = subprocess.Popen(
            [str(CAMERA_PYTHON), "-u", str(WORKER_SCRIPT), "--config", str(CONFIG_PATH), command],
            cwd=ROOT, creationflags=creation_flags)
        self.external_process = process
        self._set_busy(True, "标定向导已在独立窗口打开…")
        self._log(f"已启动 {command} 标定向导。")
        self.after(300, self._poll_external)

    def _poll_external(self):
        if self.external_process is None:
            return
        code = self.external_process.poll()
        if code is None:
            self.after(300, self._poll_external)
            return
        self.external_process = None
        self._set_busy(False, "标定窗口已关闭。")
        self._refresh_config()
        self._log(f"标定进程结束，返回码 {code}。")

    def _configure_tool(self):
        offset_text = simpledialog.askstring(
            "夹持中心", "输入 TCP 到实际夹持中心的 X Y Z（mm）：",
            parent=self.master)
        if offset_text is None:
            return
        maximum = simpledialog.askfloat(
            "最大工件外径", "输入当前手指安装方式实测可容纳的最大外径（mm）：",
            minvalue=0.1, parent=self.master)
        if maximum is None:
            return
        try:
            offset = [float(value) for value in offset_text.replace(",", " ").split()]
            if len(offset) != 3:
                raise ValueError
        except ValueError:
            messagebox.showerror("输入错误", "夹持中心必须是三个数字。", parent=self.master)
            return

        def work():
            arguments = ["configure-tool", "--offset", *map(str, offset),
                         "--max-object", str(maximum)]
            return self._worker_command(*arguments)

        def done(result):
            code, output = result
            if code:
                raise RuntimeError(output)
            self._log(output)
            self._refresh_config()
            self.operation.set("夹具几何参数已保存。")

        self._start_task("正在保存夹具几何参数…", work, done)

    def _open_config(self):
        subprocess.Popen(["notepad.exe", str(CONFIG_PATH)])

    def _close(self):
        if self.busy:
            messagebox.showwarning("操作进行中", "请等待当前操作结束后再关闭软件。", parent=self.master)
            return
        self.master.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(description="JAKA 3D 视觉抓取")
    parser.add_argument("--calibrate-without-board", action="store_true")
    args = parser.parse_args()
    root = tk.Tk()
    app = PickPlaceApp(root)
    if args.calibrate_without_board:
        root.after(100, app._open_markerless_dialog)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
