"""Small, reviewed calibration trajectories; planning sends read-only commands.

Protocol: https://www.jaka.com/docs/legacy172/en/guide/1.7.2/tcpip.html
TCP protocol units are mm/degrees (not the native SDK's radians).
Kinematic checks do NOT establish obstacle or self-collision clearance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

import arm_control
from gripper_transport import ControllerConnection
import markerless_handeye as handeye
import pick_place_3d as vision

ROOT = Path(__file__).resolve().parent
PLAN_PATH = ROOT / "calibration_data" / "auto_trajectory.json"
SITE_PATH = ROOT / "calibration_data" / "calibration_site.json"
TILT_DEG = 6.0
JOINT_SPEED = 1.0
JOINT_ACCEL = 2.0


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def vector(value, size=6):
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"expected {size} finite values")
    return array


def checked_reply(reply, name):
    if reply.get("cmdName") != name or str(reply.get("errorCode")).lower() not in ("0", "0x0"):
        raise RuntimeError(f"{name}: {reply}")
    return reply


class Robot:
    """Fresh finite-time requests; never retry movement, including ambiguous writes."""
    def call(self, name, **parameters):
        connection = ControllerConnection(arm_control.ROBOT_IP)
        try:
            return checked_reply(connection.request({"cmdName": name, **parameters}, timeout=1.5), name)
        finally:
            connection.close()

    def snapshot(self):
        last = None
        for _ in range(3):
            try:
                return arm_control.get_status_snapshot(timeout=2.0)
            except (OSError, TimeoutError, ValueError) as exc:
                last = exc
                time.sleep(0.1)
        raise last

    def motion_snapshot(self):
        """Use request/response reads while moving; the port-10000 stream is intermittent."""
        motion = self.call("get_motion_state")
        return {**motion,
                "actual_position": self.call("get_actual_tcp_pos")["position"],
                "joint_actual_position": self.call("get_actual_joint_pos")["position"]}

    def inverse(self, pose, reference):
        last = None
        for _ in range(3):
            try:
                return vector(self.call("kine_inverse", cartPosition=list(pose),
                                        jointPosition=list(reference))["jointPosition"])
            except (OSError, TimeoutError, ConnectionError) as exc:
                last = exc
                time.sleep(0.05)
        raise last

    def forward(self, joints):
        last = None
        for _ in range(3):
            try:
                return vector(self.call("kine_forward", jointPosition=list(joints))["cartPosition"])
            except (OSError, TimeoutError, ConnectionError) as exc:
                last = exc
                time.sleep(0.05)
        raise last

    def move(self, joints):
        # Finite 1-degree-or-smaller orientation steps with no blending/queueing.
        return self.call("joint_move", relFlag=0, jointPosition=list(joints),
                         speed=JOINT_SPEED, accel=JOINT_ACCEL, tol=0)

    def stop(self):
        return self.call("stop_program")


def check_state(status, stationary=True):
    for key in ("emergency_stop", "protective_stop", "on_soft_limit", "paused", "drag_status"):
        if key not in status or status[key] not in (False, 0):
            raise RuntimeError(f"robot state prevents calibration: {key}={status.get(key)}")
    if (str(status.get("errcode")).lower() not in ("0", "0x0")
            or status.get("enabled") is not True or status.get("powered_on") != 1):
        raise RuntimeError("robot is faulted, disabled or powered off")
    if status.get("current_tool_id") != 0 or status.get("current_user_id") != 0:
        raise RuntimeError("calibration trajectory requires tool 0 and base frame 0")
    if stationary and (status.get("inpos") is not True or status.get("interp_state") != 0
                       or np.max(np.abs(vector(status.get("instVel")))) > 0.01):
        raise RuntimeError("robot is not stationary and idle")
    if not 0 < float(status.get("rapidrate", 0)) <= 1:
        raise RuntimeError("invalid robot speed override")
    vector(status.get("actual_position"))
    vector(status.get("joint_actual_position"))


def check_motion_state(status, stationary=True):
    for key in ("estop", "collision", "soft_limit", "paused"):
        if status.get(key) not in (False, 0):
            raise RuntimeError(f"robot motion fault: {key}={status.get(key)}")
    if stationary and (status.get("inpos") is not True or status.get("queue")
                       or status.get("active_queue")):
        raise RuntimeError("robot is not stationary and idle")
    vector(status.get("actual_position"))
    vector(status.get("joint_actual_position"))


def make_views(origin):
    """Nine orientations, including origin; rotations about local flange XY axes."""
    origin = vector(origin)
    rotation = Rotation.from_euler("xyz", origin[3:], degrees=True)
    views = [{"name": "center", "pose": origin.tolist(), "tilt_vector_deg": [0, 0, 0]}]
    for index in range(8):
        azimuth = index * math.pi / 4
        tilt = TILT_DEG * np.array([math.cos(azimuth), math.sin(azimuth), 0])
        rpy = (rotation * Rotation.from_rotvec(np.deg2rad(tilt))).as_euler("xyz", degrees=True)
        views.append({"name": f"view_{index + 1:02d}", "pose": [*origin[:3], *rpy],
                      "tilt_vector_deg": tilt.tolist()})
    return views


def pose_error(left, right):
    return handeye.pose_difference(left, right)


def make_plan(robot):
    start = robot.snapshot()
    check_state(start)
    origin, joints0 = vector(start["actual_position"]), vector(start["joint_actual_position"])
    tool = vector(robot.call("get_curr_tool_offset")["tool_offset"])
    if np.max(np.abs(tool)) > 1e-6:
        raise RuntimeError("tool 0 offset is not zero")
    payload = robot.call("get_payload")
    collision = robot.call("get_clsn_sensitivity")["sensitivityLevel"]
    motion = robot.call("get_motion_state")
    if motion.get("queue") != 0 or motion.get("active_queue") != 0 or not motion.get("inpos"):
        raise RuntimeError("controller has queued motion")
    if any(motion.get(key, 1) for key in ("collision", "soft_limit", "estop")):
        raise RuntimeError("controller reports a motion fault")
    if any(value > limit for value, limit in zip(pose_error(robot.forward(joints0), origin), (0.1, 0.05))):
        raise RuntimeError("controller FK does not match actual pose/coordinate convention")
    views = make_views(origin)
    legs, all_joints, all_poses = [], [joints0], [origin]
    rotation = Rotation.from_euler("xyz", origin[3:], degrees=True)
    reference = joints0.copy()
    for view in views[1:]:
        outward = []
        for fraction in np.linspace(0, 1, 9)[1:]:
            delta = np.deg2rad(view["tilt_vector_deg"]) * fraction
            pose = np.r_[origin[:3], (rotation * Rotation.from_rotvec(delta)).as_euler("xyz", degrees=True)]
            joint = robot.inverse(pose, reference)
            # Use literal differences; wrapping could hide an unintended full turn.
            if np.max(np.abs(joint - reference)) > 3 or np.max(np.abs(joint - joints0)) > 20:
                raise RuntimeError("IK branch discontinuity or excessive joint motion")
            fk = robot.forward(joint)
            if any(v > limit for v, limit in zip(pose_error(fk, pose), (0.1, 0.05))):
                raise RuntimeError("IK/FK round trip failed")
            # The executor uses MoveJ: check the actual joint-interpolated path,
            # not an assumed Cartesian line. Samples are not a collision model.
            interpolated = []
            for mix in (0.25, 0.5, 0.75):
                q = reference * (1 - mix) + joint * mix
                p = robot.forward(q)
                displacement, tilt = pose_error(p, origin)
                if displacement > 0.5 or tilt > TILT_DEG + 0.2:
                    raise RuntimeError("joint interpolation exceeds small flange envelope")
                interpolated.append({"joints": q.tolist(), "pose": p.tolist()})
                all_joints.append(q)
                all_poses.append(p)
            outward.append({"joints": joint.tolist(), "pose": pose.tolist(),
                            "interpolation_checks": interpolated})
            all_joints.append(joint)
            all_poses.append(fk)
            reference = joint
        # Return on the same validated branch before observing another direction.
        reverse = [{"joints": item["joints"], "pose": item["pose"]}
                   for item in outward[-2::-1]]
        reverse.append({"joints": joints0.tolist(), "pose": origin.tolist()})
        legs.append({"view": view["name"], "outward": outward, "return": reverse})
        reference = joints0.copy()
        print(f"kinematics checked: {view['name']}", flush=True)
    end = robot.snapshot()
    check_state(end)
    if (max(pose_error(end["actual_position"], origin)) > 0.1
            or np.max(np.abs(vector(end["joint_actual_position"]) - joints0)) > 0.05):
        raise RuntimeError("robot changed pose during planning")
    errors = [pose_error(p, origin) for p in all_poses]
    return {"schema_version": 1, "created_unix": time.time(), "robot_host": arm_control.ROBOT_IP,
            "motion_sent": False, "status": "kinematics_checked_site_pending", "executable": False,
            "origin_pose": origin.tolist(), "origin_joints": joints0.tolist(),
            "tool_id": 0, "user_id": 0, "views": views, "legs": legs,
            "joint_speed_deg_s": JOINT_SPEED, "joint_accel_deg_s2": JOINT_ACCEL,
            "payload_at_planning": payload, "collision_sensitivity": collision,
            "checks": {"ik_samples": sum(len(leg["outward"]) for leg in legs),
                       "fk_samples": len(all_poses), "max_sampled_flange_shift_mm": max(e[0] for e in errors),
                       "max_sampled_flange_tilt_deg": max(e[1] for e in errors),
                       "joint_min_deg": np.min(all_joints, axis=0).tolist(),
                       "joint_max_deg": np.max(all_joints, axis=0).tolist(),
                       "max_joint_excursion_deg": np.max(np.abs(np.array(all_joints) - joints0), axis=0).tolist(),
                       "joint_limits_checked": False, "collision_clearance_checked": False},
            "remaining": ["现场相机/夹具/支架与整条机械臂、线缆的运动净空",
                          "当前机器人的实际关节软限位", "与已安装附件一致的负载和重心配置"],
            "source": "https://www.jaka.com/docs/legacy172/en/guide/1.7.2/tcpip.html"}


def validate_site(plan, site, payload):
    """Require concrete installation data; never infer clearance from IK."""
    if not site.get("measurement_source"):
        raise ValueError("缺少现场尺寸、限位及负载的来源记录")
    radius = float(site.get("attachment_radius_mm") or 0)
    clearance = float(site.get("attachment_clearance_mm") or 0)
    if not math.isfinite(radius) or radius <= 0 or not math.isfinite(clearance):
        raise ValueError("缺少法兰中心到附件最远处的实测距离")
    # Bound displacement of any rigid attachment point. Cable motion and the
    # rest of the arm need separate on-site review of the actual joint range.
    swept = 2 * radius * math.sin(math.radians(TILT_DEG + 0.2) / 2) + 0.5
    if clearance < swept + 10:
        raise ValueError(f"末端净空不足：当前轨迹要求至少 {swept + 10:.1f} mm（含余量）")
    digest = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    if site.get("reviewed_plan_sha256") != digest or site.get("arm_and_cables_clear") is not True:
        raise ValueError("尚未核实这条具体轨迹的机械臂、相机、夹具和线缆净空")
    limits = np.asarray(site.get("joint_limits_deg"), dtype=float)
    if limits.shape != (6, 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("缺少这台机器人的六轴实际软限位")
    if (np.any(np.array(plan["checks"]["joint_min_deg"]) < limits[:, 0] + 2)
            or np.any(np.array(plan["checks"]["joint_max_deg"]) > limits[:, 1] - 2)):
        raise ValueError("轨迹离关节软限位不足 2 度")
    expected = site.get("verified_payload") or {}
    mass = float(expected.get("mass", -1))
    if not math.isfinite(mass) or mass < 0 or abs(float(payload.get("mass", 0)) - mass) > 0.01:
        raise ValueError("机器人负载尚未按实际相机、夹具和支架核实配置")
    if mass == 0 and site.get("use_existing_controller_payload_for_slow_calibration") is not True:
        raise ValueError("控制器负载为 0 kg，未允许本次低速标定沿用当前控制器配置")
    if np.max(np.abs(vector(expected.get("centroid"), 3) - vector(payload.get("centroid"), 3))) > 0.1:
        raise ValueError("机器人负载重心与核实值不符")
    return {"attachment_sweep_bound_mm": swept, "required_clearance_mm": swept + 10}


def ensure_exclusive_client():
    """Legacy controller supports one command client; old GUI processes do not lock."""
    if os.name != "nt":
        raise RuntimeError("this executor requires Windows client ownership checks")
    script = ("Get-NetTCPConnection -ErrorAction Stop | Where-Object { $_.RemoteAddress -eq '"
              + arm_control.ROBOT_IP + "' -and $_.RemotePort -eq 10001 -and $_.State -eq 'Established' } | "
              "Select-Object -ExpandProperty OwningProcess | Sort-Object -Unique")
    result = subprocess.run(["powershell", "-NoProfile", "-Command", script],
                            capture_output=True, text=True, timeout=8)
    if result.returncode != 0:
        raise RuntimeError("cannot check controller connection ownership")
    owners = [int(line) for line in result.stdout.splitlines() if line.strip().isdigit()]
    if any(pid != os.getpid() for pid in owners):
        raise RuntimeError(f"控制端口正被其他程序使用（PID {owners}），先断开它们的控制连接")


def stop_and_verify(robot):
    robot.stop()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = robot.motion_snapshot()
        check_motion_state(state, stationary=False)
        if state.get("inpos") and not state.get("queue") and not state.get("active_queue"):
            return
        time.sleep(0.1)
    raise RuntimeError("STOP UNVERIFIED: 请现场按急停；软件未能确认停止")


def move_step(robot, start, target, cancelled=lambda: False):
    if cancelled():
        raise RuntimeError("采集已取消")
    before = robot.motion_snapshot()
    check_motion_state(before)
    q0, q1 = vector(start["joints"]), vector(target["joints"])
    if np.max(np.abs(vector(before["joint_actual_position"]) - q0)) > 0.15:
        raise RuntimeError("起点已变化；不会从新位置执行旧轨迹")
    if np.max(np.abs(q1 - q0)) > 3.01:
        raise RuntimeError("trajectory step exceeds joint increment limit")
    try:
        robot.move(q1)  # Exactly once. Even loss of ACK triggers stop, not retry.
        deadline, stable = time.monotonic() + 15, None
        while time.monotonic() < deadline:
            if cancelled():
                raise RuntimeError("采集已取消")
            state = robot.motion_snapshot()
            check_motion_state(state, stationary=False)
            actual = vector(state["joint_actual_position"])
            if np.any(actual < np.minimum(q0, q1) - 0.3) or np.any(actual > np.maximum(q0, q1) + 0.3):
                raise RuntimeError("关节偏离当前小步轨迹")
            position_error, angle_error = pose_error(state["actual_position"], target["pose"])
            settled = (state["inpos"] and np.max(np.abs(actual - q1)) < 0.08
                       and position_error < 0.5 and angle_error < 0.2
                       and not state.get("queue") and not state.get("active_queue"))
            if settled:
                stable = stable or time.monotonic()
                if time.monotonic() - stable >= 0.5:
                    return
            else:
                stable = None
            time.sleep(0.1)
        raise TimeoutError("等待到位超时")
    except BaseException as problem:
        try:
            stop_and_verify(robot)
        except Exception as stopping:
            raise RuntimeError(f"{problem}; STOP UNVERIFIED: {stopping}; 请现场按急停") from problem
        raise


def capture_child(config, dataset, cancelled):
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    with (dataset / "capture.log").open("a", encoding="utf-8") as log:
        child = subprocess.Popen([sys.executable, "-u", str(ROOT / "markerless_handeye.py"),
                                  "capture", "--config", str(config), "--dataset", str(dataset)],
                                 stdout=log, stderr=subprocess.STDOUT, env=environment)
        try:
            deadline = time.monotonic() + 90
            while child.poll() is None:
                if cancelled() or time.monotonic() > deadline:
                    raise RuntimeError("采集取消或超时；停止本轮，不继续运动")
                time.sleep(0.1)
            if child.returncode:
                raise RuntimeError(f"深度采集失败：{dataset / 'capture.log'}")
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=5)


def run_collection(robot, plan, site, config, dataset, cancelled=lambda: False,
                   capture=capture_child, exclusive=ensure_exclusive_client):
    if dataset.exists():
        raise ValueError("自动采集使用新的目录，避免混入移动过场景的旧数据")
    # All prerequisites are checked before the first write or camera acquisition.
    validate_site(plan, site, robot.call("get_payload"))
    if time.time() - plan["created_unix"] > 1800:
        raise ValueError("轨迹已过期，请重新生成并核实现场")
    state = robot.snapshot()
    check_state(state)
    if max(pose_error(state["actual_position"], plan["origin_pose"])) > 0.1:
        raise ValueError("机械臂已不在生成轨迹的位置")
    if np.max(np.abs(vector(state["joint_actual_position"]) - vector(plan["origin_joints"]))) > 0.05:
        raise ValueError("机械臂关节起点已变化")
    if robot.call("get_clsn_sensitivity")["sensitivityLevel"] not in (1, 2, 3, 4, 5):
        raise ValueError("碰撞检测未开启")
    exclusive()
    dataset.mkdir(parents=True)
    save_json(dataset / "trajectory.json", plan)
    save_json(dataset / "site.json", site)
    report = {"completed": False, "motion_attempts": 0, "samples": 0, "returned_to_start": False}
    cursor = {"joints": plan["origin_joints"], "pose": plan["origin_pose"]}
    try:
        if cancelled():
            raise RuntimeError("采集已取消")
        capture(config, dataset, cancelled)
        report["samples"] += 1
        for leg in plan["legs"]:
            for direction in ("outward", "return"):
                exclusive()
                for target in leg[direction]:
                    report["motion_attempts"] += 1
                    move_step(robot, cursor, target, cancelled)
                    cursor = target
                    save_json(dataset / "run_report.json", report)
                if direction == "outward":
                    capture(config, dataset, cancelled)
                    observation = handeye.read_manifest(dataset)["samples"][-1]["observation"]
                    if max(pose_error(observation["tcp_pose"], cursor["pose"])) > 0.5:
                        raise RuntimeError("采集位姿与目标不一致")
                    report["samples"] += 1
                    print(f"已采集 {report['samples']}/9", flush=True)
        report.update(completed=True, returned_to_start=True)
    except BaseException as exc:
        report["error"] = str(exc)
        # Do not start an unreviewed return motion after a failure/cancellation.
        raise
    finally:
        save_json(dataset / "run_report.json", report)
    return report


def write_preview(plan, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].set(title="9 camera views; flange center fixed", xlabel="Local X tilt (deg)", ylabel="Local Y tilt (deg)")
    for i, view in enumerate(plan["views"]):
        x, y, _ = view["tilt_vector_deg"]
        axes[0].plot([0, x], [0, y], "-", color="#b5c9db")
        axes[0].scatter(x, y, color="#1375af")
        axes[0].annotate(str(i), (x, y), xytext=(5, 5), textcoords="offset points")
    axes[0].axis("equal")
    axes[0].grid(alpha=0.2)
    axes[1].bar(np.arange(1, 7), plan["checks"]["max_joint_excursion_deg"], color="#1375af")
    axes[1].set(title="Maximum joint change from start", xlabel="Joint", ylabel="Degrees")
    figure.suptitle("KINEMATICS CHECKED - CLEARANCE / PAYLOAD / LIMITS PENDING", fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="自动无板采集：plan 只计算，run 在现场条件完整后移动")
    parser.add_argument("action", choices=("plan", "run"))
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--site", type=Path, default=SITE_PATH)
    parser.add_argument("--config", type=Path, default=vision.DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--cancel-file", type=Path)
    args = parser.parse_args()
    try:
        robot = Robot()
        if args.action == "plan":
            plan = make_plan(robot)
            save_json(args.plan, plan)
            write_preview(plan, args.plan.with_suffix(".png"))
            print(json.dumps({"plan": str(args.plan), "views": len(plan["views"]),
                              "checks": plan["checks"], "motion_sent": False}, ensure_ascii=False, indent=2))
        else:
            plan = json.loads(args.plan.read_text(encoding="utf-8"))
            site = json.loads(args.site.read_text(encoding="utf-8"))
            validate_site(plan, site, robot.call("get_payload"))
            ensure_exclusive_client()
            # Do not trust an edited saved path: recompute all targets and compare.
            fresh = make_plan(robot)
            if (pose_error(plan["origin_pose"], fresh["origin_pose"])[0] > 0.1
                    or len(plan["legs"]) != 8):
                raise ValueError("保存的轨迹与现场不符")
            for old_leg, new_leg in zip(plan["legs"], fresh["legs"]):
                for direction in ("outward", "return"):
                    if len(old_leg[direction]) != len(new_leg[direction]):
                        raise ValueError("保存的轨迹已改变")
                    for old, new in zip(old_leg[direction], new_leg[direction]):
                        if (np.max(np.abs(vector(old["joints"]) - vector(new["joints"]))) > 0.02
                                or max(pose_error(old["pose"], new["pose"])) > 0.02):
                            raise ValueError("保存的轨迹已改变")
            dataset = args.dataset or ROOT / "calibration_data" / time.strftime("auto_%Y%m%d_%H%M%S")
            cancelled = lambda: bool(args.cancel_file and args.cancel_file.exists())
            result = run_collection(robot, plan, site, args.config, dataset, cancelled)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print(f"dataset: {dataset}", flush=True)
            print(json.dumps(handeye.solve_dataset(dataset), ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"未完成：{exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
