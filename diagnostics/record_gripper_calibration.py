"""Record RGB/depth and empty-jaw command cycles at the table overview.

Never sends arm motion. Commands are labels, not measured finger separation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import calibration_motion as safety
from camera_capture import CameraCapture
from color_depth_viewer import find_sdk_dir, render_depth
from controller_lease import controller_lease
from gripper_transport import close_connections
import one_click_pick as motion
import pick_place_3d as vision
import side_pick_execute as side


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--hold-seconds", type=float, default=1.5)
    args = parser.parse_args()
    if not 1 <= args.cycles <= 5 or not 1 <= args.hold_seconds <= 5:
        parser.error("cycles must be 1..5 and hold-seconds 1..5")
    output = ROOT / "calibration_data" / time.strftime("jaw_video_%Y%m%d_%H%M%S")
    output.mkdir(parents=True)
    config = vision.load_config(ROOT / "pick_place_config.json")
    report = dict(arm_motion_sent=False, command_is_measured_gap=False,
                  output=str(output), commands=[], completed=False)
    camera = None
    worker = None
    done = threading.Event()
    healthy = threading.Event()
    label = {"command_mm": None, "step": -1, "cycle": 0}
    lock = threading.Lock()

    def record(first_pair):
        writers = []
        frame_count = 0
        fresh_count = 0
        last_pair = first_pair
        last_fresh = time.monotonic()
        next_frame = last_fresh
        try:
            color, depth = first_pair
            for name, shape in (("color.mp4", color.shape[:2]), ("depth.mp4", depth.shape[:2])):
                writer = cv2.VideoWriter(str(output / name), cv2.VideoWriter_fourcc(*"mp4v"),
                                         10, (shape[1], shape[0]))
                if not writer.isOpened():
                    raise RuntimeError("Cannot open video writer: " + name)
                writers.append(writer)
            healthy.set()
            with (output / "frames.jsonl").open("w", encoding="utf-8") as metadata:
                while not done.is_set():
                    pair = camera.poll()
                    if pair is not None:
                        last_pair = pair
                        last_fresh = time.monotonic()
                        fresh_count += 1
                    now = time.monotonic()
                    if now - last_fresh > 2:
                        raise RuntimeError("Camera stalled; no further gripper experiments")
                    if now < next_frame:
                        time.sleep(.005)
                        continue
                    next_frame += .1
                    color, depth = last_pair
                    with lock:
                        sample = dict(label)
                    sample.update(frame=frame_count, created_unix=time.time(),
                                  frame_age_seconds=now-last_fresh,
                                  command_is_measured_gap=False)
                    annotated = color.copy()
                    text = f"step {sample['step']} | command {sample['command_mm']} mm | NOT measured gap"
                    cv2.putText(annotated, text, (18, 35), cv2.FONT_HERSHEY_SIMPLEX,
                                .7, (0, 255, 255), 2, cv2.LINE_AA)
                    writers[0].write(annotated)
                    writers[1].write(render_depth(depth, 150, 800, cv2.COLORMAP_TURBO))
                    if frame_count % 5 == 0:
                        stem = f"frame_{frame_count:05d}"
                        cv2.imwrite(str(output / (stem + "_color.png")), color)
                        cv2.imwrite(str(output / (stem + "_depth_mm.png")), depth)
                        sample.update(color=stem + "_color.png", depth=stem + "_depth_mm.png")
                    metadata.write(json.dumps(sample) + "\n")
                    metadata.flush()
                    frame_count += 1
        except BaseException as exc:
            report["recording_error"] = str(exc)
        finally:
            healthy.clear()
            for writer in writers:
                writer.release()
            report.update(video_frames=frame_count, fresh_frames=fresh_count)

    try:
        with controller_lease(8):
            safety.ensure_exclusive_client()
            robot = motion.FastRobot()
            start = robot.motion_snapshot()
            safety.check_motion_state(start)
            overview = config["robot"]["table_observation_pose"]
            vision.verify_observation_unchanged(overview, vision.read_observation_pose())
            report["start_state"] = start
            camera = CameraCapture(find_sdk_dir(config["camera"].get("sdk")))
            deadline = time.monotonic() + 45
            pair = None
            while pair is None and time.monotonic() < deadline:
                pair = camera.poll()
                time.sleep(.02)
            if pair is None:
                raise RuntimeError("Camera did not provide initial frames; no command sent")
            worker = threading.Thread(target=record, args=(pair,), daemon=True)
            worker.start()
            if not healthy.wait(5):
                raise RuntimeError(report.get("recording_error", "Video recorder did not start"))
            steps = [(cycle, value) for cycle in range(1, args.cycles+1)
                     for value in (8.65, 12, 16, 20, 16, 12, 8.65)]
            steps.append((args.cycles, 20))
            gripper = side.robot_gripper()
            gripper.connect()
            close_connections()
            for index, (cycle, value) in enumerate(steps):
                if not healthy.is_set():
                    raise RuntimeError(report.get("recording_error", "Video recorder stopped"))
                current = robot.motion_snapshot()
                safety.check_motion_state(current)
                if max(safety.pose_error(start["actual_position"], current["actual_position"])) > .1:
                    raise RuntimeError("Arm moved during jaw recording")
                command = dict(step=index, cycle=cycle, command_mm=value, sent_unix=time.time(),
                               command_is_measured_gap=False)
                report["commands"].append(command)
                with lock:
                    label.update(command)
                gripper.set_opening_mm(value, settle_time=.1)
                close_connections()
                command["setpoint_readback_unix"] = time.time()
                print(f"cycle {cycle}/{args.cycles}, command {value:.2f} mm", flush=True)
                limit = time.monotonic() + args.hold_seconds
                while time.monotonic() < limit:
                    if not healthy.is_set():
                        raise RuntimeError(report.get("recording_error", "Video recorder stopped"))
                    time.sleep(.05)
            report.update(completed=True, end_state=robot.motion_snapshot())
    except BaseException as exc:
        report["error"] = str(exc)
        raise
    finally:
        done.set()
        if worker is not None:
            worker.join(5)
        if camera is not None:
            camera.close()
        close_connections()
        if report.get("recording_error"):
            report["completed"] = False
        safety.save_json(output / "experiment.json", report)
        print(json.dumps({k: value for k, value in report.items() if k not in ("commands", "start_state", "end_state")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
