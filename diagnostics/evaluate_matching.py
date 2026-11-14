"""Capture/replay camera-only matching data; never access robot or gripper I/O."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np


def capture(directory, count):
    import cv2
    from camera_capture import CameraCapture
    from color_depth_viewer import find_sdk_dir

    directory.mkdir(parents=True, exist_ok=True)
    camera = CameraCapture(find_sdk_dir(None), with_points=True)
    deadline = time.monotonic() + 60
    images = None
    warmup = 0
    saved = 0
    try:
        while saved < count and time.monotonic() < deadline:
            pair = camera.poll()
            if pair is not None:
                images = pair
            frame = camera.poll_points()
            if frame is None or images is None:
                time.sleep(0.02)
                continue
            sequence, stamp, points = frame
            warmup += 1
            if warmup <= 3:
                continue
            saved += 1
            stem = f"frame_{saved:03d}"
            np.savez_compressed(directory / (stem + ".npz"), native_points=points,
                                color=images[0], depth=images[1], captured_unix=stamp)
            cv2.imwrite(str(directory / (stem + "_color.png")), images[0])
            print(json.dumps({"saved": stem, "points": len(points), "sequence": sequence}), flush=True)
    finally:
        camera.close()
    if saved != count:
        raise RuntimeError(f"Only captured {saved}/{count} frames before timeout")


def evaluate(directory, strategy, limit=0, track=False):
    import cv2
    import open3d as o3d
    import pick_place_3d as vision
    from color_depth_viewer import render_depth
    from pose_overlay import PoseOverlay

    config = vision.load_config(ROOT / "pick_place_config.json")
    config["matching"]["strategy"] = strategy
    mesh = vision.load_fcstd_mesh(ROOT / config["model"]["path"])
    o3d.utility.random.seed(7)
    model = mesh.sample_points_uniformly(int(config["matching"]["model_sample_points"]))
    summary = []
    previous = []
    label = (strategy + ("_tracking" if track else "_cold")) if strategy == "clutter" else strategy
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    paths = sorted(directory.glob("frame_*.npz"))
    for path in paths[:limit] if limit else paths:
        data = np.load(path)
        points = np.array(data["native_points"], dtype=float)
        if config["camera"].get("point_cloud_frame") == "opencv_optical":
            points[:, 1] *= -1
        points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 0)]
        scene = vision.crop_cloud(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)),
                                  config["camera"]["roi"])
        scene = scene.voxel_down_sample(float(config["matching"]["scene_voxel_mm"]))
        o3d.utility.random.seed(7)
        started = time.monotonic()
        objects, _plane, _support = vision.remove_support_plane(scene, config)
        objects = vision.exclude_camera_geometry(objects, config)
        clusters = vision.cluster_objects(objects, config)
        matches = vision.find_matches(model, scene, clusters, config, previous_matches=previous if track else None)
        previous = matches
        duration = time.monotonic() - started
        diagnostics = config.get("_matching_diagnostics", [])
        result = {"frame": path.name, "strategy": strategy, "tracking": track, "duration_s": duration,
                  "scene_points": len(scene.points), "clusters": len(clusters),
                  "matches": [asdict(m) for m in matches], "diagnostics": diagnostics,
                  "statistics": config.get("_matching_statistics", {}), "preview_only": True}
        (directory / f"{path.stem}_{label}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        overlay = PoseOverlay(directory, ROOT / "pick_place_config.json")
        overlay.camera = config["camera"]
        overlay.model_vertices = np.asarray(mesh.vertices)
        overlay.model_triangles = np.asarray(mesh.triangles)
        overlay.matches = result["matches"]
        color = data["color"].copy()
        depth = render_depth(data["depth"], 150, 600, cv2.COLORMAP_JET)
        overlay._axes(color)
        overlay._axes(depth, "depth_projection")
        canvas = cv2.hconcat([cv2.resize(color, (800,600)), cv2.resize(depth, (800,600))])
        cv2.putText(canvas, f"{strategy} | {len(matches)} matches | {duration:.2f}s", (12,28),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
        cv2.imwrite(str(directory / f"{path.stem}_{label}.png"), canvas)
        row = {"frame": path.name, "strategy": strategy, "matches": len(matches),
               "duration_s": round(duration, 3), "clusters": len(clusters),
               "reasons": dict(Counter(d.get("reason", "") for d in diagnostics))}
        summary.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    (directory / f"summary_{label}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--capture", type=int, default=0)
    parser.add_argument("--strategy", choices=("legacy", "clutter"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--track", action="store_true", help="Revalidate previous-frame poses during replay")
    args = parser.parse_args()
    if args.capture:
        capture(args.directory, args.capture)
    if args.strategy:
        evaluate(args.directory, args.strategy, args.limit, args.track)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
