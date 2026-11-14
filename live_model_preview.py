"""Continuous CAD matching of preview XYZ frames. No robot/gripper I/O."""
from __future__ import annotations

from dataclasses import asdict
import json
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty, Full
import time

import numpy as np


def match_frame(points, config, model_cloud, previous_matches=None):
    import open3d as o3d
    import pick_place_3d as vision

    points = np.asarray(points, dtype=float).reshape(-1, 3).copy()
    if config["camera"].get("point_cloud_frame") == "opencv_optical":
        points[:, 1] *= -1
    points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 0)]
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    scene = vision.crop_cloud(cloud, config["camera"]["roi"])
    scene = scene.voxel_down_sample(float(config["matching"]["scene_voxel_mm"]))
    if len(scene.points) < 100:
        return {"matches": [], "scene_points": len(scene.points), "clusters": 0,
                "diagnostics": [{"reason": "Too few depth points inside ROI"}]}
    objects, _plane, _support = vision.remove_support_plane(scene, config)
    objects = vision.exclude_camera_geometry(objects, config)
    clusters = vision.cluster_objects(objects, config)
    # Preview results deliberately have no robot observation and are never
    # written to the executable pick_results/matches.json or plan.json.
    config.pop("_observation", None)
    matches = vision.find_matches(model_cloud, scene, clusters, config, previous_matches=previous_matches)
    return {"matches": [asdict(item) for item in matches], "scene_points": len(scene.points),
            "clusters": len(clusters), "diagnostics": config.get("_matching_diagnostics", []),
            "statistics": config.get("_matching_statistics", {})}


def matching_worker(config_path, requests, responses):
    # Bound OpenMP here, before importing Open3D, to keep the preview responsive.
    os.environ["OMP_NUM_THREADS"] = "2"
    from camera_capture import configure_logging
    configure_logging("live_model_preview.log")
    import logging
    import open3d as o3d
    import pick_place_3d as vision

    try:
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        config = vision.load_config(Path(config_path))
        mesh = vision.load_fcstd_mesh(Path(config_path).parent / config["model"]["path"])
        o3d.utility.random.seed(7)
        model = mesh.sample_points_uniformly(int(config["matching"]["model_sample_points"]))
        previous = []
        while True:
            request = requests.get()
            if request is None:
                return
            started = time.monotonic()
            result = {"sequence": request["sequence"], "captured_unix": request["captured_unix"],
                      "preview_only": True}
            try:
                result.update(match_frame(request["points"], config, model, previous))
            except Exception as exc:
                logging.exception("Live CAD frame failed")
                result.update(matches=[], error=str(exc), diagnostics=[])
            previous = result.get("matches", [])
            result.update(completed_unix=time.time(), duration_s=time.monotonic() - started)
            responses.put(result)
    except Exception as exc:
        logging.exception("Live CAD matcher stopped")
        responses.put({"matches": [], "error": str(exc), "preview_only": True})


class LiveModelMatcher:
    def __init__(self, config_path, output_dir):
        self.context = mp.get_context("spawn")
        self.requests = self.context.Queue(maxsize=1)
        self.responses = self.context.Queue(maxsize=1)
        self.process = self.context.Process(target=matching_worker,
            args=(str(config_path), self.requests, self.responses), name="live-cad-matching", daemon=True)
        self.process.start()
        self.output_dir = Path(output_dir)
        self.busy = False
        self.last_submit = 0.0
        self.result = None
        self.source_images = None
        self.pending_images = None

    def submit(self, points, stamp, sequence, color, depth):
        if self.busy or not self.process.is_alive() or time.monotonic() - self.last_submit < 0.5:
            return False
        try:
            self.requests.put_nowait({"points": points, "captured_unix": stamp, "sequence": sequence})
        except Full:
            return False
        self.pending_images = (color.copy(), depth.copy())
        self.busy = True
        self.last_submit = time.monotonic()
        return True

    def poll(self):
        try:
            result = self.responses.get_nowait()
        except Empty:
            if self.process.is_alive():
                return None
            if self.result and self.result.get("error") == "CAD matching process exited":
                return None
            result = {"matches": [], "error": "CAD matching process exited", "preview_only": True}
        self.busy = False
        self.result = result
        self.source_images = self.pending_images
        self.output_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.output_dir / "status.tmp"
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.output_dir / "status.json")
        return result

    def close(self):
        if self.process.is_alive():
            try:
                self.requests.put_nowait(None)
            except Full:
                pass
            self.process.join(0.3)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(1)
        for channel in (self.requests, self.responses):
            channel.cancel_join_thread()
            channel.close()
        self.process.close()
