"""Project CAD contours for saved scans or continuously measured preview poses."""
from __future__ import annotations

import json
import time
import textwrap
from pathlib import Path

import numpy as np


class PoseOverlay:
    def __init__(self, results: Path, config: Path, *, live=False):
        self.results = results
        self.config = config
        self.next_read = 0.0
        self.matches = []
        self.camera = {}
        self.stamp = 0.0
        self.error = ""
        self.model_vertices = None
        self.model_triangles = None
        self.model_path = None
        self.live = live
        self.live_result = None
        self.source_depth = None
        self.outline_count = 0
        self.contour_cache = {}

    def set_live_result(self, result, source_depth=None):
        self.contour_cache.clear()
        self.live_result = result
        self.source_depth = source_depth
        self.matches = result.get("matches", [])
        self.stamp = result.get("captured_unix", 0)
        self.error = result.get("error", "")

    def _load_assets(self):
        config = json.loads(self.config.read_text(encoding="utf-8-sig"))
        self.camera = config["camera"]
        model_path = self.config.parent / config["model"]["path"]
        if self.model_path != model_path:
            from pick_place_3d import load_fcstd_mesh
            mesh = load_fcstd_mesh(model_path)
            self.model_vertices = np.asarray(mesh.vertices).copy()
            self.model_triangles = np.asarray(mesh.triangles).copy()
            self.model_path = model_path
            self.contour_cache.clear()

    def _live_overlay_valid(self, current_depth, matches=None):
        """Hide an old fit when the analyzed region changes in the live depth."""
        import cv2

        # A slow registration is not stale on arrival. Give its completed
        # result a short display window, still checking the live target depth.
        duration = max(0.0, float((self.live_result or {}).get("duration_s", 0)))
        if not self.stamp or time.time() - self.stamp > duration + 15:
            return False
        if current_depth is None or self.source_depth is None:
            return False
        if current_depth.shape != self.source_depth.shape:
            return False
        projection = self.camera["depth_projection"]
        if projection["image_size"] != [current_depth.shape[1], current_depth.shape[0]]:
            return False
        intrinsic = np.asarray(projection["matrix"], dtype=float)
        extrinsic = np.asarray(projection["color_T_camera"], dtype=float)
        distortion = np.asarray(projection.get("distortion", [0] * 5), dtype=float)
        for match in self.matches if matches is None else matches:
            transform = extrinsic @ np.asarray(match["camera_T_model"])
            vertices = self.model_vertices @ transform[:3, :3].T + transform[:3, 3]
            if np.any(vertices[:, 2] <= 0):
                return False
            pixels, _ = cv2.projectPoints(vertices, np.zeros(3), np.zeros(3), intrinsic, distortion)
            pixels = pixels.reshape(-1, 2)
            if not np.isfinite(pixels).all():
                return False
            lower = np.maximum(np.floor(pixels.min(axis=0)).astype(int) - 3, 0)
            upper = np.minimum(np.ceil(pixels.max(axis=0)).astype(int) + 4,
                               [current_depth.shape[1], current_depth.shape[0]])
            x0, y0 = lower; x1, y1 = upper
            before = self.source_depth[y0:y1:2, x0:x1:2].astype(float)
            after = current_depth[y0:y1:2, x0:x1:2].astype(float)
            valid = (before > 0) & (after > 0)
            if np.count_nonzero(valid) < 20:
                return False
            if np.mean((before > 0) != (after > 0)) > 0.20:
                return False
            if np.mean(np.abs(before[valid] - after[valid]) > 5) > 0.25:
                return False
        return True

    def draw(self, image=None, height=480, depth_image=None, *, current_depth=None, camera_live=True):
        import cv2

        self.outline_count = 0
        if time.monotonic() >= self.next_read:
            self.next_read = time.monotonic() + 1.0
            try:
                path = self.results / "matches.json"
                matches = self.matches if self.live else json.loads(path.read_text(encoding="utf-8"))
                for match in matches:
                    transform = np.asarray(match["camera_T_model"], dtype=float)
                    if transform.shape != (4, 4) or not np.isfinite(transform).all():
                        raise ValueError("Invalid pose matrix")
                self._load_assets()
                self.matches = matches
                if not self.live:
                    self.stamp = path.stat().st_mtime
                    self.error = ""
            except (OSError, ValueError, KeyError, TypeError):
                self.matches = []
                self.error = "Pose unavailable - run CAD scan first"

        lines = ["MODEL POSE / CAD", "Scan snapshot (not live tracking)", ""]
        overlay_valid = True
        visible_matches = self.matches
        if self.live:
            lines = ["LIVE CAD / READ ONLY", "Continuous RGB + depth matching", ""]
            if self.live_result is None:
                lines.extend(["Waiting for first native XYZ fit...", "Robot / gripper: no commands"])
            elif self.error:
                lines.extend(["Matching unavailable", *textwrap.wrap(self.error, 42)[:3]])
            else:
                result = self.live_result
                lines.extend([f"Validated matches: {len(self.matches)}",
                              f"Input age: {max(0, time.time() - self.stamp):.1f}s",
                              f"Fit time: {result.get('duration_s', 0):.2f}s",
                              f"Scene: {result.get('scene_points', 0)} points / {result.get('clusters', 0)} clusters"])
                statistics = result.get("statistics", {})
                if statistics:
                    lines[-1] = (f"Segmented clusters: {statistics.get('segmented_clusters', 0)}")
                    lines.extend([f"Regions: {statistics.get('candidate_regions', 0)} / fits: {statistics.get('registered_candidates', 0)}",
                                  f"Previous poses refitted: {statistics.get('previous_pose_hints', 0)}"])
                if not self.matches:
                    lines.extend(["", "No match passed geometry checks"])
                    reasons = list(dict.fromkeys(d.get("reason", "") for d in result.get("diagnostics", [])))
                    for reason in reasons[:2]:
                        lines.extend(textwrap.wrap(reason, 42)[:2])
            try:
                visible_matches = ([match for match in self.matches
                                    if self._live_overlay_valid(current_depth, [match])]
                                   if camera_live else [])
                overlay_valid = bool(visible_matches)
            except (ValueError, KeyError, TypeError, cv2.error):
                overlay_valid = False
            if self.matches and not overlay_valid:
                lines.extend(["", "Overlay hidden: scene changed / stale", "Waiting for a new validated fit"])
            elif self.matches and len(visible_matches) < len(self.matches):
                lines.append(f"Unchanged contours: {len(visible_matches)}/{len(self.matches)}")
        elif self.error:
            lines.extend(["Pose unavailable", "Run CAD scan first"])
        else:
            lines.extend([f"CAD candidates: {len(self.matches)}",
                          f"Result age: {max(0, time.time() - self.stamp):.0f}s"])
            if not self.matches:
                lines.extend(["", "No match passed geometry checks"])
        for index, match in enumerate(self.matches[:3], 1):
            transform = np.asarray(match["camera_T_model"], dtype=float)
            xyz = transform[:3, 3]
            # XYZ Euler convention used by pick_place_3d.matrix_to_pose.
            from pick_place_3d import matrix_to_pose
            rpy = matrix_to_pose(transform)[3:]
            lines.extend(["", f"#{index}  Match score: {match['score']:.3f}", "XYZ mm: " + ", ".join(f"{v:.1f}" for v in xyz)])
            lines.append("   RPY deg: " + ", ".join(f"{v:.1f}" for v in rpy))
            if self.live:
                lines.append(f"   Fit RMSE: {match['rmse_mm']:.2f} mm")
        if self.matches and overlay_valid and image is not None:
            try:
                self.outline_count = self._axes(image, matches=visible_matches)
                if depth_image is not None:
                    self.outline_count = max(self.outline_count, self._axes(depth_image, "depth_projection", matches=visible_matches))
            except (ValueError, KeyError, TypeError, cv2.error):
                lines.extend(["", "Axes: calibration required", "Set camera.color_projection"])
        panel = np.full((height, 380, 3), (32, 28, 24), dtype=np.uint8)
        cv2.line(panel, (0, 0), (0, height - 1), (85, 85, 85), 2)
        spacing = min(24, max(14, (height - 40) // max(1, len(lines))))
        for index, line in enumerate(lines):
            cv2.putText(panel, line, (16, 28 + index * spacing), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (0, 220, 255) if index == 0 else (225, 225, 225), 1, cv2.LINE_AA)
        return panel

    def _axes(self, image, projection_key="color_projection", matches=None):
        import cv2

        projection = self.camera[projection_key]
        if list(projection["image_size"]) != [image.shape[1], image.shape[0]]:
            raise ValueError("Calibration image size mismatch")
        intrinsic = np.asarray(projection["matrix"], dtype=float)
        extrinsic = np.asarray(projection["color_T_camera"], dtype=float)
        distortion = np.asarray(projection.get("distortion", [0, 0, 0, 0, 0]), dtype=float)
        if (intrinsic.shape != (3, 3) or extrinsic.shape != (4, 4)
                or not np.isfinite(intrinsic).all() or not np.isfinite(extrinsic).all()
                or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0):
            raise ValueError("Invalid camera projection")
        axes = np.array([[0, 0, 0], [30, 0, 0], [0, 30, 0], [0, 0, 30]], dtype=float)
        outline_count = 0
        thickness = max(2, int(round(max(image.shape[:2]) / 400)))
        for index, match in enumerate(self.matches, 1):
            if matches is not None and not any(match is item for item in matches):
                continue
            transform = extrinsic @ np.asarray(match["camera_T_model"])
            if self.model_vertices is not None:
                key = (projection_key,image.shape[:2],transform.tobytes(),intrinsic.tobytes(),distortion.tobytes())
                contours = self.contour_cache.get(key)
                if contours is None:
                    contours = []
                    model = self.model_vertices @ transform[:3, :3].T + transform[:3, 3]
                    if self.model_triangles is not None and len(model) >= 3 and np.all(model[:, 2] > 0):
                        outline, _ = cv2.projectPoints(model, np.zeros(3), np.zeros(3), intrinsic, distortion)
                        outline = outline.reshape(-1, 2)
                        if np.isfinite(outline).all() and np.max(np.abs(outline)) < 1e6:
                            # Rasterize real CAD triangles once per pose update.
                            mask = np.zeros(image.shape[:2], dtype=np.uint8)
                            pixels = np.rint(outline).astype(np.int32)
                            for triangle in self.model_triangles:
                                cv2.fillConvexPoly(mask, pixels[triangle], 255)
                            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    if len(self.contour_cache) > 128:
                        self.contour_cache.clear()
                    self.contour_cache[key] = contours
                if contours:
                    cv2.drawContours(image, contours, -1, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                    cv2.drawContours(image, contours, -1, (0, 255, 255), thickness, cv2.LINE_AA)
                    outline_count += 1
            points = axes @ transform[:3, :3].T + transform[:3, 3]
            if np.any(points[:, 2] <= 0):
                continue
            pixels, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), intrinsic, distortion)
            pixels = pixels.reshape(-1, 2)
            if not np.isfinite(pixels).all() or np.max(np.abs(pixels)) > 1e6:
                continue
            pixels = np.rint(pixels).astype(int)
            origin = tuple(pixels[0])
            for endpoint, color, label in zip(pixels[1:], [(0, 0, 255), (0, 255, 0), (255, 0, 0)], "XYZ"):
                cv2.line(image, origin, tuple(endpoint), color, 2, cv2.LINE_AA)
                cv2.putText(image, label, tuple(endpoint), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            label = f"#{index} CAD" + (f" {match['score']:.2f}" if "score" in match else "")
            cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        return outline_count
