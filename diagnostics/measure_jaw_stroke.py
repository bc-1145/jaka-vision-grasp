"""Estimate the visible jaw contact frame from two known empty-jaw strokes.

This is an image/stroke estimate, not an independent tool-center calibration.
The arm and camera must stay rigidly mounted between the two images.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import pick_place_3d as vision


def measure(open_image, closed_image, total_stroke_mm, config):
    first = cv2.imread(str(open_image), cv2.IMREAD_GRAYSCALE)
    second = cv2.imread(str(closed_image), cv2.IMREAD_GRAYSCALE)
    if first is None or second is None or first.shape != second.shape:
        raise ValueError("Jaw calibration images are missing or have different sizes")
    projection = config["camera"]["color_projection"]
    intrinsic = np.asarray(projection["matrix"], float)
    distortion = np.asarray(projection["distortion"], float)
    color_T_tcp = np.asarray(projection["color_T_camera"]) @ np.linalg.inv(
        np.asarray(config["camera"]["tcp_T_camera"]))
    mask = np.zeros_like(first)
    mask[800:870, 830:920] = 255
    mask[800:870, 1040:1115] = 255
    corners = cv2.goodFeaturesToTrack(first, 200, .03, 5, mask=mask)
    if corners is None:
        raise RuntimeError("No jaw features")
    tracked, status, error = cv2.calcOpticalFlowPyrLK(
        first, second, corners, None, winSize=(25, 25), maxLevel=3)
    backward, back_status, _ = cv2.calcOpticalFlowPyrLK(
        second, first, tracked, None, winSize=(25, 25), maxLevel=3)
    before, after = corners.reshape(-1, 2), tracked.reshape(-1, 2)
    keep = (status.ravel().astype(bool) & back_status.ravel().astype(bool)
            & (error.ravel() < 12)
            & (np.linalg.norm(before - backward.reshape(-1, 2), axis=1) < .6)
            & (np.abs(after[:, 1] - before[:, 1]) < 2)
            & (np.abs(after[:, 0] - before[:, 0]) > total_stroke_mm * 1.8))
    before, after = before[keep], after[keep]
    if len(before) < 15:
        raise RuntimeError("Insufficient consistent moving-jaw features")
    rays = [np.c_[cv2.undistortPoints(p.reshape(-1, 1, 2), intrinsic, distortion)
                  .reshape(-1, 2), np.ones(len(p))] for p in (before, after)]
    points, residuals = [], []
    for pixel, ray0, ray1 in zip(before, *rays):
        sign = -1 if pixel[0] < 950 else 1
        displacement = color_T_tcp[:3, :3] @ [0, sign * total_stroke_mm / 2, 0]
        system = np.column_stack((-ray0, ray1))
        depths = np.linalg.lstsq(system, displacement, rcond=None)[0]
        if np.min(depths) <= 0:
            raise RuntimeError("Jaw motion has invalid triangulated depth")
        points.append(vision.transform_points([depths[1] * ray1], np.linalg.inv(color_T_tcp))[0])
        residuals.append(np.linalg.norm(system @ depths - displacement))
    return dict(open_pixels=before, closed_pixels=after,
                closed_tcp_points=np.asarray(points), residual_mm=np.asarray(residuals))
