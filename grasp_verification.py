"""Capture a close-range color/depth image for human grasp verification."""
from __future__ import annotations

import json
import time
from pathlib import Path


def _capture_verification_once(output_dir: Path, sdk_path=None) -> dict:
    import cv2
    import numpy as np
    from color_depth_viewer import (
        find_sdk_dir, frame_to_color, frame_to_depth, prepare_runtime, render_depth,
    )

    prepare_runtime(find_sdk_dir(sdk_path))
    from api.depth_stream import DepthOutputFormat, DepthPostProcessing
    from api.image_stream import ImageOutputFormat, ImagePostProcessing
    from api.inu_sensor import InuSensor
    from api.shared import StreamType

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sensor = InuSensor()
    streams = []
    started = []
    color = depth = None
    try:
        sensor.init()
        sensor.start()
        color_stream = sensor.create_stream(StreamType.GENERAL_CAMERA)
        streams.append(color_stream)
        color_stream.init(ImageOutputFormat.BGRA, ImagePostProcessing.NONE_PROCESSING)
        color_stream.start(); started.append(color_stream)
        depth_stream = sensor.create_stream(StreamType.DEPTH)
        streams.append(depth_stream)
        depth_stream.init(DepthOutputFormat.DEPTH, DepthPostProcessing.DEFAULT)
        depth_stream.start(); started.append(depth_stream)
        valid = 0
        for _ in range(50):
            cframe = color_stream.frame
            dframe = depth_stream.frame
            current_color = frame_to_color(cframe) if cframe is not None else None
            current_depth = frame_to_depth(dframe) if dframe is not None else None
            if (current_color is None or current_depth is None or
                    not np.any(current_depth > 0)):
                continue
            valid += 1
            if valid >= 4:
                color = current_color.copy()
                depth = current_depth.copy()
                break
    finally:
        for stream in reversed(started):
            try:
                stream.stop()
            except Exception:
                pass
        for stream in reversed(streams):
            try:
                stream.terminate()
            except Exception:
                pass
        for operation in (sensor.stop, sensor.terminate):
            try:
                operation()
            except Exception:
                pass
    if color is None or depth is None:
        raise RuntimeError("相机没有提供抓取验证图像")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    color_path = output_dir / f"grasp_verify_color_{stamp}.png"
    depth_path = output_dir / f"grasp_verify_depth_{stamp}.png"
    view_path = output_dir / f"grasp_verify_view_{stamp}.png"
    raw_depth_path = output_dir / f"grasp_verify_depth_raw_{stamp}.png"
    valid_depth = depth[depth > 0]
    low = max(80, int(np.percentile(valid_depth, 1)))
    high = min(800, max(low + 100, int(np.percentile(valid_depth, 90))))
    depth_view = render_depth(depth, low, high, cv2.COLORMAP_TURBO)
    cv2.putText(depth_view, f"depth {low}-{high} mm", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    if depth_view.shape[:2] != color.shape[:2]:
        depth_view = cv2.resize(depth_view, (color.shape[1], color.shape[0]))
    combined = np.hstack((color, depth_view))
    cv2.imwrite(str(color_path), color)
    cv2.imwrite(str(depth_path), depth_view)
    cv2.imwrite(str(raw_depth_path), depth)
    cv2.imwrite(str(view_path), combined)
    report = {
        "created_unix": time.time(),
        "color_path": str(color_path.resolve()),
        "depth_path": str(depth_path.resolve()),
        "raw_depth_path": str(raw_depth_path.resolve()),
        "view_path": str(view_path.resolve()),
        "depth_min_mm": int(valid_depth.min()),
        "depth_median_mm": float(np.median(valid_depth)),
        "depth_max_mm": int(valid_depth.max()),
    }
    (output_dir / "grasp_verify_latest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def capture_verification(output_dir: Path, sdk_path=None) -> dict:
    """Capture with bounded retries for the NU4000 service timeout."""
    last_error = None
    for attempt in range(3):
        try:
            return _capture_verification_once(output_dir, sdk_path)
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1.0 + attempt)
    raise RuntimeError(f"抓取验证相机连续 3 次失败：{last_error}") from last_error


def compare_verification(before: dict, after: dict, output_path: Path) -> str:
    """Write before/after/absolute-difference color comparison."""
    import cv2
    import numpy as np

    first = cv2.imread(str(before["color_path"]))
    second = cv2.imread(str(after["color_path"]))
    if first is None or second is None or first.shape != second.shape:
        raise RuntimeError("抓取前后彩色图无法比较")
    difference = cv2.absdiff(first, second)
    gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    heat = cv2.applyColorMap(np.clip(gray * 5, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    for image, title in ((first, "BEFORE"), (second, "AFTER"), (heat, "DIFFERENCE")):
        cv2.putText(image, title, (18, 42), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (255, 255, 255), 3, cv2.LINE_AA)
    output_path = Path(output_path)
    cv2.imwrite(str(output_path), np.hstack((first, second, heat)))
    return str(output_path.resolve())
