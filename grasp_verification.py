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
    from camera_cleanup import CameraResource
    sensor = InuSensor()
    sensor_lifecycle = CameraResource(sensor)
    streams = []
    color = depth = None
    try:
        sensor_lifecycle.init()
        sensor_lifecycle.start()
        color_stream = sensor.create_stream(StreamType.GENERAL_CAMERA)
        color_lifecycle = CameraResource(color_stream)
        streams.append(color_lifecycle)
        color_lifecycle.init(ImageOutputFormat.BGRA, ImagePostProcessing.NONE_PROCESSING)
        color_lifecycle.start()
        depth_stream = sensor.create_stream(StreamType.DEPTH)
        depth_lifecycle = CameraResource(depth_stream)
        streams.append(depth_lifecycle)
        depth_lifecycle.init(DepthOutputFormat.DEPTH, DepthPostProcessing.DEFAULT)
        depth_lifecycle.start()
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
        for lifecycle in reversed(streams):
            lifecycle.close()
        sensor_lifecycle.close()
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


def capture_verification_isolated(output_dir: Path, sdk_path=None, timeout_seconds=45) -> dict:
    """Bound a native SDK stall and isolate RGB/depth lifecycle from XYZ scans."""
    import subprocess
    import sys
    output_dir = Path(output_dir).resolve()
    started = time.time()
    command = [sys.executable, "-X", "utf8", "-c",
               "import sys; from pathlib import Path; from grasp_verification import _capture_verification_once; "
               "_capture_verification_once(Path(sys.argv[1]), sys.argv[2] or None)",
               str(output_dir), str(sdk_path or "")]
    try:
        subprocess.run(command, cwd=str(Path(__file__).resolve().parent), check=True,
                       timeout=timeout_seconds, stdin=subprocess.DEVNULL,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("RGB/depth camera exceeded its capture deadline") from exc
    result = json.loads((output_dir / "grasp_verify_latest.json").read_text(encoding="utf-8"))
    if result["created_unix"] < started or not Path(result["color_path"]).is_file():
        raise RuntimeError("Isolated camera returned an old or missing image")
    return result


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
