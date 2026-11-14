"""Read-only capture and target-free hand-eye calibration from a static scene.

No function in this module moves the robot or writes a gripper command.
Solutions are saved for review; they do not modify the live robot config.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import pick_place_3d as vision

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "calibration_data" / "markerless"
MINIMUM_SAMPLES = 8


def read_manifest(directory: Path) -> dict:
    path = directory / "manifest.json"
    if not path.exists():
        return {"schema_version": 1, "samples": []}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported calibration dataset version")
    return manifest


def pose_difference(first, second) -> tuple[float, float]:
    import numpy as np
    from scipy.spatial.transform import Rotation

    left, right = vision.pose_to_matrix(first), vision.pose_to_matrix(second)
    return (float(np.linalg.norm(left[:3, 3] - right[:3, 3])),
            float(np.degrees(Rotation.from_matrix(left[:3, :3].T @ right[:3, :3]).magnitude())))


def check_scene_geometry(cloud) -> None:
    import numpy as np

    points = np.asarray(cloud.points)
    if len(points) < 1000 or not np.isfinite(points).all():
        raise ValueError("场景点太少或包含无效点，请保持料框、边角和工件在视野内。")
    values = np.linalg.eigvalsh(np.cov(points.T))
    if values[0] / max(values[-1], 1e-9) < 0.002:
        raise ValueError("画面几乎只有一个平面，无法可靠配准；请包含料框边角和立体物体。")


def capture_sample(config_path: Path, directory: Path) -> dict:
    import open3d as o3d

    config = vision.load_config(config_path)
    if config["camera"].get("mount") != "eye_in_hand":
        raise ValueError("无板手眼标定仅用于机械臂上的相机。")
    manifest = read_manifest(directory)
    before = vision.read_observation_pose()
    for sample in manifest["samples"]:
        previous = sample["observation"]
        if (before["tool_id"], before["user_id"]) != (previous["tool_id"], previous["user_id"]):
            raise ValueError("工具号或用户坐标系已变化，请为新的安装关系创建新采集目录。")
        translation, angle = pose_difference(before["tcp_pose"], previous["tcp_pose"])
        if translation < 5 and angle < 3:
            raise ValueError("这个姿态已采过。请先手动改变拍摄角度，停稳后再采集。")
    cloud = vision.crop_cloud(vision.capture_point_cloud(config), config["camera"]["roi"])
    observation = config["_observation"]
    vision.verify_observation_unchanged(before, observation)
    check_scene_geometry(cloud)
    if manifest.get("roi") is not None and manifest["roi"] != config["camera"]["roi"]:
        raise ValueError("同一批标定采集不能改变相机 ROI。")
    directory.mkdir(parents=True, exist_ok=True)
    name = f"sample_{len(manifest['samples']) + 1:03d}.ply"
    path = directory / name
    if path.exists():
        raise ValueError(f"采集文件已存在，请检查 manifest.json：{path}")
    if not o3d.io.write_point_cloud(str(path), cloud):
        raise OSError(f"cannot save point cloud: {path}")
    sample = {"file": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "observation": observation, "point_count": len(cloud.points)}
    manifest["samples"].append(sample)
    manifest["roi"] = config["camera"]["roi"]
    manifest["robot_host"] = __import__("arm_control").ROBOT_IP
    manifest["instructions"] = "Keep the entire scene static; change camera viewing angles around more than one rotation axis."
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"sample_count": len(manifest["samples"]), "minimum_samples": MINIMUM_SAMPLES,
            "file": str(path), "robot_motion_sent": False,
            "message": "本组采集完成。保持料框和工件不动，手动换角度、停稳后再采一组。"}


def _fit_pairs(robot_motions, camera_motions):
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    ra = np.array([Rotation.from_matrix(value[:3, :3]).as_rotvec() for value in robot_motions])
    rb = np.array([Rotation.from_matrix(value[:3, :3]).as_rotvec() for value in camera_motions])
    for values in (ra, rb):
        singular = np.linalg.svd(values, compute_uv=False)
        if singular[0] < np.deg2rad(8) or singular[1] / max(singular[0], 1e-12) < 0.15:
            raise ValueError("姿态变化不足：不能只平移或只绕同一个方向旋转，需要绕两个不同方向改变角度。")
    if np.max(np.abs(np.linalg.norm(ra, axis=1) - np.linalg.norm(rb, axis=1))) > np.deg2rad(2):
        raise ValueError("点云估计的转角与机器人转角不一致，场景可能移动或配准错误。")
    rotation, _ = Rotation.align_vectors(ra, rb)
    rx = rotation.as_matrix()
    left = np.vstack([value[:3, :3] - np.eye(3) for value in robot_motions])
    right = np.concatenate([rx @ b[:3, 3] - a[:3, 3] for a, b in zip(robot_motions, camera_motions)])
    if np.linalg.matrix_rank(left, tol=1e-6) < 3 or np.linalg.cond(left) > 50:
        raise ValueError("平移参数不可观测，请增加不同轴的旋转姿态。")
    tx = np.linalg.lstsq(left, right, rcond=None)[0]

    def unpack(parameters):
        result = np.eye(4)
        result[:3, :3] = Rotation.from_rotvec(parameters[:3]).as_matrix()
        result[:3, 3] = parameters[3:]
        return result

    def residual(parameters):
        transform = unpack(parameters)
        result = []
        for a, b in zip(robot_motions, camera_motions):
            error = np.linalg.inv(a @ transform) @ (transform @ b)
            result.extend(error[:3, 3])
            result.extend(Rotation.from_matrix(error[:3, :3]).as_rotvec() * 100.0)
        return result

    fit = least_squares(residual, np.r_[rotation.as_rotvec(), tx], loss="soft_l1", f_scale=1.0,
                        max_nfev=500)
    if not fit.success:
        raise ValueError("手眼求解未收敛，请补采更多姿态。")
    return vision.matrix(unpack(fit.x), "candidate tcp_T_camera")


def _pair_errors(transform, robot_motions, camera_motions):
    import numpy as np
    from scipy.spatial.transform import Rotation

    position, angle = [], []
    for a, b in zip(robot_motions, camera_motions):
        error = np.linalg.inv(a @ transform) @ (transform @ b)
        position.append(float(np.linalg.norm(error[:3, 3])))
        angle.append(float(np.degrees(Rotation.from_matrix(error[:3, :3]).magnitude())))
    return {"rms_translation_mm": float(np.sqrt(np.mean(np.square(position)))),
            "max_translation_mm": max(position), "rms_rotation_deg": float(np.sqrt(np.mean(np.square(angle)))),
            "max_rotation_deg": max(angle)}


def solve_motion_pairs(robot_motions, camera_motions) -> dict:
    """Solve A X = X B; B maps each observed camera into the reference camera."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    if len(robot_motions) != len(camera_motions) or len(robot_motions) < MINIMUM_SAMPLES - 1:
        raise ValueError(f"至少需要 {MINIMUM_SAMPLES} 组不同姿态的采集。")
    a = [vision.matrix(item, "robot motion") for item in robot_motions]
    b = [vision.matrix(item, "camera motion") for item in camera_motions]
    held_out = {len(a) // 3, 2 * len(a) // 3}
    train = [index for index in range(len(a)) if index not in held_out]
    partial = _fit_pairs([a[i] for i in train], [b[i] for i in train])
    holdout = _pair_errors(partial, [a[i] for i in held_out], [b[i] for i in held_out])
    transform = _fit_pairs(a, b)
    fit = _pair_errors(transform, a, b)
    shift = np.linalg.inv(partial) @ transform
    stability = {"translation_mm": float(np.linalg.norm(shift[:3, 3])),
                 "rotation_deg": float(np.degrees(Rotation.from_matrix(shift[:3, :3]).magnitude()))}
    accepted = (fit["rms_translation_mm"] <= 1.5 and fit["max_translation_mm"] <= 3.0
                and fit["max_rotation_deg"] <= 1.0 and holdout["max_translation_mm"] <= 3.0
                and holdout["max_rotation_deg"] <= 1.0 and stability["translation_mm"] <= 3.0
                and stability["rotation_deg"] <= 1.0)
    return {"accepted": bool(accepted), "candidate_tcp_T_camera": transform.tolist(),
            "fit": fit, "held_out_validation": holdout, "solution_stability": stability,
            "applied_to_robot_config": False}


def register_scene_pair(reference, current) -> tuple:
    import numpy as np
    import open3d as o3d

    check_scene_geometry(reference)
    check_scene_geometry(current)
    target, target_features = vision._features(reference, 2.5)
    source, source_features = vision._features(current, 2.5)
    o3d.utility.random.seed(17)
    registration = o3d.pipelines.registration
    coarse = registration.registration_ransac_based_on_feature_matching(
        source, target, source_features, target_features, True, 7.5,
        registration.TransformationEstimationPointToPoint(False), 4,
        [registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         registration.CorrespondenceCheckerBasedOnDistance(7.5)],
        registration.RANSACConvergenceCriteria(100000, 0.999))
    fine = registration.registration_icp(
        source, target, 3.0, coarse.transformation,
        registration.TransformationEstimationPointToPlane(),
        registration.ICPConvergenceCriteria(max_iteration=100))
    reverse = registration.evaluate_registration(target, source, 3.0, np.linalg.inv(fine.transformation))
    metrics = {"fitness": float(fine.fitness), "reverse_fitness": float(reverse.fitness),
               "rmse_mm": float(fine.inlier_rmse)}
    if fine.fitness < 0.65 or reverse.fitness < 0.65 or fine.inlier_rmse > 1.5:
        raise ValueError(f"相邻采集重叠不足或场景发生变化：{metrics}；请缩小单次视角变化并保持场景静止。")
    return vision.matrix(fine.transformation, "reference_camera_T_current_camera"), metrics


def solve_dataset(directory: Path) -> dict:
    import numpy as np
    import open3d as o3d

    manifest = read_manifest(directory)
    samples = manifest["samples"]
    if len(samples) < MINIMUM_SAMPLES:
        raise ValueError(f"目前只有 {len(samples)} 组，至少需要 {MINIMUM_SAMPLES} 组。工件不需要摆整齐，只需在采集期间保持不动。")
    frames = {(sample["observation"]["tool_id"], sample["observation"]["user_id"]) for sample in samples}
    if len(frames) != 1 or next(iter(frames))[1] != 0:
        raise ValueError("采集使用了不同工具/坐标系，不能合并求解。")
    poses, clouds = [], []
    for sample in samples:
        path = directory / sample["file"]
        if path.resolve().parent != directory.resolve() or hashlib.sha256(path.read_bytes()).hexdigest() != sample["sha256"]:
            raise ValueError("采集文件路径或校验值不一致。")
        poses.append(vision.pose_to_matrix(sample["observation"]["tcp_pose"]))
        clouds.append(o3d.io.read_point_cloud(str(path)))
    robot_motions, camera_motions, registrations = [], [], []
    for index in range(1, len(samples)):
        print(f"正在匹配点云 {index + 1}/{len(samples)}…", flush=True)
        transform, metrics = register_scene_pair(clouds[0], clouds[index])
        robot_motions.append(np.linalg.inv(poses[0]) @ poses[index])
        camera_motions.append(transform)
        registrations.append(metrics)
    report = solve_motion_pairs(robot_motions, camera_motions)
    report.update({"created_unix": time.time(), "sample_count": len(samples),
                   "tool_id": samples[0]["observation"]["tool_id"], "user_id": 0,
                   "registration_metrics": registrations,
                   "manifest_sha256": hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest(),
                   "method": "static_scene_pointcloud_AX_XB",
                   "message": "结果已保存供复核，尚未写入真机配置；夹持中心偏移仍需单独确定。"})
    (directory / "handeye_candidate.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not report["accepted"]:
        raise ValueError("求解误差或独立验证误差超限，结果未采用。详见 handeye_candidate.json。")
    return report


def main() -> int:
    runtime = ROOT / ".camera-runtime" / "python.exe"
    if sys.platform == "win32" and sys.version_info[:2] != (3, 10) and runtime.is_file():
        return subprocess.call([str(runtime), str(Path(__file__).resolve()), *sys.argv[1:]])
    parser = argparse.ArgumentParser(description="不使用标定板的静止场景手眼标定；不会移动机械臂")
    parser.add_argument("action", choices=("capture", "solve", "status"))
    parser.add_argument("--config", type=Path, default=vision.DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    try:
        if args.action == "capture":
            result = capture_sample(args.config, args.dataset)
        elif args.action == "solve":
            result = solve_dataset(args.dataset)
        else:
            result = {"sample_count": len(read_manifest(args.dataset)["samples"]), "minimum_samples": MINIMUM_SAMPLES}
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as exc:
        print(f"未完成：{exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
