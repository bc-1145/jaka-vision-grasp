# -*- coding: utf-8 -*-
"""Depth-camera CAD matching and guarded JAKA pick/place workflow.

The default commands are vision-only. Real motion requires --execute, an
explicit confirmation token, and a complete calibration in
pick_place_config.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "pick_place_config.json"
CAMERA_RUNTIME = PROJECT_DIR / ".camera-runtime" / "python.exe"


def _rerun_in_camera_runtime() -> int | None:
    if sys.platform == "win32" and sys.version_info[:2] != (3, 10) and CAMERA_RUNTIME.is_file():
        return subprocess.call([str(CAMERA_RUNTIME), str(Path(__file__).resolve()), *sys.argv[1:]])
    return None


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as stream:
        config = json.load(stream)
    if config.get("schema_version") != 1:
        raise ValueError("pick_place_config.json schema_version must be 1")
    return config


def save_config(path: Path, config: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(config, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def matrix(value: Any, name: str) -> "np.ndarray":
    import numpy as np

    result = np.asarray(value, dtype=float)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(result[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"{name} must be a homogeneous transform")
    rotation = result[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{name} must contain a proper rotation")
    return result


def read_observation_pose() -> dict[str, Any]:
    """Record the active TCP/base frames while the robot is stationary."""
    import arm_control
    from gripper_transport import send_command

    motion = send_command({"cmdName": "get_motion_state"}, ip=arm_control.ROBOT_IP)
    if (not motion.get("inpos") or motion.get("queue") or motion.get("active_queue")
            or motion.get("estop") or motion.get("collision") or motion.get("soft_limit")):
        raise RuntimeError("robot must be stationary and fault-free for eye-in-hand capture")
    status_error = None
    for _ in range(3):
        try:
            status = arm_control.get_status_snapshot(timeout=3.0)
            break
        except (OSError, TimeoutError, ValueError) as exc:
            status_error = exc
            time.sleep(0.1)
    else:
        raise RuntimeError(f"cannot read robot status stream: {status_error}")
    tool_id = status.get("current_tool_id")
    user_id = status.get("current_user_id")
    if tool_id is None or user_id != 0:
        raise RuntimeError("eye-in-hand capture requires a known tool and robot base user frame 0")
    tcp = send_command({"cmdName": "get_actual_tcp_pos"}, ip=arm_control.ROBOT_IP).get("position")
    if not isinstance(tcp, list) or len(tcp) != 6 or not all(math.isfinite(float(value)) for value in tcp):
        raise RuntimeError("robot returned an invalid TCP position")
    return {"created_unix": time.time(), "tcp_pose": tcp,
            "tool_id": tool_id, "user_id": user_id}


def verify_observation_unchanged(before: dict, after: dict) -> None:
    import numpy as np
    from scipy.spatial.transform import Rotation

    if (before["tool_id"], before["user_id"]) != (after["tool_id"], after["user_id"]):
        raise RuntimeError("active robot tool/user frame changed during camera acquisition")
    first, last = pose_to_matrix(before["tcp_pose"]), pose_to_matrix(after["tcp_pose"])
    translation = np.linalg.norm(first[:3, 3] - last[:3, 3])
    angle = math.degrees(Rotation.from_matrix(first[:3, :3].T @ last[:3, :3]).magnitude())
    if translation > 1.0 or angle > 0.5:
        raise RuntimeError("robot moved during eye-in-hand acquisition; scan cannot be used for motion")


def camera_transform(config: dict[str, Any], observation: dict | None = None):
    """Resolve the camera frame at capture time, never at execution time."""
    camera = config["camera"]
    mount = camera.get("mount", "fixed")
    if mount == "fixed":
        return matrix(camera.get("base_T_camera"), "camera.base_T_camera")
    if mount != "eye_in_hand":
        raise ValueError(f"unsupported camera mount: {mount}")
    if observation is None:
        raise RuntimeError("eye-in-hand match has no recorded acquisition pose; scan again")
    if camera.get("calibration_tool_id") is None or observation["tool_id"] != camera["calibration_tool_id"]:
        raise RuntimeError("camera hand-eye calibration does not match the active robot tool")
    if observation.get("user_id") != 0:
        raise RuntimeError("eye-in-hand observation must use robot base user frame 0")
    expected_frame = camera.get("point_cloud_frame")
    if expected_frame and observation.get("camera_frame") != expected_frame:
        raise RuntimeError("point-cloud coordinate frame changed; scan again before planning motion")
    return pose_to_matrix(observation["tcp_pose"]) @ matrix(camera.get("tcp_T_camera"), "camera.tcp_T_camera")


def pose_to_matrix(pose: list[float]) -> "np.ndarray":
    import numpy as np
    from scipy.spatial.transform import Rotation

    if len(pose) != 6:
        raise ValueError("robot pose must contain x, y, z, rx, ry, rz")
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("xyz", pose[3:], degrees=True).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def matrix_to_pose(transform: "np.ndarray") -> list[float]:
    from scipy.spatial.transform import Rotation

    angles = Rotation.from_matrix(transform[:3, :3]).as_euler("xyz", degrees=True)
    return [*map(float, transform[:3, 3]), *map(float, angles)]


def transform_points(points: "np.ndarray", transform: "np.ndarray") -> "np.ndarray":
    import numpy as np

    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def _final_brep_name(archive: zipfile.ZipFile) -> str:
    root = ElementTree.fromstring(archive.read("Document.xml"))
    body = next((obj for obj in root.findall(".//ObjectData/Object")
                 if obj.attrib.get("name") == "Body"), None)
    if body is None:
        raise ValueError("FCStd does not contain a PartDesign Body")
    shape = next((prop for prop in body.findall("./Properties/Property")
                  if prop.attrib.get("name") == "Shape"), None)
    if shape is None:
        raise ValueError("FCStd Body does not contain a final Shape")
    part = shape.find("Part")
    if part is None or not part.attrib.get("file"):
        raise ValueError("FCStd Body Shape does not reference a BREP")
    return part.attrib["file"]


def load_fcstd_mesh(path: Path, linear_deflection_mm: float = 0.25):
    """Read the final Body BREP from FCStd and tessellate it with OpenCascade."""
    import numpy as np
    import open3d as o3d
    from OCP.BRep import BRep_Builder, BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepTools import BRepTools
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS, TopoDS_Shape

    with zipfile.ZipFile(path) as archive, tempfile.TemporaryDirectory() as temp_dir:
        brep_name = _final_brep_name(archive)
        brep_path = Path(temp_dir) / "model.brp"
        brep_path.write_bytes(archive.read(brep_name))
        shape = TopoDS_Shape()
        if not BRepTools.Read_s(shape, str(brep_path), BRep_Builder()):
            raise ValueError(f"unable to read {brep_name} from {path}")
        mesher = BRepMesh_IncrementalMesh(shape, linear_deflection_mm, False, 0.2, True)
        mesher.Perform()

        vertices: list[tuple[float, float, float]] = []
        triangles: list[tuple[int, int, int]] = []
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face = TopoDS.Face_s(explorer.Current())
            location = TopLoc_Location()
            triangulation = BRep_Tool.Triangulation_s(face, location)
            if triangulation:
                offset = len(vertices)
                local_to_world = location.Transformation()
                for index in range(1, triangulation.NbNodes() + 1):
                    point = triangulation.Node(index).Transformed(local_to_world)
                    vertices.append((point.X(), point.Y(), point.Z()))
                for index in range(1, triangulation.NbTriangles() + 1):
                    a, b, c = triangulation.Triangle(index).Get()
                    triangles.append((offset + a - 1, offset + b - 1, offset + c - 1))
            explorer.Next()
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(vertices, dtype=float)),
        o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32)),
    )
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    if mesh.is_empty():
        raise ValueError(f"{path} produced an empty mesh")
    return mesh


def capture_point_cloud(config: dict[str, Any]):
    """Capture and fuse native NU4000 point-cloud frames in camera millimetres."""
    import numpy as np
    import open3d as o3d
    from color_depth_viewer import find_sdk_dir, prepare_runtime

    prepare_runtime(find_sdk_dir(config["camera"].get("sdk")))
    from api.inu_sensor import InuSensor
    from api.shared import StreamType

    config.pop("_observation", None)
    observation = read_observation_pose() if config["camera"].get("mount") == "eye_in_hand" else None
    if observation is not None and config["camera"].get("point_cloud_frame"):
        observation["camera_frame"] = config["camera"]["point_cloud_frame"]
    from camera_cleanup import CameraResource
    sensor = InuSensor()
    sensor_lifecycle = CameraResource(sensor)
    stream = None
    stream_lifecycle = None
    frames: list[np.ndarray] = []
    try:
        sensor_lifecycle.init()
        sensor_lifecycle.start()
        stream = sensor.create_stream(StreamType.POINT_CLOUD)
        stream_lifecycle = CameraResource(stream)
        stream_lifecycle.init()
        stream_lifecycle.start()
        warmup = int(config["camera"].get("warmup_frames", 3))
        count = int(config["camera"].get("frames", 5))
        complete_frames = 0
        for _attempt in range(warmup + count + 30):
            frame = stream.frame
            raw = np.asarray(frame.data)
            # The SDK's NumOfPoints metadata can advance before the Python
            # buffer view is materialized. The buffer itself is authoritative.
            if raw.size < 300 or raw.size % 3:
                continue
            points = raw.reshape(-1, 3).astype(np.float64)
            if config["camera"].get("point_cloud_frame") == "opencv_optical":
                # NU4000 native XYZ has +Y upward. Projection/calibration here
                # use the optical convention: +X right, +Y down, +Z forward.
                points[:, 1] *= -1.0
            valid = np.isfinite(points).all(axis=1) & (np.linalg.norm(points, axis=1) > 0)
            if np.count_nonzero(valid) < 100:
                continue
            complete_frames += 1
            if complete_frames > warmup:
                frames.append(points[valid])
            if len(frames) >= count:
                break
    finally:
        if stream_lifecycle is not None:
            stream_lifecycle.close()
        sensor_lifecycle.close()
    if len(frames) < count:
        raise RuntimeError(f"camera supplied only {len(frames)} complete point-cloud frames")
    if observation is not None:
        verify_observation_unchanged(observation, read_observation_pose())
        config["_observation"] = observation
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack(frames)))
    return cloud.voxel_down_sample(float(config["matching"]["scene_voxel_mm"]))


def crop_cloud(cloud, limits: dict[str, list[float]]):
    import numpy as np
    import open3d as o3d

    minimum = np.asarray(limits["min_mm"], dtype=float)
    maximum = np.asarray(limits["max_mm"], dtype=float)
    return cloud.crop(o3d.geometry.AxisAlignedBoundingBox(minimum, maximum))


def remove_support_plane(cloud, config: dict[str, Any]):
    if len(cloud.points) < 100:
        raise RuntimeError("too few scene points after ROI crop")
    threshold = float(config["matching"]["plane_distance_mm"])
    plane, inliers = cloud.segment_plane(threshold, 3, 1200)
    objects = cloud.select_by_index(inliers, invert=True)
    return objects, plane, cloud.select_by_index(inliers)


def cluster_objects(cloud, config: dict[str, Any]) -> list[Any]:
    import numpy as np

    labels = np.asarray(cloud.cluster_dbscan(
        eps=float(config["matching"]["cluster_eps_mm"]),
        min_points=int(config["matching"]["cluster_min_points"]),
        print_progress=False,
    ))
    result = []
    for label in range(int(labels.max()) + 1 if labels.size else 0):
        indexes = np.flatnonzero(labels == label)
        if len(indexes) >= int(config["matching"]["min_cluster_points"]):
            result.append(cloud.select_by_index(indexes.tolist()))
    return result


def exclude_camera_geometry(cloud, config: dict[str, Any]):
    """Remove configured tool volumes in the wrist camera's fixed coordinates."""
    import numpy as np

    points = np.asarray(cloud.points)
    keep = np.ones(len(points), dtype=bool)
    for box in config["camera"].get("excluded_boxes_mm", []):
        lower = np.asarray(box["min_mm"], dtype=float)
        upper = np.asarray(box["max_mm"], dtype=float)
        if lower.shape != (3,) or upper.shape != (3,) or not np.all(lower < upper):
            raise ValueError("invalid camera exclusion box")
        keep &= ~np.all((points >= lower) & (points <= upper), axis=1)
    return cloud.select_by_index(np.flatnonzero(keep).tolist())


def _features(cloud, voxel_mm: float):
    import open3d as o3d

    down = cloud.voxel_down_sample(voxel_mm)
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_mm * 3.0, max_nn=40))
    feature = o3d.pipelines.registration.compute_fpfh_feature(
        down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_mm * 6.0, max_nn=100))
    return down, feature


@dataclass
class MatchResult:
    cluster_index: int
    score: float
    fitness: float
    rmse_mm: float
    camera_T_model: list[list[float]]
    cluster_points: int
    clearance_mm: float
    observation: dict[str, Any] | None = None


@dataclass
class RegistrationResult:
    fitness: float
    inlier_rmse: float
    transformation: "np.ndarray"


def _rotation_between(source: "np.ndarray", target: "np.ndarray") -> "np.ndarray":
    import numpy as np
    from scipy.spatial.transform import Rotation

    source = _normalize(source)
    target = _normalize(target)
    cross = np.cross(source, target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if np.linalg.norm(cross) < 1e-9:
        if dot > 0:
            return np.eye(3)
        helper = np.array([1.0, 0.0, 0.0])
        if abs(source[0]) > 0.8:
            helper = np.array([0.0, 1.0, 0.0])
        return Rotation.from_rotvec(_normalize(np.cross(source, helper)) * math.pi).as_matrix()
    return Rotation.from_rotvec(_normalize(cross) * math.acos(dot)).as_matrix()


def _partial_view_registration(model_cloud, target, config: dict[str, Any]) -> RegistrationResult:
    """Fit visible cluster points to CAD, tolerating an unseen model backside."""
    import numpy as np
    import open3d as o3d
    from scipy.spatial.transform import Rotation

    model = model_cloud.voxel_down_sample(float(config["matching"]["model_voxel_mm"]))
    observed = target.voxel_down_sample(float(config["matching"]["model_voxel_mm"]))
    radius = float(config["matching"]["model_voxel_mm"]) * 3.0
    for cloud in (model, observed):
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=50))
    model_points = np.asarray(model.points)
    observed_points = np.asarray(observed.points)
    model_center = model_points.mean(axis=0)
    observed_center = observed_points.mean(axis=0)
    model_values, model_vectors = np.linalg.eigh(np.cov((model_points - model_center).T))
    observed_values, observed_vectors = np.linalg.eigh(np.cov((observed_points - observed_center).T))
    model_axis = model_vectors[:, np.argmax(model_values)]
    observed_axis = observed_vectors[:, np.argmax(observed_values)]
    coarse_distance = max(float(config["matching"]["ransac_distance_mm"]), 5.0)
    fine_distance = float(config["matching"]["icp_distance_mm"])
    best = RegistrationResult(0.0, float("inf"), np.eye(4))
    for sign in (-1.0, 1.0):
        align = _rotation_between(sign * observed_axis, model_axis)
        for angle_deg in range(0, 360, 30):
            around_axis = Rotation.from_rotvec(
                _normalize(model_axis) * math.radians(angle_deg)).as_matrix()
            rotation = around_axis @ align
            initial = np.eye(4)
            initial[:3, :3] = rotation
            initial[:3, 3] = model_center - rotation @ observed_center
            coarse = o3d.pipelines.registration.registration_icp(
                observed, model, coarse_distance, initial,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=35),
            )
            fine = o3d.pipelines.registration.registration_icp(
                observed, model, fine_distance, coarse.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
            )
            quality = fine.fitness - 0.08 * fine.inlier_rmse / max(fine_distance, 1e-6)
            best_quality = best.fitness - 0.08 * best.inlier_rmse / max(fine_distance, 1e-6)
            if quality > best_quality:
                best = RegistrationResult(float(fine.fitness), float(fine.inlier_rmse),
                                          np.linalg.inv(fine.transformation))
    return best


def register_model(model_cloud, target, config: dict[str, Any], *, scene_cloud=None, local_window=False):
    if config["matching"].get("strategy") == "clutter" and scene_cloud is not None:
        from clutter_matching import register_model as register_clutter
        return register_clutter(model_cloud, target, config, scene_cloud, local_window)
    import open3d as o3d

    voxel = float(config["matching"]["model_voxel_mm"])
    source_down, source_feature = _features(model_cloud, voxel)
    target_down, target_feature = _features(target, voxel)
    distance = float(config["matching"]["ransac_distance_mm"])
    coarse = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_feature, target_feature, True, distance,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.85),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(30000, 0.999),
    )
    model_cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3.0, max_nn=50))
    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3.0, max_nn=50))
    refined = o3d.pipelines.registration.registration_icp(
        model_cloud, target, float(config["matching"]["icp_distance_mm"]),
        coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80),
    )
    partial = _partial_view_registration(model_cloud, target, config)
    full_quality = refined.fitness - 0.08 * refined.inlier_rmse / max(
        float(config["matching"]["icp_distance_mm"]), 1e-6)
    partial_quality = partial.fitness - 0.08 * partial.inlier_rmse / max(
        float(config["matching"]["icp_distance_mm"]), 1e-6)
    full_valid = validate_model_support(model_cloud, target, refined.transformation, config)["accepted"]
    partial_valid = validate_model_support(model_cloud, target, partial.transformation, config)["accepted"]
    if full_valid != partial_valid:
        if partial_valid:
            return partial
        return RegistrationResult(float(refined.fitness), float(refined.inlier_rmse), refined.transformation)
    if partial_quality > full_quality:
        return partial
    return RegistrationResult(float(refined.fitness), float(refined.inlier_rmse),
                              refined.transformation)


def _extent_is_plausible(model_cloud, cluster, tolerance: float) -> bool:
    import numpy as np

    model_extent = np.sort(model_cloud.get_axis_aligned_bounding_box().get_extent())
    cluster_extent = np.sort(cluster.get_axis_aligned_bounding_box().get_extent())
    ratio = cluster_extent / np.maximum(model_extent, 1e-6)
    # A single depth view often sees only a fraction of the extent normal to
    # the visible surface. Preserve strong checks on the two larger axes while
    # allowing that one occluded dimension to be partial.
    return bool(0.20 <= ratio[0] <= 1.0 + tolerance and
                0.45 <= ratio[1] <= 1.0 + tolerance and
                0.65 <= ratio[2] <= 1.0 + tolerance)


def scene_clearance_mm(model_cloud, scene_cloud, camera_T_model, grasp_model, tolerance_mm: float) -> float:
    """Distance to observed non-model surfaces, including touching/noise points.

    This is a candidate ranking metric, not a gripper collision certificate.
    A cluster can contain multiple touching parts; its label cannot identify
    all of its points as belonging to the selected workpiece.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    model_points = transform_points(np.asarray(model_cloud.points), np.asarray(camera_T_model))
    scene_points = np.asarray(scene_cloud.points)
    if len(model_points) == 0 or len(scene_points) == 0:
        raise ValueError("clearance requires nonempty model and scene point clouds")
    distance, _ = cKDTree(model_points).query(scene_points)
    other_points = scene_points[distance > tolerance_mm]
    if not len(other_points):
        return 100.0
    center = transform_points([grasp_model], np.asarray(camera_T_model))[0]
    return float(np.linalg.norm(other_points - center, axis=1).min())


def find_matches(model_cloud, scene_cloud, clusters, config: dict[str, Any], *, previous_matches=None) -> list[MatchResult]:
    if config["matching"].get("strategy") == "clutter":
        from clutter_matching import find_matches as find_clutter
        return find_clutter(model_cloud, scene_cloud, clusters, config, previous_matches)
    import numpy as np
    import open3d as o3d

    results: list[MatchResult] = []
    diagnostics = config["_matching_diagnostics"] = []
    tolerance = float(config["matching"]["extent_tolerance"])
    for index, cluster in enumerate(clusters):
        filtered = exclude_camera_geometry(cluster, config)
        if len(filtered.points) < 0.8 * len(cluster.points):
            diagnostics.append({"cluster_index": index, "accepted": False,
                                "reason": "intersects camera/tool exclusion volume"})
            continue
        if not _extent_is_plausible(model_cloud, cluster, tolerance):
            diagnostics.append({"cluster_index": index, "accepted": False,
                                "reason": "cluster dimensions incompatible with CAD"})
            continue
        registration = register_model(model_cloud, cluster, config)
        validation = validate_model_support(model_cloud, cluster, registration.transformation, config)
        diagnostics.append({"cluster_index": index, **validation})
        if not validation["accepted"]:
            continue
        clearance = scene_clearance_mm(
            model_cloud, scene_cloud, registration.transformation,
            config["model"]["grasp_point_mm"], float(config["matching"]["icp_distance_mm"]))
        rmse_scale = max(float(config["matching"]["icp_distance_mm"]), 1e-6)
        score = (0.78 * float(registration.fitness) +
                 0.17 * max(0.0, 1.0 - float(registration.inlier_rmse) / rmse_scale) +
                 0.05 * min(clearance / 30.0, 1.0))
        if score < float(config["matching"]["minimum_score"]):
            diagnostics[-1].update(accepted=False, reason="score below threshold")
            continue
        results.append(MatchResult(
            cluster_index=index,
            score=score,
            fitness=float(registration.fitness),
            rmse_mm=float(registration.inlier_rmse),
            camera_T_model=np.asarray(registration.transformation).tolist(),
            cluster_points=len(cluster.points),
            clearance_mm=clearance,
            observation=config.get("_observation"),
        ))
    return sorted(results, key=lambda item: item.score, reverse=True)


def validate_model_support(model_cloud, cluster, camera_T_model, config: dict[str, Any], *, visibility=None) -> dict[str, Any]:
    """Require support across the visible CAD and each configured bolt region."""
    import numpy as np
    import open3d as o3d
    from scipy.spatial import cKDTree

    tolerance = float(config["matching"]["icp_distance_mm"])
    model_points = np.asarray(model_cloud.points)
    aligned = transform_points(model_points, np.asarray(camera_T_model))
    observed = np.asarray(cluster.points)
    if not len(aligned) or not len(observed):
        return {"accepted": False, "reason": "empty geometry"}
    distances, _ = cKDTree(observed).query(aligned)
    observed_distances, _ = cKDTree(aligned).query(observed)
    visible_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(aligned))
    radius = max(float(np.linalg.norm(aligned, axis=1).max()) * 100, 1.0)
    _, visible = visible_cloud.hidden_point_removal([0, 0, 0], radius)
    visible = np.asarray(visible, dtype=int)
    if not len(visible):
        return {"accepted": False, "reason": "no visible CAD surface"}
    self_visible = visible.copy()
    if visibility is not None:
        visible = visibility.unoccluded(aligned, visible)
        if len(visible) < max(30, .25 * len(self_visible)):
            return {"accepted": False, "reason": "too little unoccluded CAD evidence"}
    supported = distances <= tolerance
    metrics = {"observed_support": float(np.mean(observed_distances <= tolerance)),
               "visible_support": float(np.mean(supported[visible]))}
    limits = {"observed_support": config["matching"].get("minimum_observed_support", 0.65),
              "visible_support": config["matching"].get("minimum_visible_support", 0.60)}
    for region in ("head", "shaft"):
        bounds = config["model"].get(f"{region}_z_range_mm")
        if bounds is None:
            continue
        indexes = visible[(model_points[visible, 2] >= bounds[0]) &
                          (model_points[visible, 2] <= bounds[1])]
        region_visible = self_visible[(model_points[self_visible, 2] >= bounds[0]) &
                                     (model_points[self_visible, 2] <= bounds[1])]
        # A hidden end cannot establish the distinctive geometry of this bolt.
        required = max(10, int(.2 * len(region_visible))) if visibility is not None else 10
        metrics[f"{region}_support"] = float(np.mean(supported[indexes])) if len(indexes) >= required else 0.0
        limits[f"{region}_support"] = config["matching"].get(f"minimum_{region}_support", 0.65)
    rejected = [name for name, minimum in limits.items() if metrics[name] < float(minimum)]
    return {**metrics, "accepted": not rejected,
            "reason": "insufficient " + ", ".join(rejected) if rejected else "geometry supported"}


def scan_scene(config: dict[str, Any]):
    import numpy as np
    import open3d as o3d

    mesh = load_fcstd_mesh(PROJECT_DIR / config["model"]["path"])
    o3d.utility.random.seed(7)
    model_cloud = mesh.sample_points_uniformly(int(config["matching"]["model_sample_points"]))
    scene = crop_cloud(capture_point_cloud(config), config["camera"]["roi"])
    objects, plane, support = remove_support_plane(scene, config)
    objects = exclude_camera_geometry(objects, config)
    clusters = cluster_objects(objects, config)
    matches = find_matches(model_cloud, scene, clusters, config)
    return mesh, model_cloud, scene, objects, support, clusters, matches, plane


def _write_preview(path: Path, scene, clusters, model_cloud, matches: list[MatchResult]) -> None:
    import copy
    import cv2
    import numpy as np

    width, height = 900, 540
    image = np.full((height, width, 3), 247, np.uint8)
    points = np.asarray(scene.points)
    if not len(points):
        cv2.imwrite(str(path), image)
        return
    minimum = np.percentile(points[:, :2], 1, axis=0)
    maximum = np.percentile(points[:, :2], 99, axis=0)
    span = np.maximum(maximum - minimum, 1.0)

    def draw(point_array, color, radius=1):
        values = np.asarray(point_array)
        if not len(values):
            return
        pixels = ((values[:, :2] - minimum) / span * [width - 40, height - 70] + [20, 20]).astype(int)
        pixels[:, 1] = height - 50 - pixels[:, 1]
        valid = ((pixels[:, 0] >= 0) & (pixels[:, 0] < width) &
                 (pixels[:, 1] >= 0) & (pixels[:, 1] < height - 30))
        for x, y in pixels[valid][::max(1, len(pixels) // 40000)]:
            cv2.circle(image, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)

    draw(points, (175, 175, 175))
    caption = "No CAD match above threshold"
    if matches:
        best = matches[0]
        draw(np.asarray(clusters[best.cluster_index].points), (40, 190, 60), 2)
        aligned = copy.deepcopy(model_cloud)
        aligned.transform(best.camera_T_model)
        draw(np.asarray(aligned.points), (35, 35, 220), 2)
        caption = f"Best score {best.score:.3f}   fitness {best.fitness:.3f}   RMSE {best.rmse_mm:.2f} mm"
    cv2.rectangle(image, (0, height - 34), (width, height), (35, 35, 35), -1)
    cv2.putText(image, caption, (14, height - 11), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(image, "TOP VIEW   gray=scene  green=selected  red=CAD", (14, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (45, 45, 45), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), image)


def save_scan(output_dir: Path, scene, objects, model_cloud, clusters,
              matches: list[MatchResult]) -> None:
    import open3d as o3d

    output_dir.mkdir(parents=True, exist_ok=True)
    # A motion plan belongs to one scan and must not survive a new acquisition.
    for name in ("plan.json", "grasp_plan.md", "grasp_preview.png"):
        (output_dir / name).unlink(missing_ok=True)
    o3d.io.write_point_cloud(str(output_dir / "scene.ply"), scene)
    o3d.io.write_point_cloud(str(output_dir / "objects.ply"), objects)
    if matches:
        aligned = type(model_cloud)(model_cloud)
        aligned.transform(matches[0].camera_T_model)
        o3d.io.write_point_cloud(str(output_dir / "best_model.ply"), aligned)
    else:
        (output_dir / "best_model.ply").unlink(missing_ok=True)
    with (output_dir / "matches.json").open("w", encoding="utf-8") as stream:
        json.dump([asdict(item) for item in matches], stream, ensure_ascii=False, indent=2)
    _write_preview(output_dir / "preview.png", scene, clusters, model_cloud, matches)


def show_scan(scene, model_cloud, clusters, matches: list[MatchResult]) -> None:
    import copy
    import open3d as o3d

    scene_view = copy.deepcopy(scene)
    scene_view.paint_uniform_color([0.55, 0.55, 0.55])
    geometry = [scene_view]
    if matches:
        selected = copy.deepcopy(clusters[matches[0].cluster_index])
        selected.paint_uniform_color([0.1, 0.8, 0.2])
        aligned = copy.deepcopy(model_cloud)
        aligned.transform(matches[0].camera_T_model)
        aligned.paint_uniform_color([0.9, 0.1, 0.1])
        geometry.extend((selected, aligned))
    o3d.visualization.draw_geometries(geometry, window_name="3D match: gray scene, green object, red CAD")


def show_saved_results(input_dir: Path) -> None:
    import open3d as o3d

    scene_path = input_dir / "scene.ply"
    if not scene_path.is_file():
        raise FileNotFoundError(f"missing scan result: {scene_path}")
    scene = o3d.io.read_point_cloud(str(scene_path))
    scene.paint_uniform_color([0.55, 0.55, 0.55])
    geometry = [scene]
    model_path = input_dir / "best_model.ply"
    if model_path.is_file():
        model = o3d.io.read_point_cloud(str(model_path))
        model.paint_uniform_color([0.9, 0.1, 0.1])
        geometry.append(model)
    o3d.visualization.draw_geometries(geometry, window_name="Saved 3D result: gray scene, red CAD")


def rigid_transform(source_points: "np.ndarray", target_points: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
    """Least-squares transform from source to target and per-point residuals."""
    import numpy as np

    source = np.asarray(source_points, dtype=float)
    target = np.asarray(target_points, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError("calibration requires matching Nx3 point sets with N >= 3")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    centered_source = source - source_center
    if np.linalg.matrix_rank(centered_source) < 2:
        raise ValueError("calibration points must not be collinear")
    u, _values, vt = np.linalg.svd(centered_source.T @ (target - target_center))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    residuals = np.linalg.norm(transform_points(source, transform) - target, axis=1)
    return transform, residuals


def calibrate_camera(config_path: Path, config: dict[str, Any], minimum_points: int) -> None:
    import numpy as np
    import open3d as o3d
    import arm_control

    scene = crop_cloud(capture_point_cloud(config), config["camera"]["roi"])
    print("Shift+left-click fixed landmarks in order; pick at least", minimum_points,
          "points, then press Q to close.")
    editor = o3d.visualization.VisualizerWithEditing()
    editor.create_window(window_name="Pick camera calibration landmarks")
    editor.add_geometry(scene)
    editor.run()
    editor.destroy_window()
    indexes = editor.get_picked_points()
    if len(indexes) < minimum_points:
        raise RuntimeError(f"only {len(indexes)} points selected; need at least {minimum_points}")
    camera_points = np.asarray(scene.points)[indexes]
    base_points = []
    for index, camera_point in enumerate(camera_points, 1):
        input(f"Jog the robot TCP origin to physical landmark {index} at camera {camera_point.round(2).tolist()}, then press Enter: ")
        landmark = read_observation_pose()
        captured = config.get("_observation")
        if captured is not None and (landmark["tool_id"], landmark["user_id"]) != (captured["tool_id"], captured["user_id"]):
            raise RuntimeError("robot tool/user frame changed during landmark calibration")
        base_points.append(landmark["tcp_pose"][:3])
    transform, residuals = rigid_transform(camera_points, np.asarray(base_points))
    print("base_T_camera:")
    print(np.array2string(transform, precision=8, suppress_small=True))
    print(f"residual RMS={math.sqrt(float(np.mean(residuals ** 2))):.3f} mm, max={float(residuals.max()):.3f} mm")
    if float(residuals.max()) > 3.0:
        raise RuntimeError("calibration residual exceeds 3 mm; points were not saved")
    if input("Type SAVE to write this transform: ").strip() != "SAVE":
        print("Calibration was not saved.")
        return
    if config["camera"].get("mount") == "eye_in_hand":
        observation = config.get("_observation")
        if observation is None:
            raise RuntimeError("eye-in-hand calibration has no acquisition pose")
        config["camera"]["tcp_T_camera"] = (np.linalg.inv(pose_to_matrix(observation["tcp_pose"])) @ transform).tolist()
        config["camera"]["calibration_tool_id"] = observation["tool_id"]
    else:
        config["camera"]["base_T_camera"] = transform.tolist()
    config.pop("_observation", None)
    save_config(config_path, config)
    print("Camera extrinsic saved.")


def calibrate_platform(config_path: Path, config: dict[str, Any]) -> None:
    import numpy as np

    scene = crop_cloud(capture_point_cloud(config), config["camera"]["roi"])
    _objects, plane, support = remove_support_plane(scene, config)
    base_T_camera = camera_transform(config, config.get("_observation"))
    base_points = transform_points(np.asarray(support.points), base_T_camera)
    base_normal = _normalize(base_T_camera[:3, :3] @ np.asarray(plane[:3], dtype=float))
    if abs(float(base_normal[2])) < 0.95:
        raise RuntimeError(f"largest plane is not horizontal in robot base coordinates: normal={base_normal}")
    z = float(np.median(base_points[:, 2]))
    spread = float(np.percentile(np.abs(base_points[:, 2] - z), 95))
    print(f"Detected platform Z={z:.3f} mm; 95% vertical spread={spread:.3f} mm")
    if spread > 3.0:
        raise RuntimeError("support plane spread exceeds 3 mm; platform height was not saved")
    if input("Clear the placement area and type SAVE to use this platform height: ").strip() != "SAVE":
        print("Platform height was not saved.")
        return
    config["placement"]["platform_z_mm"] = z
    config.pop("_observation", None)
    save_config(config_path, config)
    print("Platform height saved.")


def configure_tool(config_path: Path, config: dict[str, Any], offset: list[float], max_object: float) -> None:
    if len(offset) != 3 or not all(math.isfinite(value) for value in offset):
        raise ValueError("tool offset must contain three finite millimetre values")
    if not math.isfinite(max_object) or max_object <= 0:
        raise ValueError("measured maximum object diameter must be positive")
    config["gripper"]["tcp_to_grasp_center_mm"] = list(map(float, offset))
    config["gripper"]["measured_max_object_mm"] = float(max_object)
    save_config(config_path, config)
    print("Tool geometry saved. Run validate-config to review readiness.")


def missing_execution_calibration(config: dict[str, Any]) -> list[str]:
    missing = []
    required = {
        "gripper.tcp_to_grasp_center_mm": config["gripper"].get("tcp_to_grasp_center_mm"),
        "gripper.measured_max_object_mm": config["gripper"].get("measured_max_object_mm"),
    }
    mount = config["camera"].get("mount", "fixed")
    if mount == "fixed":
        required["camera.base_T_camera"] = config["camera"].get("base_T_camera")
    elif mount == "eye_in_hand":
        required["camera.tcp_T_camera"] = config["camera"].get("tcp_T_camera")
        required["camera.calibration_tool_id"] = config["camera"].get("calibration_tool_id")
    else:
        raise ValueError(f"unsupported camera mount: {mount}")
    if config["placement"].get("mode", "platform") == "platform":
        required["placement.platform_z_mm"] = config["placement"].get("platform_z_mm")
    elif config["placement"]["mode"] != "home_tcp":
        raise ValueError("unsupported placement mode")
    for name, value in required.items():
        if value is None:
            missing.append(name)
    if config["fixture"].get("enabled"):
        for name in ("deposit_approach_pose", "deposit_pose", "pickup_approach_pose",
                     "pickup_pose", "model_pose_when_clamped"):
            if config["fixture"].get(name) is None:
                missing.append(f"fixture.{name}")
    return missing


def _normalize(vector: "np.ndarray") -> "np.ndarray":
    import numpy as np

    length = np.linalg.norm(vector)
    if length < 1e-9:
        raise ValueError("cannot normalize a zero vector")
    return vector / length


def generate_grasp_plan(matches: list[MatchResult], config: dict[str, Any], scene, model_cloud) -> dict[str, Any]:
    """Generate a reviewable shaft grasp even before robot calibration exists."""
    import copy
    import numpy as np

    region = config["model"].get("grasp_region")
    if region != "shaft":
        raise ValueError("this bolt planner requires the configured shaft grasp region")
    lower, upper = map(float, config["model"]["shaft_z_range_mm"])
    if not math.isfinite(lower + upper) or upper <= lower:
        raise ValueError("invalid shaft grasp range")
    head_upper = float(config["model"]["head_z_range_mm"][1])
    if lower <= head_upper:
        raise ValueError("shaft grasp range overlaps the bolt head")
    proposals = []
    for match in matches:
        if match.score < float(config["matching"]["minimum_score"]):
            continue
        transform = matrix(match.camera_T_model, "match.camera_T_model")
        for fraction in (0.35, 0.5, 0.65):
            grasp = [0.0, 0.0, lower + fraction * (upper - lower)]
            clearance = scene_clearance_mm(model_cloud, scene, transform, grasp,
                                           float(config["matching"]["icp_distance_mm"]))
            proposals.append({
                "cluster_index": match.cluster_index, "match_score": match.score,
                "fitness": match.fitness, "rmse_mm": match.rmse_mm,
                "grasp_region": region, "grasp_point_model_mm": grasp,
                "grasp_center_camera_mm": transform_points([grasp], transform)[0].tolist(),
                "shaft_axis_camera": _normalize(transform[:3, :3] @ np.asarray(config["model"]["axis_model"])).tolist(),
                "scene_clearance_mm": clearance,
                "camera_T_model": match.camera_T_model,
            })
    if not proposals:
        raise RuntimeError("no shaft grasp candidate passed the model match threshold")
    # Clearance is a ranking metric until actual finger geometry is measured.
    proposals.sort(key=lambda item: (item["scene_clearance_mm"], item["match_score"]), reverse=True)
    selected = proposals[0]
    match = next(item for item in matches if item.cluster_index == selected["cluster_index"])
    missing = missing_execution_calibration(config)
    home = list(config["placement"]["home_pose"])
    home_approach = home.copy()
    home_approach[2] += float(config["robot"]["approach_mm"])
    if missing:
        plan = {"route": "requires_calibration", "status": "preview", "executable": False,
                "direct_rotation_deg": None, "observation": match.observation,
                "pick_approach": None, "pick": None, "pick_lift": None,
                "place_approach": home_approach, "place": home, "retreat": home_approach}
    else:
        selected_config = copy.deepcopy(config)
        selected_config["model"]["grasp_point_mm"] = selected["grasp_point_model_mm"]
        plan = plan_pick_place(match, selected_config)
        plan.update(status="planned", executable=plan["route"] != "blocked")
    plan.update({
        "created_unix": time.time(), "model_path": config["model"]["path"],
        "match_score": match.score, "missing_calibration": missing,
        "grasp_region": region, "grasp_width_mm": float(config["model"]["grasp_width_mm"]),
        "shaft_z_range_mm": [lower, upper],
        "open_command_mm": float(config["gripper"]["open_command_mm"]),
        "close_command_mm": float(config["gripper"]["close_command_mm"]),
        "selected_grasp": selected, "shaft_grasp_candidates": proposals,
        "finger_collision_checked": False, "ik_checked": False,
        "sequence": ["observe", "open", "approach_shaft", "descend_to_shaft", "close",
                     "lift", "confirm_hold", "transfer_above_home", "descend_home",
                     "open_to_release", "retreat", "rescan_before_next_pick"],
    })
    return plan


def write_grasp_plan(output_dir: Path, plan: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "plan.json").open("w", encoding="utf-8") as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2)
    selected = plan["selected_grasp"]
    xyz = ", ".join(f"{value:.2f}" for value in selected["grasp_center_camera_mm"])
    home = ", ".join(f"{value:g}" for value in plan["place"])
    lines = ["# 螺栓杆身抓取方案", "",
             f"模型：{plan['model_path']}。抓杆身，避开大头。模型杆身外径 {plan['grasp_width_mm']:g} mm。",
             f"张开设定：{plan['open_command_mm']:g} mm；夹持杆身设定：{plan['close_command_mm']:g} mm（用户已确认夹住时的指令值）。",
             f"选中候选簇：{selected['cluster_index']}；模型匹配分：{selected['match_score']:.3f}；拟合误差：{selected['rmse_mm']:.2f} mm。",
             f"杆身抓点（模型坐标，mm）：{selected['grasp_point_model_mm']}。",
             f"抓取中心（相机坐标，mm）：[{xyz}]。",
             f"距非模型场景点的最小距离：{selected['scene_clearance_mm']:.2f} mm，仅用于排序，尚未验证夹指碰撞。",
             f"home 释放 TCP 位姿（mm / deg）：[{home}]。", "",
             "动作顺序：拍照 → 张开 → 对准杆身 → 下降 → 夹持 → 抬升 → 确认夹牢 → 移到 home 上方 → 下降到 home → 张开释放 → 退回 → 再次拍照。", ""]
    if not plan["executable"]:
        lines += ["当前为预览方案，不能发送真机动作。",
                  "尚缺：" + ", ".join(plan["missing_calibration"]) + "。",
                  "机器人抓取位姿尚未计算；不会用相机坐标直接代替机器人坐标。"]
    else:
        lines += ["已计算机器人位姿。实际执行仍需核对可达性、夹指与料框碰撞及现场状态。"]
    (output_dir / "grasp_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_grasp_preview(output_dir: Path, plan: dict[str, Any], scene, model_cloud) -> None:
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    figure = Figure(figsize=(9, 5.4), dpi=100)
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(111)
    points = np.asarray(scene.points)
    shown = points[::max(1, len(points) // 30000)]
    axes.scatter(shown[:, 0], shown[:, 1], s=1, c="#b6babf", label="Scene")
    selected = plan["selected_grasp"]
    model_points = np.asarray(model_cloud.points)
    aligned = transform_points(model_points, np.asarray(selected["camera_T_model"]))
    low, high = plan["shaft_z_range_mm"]
    shaft = (model_points[:, 2] >= low) & (model_points[:, 2] <= high)
    axes.scatter(aligned[~shaft, 0], aligned[~shaft, 1], s=3, c="#d25442", label="Head - do not grip")
    axes.scatter(aligned[shaft, 0], aligned[shaft, 1], s=3, c="#287ba8", label="Shaft")
    center = selected["grasp_center_camera_mm"]
    axes.scatter([center[0]], [center[1]], s=160, marker="*", c="#ffcf33", edgecolors="black", label="Selected shaft grasp", zorder=5)
    axes.annotate(f"Grip shaft\ncommand {plan['close_command_mm']:g} mm", xy=center[:2],
                  xytext=(25, -35), textcoords="offset points", fontsize=10,
                  arrowprops={"arrowstyle": "->", "color": "black"},
                  bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9})
    axes.set(xlabel="Camera X (mm)", ylabel="Camera Y (mm)",
             title="Shaft grasp preview | camera coordinates | robot motion pending calibration")
    axes.set_aspect("equal", adjustable="datalim")
    axes.legend(loc="upper right", fontsize=8, markerscale=1.3)
    figure.tight_layout()
    figure.savefig(output_dir / "grasp_preview.png")


def plan_saved_results(input_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    import open3d as o3d

    matches = [MatchResult(**item) for item in json.loads((input_dir / "matches.json").read_text(encoding="utf-8"))]
    scene = o3d.io.read_point_cloud(str(input_dir / "scene.ply"))
    o3d.utility.random.seed(7)
    model = load_fcstd_mesh(PROJECT_DIR / config["model"]["path"]).sample_points_uniformly(
        int(config["matching"]["model_sample_points"]))
    plan = generate_grasp_plan(matches, config, scene, model)
    write_grasp_plan(input_dir, plan)
    write_grasp_preview(input_dir, plan, scene, model)
    return plan


def plan_pick_place(match: MatchResult, config: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from scipy.spatial.transform import Rotation

    missing = missing_execution_calibration(config)
    if missing:
        raise RuntimeError("execution calibration is incomplete: " + ", ".join(missing))
    if float(config["gripper"]["measured_max_object_mm"]) < float(config["model"]["grasp_width_mm"]) + 0.5:
        raise RuntimeError("measured finger opening is too small for the configured grasp width")
    base_T_camera = camera_transform(config, match.observation)
    camera_T_model = matrix(match.camera_T_model, "match.camera_T_model")
    base_T_model = base_T_camera @ camera_T_model
    grasp_model = np.asarray(config["model"]["grasp_point_mm"], dtype=float)
    grasp_base = transform_points([grasp_model], base_T_model)[0]
    axis_base = _normalize(base_T_model[:3, :3] @
                           np.asarray(config["model"]["axis_model"], dtype=float))

    down_rpy = config["gripper"]["down_rpy_deg"]
    base_rotation = Rotation.from_euler("xyz", down_rpy, degrees=True).as_matrix()
    jaw_axis = _normalize(base_rotation @ np.asarray(config["gripper"]["jaw_axis_tcp"], dtype=float))
    axis_xy = np.array([axis_base[0], axis_base[1], 0.0])
    if np.linalg.norm(axis_xy) > 1e-6:
        desired_jaw = _normalize(np.cross([0.0, 0.0, 1.0], axis_xy))
        cross_z = jaw_axis[0] * desired_jaw[1] - jaw_axis[1] * desired_jaw[0]
        angle = math.atan2(float(cross_z), float(np.dot(jaw_axis[:2], desired_jaw[:2])))
        yaw_rotation = Rotation.from_rotvec([0.0, 0.0, float(angle)]).as_matrix()
        base_rotation = yaw_rotation @ base_rotation

    tcp_to_center = np.asarray(config["gripper"]["tcp_to_grasp_center_mm"], dtype=float)
    base_T_tcp_pick = np.eye(4)
    base_T_tcp_pick[:3, :3] = base_rotation
    base_T_tcp_pick[:3, 3] = grasp_base - base_rotation @ tcp_to_center
    object_in_tcp = np.linalg.inv(base_T_tcp_pick) @ base_T_model

    home = config["placement"]["home_pose"]
    if config["placement"].get("mode") == "home_tcp":
        final_tcp = pose_to_matrix(home)
    else:
        final_model = np.eye(4)
        final_model[:3, :3] = Rotation.from_euler(
            "z", float(config["placement"].get("yaw_deg", 0.0)), degrees=True).as_matrix()
        final_model[:3, 3] = [
            home[0], home[1],
            float(config["placement"]["platform_z_mm"]) - float(config["model"]["large_end_z_mm"]),
        ]
        final_tcp = final_model @ np.linalg.inv(object_in_tcp)

    approach = float(config["robot"]["approach_mm"])
    lift = float(config["robot"]["lift_mm"])
    pick_approach = base_T_tcp_pick.copy(); pick_approach[2, 3] += approach
    pick_lift = base_T_tcp_pick.copy(); pick_lift[2, 3] += lift
    place_approach = final_tcp.copy(); place_approach[2, 3] += approach
    direct_tilt = math.degrees(math.acos(float(np.clip(np.dot(
        final_tcp[:3, 2], base_rotation[:, 2]), -1.0, 1.0))))
    route = "direct"
    if direct_tilt > float(config["robot"]["max_direct_rotation_deg"]):
        route = "fixture" if config["fixture"].get("enabled") else "blocked"
    if route == "fixture":
        fixture = config["fixture"]
        clamped_model = matrix(fixture["model_pose_when_clamped"],
                               "fixture.model_pose_when_clamped")
        second_pick_tcp = pose_to_matrix(fixture["pickup_pose"])
        object_in_second_tcp = np.linalg.inv(second_pick_tcp) @ clamped_model
        final_tcp = (pose_to_matrix(home) if config["placement"].get("mode") == "home_tcp"
                     else final_model @ np.linalg.inv(object_in_second_tcp))
        place_approach = final_tcp.copy(); place_approach[2, 3] += approach
    return {
        "route": route,
        "direct_rotation_deg": direct_tilt,
        "model_pose_base": base_T_model.tolist(),
        "pick_approach": matrix_to_pose(pick_approach),
        "pick": matrix_to_pose(base_T_tcp_pick),
        "pick_lift": matrix_to_pose(pick_lift),
        "place_approach": matrix_to_pose(place_approach),
        "place": matrix_to_pose(final_tcp),
        "retreat": matrix_to_pose(place_approach),
        "observation": match.observation,
    }


def _move_verified(pose: list[float], config: dict[str, Any]) -> None:
    import numpy as np
    import arm_control

    motion = arm_control.get_motion_state()
    if (motion.get("estop") or motion.get("collision") or motion.get("soft_limit") or
            motion.get("queue") or motion.get("active_queue") or not motion.get("inpos")):
        raise RuntimeError(f"robot has an active/faulted motion state: {motion}")
    command = arm_control.move_abs(pose, speed=float(config["robot"]["speed_mm_s"]),
                                   accel=float(config["robot"]["accel_mm_s2"]))
    if str(command.get("errorCode", "")) not in ("0", "0x0", "OK"):
        raise RuntimeError(f"moveL rejected: {command}")
    if not arm_control.wait_inpos(float(config["robot"]["motion_timeout_s"])):
        raise TimeoutError("robot motion timed out")
    actual = np.asarray(arm_control.get_tcp(), dtype=float)
    target = np.asarray(pose, dtype=float)
    position_error = np.max(np.abs(actual[:3] - target[:3]))
    angle_error = np.max(np.abs((actual[3:] - target[3:] + 180.0) % 360.0 - 180.0))
    if position_error > 5.0 or angle_error > 3.0:
        raise RuntimeError(f"robot stopped away from target: actual={actual.tolist()}")


def _load_grippers(config: dict[str, Any]):
    from gripper_backend import GripperClient, GripperProfile

    with (PROJECT_DIR / "gripper_devices.json").open("r", encoding="utf-8-sig") as stream:
        devices = json.load(stream)["devices"]
    clients = []
    for device in devices:
        profile = GripperProfile(**device)
        clients.append(GripperClient(profile=profile))
    if not clients:
        raise RuntimeError("no gripper profiles are configured")
    return clients


def execute_plan(plan: dict[str, Any], config: dict[str, Any], confirm_grasp=None) -> None:
    import arm_control

    if plan.get("executable") is False or plan.get("status") == "preview":
        raise RuntimeError("preview grasp plan cannot execute; robot grasp poses require calibration")
    missing = missing_execution_calibration(config)
    if missing:
        raise RuntimeError("execution calibration is incomplete: " + ", ".join(missing))
    if config["camera"].get("mount") == "eye_in_hand":
        camera_transform(config, plan.get("observation"))
    if plan["route"] == "blocked":
        raise RuntimeError("direct wrist rotation exceeds the configured limit and fixture regrasp is unavailable")
    state = arm_control.get_status_snapshot()
    if (not state.get("enabled") or not state.get("powered_on") or
            state.get("protective_stop") or state.get("emergency_stop")):
        raise RuntimeError(f"robot is not ready: {state.get('errmsg', '')}")
    robot_gripper, *remaining = _load_grippers(config)
    fixture_gripper = remaining[0] if remaining else None
    open_mm = float(config["gripper"]["open_command_mm"])
    close_mm = float(config["gripper"]["close_command_mm"])

    if plan.get("observation") is not None:
        verify_observation_unchanged(plan["observation"], read_observation_pose())
    else:
        _move_verified(config["placement"]["home_pose"], config)
    robot_gripper.set_opening_mm(open_mm)
    _move_verified(plan["pick_approach"], config)
    _move_verified(plan["pick"], config)
    robot_gripper.set_opening_mm(close_mm)
    _move_verified(plan["pick_lift"], config)
    confirmed = (confirm_grasp() if confirm_grasp is not None else
                 input("Confirm that the part is securely held, then type HOLD: ").strip() == "HOLD")
    if not confirmed:
        raise RuntimeError("operator did not confirm the grasp")

    if plan["route"] == "fixture":
        fixture = config["fixture"]
        if fixture_gripper is None:
            raise RuntimeError("fixture regrasp requested but the positioning gripper is unavailable")
        fixture_gripper.set_opening_mm(float(fixture["open_command_mm"]))
        _move_verified(fixture["deposit_approach_pose"], config)
        _move_verified(fixture["deposit_pose"], config)
        fixture_gripper.set_opening_mm(float(fixture["close_command_mm"]))
        robot_gripper.set_opening_mm(open_mm)
        _move_verified(fixture["deposit_approach_pose"], config)
        _move_verified(fixture["pickup_approach_pose"], config)
        _move_verified(fixture["pickup_pose"], config)
        robot_gripper.set_opening_mm(close_mm)
        fixture_gripper.set_opening_mm(float(fixture["open_command_mm"]))
        _move_verified(fixture["pickup_approach_pose"], config)

    _move_verified(plan["place_approach"], config)
    _move_verified(plan["place"], config)
    robot_gripper.set_opening_mm(open_mm)
    _move_verified(plan["retreat"], config)
    _move_verified(config["placement"]["home_pose"], config)


def inspect_model(config: dict[str, Any]) -> None:
    mesh = load_fcstd_mesh(PROJECT_DIR / config["model"]["path"])
    print(json.dumps({
        "min_mm": list(map(float, mesh.get_min_bound())),
        "max_mm": list(map(float, mesh.get_max_bound())),
        "extent_mm": list(map(float, mesh.get_axis_aligned_bounding_box().get_extent())),
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
    }, indent=2))


def main() -> int:
    rerun = _rerun_in_camera_runtime()
    if rerun is not None:
        return rerun
    parser = argparse.ArgumentParser(description="NU4000 CAD matching and guarded pick/place")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inspect-model")
    commands.add_parser("validate-config")
    camera_cal = commands.add_parser("calibrate-camera")
    camera_cal.add_argument("--points", type=int, default=4)
    commands.add_parser("calibrate-platform")
    tool_cal = commands.add_parser("configure-tool")
    tool_cal.add_argument("--offset", type=float, nargs=3, required=True,
                          metavar=("X", "Y", "Z"))
    tool_cal.add_argument("--max-object", type=float, required=True)
    viewer = commands.add_parser("view-results")
    viewer.add_argument("--input", type=Path, default=PROJECT_DIR / "pick_results")
    planner = commands.add_parser("plan-results")
    planner.add_argument("--input", type=Path, default=PROJECT_DIR / "pick_results")
    scan = commands.add_parser("scan")
    scan.add_argument("--output", type=Path, default=PROJECT_DIR / "pick_results")
    scan.add_argument("--visualize", action="store_true")
    run = commands.add_parser("run")
    run.add_argument("--output", type=Path, default=PROJECT_DIR / "pick_results")
    run.add_argument("--visualize", action="store_true")
    run.add_argument("--execute", action="store_true")
    run.add_argument("--confirm", default="")
    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "inspect-model":
        inspect_model(config)
        return 0
    if args.command == "validate-config":
        missing = missing_execution_calibration(config)
        print("vision: READY")
        print("execution:", "READY" if not missing else "BLOCKED")
        if missing:
            print("missing:", ", ".join(missing))
        return 0
    if args.command == "calibrate-camera":
        calibrate_camera(args.config, config, max(3, args.points))
        return 0
    if args.command == "calibrate-platform":
        calibrate_platform(args.config, config)
        return 0
    if args.command == "configure-tool":
        configure_tool(args.config, config, args.offset, args.max_object)
        return 0
    if args.command == "view-results":
        show_saved_results(args.input)
        return 0
    if args.command == "plan-results":
        plan = plan_saved_results(args.input, config)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    mesh, model_cloud, scene, objects, support, clusters, matches, plane = scan_scene(config)
    save_scan(args.output, scene, objects, model_cloud, clusters, matches)
    (args.output / "matching_diagnostics.json").write_text(
        json.dumps(config.get("_matching_diagnostics", []), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"scene={len(scene.points)} object_points={len(objects.points)} clusters={len(clusters)} matches={len(matches)}")
    print("support_plane_camera:", [round(float(value), 6) for value in plane])
    if not matches:
        print("No model instance passed the configured geometry filters.")
        return 2
    best = matches[0]
    if best.score < float(config["matching"]["minimum_score"]):
        print(f"Best candidate score {best.score:.3f} is below the configured minimum.")
        return 2
    print("best:", json.dumps(asdict(best), ensure_ascii=False, indent=2))
    if args.visualize:
        show_scan(scene, model_cloud, clusters, matches)
    if args.command == "scan":
        return 0

    try:
        plan = generate_grasp_plan(matches, config, scene, model_cloud)
        write_grasp_plan(args.output, plan)
        write_grasp_preview(args.output, plan, scene, model_cloud)
        print("plan:", json.dumps(plan, ensure_ascii=False, indent=2))
    except RuntimeError as exc:
        print(f"Motion plan blocked: {exc}")
        return 3
    if not args.execute:
        print("Preview only. Real motion requires --execute --confirm EXECUTE.")
        return 0
    if args.confirm != "EXECUTE":
        raise SystemExit("real motion requires the exact argument: --confirm EXECUTE")
    execute_plan(plan, config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
