import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

import markerless_handeye as calibration
import pick_place_3d as vision


def synthetic_motions():
    rng = np.random.default_rng(21)
    transform = vision.pose_to_matrix([42, -35, 85, 12, -9, 32])
    robot = [vision.pose_to_matrix([*rng.uniform(-35, 35, 3), *rng.uniform(-22, 22, 3)])
             for _ in range(9)]
    camera = [np.linalg.inv(transform) @ a @ transform for a in robot]
    return transform, robot, camera


class HandeyeTests(unittest.TestCase):
    def test_known_handeye_rotation_translation_and_frame_direction_are_recovered(self):
        expected, a, b = synthetic_motions()
        report = calibration.solve_motion_pairs(a, b)
        self.assertTrue(report["accepted"])
        self.assertFalse(report["applied_to_robot_config"])
        np.testing.assert_allclose(report["candidate_tcp_T_camera"], expected, atol=1e-7)
        self.assertLess(report["held_out_validation"]["max_translation_mm"], 1e-7)

    def test_small_measurement_noise_does_not_destroy_solution(self):
        expected, a, b = synthetic_motions()
        rng = np.random.default_rng(5)
        noisy = []
        for transform in b:
            transform = transform.copy()
            transform[:3, :3] = Rotation.from_rotvec(rng.normal(0, 0.0001, 3)).as_matrix() @ transform[:3, :3]
            transform[:3, 3] += rng.normal(0, 0.03, 3)
            noisy.append(transform)
        report = calibration.solve_motion_pairs(a, noisy)
        self.assertTrue(report["accepted"])
        actual = np.asarray(report["candidate_tcp_T_camera"])
        self.assertLess(np.linalg.norm(actual[:3, 3] - expected[:3, 3]), 0.5)

    def test_translation_only_and_single_rotation_axis_are_rejected(self):
        for only_translation in (True, False):
            with self.subTest(translation_only=only_translation):
                a = [vision.pose_to_matrix([i, 2 * i, i / 2, 0, 0, 0 if only_translation else i * 3])
                     for i in range(1, 9)]
                with self.assertRaisesRegex(ValueError, "姿态变化不足"):
                    calibration.solve_motion_pairs(a, a)

    def test_inconsistent_camera_motion_cannot_be_accepted(self):
        _, a, b = synthetic_motions()
        b = copy.deepcopy(b)
        b[3][:3, 3] += [40, -30, 20]
        report = calibration.solve_motion_pairs(a, b)
        self.assertFalse(report["accepted"])

    def test_too_few_samples_are_rejected(self):
        _, a, b = synthetic_motions()
        with self.assertRaisesRegex(ValueError, "至少需要"):
            calibration.solve_motion_pairs(a[:3], b[:3])

    def test_flat_scene_is_rejected(self):
        rng = np.random.default_rng(2)
        points = np.column_stack([rng.uniform(-100, 100, 2000), rng.uniform(-100, 100, 2000), np.zeros(2000)])
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        with self.assertRaisesRegex(ValueError, "一个平面"):
            calibration.check_scene_geometry(cloud)

    def test_duplicate_pose_is_rejected_before_accessing_camera(self):
        observation = {"tcp_pose": [100, 20, 300, 180, 0, 0], "tool_id": 0, "user_id": 0}
        with patch("markerless_handeye.read_manifest", return_value={"samples": [{"observation": observation}]}), \
             patch("markerless_handeye.vision.load_config", return_value={"camera": {"mount": "eye_in_hand"}}), \
             patch("markerless_handeye.vision.read_observation_pose", return_value=observation), \
             patch("markerless_handeye.vision.capture_point_cloud") as capture:
            with self.assertRaisesRegex(ValueError, "姿态已采过"):
                calibration.capture_sample(Path("unused"), Path("unused"))
            capture.assert_not_called()

    def test_dataset_solution_does_not_modify_live_robot_config(self):
        expected, robot, camera = synthetic_motions()
        poses = [np.eye(4), *robot]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            samples = []
            for index, pose in enumerate(poses):
                filename = f"sample_{index:03d}.ply"
                cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.eye(3)))
                o3d.io.write_point_cloud(str(path / filename), cloud)
                samples.append({"file": filename, "sha256": hashlib.sha256((path / filename).read_bytes()).hexdigest(),
                                "observation": {"tcp_pose": vision.matrix_to_pose(pose), "tool_id": 0, "user_id": 0}})
            (path / "manifest.json").write_text(json.dumps({"schema_version": 1, "samples": samples}), encoding="utf-8")
            with patch("markerless_handeye.register_scene_pair", side_effect=[(b, {"fitness": 1}) for b in camera]), \
                 patch("markerless_handeye.vision.save_config") as save:
                report = calibration.solve_dataset(path)
                save.assert_not_called()
            self.assertTrue(report["accepted"])
            self.assertTrue((path / "handeye_candidate.json").is_file())
            np.testing.assert_allclose(report["candidate_tcp_T_camera"], expected, atol=1e-7)

    def test_scene_registration_recovers_camera_motion_without_a_marker(self):
        o3d.utility.random.seed(9)
        mesh = o3d.geometry.TriangleMesh.create_box(100, 65, 40)
        sphere = o3d.geometry.TriangleMesh.create_sphere(18).translate([18, 12, 53])
        cone = o3d.geometry.TriangleMesh.create_cone(12, 35).translate([78, 48, 40])
        mesh += sphere
        mesh += cone
        reference = mesh.sample_points_uniformly(10000)
        expected = vision.pose_to_matrix([15, -8, 12, 5, -7, 9])
        current = copy.deepcopy(reference).transform(np.linalg.inv(expected))
        result, metrics = calibration.register_scene_pair(reference, current)
        error = np.linalg.inv(expected) @ result
        self.assertLess(np.linalg.norm(error[:3, 3]), 1.0)
        self.assertLess(np.degrees(Rotation.from_matrix(error[:3, :3]).magnitude()), 1.0)
        self.assertGreater(metrics["fitness"], 0.9)


if __name__ == "__main__":
    unittest.main()
