import json
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
try:
    import open3d as o3d
    from scipy.spatial.transform import Rotation
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("3D tests require the project .camera-runtime") from exc

import pick_place_3d as vision


ROOT = Path(__file__).resolve().parents[1]


class CadAndMatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = vision.load_config(ROOT / "pick_place_config.json")
        cls.mesh = vision.load_fcstd_mesh(ROOT / "model.FCStd")

    def test_fcstd_final_body_dimensions(self):
        extent = self.mesh.get_axis_aligned_bounding_box().get_extent()
        np.testing.assert_allclose(extent, [34.0, 34.0, 51.2], atol=0.05)

    def test_pose_matrix_round_trip_handles_wrapped_angles(self):
        pose = [-90.0, -180.0, 280.0, -180.0, 0.0, 90.0]
        first = vision.pose_to_matrix(pose)
        second = vision.pose_to_matrix(vision.matrix_to_pose(first))
        np.testing.assert_allclose(first, second, atol=1e-8)

    def test_synthetic_cad_registration(self):
        o3d.utility.random.seed(7)
        source = self.mesh.sample_points_uniformly(5000)
        expected = np.eye(4)
        expected[:3, :3] = Rotation.from_euler(
            "xyz", [27.0, -18.0, 43.0], degrees=True).as_matrix()
        expected[:3, 3] = [53.0, -21.0, 260.0]
        target = o3d.geometry.PointCloud(source)
        target.transform(expected)
        points = np.asarray(target.points)
        points += np.random.default_rng(2).normal(0.0, 0.15, points.shape)
        target.points = o3d.utility.Vector3dVector(points)

        result = vision.register_model(source, target, self.config)

        self.assertGreater(result.fitness, 0.95)
        self.assertLess(result.inlier_rmse, 0.5)
        np.testing.assert_allclose(result.transformation[:3, 3], expected[:3, 3], atol=0.5)

    def test_partial_single_view_registration(self):
        o3d.utility.random.seed(5)
        source = self.mesh.sample_points_uniformly(7000)
        expected = np.eye(4)
        expected[:3, :3] = Rotation.from_euler(
            "xyz", [72.0, 9.0, -35.0], degrees=True).as_matrix()
        expected[:3, 3] = [-120.0, 40.0, 330.0]
        complete = o3d.geometry.PointCloud(source)
        complete.transform(expected)
        points = np.asarray(complete.points)
        points = points[points[:, 2] <= np.quantile(points[:, 2], 0.45)]
        points += np.random.default_rng(3).normal(0.0, 0.18, points.shape)
        observed = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

        result = vision.register_model(source, observed, self.config)

        self.assertGreater(result.fitness, 0.90)
        self.assertLess(result.inlier_rmse, 1.0)
        np.testing.assert_allclose(result.transformation[:3, 3], expected[:3, 3], atol=1.0)

    def test_execution_stays_blocked_without_physical_calibration(self):
        config = self.uncalibrated_config()
        missing = vision.missing_execution_calibration(config)
        self.assertIn("camera.tcp_T_camera", missing)
        self.assertIn("gripper.tcp_to_grasp_center_mm", missing)
        self.assertIn("gripper.measured_max_object_mm", missing)

    def uncalibrated_config(self):
        # Tests must not depend on whether the operator has calibrated this site.
        config = copy.deepcopy(self.config)
        config["camera"]["tcp_T_camera"] = None
        config["gripper"].update(tcp_to_grasp_center_mm=None, measured_max_object_mm=None)
        return config

    def eye_in_hand_config(self):
        config = copy.deepcopy(self.config)
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler("xyz", [8, -13, 35], degrees=True).as_matrix()
        transform[:3, 3] = [35, -20, 80]
        config["camera"].update(mount="eye_in_hand", tcp_T_camera=transform.tolist(), calibration_tool_id=0)
        config["gripper"].update(tcp_to_grasp_center_mm=[0, 0, 100], measured_max_object_mm=30)
        config["placement"].update(mode="home_tcp", platform_z_mm=None)
        return config

    def observation(self):
        return {"tcp_pose": [192, -60, 280, 178, 0.5, -179], "tool_id": 0, "user_id": 0,
                "camera_frame": self.config["camera"].get("point_cloud_frame")}

    def test_eye_in_hand_uses_acquisition_pose_and_tool_relative_transform(self):
        config = self.eye_in_hand_config()
        observation = self.observation()
        expected = vision.pose_to_matrix(observation["tcp_pose"]) @ config["camera"]["tcp_T_camera"]
        with patch("arm_control.get_tcp", side_effect=AssertionError("must not read the live pose")):
            np.testing.assert_allclose(vision.camera_transform(config, observation), expected)

    def test_old_or_wrong_tool_match_cannot_use_eye_in_hand_calibration(self):
        config = self.eye_in_hand_config()
        with self.assertRaisesRegex(RuntimeError, "no recorded acquisition pose"):
            vision.camera_transform(config)
        observation = self.observation()
        observation["tool_id"] = 2
        with self.assertRaisesRegex(RuntimeError, "active robot tool"):
            vision.camera_transform(config, observation)

    def test_robot_motion_during_acquisition_is_rejected(self):
        before = self.observation()
        after = copy.deepcopy(before)
        after["tcp_pose"][0] += 2
        with self.assertRaisesRegex(RuntimeError, "robot moved"):
            vision.verify_observation_unchanged(before, after)
        after = copy.deepcopy(before)
        after["tool_id"] = 1
        with self.assertRaisesRegex(RuntimeError, "frame changed"):
            vision.verify_observation_unchanged(before, after)

    def test_home_target_is_exact_tcp_pose_without_assuming_table_height(self):
        config = self.eye_in_hand_config()
        match = vision.MatchResult(0, 0.9, 0.9, 0.5, np.eye(4).tolist(), 1000, 30,
                                   observation=self.observation())
        plan = vision.plan_pick_place(match, config)
        np.testing.assert_allclose(vision.pose_to_matrix(plan["place"]),
                                   vision.pose_to_matrix(config["placement"]["home_pose"]), atol=1e-9)
        self.assertEqual(plan["observation"], match.observation)

    def test_fixed_camera_transform_still_works(self):
        config = self.eye_in_hand_config()
        config["camera"].update(mount="fixed", base_T_camera=np.eye(4).tolist())
        np.testing.assert_allclose(vision.camera_transform(config), np.eye(4))

    def test_old_native_frame_cannot_use_optical_handeye(self):
        config = self.eye_in_hand_config()
        config["camera"]["point_cloud_frame"] = "opencv_optical"
        observation = self.observation()
        observation.pop("camera_frame", None)
        with self.assertRaisesRegex(RuntimeError, "coordinate frame changed"):
            vision.camera_transform(config, observation)

    def test_scaled_or_reflected_calibration_matrix_is_rejected(self):
        for factor in (2, -1):
            transform = np.eye(4)
            transform[0, 0] = factor
            with self.assertRaisesRegex(ValueError, "proper rotation"):
                vision.matrix(transform, "test")

    def test_clearance_counts_foreign_surfaces_inside_same_cluster(self):
        angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        points = np.column_stack([10 * np.cos(angles), 10 * np.sin(angles), np.zeros(100)])
        model = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        scene = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack([points, [0, 0, 3]])))
        self.assertAlmostEqual(vision.scene_clearance_mm(model, scene, np.eye(4), [0, 0, 0], 0.5), 3.0)

    def test_clearance_counts_unclustered_points_and_support_surface(self):
        points = np.array([[10, 0, 0], [-10, 0, 0], [0, 10, 0], [0, -10, 0]], dtype=float)
        model = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        transform = np.eye(4)
        transform[:3, 3] = [90, -20, 250]
        shifted = vision.transform_points(points, transform)
        scene = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
            np.vstack([shifted, [90, -20, 245], [90, -20, 252]])))
        self.assertAlmostEqual(vision.scene_clearance_mm(model, scene, transform, [0, 0, 0], 0.5), 2.0)

    def test_clearance_ignores_selected_model_surface(self):
        points = np.array([[10, 0, 0], [-10, 0, 0], [0, 10, 0], [0, -10, 0]], dtype=float)
        model = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        self.assertEqual(vision.scene_clearance_mm(model, model, np.eye(4), [0, 0, 0], 0.5), 100.0)

    def test_execution_with_missing_calibration_never_calls_motion_or_grippers(self):
        with patch("pick_place_3d._move_verified") as move, patch("pick_place_3d._load_grippers") as grippers:
            with self.assertRaisesRegex(RuntimeError, "calibration is incomplete"):
                vision.execute_plan({"route": "direct"}, self.uncalibrated_config())
            move.assert_not_called()
            grippers.assert_not_called()

    def draft_grasp_plan(self):
        config = copy.deepcopy(self.config)
        config["camera"].update(mount="eye_in_hand", tcp_T_camera=None, calibration_tool_id=None)
        model = self.mesh.sample_points_uniformly(1500)
        first, second = np.eye(4), np.eye(4)
        first[:3, 3], second[:3, 3] = [-60, 0, 300], [60, 0, 300]
        first_points = vision.transform_points(np.asarray(model.points), first)
        second_points = vision.transform_points(np.asarray(model.points), second)
        scene = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack([
            first_points, second_points, [-60, 0, 320], [-60, 0, 327], [-60, 0, 334]])))
        matches = [vision.MatchResult(index, 0.9, 0.95, 0.4, transform.tolist(), 1500, 100,
                                     observation=self.observation())
                   for index, transform in enumerate((first, second))]
        return vision.generate_grasp_plan(matches, config, scene, model), config

    def test_preview_shaft_grasps_are_generated_without_inventing_robot_coordinates(self):
        plan, config = self.draft_grasp_plan()
        self.assertEqual(plan["status"], "preview")
        self.assertFalse(plan["executable"])
        self.assertIsNone(plan["pick"])
        self.assertIsNone(plan["pick_approach"])
        self.assertEqual(plan["close_command_mm"], config["gripper"]["close_command_mm"])
        self.assertEqual(plan["grasp_width_mm"], 21)
        self.assertEqual(plan["place"], config["placement"]["home_pose"])
        self.assertEqual(plan["selected_grasp"]["cluster_index"], 1)
        self.assertIn("camera.tcp_T_camera", plan["missing_calibration"])
        for candidate in plan["shaft_grasp_candidates"]:
            self.assertEqual(candidate["grasp_region"], "shaft")
            self.assertGreater(candidate["grasp_point_model_mm"][2], config["model"]["head_z_range_mm"][1])
            self.assertLess(candidate["grasp_point_model_mm"][2], config["model"]["shaft_z_range_mm"][1])

    def test_preview_plan_is_never_executed_even_if_calibration_is_later_added(self):
        plan, _ = self.draft_grasp_plan()
        with patch("arm_control.get_status_snapshot") as robot, patch("pick_place_3d._load_grippers") as grippers:
            with self.assertRaisesRegex(RuntimeError, "preview grasp plan cannot execute"):
                vision.execute_plan(plan, self.eye_in_hand_config())
            robot.assert_not_called()
            grippers.assert_not_called()

    def test_draft_plan_writes_json_and_readable_shaft_grasp_instructions(self):
        plan, config = self.draft_grasp_plan()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            vision.write_grasp_plan(path, plan)
            saved = json.loads((path / "plan.json").read_text(encoding="utf-8"))
            self.assertFalse(saved["executable"])
            text = (path / "grasp_plan.md").read_text(encoding="utf-8")
            self.assertIn("杆身", text)
            self.assertIn(str(config["gripper"]["close_command_mm"]), text)
            self.assertIn("相机坐标", text)

    def test_shaft_region_overlapping_large_head_is_rejected(self):
        config = copy.deepcopy(self.config)
        config["model"]["shaft_z_range_mm"] = [1.0, 30.0]
        with self.assertRaisesRegex(ValueError, "overlaps the bolt head"):
            vision.generate_grasp_plan([], config, None, None)

    def test_rigid_transform_recovers_camera_extrinsic(self):
        source = np.array([[0, 0, 0], [100, 0, 0], [0, 80, 0], [30, 20, 50]], dtype=float)
        expected = np.eye(4)
        expected[:3, :3] = Rotation.from_euler("xyz", [4, -8, 31], degrees=True).as_matrix()
        expected[:3, 3] = [-90, -180, 70]
        target = vision.transform_points(source, expected)

        actual, residuals = vision.rigid_transform(source, target)

        np.testing.assert_allclose(actual, expected, atol=1e-9)
        np.testing.assert_allclose(residuals, 0, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
