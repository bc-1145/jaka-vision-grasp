"""Geometry regressions for CAD overlays and rejection of partial tool fits."""
import copy
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

import pick_place_3d as vision
from pose_overlay import PoseOverlay


ROOT = Path(__file__).resolve().parents[1]


class PoseMatchingRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = vision.load_config(ROOT / "pick_place_config.json")
        mesh = vision.load_fcstd_mesh(ROOT / cls.config["model"]["path"])
        o3d.utility.random.seed(11)
        cls.model = mesh.sample_points_uniformly(7000)
        cls.transform = np.eye(4)
        cls.transform[:3, :3] = Rotation.from_euler("xyz", [70, 15, 20], degrees=True).as_matrix()
        cls.transform[:3, 3] = [0, -60, 400]

    def test_visible_bolt_head_and_shaft_are_accepted(self):
        aligned = vision.transform_points(np.asarray(self.model.points), self.transform)
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(aligned))
        _, visible = cloud.hidden_point_removal([0, 0, 0], np.linalg.norm(aligned, axis=1).max() * 100)
        observed = cloud.select_by_index(visible)
        result = vision.validate_model_support(self.model, observed, self.transform, self.config)
        self.assertTrue(result["accepted"], result)

    def test_shaft_only_patch_cannot_identify_the_bolt(self):
        points = np.asarray(self.model.points)
        shaft = points[points[:, 2] > 15]
        observed = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
            vision.transform_points(shaft, self.transform)))
        result = vision.validate_model_support(self.model, observed, self.transform, self.config)
        self.assertFalse(result["accepted"], result)
        self.assertLess(result["head_support"], 0.1)

    def test_tool_exclusion_keeps_surrounding_workpiece_points(self):
        config = copy.deepcopy(self.config)
        config["camera"]["excluded_boxes_mm"] = [{"min_mm": [-1, -1, -1], "max_mm": [1, 1, 1]}]
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector([[0, 0, 0], [2, 0, 0], [0, 0, 3]]))
        filtered = vision.exclude_camera_geometry(cloud, config)
        np.testing.assert_array_equal(np.asarray(filtered.points), [[2, 0, 0], [0, 0, 3]])

    def test_low_score_does_not_enter_saved_candidates(self):
        config = copy.deepcopy(self.config)
        config["matching"]["strategy"] = "legacy"
        config["camera"]["excluded_boxes_mm"] = []
        registration = vision.RegistrationResult(0.1, 2.0, np.eye(4))
        with patch.object(vision, "_extent_is_plausible", return_value=True), \
             patch.object(vision, "register_model", return_value=registration), \
             patch.object(vision, "validate_model_support", return_value={"accepted": True}), \
             patch.object(vision, "scene_clearance_mm", return_value=30.0):
            self.assertEqual(vision.find_matches(self.model, self.model, [self.model], config), [])

    def test_projected_mesh_preserves_step_instead_of_convex_hull(self):
        overlay = PoseOverlay(ROOT / "pick_results", ROOT / "pick_place_config.json")
        # Two rectangles form a stepped head and shaft in the projected plane.
        overlay.model_vertices = np.array([[-30,-20,300],[30,-20,300],[30,0,300],[-30,0,300],
                                           [-10,0,300],[10,0,300],[10,60,300],[-10,60,300]], dtype=float)
        overlay.model_triangles = np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7]])
        overlay.camera = {"color_projection": {"image_size": [400,400],
                          "matrix": [[600,0,200],[0,600,100],[0,0,1]],
                          "color_T_camera": np.eye(4).tolist()}}
        overlay.matches = [{"camera_T_model": np.eye(4).tolist()}]
        image = np.zeros((400,400,3), dtype=np.uint8)
        overlay._axes(image)
        # True step at x=180 exists; a hull would draw a diagonal across x=160 here.
        self.assertTrue(image[150:170,178:183].any())
        self.assertFalse(image[150:170,155:165].any())


if __name__ == "__main__":
    unittest.main()
