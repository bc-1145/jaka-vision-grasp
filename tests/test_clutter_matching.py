"""Known-geometry and occlusion regressions; no hardware connections."""
import copy
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
from pathlib import Path
import unittest

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import clutter_matching as clutter
import pick_place_3d as vision

ROOT = Path(__file__).resolve().parents[1]


def cloud(points):
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))


class ClutterMatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = vision.load_config(ROOT / "pick_place_config.json")
        cls.config["matching"].update(strategy="clutter", local_candidate_budget=10)
        mesh = vision.load_fcstd_mesh(ROOT / cls.config["model"]["path"])
        o3d.utility.random.seed(11)
        cls.model = mesh.sample_points_uniformly(7000)
        cls.pose = np.eye(4)
        cls.pose[:3,:3] = Rotation.from_euler("xyz", [70,15,20], degrees=True).as_matrix()
        cls.pose[:3,3] = [0,0,330]
        cls.aligned = vision.transform_points(np.asarray(cls.model.points), cls.pose)
        complete = cloud(cls.aligned)
        _, visible = complete.hidden_point_removal([0,0,0], np.linalg.norm(cls.aligned,axis=1).max()*100)
        cls.observed = np.asarray(complete.select_by_index(visible).points)

    def test_partial_single_view_recovers_translation_and_axis(self):
        result = vision.register_model(self.model, cloud(self.observed), self.config,
                                       scene_cloud=cloud(self.observed))
        np.testing.assert_allclose(result.transformation[:3,3], self.pose[:3,3], atol=2)
        self.assertGreater(float(result.transformation[:3,2] @ self.pose[:3,2]), .99)
        self.assertTrue(vision.validate_model_support(self.model, cloud(self.observed),
                                                     result.transformation, self.config)["accepted"])

    def test_measured_foreground_excuses_occlusion_but_missing_depth_does_not(self):
        pose = np.eye(4)
        pose[:3,:3] = Rotation.from_euler("x",70,degrees=True).as_matrix()
        pose[:3,3] = [0,0,330]
        aligned = vision.transform_points(np.asarray(self.model.points),pose)
        complete = cloud(aligned)
        _,visible = complete.hidden_point_removal([0,0,0],np.linalg.norm(aligned,axis=1).max()*100)
        observed = aligned[visible]
        observed = observed[observed[:,0] < -1]
        x,y = np.meshgrid(np.arange(0,40,.5),np.arange(-80,40,.5))
        foreground = np.column_stack([x.ravel(),y.ravel(),np.full(x.size,300)])
        scene = cloud(np.vstack([observed,foreground]))
        validation = vision.validate_model_support(self.model, cloud(observed), pose, self.config,
                                                    visibility=clutter.DepthVisibility(scene,self.config))
        self.assertTrue(validation["accepted"], validation)
        unknown = vision.validate_model_support(self.model, cloud(observed), pose, self.config,
                                                visibility=clutter.DepthVisibility(cloud(observed),self.config))
        self.assertFalse(unknown["accepted"], unknown)

    def test_fully_hidden_head_cannot_establish_part_identity(self):
        model_points = np.asarray(self.model.points)
        observed = vision.transform_points(model_points[model_points[:,2] > 15], self.pose)
        x,y = np.meshgrid(np.arange(-40,40,.5),np.arange(-15,30,.5))
        foreground = np.column_stack([x.ravel(),y.ravel(),np.full(x.size,300)])
        scene = cloud(np.vstack([observed,foreground]))
        validation = vision.validate_model_support(self.model, cloud(observed), self.pose, self.config,
                                                    visibility=clutter.DepthVisibility(scene,self.config))
        self.assertFalse(validation["accepted"], validation)

    def test_axial_spin_duplicate_is_removed_but_neighbor_is_distinct(self):
        first = vision.MatchResult(0,.8,.8,1,self.pose.tolist(),1000,10)
        spin = self.pose.copy()
        spin[:3,:3] = spin[:3,:3] @ Rotation.from_euler("z",95,degrees=True).as_matrix()
        second = vision.MatchResult(1,.8,.8,1,spin.tolist(),1000,10)
        self.assertTrue(clutter.same_instance(first,second,self.model,self.config))
        spin[:3,3] += [0,35,0]
        second.camera_T_model = spin.tolist()
        self.assertFalse(clutter.same_instance(first,second,self.model,self.config))

    def test_fast_clearance_matches_full_scene_distance(self):
        aligned = self.aligned[::2]
        grasp = vision.transform_points([self.config["model"]["grasp_point_mm"]],self.pose)[0]
        points = np.vstack([aligned,grasp+[1,1,1],grasp+[0,0,4]])
        model = cloud(aligned)
        expected = vision.scene_clearance_mm(model,cloud(points),np.eye(4),grasp.tolist(),2.5)
        actual = clutter.nearest_foreign_clearance(aligned,cKDTree(points),grasp,2.5)
        self.assertAlmostEqual(actual,expected)

    def test_touching_parts_can_be_recovered_from_one_merged_cluster(self):
        pose = self.pose.copy(); pose[:3,3] += [0,35,0]
        other = vision.transform_points(np.asarray(self.model.points),pose)
        complete = cloud(np.vstack([self.aligned,other]))
        _, visible = complete.hidden_point_removal([0,0,0], np.linalg.norm(np.asarray(complete.points),axis=1).max()*100)
        scene = complete.select_by_index(visible)
        candidates = [scene]
        config = copy.deepcopy(self.config)
        matches = vision.find_matches(self.model,scene,candidates,config)
        self.assertEqual(len(matches),2,config.get("_matching_diagnostics"))
        for expected in (self.pose,pose):
            self.assertLess(min(np.linalg.norm(np.asarray(m.camera_T_model)[:3,3]-expected[:3,3]) for m in matches),3)
        self.assertTrue(all(m.cluster_index < len(candidates) for m in matches))

    def test_previous_pose_is_discarded_when_object_disappears(self):
        hint = vision.MatchResult(0,.9,.9,.2,self.pose.tolist(),1000,20)
        empty_bin = cloud(np.array([[x,y,800] for x in range(20) for y in range(20)]))
        config = copy.deepcopy(self.config)
        matches = vision.find_matches(self.model,empty_bin,[],config,previous_matches=[hint])
        self.assertEqual(matches,[])
        self.assertTrue(config["_matching_statistics"]["tracking_fallback"])

    def test_headless_shaft_cannot_be_accepted_as_complete_part(self):
        model_points = np.asarray(self.model.points)
        shaft = cloud(vision.transform_points(model_points[model_points[:,2] > 15],self.pose))
        config = copy.deepcopy(self.config)
        config["matching"]["local_candidate_budget"] = 4
        self.assertEqual(vision.find_matches(self.model,shaft,[shaft],config),[])

    def test_flat_patch_cannot_supply_shaft_curvature(self):
        x,z = np.meshgrid(np.linspace(-5,5,30),np.linspace(8,45,40))
        plane = np.column_stack([x.ravel(),np.full(x.size,10.5),z.ravel()])
        result = clutter.validate_shaft_curvature(cloud(plane),np.eye(4),self.config)
        self.assertFalse(result["curvature_accepted"],result)
        genuine = clutter.validate_shaft_curvature(cloud(self.observed),self.pose,self.config)
        self.assertTrue(genuine["curvature_accepted"],genuine)


if __name__ == "__main__":
    unittest.main()
