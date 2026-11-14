"""Live preview freshness and native XYZ handling, without camera/robot I/O."""
import copy
import multiprocessing as mp
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import numpy as np
import open3d as o3d

from camera_capture import PointCloudMailbox
from live_model_preview import match_frame
from pose_overlay import PoseOverlay
import pick_place_3d as vision

ROOT = Path(__file__).resolve().parents[1]


class LivePreviewTests(unittest.TestCase):
    def test_xyz_mailbox_copies_latest_frame_and_never_waits_on_busy_writer(self):
        mailbox = PointCloudMailbox(mp.get_context("spawn"), capacity=48)
        points = np.array([[1, 2, 300], [4, 5, 600]], dtype=np.float32)
        mailbox.publish(points)
        sequence, stamp, result = mailbox.read(0)
        points[:] = 0
        self.assertEqual(sequence, 1)
        self.assertLess(abs(time.time() - stamp), 1)
        self.assertEqual(result[0, 2], 300)
        mailbox.lock.acquire()
        try:
            self.assertIsNone(mailbox.read(0))
        finally:
            mailbox.lock.release()
        with self.assertRaises(ValueError):
            mailbox.publish(np.zeros((5, 3), np.float32))
        self.assertEqual(mailbox.read(0)[0], sequence)

    def test_live_matching_uses_optical_y_and_never_reads_or_commands_robot(self):
        config = vision.load_config(ROOT / "pick_place_config.json")
        config["_observation"] = {"old": "scan"}
        points = np.array([[x, 20, 300] for x in range(150)], dtype=float)
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        def capture_input(incoming, limits):
            np.testing.assert_allclose(np.asarray(incoming.points)[:, 1], -20)
            return incoming
        with patch.object(vision, "crop_cloud", side_effect=capture_input), \
             patch.object(vision, "remove_support_plane", return_value=(cloud, [], cloud)), \
             patch.object(vision, "cluster_objects", return_value=[cloud]), \
             patch.object(vision, "find_matches", return_value=[]) as matches, \
             patch.object(vision, "read_observation_pose", side_effect=AssertionError("robot I/O")), \
             patch("gripper_transport.send_command", side_effect=AssertionError("controller I/O")):
            result = match_frame(points, config, cloud)
        self.assertEqual(result["matches"], [])
        self.assertNotIn("_observation", matches.call_args.args[3])
        np.testing.assert_allclose(points[:, 1], 20)

    def test_live_mode_does_not_read_saved_candidates(self):
        overlay = PoseOverlay(ROOT / "nonexistent-preview-results", ROOT / "pick_place_config.json", live=True)
        with patch.object(overlay, "_load_assets"):
            panel = overlay.draw()
        self.assertEqual(panel.shape, (480, 380, 3))
        self.assertEqual(overlay.matches, [])
        self.assertEqual(overlay.error, "")

    def test_depth_changes_and_age_hide_previous_pose(self):
        overlay = PoseOverlay(ROOT, ROOT / "pick_place_config.json", live=True)
        overlay.camera = {"depth_projection": {"image_size": [80, 60],
                          "matrix": [[100, 0, 40], [0, 100, 30], [0, 0, 1]],
                          "color_T_camera": np.eye(4).tolist()}}
        overlay.model_vertices = np.array([[-20, -20, 300], [20, 20, 300]])
        source = np.full((60, 80), 300, dtype=np.uint16)
        result = {"matches": [{"camera_T_model": np.eye(4).tolist()}], "captured_unix": time.time()}
        overlay.set_live_result(result, source)
        self.assertTrue(overlay._live_overlay_valid(source.copy()))
        changed = source.copy(); changed[15:45, 25:55] += 15
        self.assertFalse(overlay._live_overlay_valid(changed))
        expired = copy.deepcopy(result); expired["captured_unix"] -= 20
        overlay.set_live_result(expired, source)
        self.assertFalse(overlay._live_overlay_valid(source))

    def test_slow_fit_can_draw_contours_on_unchanged_live_images(self):
        overlay = PoseOverlay(ROOT, ROOT / "pick_place_config.json", live=True)
        projection = {"image_size": [80, 60],
                      "matrix": [[100, 0, 40], [0, 100, 30], [0, 0, 1]],
                      "color_T_camera": np.eye(4).tolist()}
        overlay.camera = {"depth_projection": projection, "color_projection": projection}
        overlay.model_vertices = np.array([[-20,-20,300], [20,-20,300], [20,20,300], [-20,20,300]])
        overlay.model_triangles = np.array([[0,1,2], [0,2,3]])
        depth = np.full((60, 80), 300, dtype=np.uint16)
        result = {"matches": [{"camera_T_model": np.eye(4).tolist(), "score": 0.8, "rmse_mm": 1}],
                  "captured_unix": time.time() - 22, "duration_s": 22}
        overlay.set_live_result(result, depth)
        overlay.next_read = float("inf")
        color, depth_image = [np.zeros((60,80,3), np.uint8) for _ in range(2)]
        with patch("gripper_transport.send_command", side_effect=AssertionError("controller I/O")):
            overlay.draw(color, depth_image=depth_image, current_depth=depth)
        self.assertEqual(overlay.outline_count, 1)
        self.assertTrue(color.any())
        self.assertTrue(depth_image.any())
        changed = depth.copy(); changed[15:45, 25:55] += 15
        color[:] = 0; depth_image[:] = 0
        overlay.draw(color, depth_image=depth_image, current_depth=changed)
        self.assertEqual(overlay.outline_count, 0)
        self.assertFalse(color.any())

    def test_changed_part_does_not_hide_other_unchanged_contours(self):
        overlay = PoseOverlay(ROOT, ROOT / "pick_place_config.json", live=True)
        projection = {"image_size":[80,60], "matrix":[[100,0,40],[0,100,30],[0,0,1]],
                      "color_T_camera":np.eye(4).tolist()}
        overlay.camera = {"color_projection":projection,"depth_projection":projection}
        overlay.model_vertices = np.array([[-20,-20,300],[20,-20,300],[20,20,300],[-20,20,300]])
        overlay.model_triangles = np.array([[0,1,2],[0,2,3]])
        poses = []
        for x in (-60,60):
            pose = np.eye(4); pose[0,3] = x
            poses.append({"camera_T_model":pose.tolist(),"score":.8,"rmse_mm":1})
        source = np.full((60,80),300,np.uint16)
        overlay.set_live_result({"matches":poses,"captured_unix":time.time()},source)
        overlay.next_read = float("inf")
        current = source.copy(); current[15:45,5:35] += 15
        image = np.zeros((60,80,3),np.uint8)
        overlay.draw(image,current_depth=current)
        self.assertEqual(overlay.outline_count,1)
        self.assertFalse(image[:,:35].any())
        self.assertTrue(image[:,45:].any())


if __name__ == "__main__":
    unittest.main()
