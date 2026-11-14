import json
import unittest
import numpy as np
from bin_target_report import build_target_report


class TargetReportTests(unittest.TestCase):
    def fixture(self):
        config = dict(camera=dict(mount="eye_in_hand", calibration_tool_id=0,
            point_cloud_frame="opencv_optical", tcp_T_camera=np.eye(4).tolist()),
            gripper=dict(tcp_to_grasp_center_mm=[10, 20, 30], close_command_mm=8.65),
            model=dict(shaft_z_range_mm=[3.6, 51.2], large_end_z_mm=0))
        observation = dict(tcp_pose=[100, 200, 300, 0, 0, 90], tool_id=0,
                           user_id=0, camera_frame="opencv_optical")
        transform = np.eye(4)
        transform[:3, 3] = [40, 50, 60]
        match = dict(camera_T_model=transform, score=.95, cluster_index=2)
        return config, observation, match

    def test_origins_and_rotated_axes(self):
        report = build_target_report(*self.fixture())
        np.testing.assert_allclose(report["target"]["center"]["tcp_origin_mm"], [40, 50, 87.4])
        np.testing.assert_allclose(report["target"]["center"]["grasp_origin_tcp_axes_mm"], [30, 30, 57.4])
        np.testing.assert_allclose(report["target"]["center"]["base_mm"], [50, 240, 387.4])
        self.assertAlmostEqual(report["downward_depth_from_current_contact_mm"], -57.4)
        self.assertFalse(report["gripper"]["command_is_measured_gap"])
        self.assertIsNone(report["gripper"]["actual_jaw_gap_mm"])
        json.dumps(report, allow_nan=False)

    def test_failure_preserves_geometry(self):
        report = build_target_report(*self.fixture(), planning_error="IK unavailable")
        self.assertFalse(report["executable"])
        self.assertEqual(report["planning_error"], "IK unavailable")
        self.assertIn("large_head", report["target"])
        self.assertTrue(all(not item["executable"] for item in report["flange_tilt_candidates"]))

    def test_plan_native_poses_and_vertical_descent(self):
        plan = dict(poses=dict(prepick=[100, 200, 400, 0, 0, 90],
                               pick=[100, 200, 350, 0, 0, 90]),
                    route_checks={"checked": True}, motion_modes={"pick": "MoveL"})
        report = build_target_report(*self.fixture(), plan=plan)
        self.assertTrue(report["executable"])
        self.assertEqual(report["motion"]["poses"], plan["poses"])
        self.assertEqual(report["planned_descent"]["depth_mm"], 50)
        self.assertTrue(report["planned_descent"]["base_vertical"])

    def test_head_direction_rotates_with_model(self):
        config, observation, match = self.fixture()
        match["camera_T_model"][:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
        report = build_target_report(config, observation, match)
        np.testing.assert_allclose(report["shaft_axis_tcp"], [1, 0, 0], atol=1e-10)
        np.testing.assert_allclose(report["head_relative_target_center_tcp_axes_mm"], [-27.4, 0, 0])


if __name__ == "__main__":
    unittest.main()
