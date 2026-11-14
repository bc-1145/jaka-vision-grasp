"""Highest-score selection must fail closed instead of substituting another part."""
import unittest
from types import SimpleNamespace
import numpy as np

from side_pick_execute import select_highest_score_match, shaft_middle_model_z, grasp_rotation, flange_position


def match(index, score, clearance):
    return SimpleNamespace(cluster_index=index, score=score, clearance_mm=clearance)


class HighestScorePickTests(unittest.TestCase):
    def test_highest_score_wins_over_larger_clearance(self):
        highest = match(19, 0.85, 12.0)
        self.assertIs(select_highest_score_match([match(4, 0.80, 30.0), highest]), highest)

    def test_unsafe_highest_score_does_not_fall_back(self):
        with self.assertRaisesRegex(RuntimeError, "未改选其他零件"):
            select_highest_score_match([match(4, 0.80, 30.0), match(19, 0.85, 9.0)])

    def test_requested_graspable_policy_selects_highest_among_clear_targets(self):
        highest_clear = match(4, 0.80, 15.0)
        selected = select_highest_score_match(
            [match(19, 0.85, 4.0), match(9, 0.79, 30.0), highest_clear], highest_graspable=True)
        self.assertIs(selected, highest_clear)

    def test_graspable_policy_preserves_score_and_clearance_thresholds(self):
        with self.assertRaisesRegex(RuntimeError, "没有同时满足"):
            select_highest_score_match([match(19, 0.85, 4.0), match(4, 0.77, 30.0)],
                                      highest_graspable=True)

    def test_low_score_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "未通过抓取门槛"):
            select_highest_score_match([match(4, 0.77, 30.0)])

    def test_nonfinite_scores_cannot_affect_ranking(self):
        for score in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(score=score), self.assertRaisesRegex(RuntimeError, "分数无效"):
                select_highest_score_match([match(4, 0.80, 30.0), match(19, score, 12.0)])

    def test_nonfinite_clearance_is_rejected(self):
        for clearance in (float("nan"), float("inf")):
            with self.subTest(clearance=clearance), self.assertRaisesRegex(RuntimeError, "未通过"):
                select_highest_score_match([match(19, 0.85, clearance)])

    def test_empty_scan_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "没有可供抓取"):
            select_highest_score_match([])

    def test_grasp_is_fixed_at_tube_middle(self):
        config = {"model": {"shaft_z_range_mm": [3.6, 51.2], "head_z_range_mm": [0, 3.2]}}
        self.assertAlmostEqual(shaft_middle_model_z(config), 27.4)

    def test_invalid_shaft_range_cannot_define_a_grasp(self):
        for limits in ([51.2, 3.6], [3.6, 3.6], [float("nan"), 51.2]):
            config = {"model": {"shaft_z_range_mm": limits, "head_z_range_mm": [0, 3.2]}}
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                shaft_middle_model_z(config)

    def test_physical_y_jaw_axis_is_perpendicular_to_shaft(self):
        approach = np.array([0, 0, -1.0])
        shaft = np.array([1.0, 0, 0])
        desired_jaw = np.cross(shaft, approach)
        rotation = grasp_rotation(approach, desired_jaw, [0, 1, 0])
        np.testing.assert_allclose(rotation @ [0, 1, 0], desired_jaw)
        np.testing.assert_allclose(rotation @ [0, 0, 1], approach)
        self.assertAlmostEqual(np.linalg.det(rotation), 1)

    def test_contact_offset_includes_lateral_mounting_offset(self):
        rotation = grasp_rotation([0, 0, -1], [0, 1, 0], [0, 1, 0])
        center = np.array([43.0, -536.0, 84.0])
        offset = np.array([-22.0, 0, 166.0])
        flange = flange_position(center, rotation, offset)
        np.testing.assert_allclose(flange + rotation @ offset, center)

    def test_parallel_tool_and_jaw_axes_are_rejected(self):
        with self.assertRaises(ValueError):
            grasp_rotation([0, 0, 1], [0, 0, 1], [1, 0, 0])


if __name__ == "__main__":
    unittest.main()
