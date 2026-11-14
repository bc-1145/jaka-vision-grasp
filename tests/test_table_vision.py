"""Visual supervision must reject empty tools, reversed ends and fallen parts."""
import unittest
from unittest.mock import patch
from pathlib import Path
import json
import numpy as np
import cv2

import table_vision as visual


def geometry():
    angles = np.linspace(0, 2*np.pi, 64, endpoint=False)
    vertices = np.concatenate([np.c_[radius*np.cos(angles), radius*np.sin(angles), np.full(64, z)]
                               for z, radius in ((0, 17), (3.2, 17), (3.6, 10.5), (51.2, 10.5))])
    return vertices, 10.5, 17., 51.2


CONFIG = {"model": {"path": "unused.FCStd", "grasp_width_mm": 21,
                    "head_z_range_mm": [0, 3.2], "shaft_z_range_mm": [3.6, 51.2]},
          "camera": {"color_projection": {"image_size": [640, 480],
                     "matrix": [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
                     "distortion": [0, 0, 0, 0, 0], "color_T_camera": np.eye(4).tolist()}}}


def upright_cloud():
    angles = np.linspace(0, 2*np.pi, 64, endpoint=False)
    points = np.concatenate([np.c_[10.5*np.cos(angles), 10.5*np.sin(angles), np.full(64, z)]
                             for z in np.linspace(5, 51.2, 50)])
    flange = np.concatenate([np.c_[r*np.cos(angles), r*np.sin(angles), np.full(64, 2.7)]
                             for r in (13, 15, 17)])
    return np.vstack((points, flange)) + [40, -530, 67]


def images(background_shift=(8, 6), move_part=False):
    rng = np.random.default_rng(731)
    background = rng.integers(155, 235, (480, 640), dtype=np.uint8)
    background = cv2.GaussianBlur(background, (3, 3), 0)
    transform = np.eye(4)
    angle = np.deg2rad(80)
    transform[:3, :3] = [[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]
    transform[:3, 3] = [0, 0, 250]
    vertices = visual._transform(geometry()[0], transform)
    uv = cv2.projectPoints(vertices, np.zeros(3), np.zeros(3),
                           np.array(CONFIG["camera"]["color_projection"]["matrix"], float), np.zeros(5))[0]
    mask = np.zeros(background.shape, np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(np.round(uv).astype(np.int32)), 255)
    texture = rng.integers(25, 90, background.shape, dtype=np.uint8)
    first = background.copy()
    first[mask > 0] = texture[mask > 0]
    shift = np.array([[1, 0, background_shift[0]], [0, 1, background_shift[1]]], float)
    second = cv2.warpAffine(background, shift, (640, 480), borderMode=cv2.BORDER_REFLECT)
    if move_part:
        moved_mask = cv2.warpAffine(mask, shift, (640, 480))
        moved_texture = cv2.warpAffine(texture, shift, (640, 480))
        second[moved_mask > 0] = moved_texture[moved_mask > 0]
    else:
        second[mask > 0] = texture[mask > 0]
    return [cv2.cvtColor(value, cv2.COLOR_GRAY2BGR) for value in (first, second, background)], transform


class TableVisionTests(unittest.TestCase):
    def setUp(self):
        self.model = patch.object(visual, "_dimensions", side_effect=lambda config: geometry())
        self.model.start()
        self.addCleanup(self.model.stop)
        self.pose = np.eye(4)
        self.pose[:3, 3] = [40, -530, 67]
        self.plane = {"plane_base": [0, 0, 1, -67]}

    def test_empty_jaws_cannot_prove_presence(self):
        (part, _, empty), transform = images()
        self.assertTrue(visual.verify_rgb_presence(part, transform, CONFIG, empty_image=empty)["accepted"])
        self.assertFalse(visual.verify_rgb_presence(empty, transform, CONFIG, empty_image=empty)["accepted"])

    def test_missing_reference_is_unknown(self):
        (part, _, _), transform = images()
        result = visual.verify_rgb_presence(part, transform, CONFIG)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["status"], "unknown")

    def test_reflective_silver_part_uses_shape_not_dark_threshold(self):
        (_, _, empty), transform = images()
        empty[:] = 190
        vertices = geometry()[0]
        xyz = visual._transform(vertices, transform)
        uv = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3),
                               np.array(CONFIG["camera"]["color_projection"]["matrix"], float), np.zeros(5))[0]
        mask = np.zeros(empty.shape[:2], np.uint8)
        for lo, hi in ((0, 3.2), (3.6, 51.2)):
            selected = (vertices[:, 2] >= lo-.01) & (vertices[:, 2] <= hi+.01)
            cv2.fillConvexPoly(mask, cv2.convexHull(np.round(uv[selected]).astype(np.int32)), 255)
        image = empty.copy()
        yy, xx = np.indices(mask.shape)
        surface = np.clip(178+55*np.sin(xx*.13)+10*np.sin(yy*.2), 115, 240).astype(np.uint8)
        image[mask > 0] = surface[mask > 0, None]
        border = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        image[border] = 100
        result = visual.verify_rgb_presence(image, transform, CONFIG, empty_image=empty)
        self.assertTrue(result["accepted"], result)
        self.assertEqual(result["head_appearance_method"], "shape_and_reference_change")
        self.assertEqual(result["shaft_appearance_method"], "shape_and_reference_change")
        self.assertFalse(result["confirms_physical_grasp"])

    def test_uniform_bright_background_is_not_a_silver_part(self):
        (_, _, empty), transform = images()
        image = np.full_like(empty, 230)
        reference = np.full_like(empty, 170)
        self.assertFalse(visual.verify_rgb_presence(image, transform, CONFIG, empty_image=reference)["accepted"])

    def test_dark_tool_itself_does_not_count_as_part(self):
        (part, _, _), transform = images()
        self.assertFalse(visual.verify_rgb_presence(part, transform, CONFIG, empty_image=part)["accepted"])

    def test_measured_wide_end_supports_only_one_direction(self):
        points = upright_cloud()
        self.assertTrue(visual.flange_direction_evidence(points, self.pose, self.plane, CONFIG)["accepted"])
        reverse = self.pose.copy()
        reverse[:3, :3] = np.diag([-1., 1., -1.])
        reverse[:3, 3] += [0, 0, 51.2]
        result = visual.flange_direction_evidence(points, reverse, self.plane, CONFIG)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["status"], "reversed")

    def test_horizontal_direction_excludes_near_table_flying_points(self):
        local = upright_cloud()-[40, -530, 67]
        rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)
        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = [40, -530, 84.5]
        flying = np.c_[np.full(600, 15.1), np.linspace(-4, 4, 600), np.full(600, 49)]
        cloud = np.vstack((local, flying)) @ rotation.T + pose[:3, 3]
        result = visual.flange_direction_evidence(cloud, pose, self.plane, CONFIG)
        self.assertTrue(result["accepted"], result)
        self.assertEqual(result["table_exclusion_mm"], 3)

    def test_upright_direction_keeps_thin_lower_flange(self):
        result = visual.flange_direction_evidence(upright_cloud(), self.pose, self.plane, CONFIG)
        self.assertTrue(result["accepted"], result)
        self.assertEqual(result["table_exclusion_mm"], 1.5)

    def test_upright_cylinder_with_lower_flange_is_verified(self):
        result = visual.verify_upright(upright_cloud(), self.pose, self.plane, CONFIG)
        self.assertTrue(result["accepted"], result)

    def test_upright_without_distinct_lower_flange_is_not_verified(self):
        points = upright_cloud()[:3200]
        self.assertFalse(visual.verify_upright(points, self.pose, self.plane, CONFIG)["accepted"])

    def test_fallen_part_is_rejected(self):
        cloud = upright_cloud() - [40, -530, 67]
        cloud = cloud @ np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float).T + [40, -530, 77.5]
        result = visual.verify_upright(cloud, self.pose, self.plane, CONFIG)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["status"], "not_upright")

    def test_part_fixed_to_camera_with_moving_table_follows_lift(self):
        (first, second, empty), transform = images()
        result = visual.verify_follow_motion(first, second, transform, CONFIG, empty_image=empty)
        self.assertTrue(result["accepted"], result)

    def test_stationary_background_cannot_establish_lift(self):
        (first, second, empty), transform = images(background_shift=(0, 0))
        self.assertFalse(visual.verify_follow_motion(first, second, transform, CONFIG, empty_image=empty)["accepted"])

    def test_part_moving_with_table_is_not_held(self):
        (first, second, empty), transform = images(move_part=True)
        self.assertFalse(visual.verify_follow_motion(first, second, transform, CONFIG, empty_image=empty)["accepted"])

    def test_plane_is_transformed_into_robot_base_coordinates(self):
        rng = np.random.default_rng(18)
        points = np.c_[rng.uniform(-200, 200, 2500), rng.uniform(-180, 180, 2500),
                       rng.normal(300, .15, 2500)]
        transform = np.eye(4)
        transform[:3, :3] = np.diag([1, -1, -1])
        transform[:3, 3] = [40, -530, 367]
        result = visual.estimate_table_plane(points, transform, [40, -530])
        self.assertTrue(result["accepted"], result)
        self.assertAlmostEqual(result["height_at_grasp_xy_mm"], 67, delta=.1)


ROOT = Path(__file__).resolve().parents[1]
HELD = ROOT / "pick_results/table_upright/after_lift/grasp_verify_color_20261115_001302.png"
EMPTY = ROOT / "pick_results/table_upright/after_lift/grasp_verify_color_20261114_234812.png"
PLAN = ROOT / "pick_results/table_upright/stand_plan.json"


@unittest.skipUnless(HELD.exists() and EMPTY.exists() and PLAN.exists(), "Recorded hardware examples unavailable")
class RecordedImageRegressionTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "pick_place_config.json").read_text(encoding="utf-8-sig"))
        self.open_image = ROOT / "pick_results/table_upright/overview/grasp_verify_color_20261114_232409.png"

    def test_previous_success_and_empty_grasp_are_distinguished(self):
        config = json.loads((ROOT / "pick_place_config.json").read_text(encoding="utf-8-sig"))
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
        expected = np.linalg.inv(np.asarray(config["camera"]["tcp_T_camera"])) @ np.asarray(plan["held_model"])
        held = visual.verify_rgb_presence(HELD, expected, config, empty_image=EMPTY)
        empty = visual.verify_rgb_presence(EMPTY, expected, config, empty_image=EMPTY)
        self.assertTrue(held["accepted"], held)
        self.assertFalse(held["confirms_physical_grasp"])
        self.assertFalse(empty["accepted"], empty)

    def test_known_open_and_closed_tools_are_visually_empty(self):
        for image in (self.open_image, EMPTY):
            result = visual.verify_empty_jaws(image, self.config, open_reference=self.open_image, closed_reference=EMPTY)
            self.assertTrue(result["accepted"], result)

    def test_intermediate_finger_position_is_not_a_new_object(self):
        mid = ROOT / "pick_results/table_upright/tool_check/open15/grasp_verify_color_20261115_000105.png"
        if not mid.exists():
            self.skipTest("Intermediate-jaw capture unavailable")
        result = visual.verify_empty_jaws(mid, self.config, open_reference=self.open_image, closed_reference=EMPTY)
        self.assertTrue(result["accepted"], result)

    def test_held_tube_cannot_be_classified_as_empty_jaws(self):
        result = visual.verify_empty_jaws(HELD, self.config, open_reference=self.open_image, closed_reference=EMPTY)
        self.assertFalse(result["accepted"], result)
        self.assertEqual(result["status"], "occupied_or_uncertain")

    def test_dark_background_object_prevents_empty_confirmation(self):
        image = cv2.imread(str(self.open_image))
        cv2.circle(image, (980, 730), 18, (30, 30, 30), -1)
        result = visual.verify_empty_jaws(image, self.config, open_reference=self.open_image, closed_reference=EMPTY)
        self.assertFalse(result["accepted"], result)

    def test_blank_image_without_fixed_tool_is_unknown(self):
        image = np.full_like(cv2.imread(str(self.open_image)), 200)
        result = visual.verify_empty_jaws(image, self.config, open_reference=self.open_image, closed_reference=EMPTY)
        self.assertFalse(result["accepted"], result)
        self.assertEqual(result["status"], "unknown")

    def test_dark_image_and_missing_reference_never_confirm_empty(self):
        image = np.zeros_like(cv2.imread(str(self.open_image)))
        result = visual.verify_empty_jaws(image, self.config, open_reference=self.open_image, closed_reference=EMPTY)
        self.assertFalse(result["accepted"], result)
        result = visual.verify_empty_jaws(self.open_image, self.config, open_reference=None, closed_reference=EMPTY)
        self.assertFalse(result["accepted"], result)

    def motion_images(self, *, held=False, shift=(13, 9), texture=True):
        opened = cv2.imread(str(self.open_image))
        closed = cv2.imread(str(EMPTY))
        gray_open, gray_closed = [cv2.cvtColor(value, cv2.COLOR_BGR2GRAY) for value in (opened, closed)]
        visible, _ = visual._empty_motion_masks(self.config, gray_open, gray_closed)
        rng = np.random.default_rng(839)
        if texture:
            background = cv2.resize(rng.integers(135, 230, (60, 80), dtype=np.uint8),
                                     gray_open.shape[::-1], interpolation=cv2.INTER_NEAREST)
            background = cv2.GaussianBlur(background, (3, 3), 0)
            cv2.rectangle(background, (910, 680), (1030, 760), 70, -1)
            for _ in range(30):
                point = tuple(rng.integers([915, 685], [1025, 755]))
                cv2.circle(background, point, 2, 170, -1)
        else:
            background = np.full(gray_open.shape, 160, np.uint8)
        shifted = cv2.warpAffine(background, np.array([[1, 0, shift[0]], [0, 1, shift[1]]], float),
                                 gray_open.shape[::-1], borderMode=cv2.BORDER_REFLECT)
        # The scene moves across the entire image. Pasting only inside the
        # inspection ROI would invent a stationary elliptical seam and make
        # optical flow correctly reject the supposedly empty positive case.
        first, second = background.copy(), shifted.copy()
        tool = cv2.dilate(((gray_open < 110) | (gray_closed < 110)).astype(np.uint8),
                          np.ones((11, 11), np.uint8)) > 0
        first[tool], second[tool] = gray_open[tool], gray_open[tool]
        if held:
            # Camera-rigid object remains while the clutter behind it moves.
            cv2.circle(first, (975, 785), 23, 40, -1)
            cv2.circle(second, (975, 785), 23, 40, -1)
            for point in ((962, 775), (985, 779), (968, 793), (983, 795)):
                cv2.circle(first, point, 2, 90, -1)
                cv2.circle(second, point, 2, 90, -1)
        return cv2.cvtColor(first, cv2.COLOR_GRAY2BGR), cv2.cvtColor(second, cv2.COLOR_GRAY2BGR)

    def test_moving_dark_scene_can_establish_visible_empty_jaws(self):
        first, second = self.motion_images()
        result = visual.verify_empty_from_motion(first, second, self.config,
                     open_reference=self.open_image, closed_reference=EMPTY)
        self.assertTrue(result["accepted"], result)
        self.assertFalse(result["hidden_volume_observed"])

    def test_camera_fixed_object_over_moving_clutter_is_not_empty(self):
        first, second = self.motion_images(held=True)
        result = visual.verify_empty_from_motion(first, second, self.config,
                     open_reference=self.open_image, closed_reference=EMPTY)
        self.assertFalse(result["accepted"], result)

    def test_same_image_cannot_prove_empty_from_motion(self):
        first, second = self.motion_images(shift=(0, 0))
        result = visual.verify_empty_from_motion(first, second, self.config,
                     open_reference=self.open_image, closed_reference=EMPTY)
        self.assertFalse(result["accepted"], result)

    def test_textureless_volume_cannot_prove_empty_from_motion(self):
        first, second = self.motion_images(texture=False)
        result = visual.verify_empty_from_motion(first, second, self.config,
                     open_reference=self.open_image, closed_reference=EMPTY)
        self.assertFalse(result["accepted"], result)

    def test_recorded_bin_retreat_has_moving_evidence_in_each_dark_region(self):
        folder = ROOT / "pick_results/bin_grasp/run_20261115_014515"
        before = folder / "bin_overview/grasp_verify_color_20261115_014533.png"
        after = folder / "bin_overview_retreat/grasp_verify_color_20261115_014553.png"
        if not before.exists() or not after.exists():
            self.skipTest("Recorded bin retreat pair unavailable")
        result = visual.verify_empty_from_motion(before, after, self.config,
                     open_reference=self.open_image, closed_reference=EMPTY)
        self.assertTrue(result["accepted"], result)
        self.assertEqual(result["stationary_tracks"], 0)
        for region in result["dark_regions"]:
            self.assertGreaterEqual(region["tracks"], 5)
            self.assertGreaterEqual(region["moving_fraction"], .9)


if __name__ == "__main__":
    unittest.main()
