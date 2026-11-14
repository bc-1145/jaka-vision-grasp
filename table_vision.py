"""Measured visual evidence for the table workflow; no hardware access.

All functions fail closed when evidence is missing. A commanded pose or a CAD
projection alone is never treated as proof that a physical part is present.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import numpy as np


def _result(accepted, reason, *, status=None, **metrics):
    return {"accepted": bool(accepted), "status": status or ("verified" if accepted else "unknown"),
            "reason": reason, **metrics}


def _points(value):
    points = np.asarray(value.points if hasattr(value, "points") else value, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected Nx3 point coordinates in millimetres")
    return points[np.isfinite(points).all(axis=1)]


def _transform(points, transform):
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("Expected a finite 4x4 transform")
    return points @ transform[:3, :3].T + transform[:3, 3]


def _plane(plane):
    coefficients = np.asarray(plane["plane_base"] if isinstance(plane, dict) else plane, dtype=float)
    if coefficients.shape != (4,) or not np.isfinite(coefficients).all():
        raise ValueError("Invalid base-frame support plane")
    length = np.linalg.norm(coefficients[:3])
    if length < 1e-9:
        raise ValueError("Zero plane normal")
    coefficients = coefficients / length
    return coefficients if coefficients[2] > 0 else -coefficients


@lru_cache(maxsize=4)
def _model_vertices(path):
    import pick_place_3d as vision
    mesh = vision.load_fcstd_mesh(Path(path))
    return np.asarray(mesh.vertices).copy()


def _dimensions(config):
    root = Path(__file__).resolve().parent
    vertices = _model_vertices(str(root / config["model"]["path"]))
    return vertices, float(config["model"]["grasp_width_mm"]) / 2, float(np.max(np.linalg.norm(vertices[:, :2], axis=1))), float(np.max(vertices[:, 2]))


def estimate_table_plane(scene_camera, base_T_camera, grasp_xy=None):
    """Fit the current table, in the base frame, rather than reuse an old Z."""
    import open3d as o3d
    base = _transform(_points(scene_camera), base_T_camera)
    base = base[(base[:, 2] > 0) & (base[:, 2] < 200)]
    if grasp_xy is not None:
        xy = np.asarray(grasp_xy, float)
        base = base[(np.abs(base[:, 0] - xy[0]) < 230) & (np.abs(base[:, 1] - xy[1]) < 260)]
    if len(base) < 500:
        return _result(False, "Too few measured table points")
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(base)).voxel_down_sample(1.5)
    o3d.utility.random.seed(7)
    coefficients, inliers = cloud.segment_plane(1.5, 3, 1000)
    coefficients = _plane(coefficients)
    support = np.asarray(cloud.points)[inliers]
    tilt = float(np.degrees(np.arccos(np.clip(coefficients[2], -1, 1))))
    residual = float(np.median(np.abs(support @ coefficients[:3] + coefficients[3])))
    xy = np.asarray(grasp_xy, float) if grasp_xy is not None else np.median(support[:, :2], axis=0)
    height = float(-(coefficients[3] + coefficients[:2] @ xy) / coefficients[2]) if abs(coefficients[2]) > .1 else None
    metrics = dict(plane_base=coefficients.tolist(), support_points=len(support),
                   normal_tilt_deg=tilt, residual_median_mm=residual,
                   height_at_grasp_xy_mm=height, fit_xy_mm=xy.tolist())
    if len(support) < 500 or tilt > 5 or residual > 1.2:
        return _result(False, "Current support plane is insufficient or tilted", **metrics)
    return _result(True, "Current tabletop measured", **metrics)


def flange_direction_evidence(scene_base, base_T_model, plane, config):
    """Compare measured wide annuli at the two ends, excluding the table."""
    _, radius, flange_radius, length = _dimensions(config)
    base = _points(scene_base)
    support_plane = _plane(plane)
    local = _transform(base, np.linalg.inv(np.asarray(base_T_model, float)))
    radial = np.linalg.norm(local[:, :2], axis=1)
    axis_alignment = abs(float(np.asarray(base_T_model, float)[:3, 2] @ support_plane[:3]))
    # Horizontal tubes expose the upper flange well above the table; reject
    # near-table flying points there. Keep the thin lower flange of an upright
    # part observable instead of discarding its 3.2 mm thickness.
    table_exclusion = 3.0 if axis_alignment < .5 else 1.5
    above = base @ support_plane[:3] + support_plane[3] > table_exclusion
    annulus = above & (radial > radius + 1.5) & (radial < flange_radius + 2.5)
    head_z = float(config["model"]["head_z_range_mm"][1])
    head = int(np.count_nonzero(annulus & (local[:, 2] > -3) & (local[:, 2] < head_z + 3)))
    reverse = int(np.count_nonzero(annulus & (local[:, 2] > length-head_z-3) & (local[:, 2] < length+3)))
    ratio = (head + 1) / (reverse + 1)
    metrics = dict(head_annulus_points=head, reverse_annulus_points=reverse, direction_ratio=float(ratio),
                   table_exclusion_mm=table_exclusion)
    if reverse >= 15 and ratio < .5:
        return _result(False, "Measured wide end is opposite to the CAD direction", status="reversed", **metrics)
    if head < 15 or ratio < 2:
        return _result(False, "Insufficient independent wide-end evidence", **metrics)
    return _result(True, "Measured wide flange supports CAD direction", **metrics)


def verify_upright(scene_base, base_T_model, plane, config):
    """Fit the observed shaft near the release point and require a lower flange.

    Allows up to 10 mm horizontal settling; requires measured height and a
    vertical cylinder. No match or pose command can manufacture this evidence.
    """
    from scipy.optimize import least_squares
    _, radius, flange_radius, length = _dimensions(config)
    normal = _plane(plane)
    base = _points(scene_base)
    expected = np.asarray(base_T_model, float)
    origin = expected[:3, 3]
    horizontal = base[:, :2] - origin[:2]
    heights = base @ normal[:3] + normal[3]
    region = (np.linalg.norm(horizontal, axis=1) < flange_radius + 13) & (heights > 2) & (heights < length + 12)
    observed = base[region]
    h = heights[region]
    if len(observed) < 100:
        return _result(False, "Too few measured points at release location")
    top = float(np.percentile(h, 97))
    if top < length * .72:
        return _result(False, "Measured part height is consistent with a fallen or missing tube", status="not_upright", height_97_mm=top)
    shaft = observed[(h > 10) & (h < length - 5)]
    if len(shaft) < 80:
        return _result(False, "Insufficient measured vertical shaft")
    fit = least_squares(lambda xy: np.linalg.norm(shaft[:, :2]-xy, axis=1)-radius,
                        origin[:2], bounds=(origin[:2]-10, origin[:2]+10), loss="soft_l1", f_scale=1.5)
    radial_error = np.abs(np.linalg.norm(shaft[:, :2]-fit.x, axis=1)-radius)
    supported = radial_error < 3
    fraction = float(np.mean(supported))
    median_error = float(np.median(radial_error))
    shaft_heights = shaft @ normal[:3] + normal[3]
    edges = np.linspace(10, length-5, 5)
    bands = [int(np.count_nonzero(supported & (shaft_heights >= lo) & (shaft_heights < hi)))
             for lo, hi in zip(edges[:-1], edges[1:])]
    metrics = dict(height_97_mm=top, shaft_radial_median_error_mm=median_error,
                   shaft_supported_fraction=fraction, shaft_height_band_points=bands,
                   measured_center_xy_mm=fit.x.tolist(), settling_shift_mm=float(np.linalg.norm(fit.x-origin[:2])))
    if fraction < .7 or median_error > 2.5 or min(bands) < 8:
        return _result(False, "Observed shaft does not establish a continuous vertical cylinder", **metrics)
    settled = expected.copy()
    settled[:2, 3] = fit.x
    settled[2, 3] = -(normal[3]+normal[:2]@fit.x)/normal[2]
    flange = flange_direction_evidence(base, settled, plane, config)
    metrics["flange"] = flange
    if not flange["accepted"]:
        return _result(False, "Vertical shaft observed, but lower large flange is not established", **metrics)
    return _result(True, "Measured vertical shaft and lower large flange verified", **metrics)


def verify_rgb_presence(image, camera_T_model, config, *, empty_image=None):
    """Conservative projected-object appearance check, with rigid-tool exclusion.

    Only verifies appearance, not a held object. The caller must combine it
    with gripper feedback or source disappearance / motion evidence.
    """
    import cv2
    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image))
    if isinstance(empty_image, (str, Path)):
        empty_image = cv2.imread(str(empty_image))
    if image is None or empty_image is None or image.shape != empty_image.shape:
        return _result(False, "A compatible empty-jaw reference image is required")
    projection = config["camera"]["color_projection"]
    if list(image.shape[1::-1]) != projection["image_size"]:
        return _result(False, "RGB frame dimensions do not match calibrated projection")
    vertices, _, _, _ = _dimensions(config)
    transformed = _transform(vertices, np.asarray(projection["color_T_camera"]) @ np.asarray(camera_T_model, float))
    if np.any(transformed[:, 2] <= 0):
        return _result(False, "Expected CAD lies behind the camera")
    pixels = cv2.projectPoints(transformed, np.zeros(3), np.zeros(3),
                               np.asarray(projection["matrix"], float), np.asarray(projection["distortion"], float))[0].reshape(-1, 2)
    if not np.isfinite(pixels).all():
        return _result(False, "Invalid projected CAD")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    empty = cv2.cvtColor(empty_image, cv2.COLOR_BGR2GRAY)
    # Dark rigid fingers and their borders are excluded using the empty image.
    tool = cv2.dilate((empty < 110).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    measured_edges = cv2.Canny(gray, 40, 100)
    edge_distance = cv2.distanceTransform((measured_edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
    metrics = {}
    for region in ("head", "shaft"):
        lo, hi = config["model"][f"{region}_z_range_mm"]
        indexes = (vertices[:, 2] >= lo-.01) & (vertices[:, 2] <= hi+.01)
        mask = np.zeros(gray.shape, np.uint8)
        hull = cv2.convexHull(np.round(pixels[indexes]).astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 1)
        boundary = (cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0) & ~tool
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8)).astype(bool) & ~tool
        count = int(np.count_nonzero(mask))
        metrics[region+"_usable_pixels"] = count
        if count < 120:
            return _result(False, "Too little unoccluded CAD appearance to distinguish tool from part", **metrics)
        dark = gray[mask] < 110
        contrast = empty[mask].astype(float)-gray[mask].astype(float)
        metrics[region+"_dark_fraction"] = float(np.mean(dark))
        metrics[region+"_reference_contrast_median"] = float(np.median(contrast))
        boundary_count = int(np.count_nonzero(boundary))
        boundary_support = float(np.mean(edge_distance[boundary] <= 4)) if boundary_count else 0.
        changed_fraction = float(np.mean(np.abs(contrast) > 25))
        texture_spread = float(np.percentile(gray[mask], 85)-np.percentile(gray[mask], 15))
        metrics.update({region+"_boundary_pixels": boundary_count,
                        region+"_boundary_edge_support": boundary_support,
                        region+"_reference_changed_fraction": changed_fraction,
                        region+"_intensity_spread": texture_spread})
        dark_supported = np.mean(dark) >= .55 and np.median(contrast) >= 25
        # Reflective silver changes from dark to bright with illumination. Its
        # measured silhouette and surface variation must independently support
        # the projected shape; brightness alone or a rendered CAD is not proof.
        reflective_supported = (boundary_count >= 80 and boundary_support >= .55
                                and changed_fraction >= .4 and texture_spread >= 20)
        metrics[region+"_appearance_method"] = "dark_contrast" if dark_supported else "shape_and_reference_change"
        if not dark_supported and not reflective_supported:
            return _result(False, "No independently supported part appearance at the expected "+region,
                           status="absent_or_misaligned", **metrics)
    return _result(True, "Measured head and shaft appearance, excluding rigid tool", status="appearance_present",
                   confirms_physical_grasp=False, **metrics)


def verify_follow_motion(before_image, after_image, camera_T_model, config, *, empty_image=None):
    """Require a camera-stationary part while nearby table features move.

    Intended for two stationary captures separated by a checked short lift.
    Missing texture, stationary background or moving part returns unknown.
    """
    import cv2
    images = [cv2.imread(str(value)) if isinstance(value, (str, Path)) else value
              for value in (before_image, after_image, empty_image)]
    first, second, empty = images
    presence = [verify_rgb_presence(value, camera_T_model, config, empty_image=empty)
                for value in (first, second)]
    if not all(value["accepted"] for value in presence):
        return _result(False, "Both lift images must contain the expected part", appearance=presence)
    vertices, _, _, _ = _dimensions(config)
    projection = config["camera"]["color_projection"]
    xyz = _transform(vertices, np.asarray(projection["color_T_camera"]) @ np.asarray(camera_T_model, float))
    uv = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), np.asarray(projection["matrix"], float),
                           np.asarray(projection["distortion"], float))[0].reshape(-1, 2)
    gray0, gray1, reference = [cv2.cvtColor(value, cv2.COLOR_BGR2GRAY) for value in images]
    silhouette = np.zeros(gray0.shape, np.uint8)
    cv2.fillConvexPoly(silhouette, cv2.convexHull(np.round(uv).astype(np.int32)), 255)
    tool = cv2.dilate((reference < 110).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    part = cv2.dilate(silhouette, np.ones((5, 5), np.uint8))
    part[tool] = 0
    bounds0 = np.maximum(np.floor(uv.min(axis=0)).astype(int)-180, 0)
    bounds1 = np.minimum(np.ceil(uv.max(axis=0)).astype(int)+180, gray0.shape[::-1])
    background = np.zeros(gray0.shape, np.uint8)
    background[bounds0[1]:bounds1[1], bounds0[0]:bounds1[0]] = 255
    background[(cv2.dilate(silhouette, np.ones((51, 51), np.uint8)) > 0) | tool] = 0

    def tracks(mask):
        features = cv2.goodFeaturesToTrack(gray0, 300, .003, 5, mask=mask)
        if features is None:
            return np.empty((0, 2))
        forward, status, error = cv2.calcOpticalFlowPyrLK(gray0, gray1, features, None,
                                                        winSize=(31, 31), maxLevel=4)
        if forward is None:
            return np.empty((0, 2))
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(gray1, gray0, forward, None,
                                                          winSize=(31, 31), maxLevel=4)
        if backward is None:
            return np.empty((0, 2))
        keep = (status.ravel().astype(bool) & back_status.ravel().astype(bool)
                & (error.ravel() < 20)
                & (np.linalg.norm((features-backward).reshape(-1, 2), axis=1) < 1))
        return (forward-features).reshape(-1, 2)[keep]

    object_flow, table_flow = tracks(part), tracks(background)
    metrics = dict(part_tracks=len(object_flow), background_tracks=len(table_flow), appearance=presence)
    if len(object_flow) < 8 or len(table_flow) < 15:
        return _result(False, "Insufficient independent part/table texture for lift verification", **metrics)
    object_distance = np.linalg.norm(object_flow, axis=1)
    table_distance = np.linalg.norm(table_flow, axis=1)
    object_median, table_median = float(np.median(object_distance)), float(np.median(table_distance))
    background_moving = float(np.mean(table_distance > 4))
    object_stationary = float(np.mean(object_distance < 3))
    metrics.update(part_motion_median_px=object_median, background_motion_median_px=table_median,
                   background_moving_fraction=background_moving, part_stationary_fraction=object_stationary)
    if table_median < 4 or background_moving < .65:
        return _result(False, "Nearby table motion does not establish the commanded lift", **metrics)
    if object_median > 2 or object_stationary < .75 or table_median < object_median*3:
        return _result(False, "Expected part did not remain fixed to the wrist camera", status="not_following", **metrics)
    return _result(True, "Part remains with the camera while nearby table moves", status="follows_lift", **metrics)


def verify_empty_jaws(image, config, *, open_reference, closed_reference):
    """Check visible grasp space before the first open command.

    The union of known open/closed tool silhouettes excludes moving fingers.
    Any new dark object, unobserved reference region, missing tool or dim frame
    is uncertain, never permission to release a potentially held workpiece.
    """
    import cv2
    values = [cv2.imread(str(value)) if isinstance(value, (str, Path)) else value
              for value in (image, open_reference, closed_reference)]
    if any(value is None for value in values) or len({value.shape for value in values}) != 1:
        return _result(False, "Compatible open and closed empty-tool references are required")
    if values[0].ndim != 3 or values[0].shape[2] != 3:
        return _result(False, "Expected three-channel calibrated RGB images")
    projection = config["camera"]["color_projection"]
    if list(values[0].shape[1::-1]) != projection["image_size"]:
        return _result(False, "Empty-tool frame does not match calibrated RGB dimensions")
    current, opened, closed = [cv2.cvtColor(value, cv2.COLOR_BGR2GRAY) for value in values]
    center = np.asarray(config["gripper"]["tcp_to_grasp_center_mm"], float)
    _, _, flange_radius, length = _dimensions(config)
    # This sphere covers the current model around the shaft-middle grasp.
    # Use deterministic directions so coverage cannot vary between runs.
    azimuth = np.linspace(0, 2*np.pi, 80, endpoint=False)
    polar = np.linspace(0, np.pi, 41)
    directions = np.array([[np.sin(p)*np.cos(a), np.sin(p)*np.sin(a), np.cos(p)]
                           for p in polar for a in azimuth])
    radius = float(np.hypot(length/2 + 3, flange_radius) + 1)
    color_T_tcp = np.asarray(projection["color_T_camera"]) @ np.linalg.inv(
        np.asarray(config["camera"]["tcp_T_camera"]))
    points = _transform(center + radius*directions, color_T_tcp)
    if np.any(points[:, 2] <= 0):
        return _result(False, "Grasp inspection volume lies behind the RGB camera")
    uv = cv2.projectPoints(points, np.zeros(3), np.zeros(3), np.asarray(projection["matrix"], float),
                           np.asarray(projection["distortion"], float))[0].reshape(-1, 2)
    if not np.isfinite(uv).all() or np.any(uv.min(axis=0) < 5) or np.any(uv.max(axis=0) > np.asarray(current.shape[::-1])-5):
        return _result(False, "Grasp inspection volume is not fully within the RGB frame")
    volume = np.zeros(current.shape, np.uint8)
    cv2.fillConvexPoly(volume, cv2.convexHull(np.round(uv).astype(np.int32)), 1)
    known_tool = (opened < 110) | (closed < 110)
    # Endpoint silhouettes leave holes where an intermediate jaw's dark side
    # crosses a highlight in both references. Bridge only along the physical
    # closing direction; do not classify these moving finger edges as a part.
    jaw = np.asarray(config["gripper"]["jaw_axis_tcp"], float)
    axis_points = _transform(np.vstack((center, center+jaw*20)), color_T_tcp)
    axis_pixels = cv2.projectPoints(axis_points, np.zeros(3), np.zeros(3),
                                    np.asarray(projection["matrix"], float),
                                    np.asarray(projection["distortion"], float))[0].reshape(-1, 2)
    direction = axis_pixels[1]-axis_pixels[0]
    direction_length = float(np.linalg.norm(direction))
    if not np.isfinite(direction_length) or direction_length < 1:
        return _result(False, "Physical jaw closing direction is not observable in RGB")
    direction /= direction_length
    span = int(np.clip(direction_length*.4, 15, 45))
    kernel = np.zeros((2*span+11, 2*span+11), np.uint8)
    origin = np.array([span+5, span+5])
    cv2.line(kernel, tuple(np.round(origin-direction*span).astype(int)),
             tuple(np.round(origin+direction*span).astype(int)), 1, 11)
    tool_sweep = cv2.dilate(known_tool.astype(np.uint8), kernel) > 0
    visible = (volume > 0) & ~tool_sweep
    fixed_tool = ((volume > 0) &
                  (cv2.erode(((opened < 110) & (closed < 110)).astype(np.uint8),
                             np.ones((7, 7), np.uint8)) > 0))
    free_count, tool_count = int(np.count_nonzero(visible)), int(np.count_nonzero(fixed_tool))
    metrics = dict(visible_grasp_pixels=free_count, fixed_tool_pixels=tool_count,
                   visible_volume_fraction=float(free_count/max(1, np.count_nonzero(volume))),
                   inspection_radius_mm=radius)
    if free_count < 1200 or tool_count < 500 or metrics["visible_volume_fraction"] < .3:
        return _result(False, "Insufficient visible grasp volume or fixed-tool reference", **metrics)
    tool_visible = float(np.mean(current[fixed_tool] < 135))
    tool_error = float(np.median(np.minimum(np.abs(current[fixed_tool].astype(float)-opened[fixed_tool]),
                                           np.abs(current[fixed_tool].astype(float)-closed[fixed_tool]))))
    dark = visible & (current < 110)
    _, _, stats, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8), 8)
    largest = int(np.max(stats[1:, cv2.CC_STAT_AREA])) if len(stats) > 1 else 0
    dark_fraction = float(np.mean(current[visible] < 110))
    bright_fraction = float(np.mean(current[visible] > 130))
    metrics.update(tool_visible_fraction=tool_visible, tool_reference_error_median=tool_error,
                   new_dark_fraction=dark_fraction, largest_new_dark_component_pixels=largest,
                   bright_grasp_fraction=bright_fraction)
    if tool_visible < .8 or tool_error > 35:
        return _result(False, "Fixed gripper appearance does not match the empty-tool references", **metrics)
    if dark_fraction > .015 or largest >= 150:
        return _result(False, "A dark object or occlusion occupies the visible grasp space", status="occupied_or_uncertain", **metrics)
    if bright_fraction < .92 or float(np.median(current[visible])) < 135:
        return _result(False, "Grasp space is too dim or ambiguous to establish empty jaws", **metrics)
    return _result(True, "Visible grasp space is clear with the known tool present", status="visually_empty",
                   hidden_volume_observed=False, **metrics)


def _empty_motion_masks(config, opened, closed):
    """Visible grasp volume, excluding the entire known jaw sweep."""
    import cv2
    projection = config["camera"]["color_projection"]
    center = np.asarray(config["gripper"]["tcp_to_grasp_center_mm"], float)
    _, _, flange_radius, length = _dimensions(config)
    radius = float(np.hypot(length/2+3, flange_radius)+1)
    directions = np.array([[np.sin(p)*np.cos(a), np.sin(p)*np.sin(a), np.cos(p)]
                           for p in np.linspace(0, np.pi, 41)
                           for a in np.linspace(0, 2*np.pi, 80, endpoint=False)])
    transform = np.asarray(projection["color_T_camera"]) @ np.linalg.inv(np.asarray(config["camera"]["tcp_T_camera"]))
    points = _transform(center+radius*directions, transform)
    if np.any(points[:, 2] <= 0):
        raise ValueError("Unobservable grasp volume")
    def project(points):
        return cv2.projectPoints(points, np.zeros(3), np.zeros(3), np.asarray(projection["matrix"], float),
                                  np.asarray(projection["distortion"], float))[0].reshape(-1, 2)
    pixels = project(points)
    if (not np.isfinite(pixels).all() or np.any(pixels.min(axis=0) < 5)
            or np.any(pixels.max(axis=0) > np.asarray(opened.shape[::-1])-5)):
        raise ValueError("Grasp volume is outside the RGB image")
    volume = np.zeros(opened.shape, np.uint8)
    cv2.fillConvexPoly(volume, cv2.convexHull(np.round(pixels).astype(np.int32)), 1)
    jaw = np.asarray(config["gripper"]["jaw_axis_tcp"], float)
    axis = project(_transform(np.vstack((center, center+jaw*20)), transform))
    direction = axis[1]-axis[0]
    magnitude = float(np.linalg.norm(direction))
    if magnitude < 1 or not np.isfinite(magnitude):
        raise ValueError("Unobservable jaw closing direction")
    direction /= magnitude
    span = int(np.clip(magnitude*.4, 15, 45))
    kernel = np.zeros((2*span+11, 2*span+11), np.uint8)
    origin = np.array([span+5, span+5])
    cv2.line(kernel, tuple(np.round(origin-direction*span).astype(int)),
             tuple(np.round(origin+direction*span).astype(int)), 1, 11)
    sweep = cv2.dilate(((opened < 110) | (closed < 110)).astype(np.uint8), kernel) > 0
    visible = (volume > 0) & ~sweep
    fixed = ((volume > 0) & (cv2.erode(((opened < 110) & (closed < 110)).astype(np.uint8),
                                      np.ones((7, 7), np.uint8)) > 0))
    return visible, fixed


def verify_empty_from_motion(before_image, after_image, config, *, open_reference, closed_reference):
    """Verify visible empty jaws over clutter using an unchanged jaw opening.

    Caller must establish the camera's short measured retreat independently.
    Every substantial dark region needs moving visual evidence. A stationary
    object, missing region texture, identical frames or inconsistent tool
    appearance produces unknown rather than permission to open.
    """
    import cv2
    values = [cv2.imread(str(value)) if isinstance(value, (str, Path)) else value
              for value in (before_image, after_image, open_reference, closed_reference)]
    if any(value is None for value in values) or len({value.shape for value in values}) != 1:
        return _result(False, "Compatible motion images and empty-tool references are required")
    if values[0].ndim != 3 or values[0].shape[2] != 3:
        return _result(False, "Expected calibrated RGB images")
    if list(values[0].shape[1::-1]) != config["camera"]["color_projection"]["image_size"]:
        return _result(False, "Motion image size does not match calibration")
    first, second, opened, closed = [cv2.cvtColor(value, cv2.COLOR_BGR2GRAY) for value in values]
    try:
        visible, fixed = _empty_motion_masks(config, opened, closed)
    except ValueError as exc:
        return _result(False, str(exc))
    metrics = dict(visible_grasp_pixels=int(np.count_nonzero(visible)), hidden_volume_observed=False)
    if metrics["visible_grasp_pixels"] < 1200 or np.count_nonzero(fixed) < 500:
        return _result(False, "Insufficient visible grasp volume or fixed tool", **metrics)
    for name, image in (("before", first), ("after", second)):
        fraction = float(np.mean(image[fixed] < 135))
        error = float(np.median(np.minimum(np.abs(image[fixed].astype(float)-opened[fixed]),
                                           np.abs(image[fixed].astype(float)-closed[fixed]))))
        metrics[name+"_tool_visible_fraction"] = fraction
        metrics[name+"_tool_error"] = error
        if fraction < .8 or error > 35:
            return _result(False, "Fixed tool does not match the known empty-tool installation", **metrics)
    tracking_mask = cv2.erode(visible.astype(np.uint8), np.ones((17, 17), np.uint8))*255
    features = cv2.goodFeaturesToTrack(first, 1200, .002, 4, mask=tracking_mask)
    if features is None or len(features) < 35:
        return _result(False, "Insufficient grasp-space texture for motion-based empty verification", **metrics)
    forward, good0, error = cv2.calcOpticalFlowPyrLK(first, second, features, None, winSize=(31, 31), maxLevel=4)
    if forward is None:
        return _result(False, "No reliable grasp-space motion tracks", **metrics)
    backward, good1, _ = cv2.calcOpticalFlowPyrLK(second, first, forward, None, winSize=(31, 31), maxLevel=4)
    if backward is None:
        return _result(False, "No bidirectional motion tracks", **metrics)
    valid = (good0.ravel().astype(bool) & good1.ravel().astype(bool) & (error.ravel() < 20)
             & (np.linalg.norm((features-backward).reshape(-1, 2), axis=1) < 1))
    positions = features.reshape(-1, 2)[valid]
    flow = (forward-features).reshape(-1, 2)[valid]
    distances = np.linalg.norm(flow, axis=1)
    metrics["motion_tracks"] = len(distances)
    if len(distances) < 35:
        return _result(False, "Too few reliable background tracks to establish visible empty jaws", **metrics)
    stationary = distances < 2.5
    moving = distances > 5
    metrics.update(motion_median_px=float(np.median(distances)), moving_fraction=float(np.mean(moving)),
                   stationary_tracks=int(np.count_nonzero(stationary)))
    if metrics["motion_median_px"] < 6 or metrics["moving_fraction"] < .85:
        return _result(False, "Grasp-space objects do not consistently move with the background", **metrics)
    # Nearby stationary tracks are more concerning than a small global ratio:
    # a held tube may occupy only a fraction of a textured bin background.
    stationary_positions = positions[stationary]
    if len(stationary_positions) >= 3:
        from scipy.spatial import cKDTree
        counts = cKDTree(stationary_positions).query_ball_point(stationary_positions, 35, return_length=True)
        metrics["largest_stationary_neighborhood_tracks"] = int(np.max(counts))
        if np.max(counts) >= 3:
            return _result(False, "A stationary grasp-space feature cluster may be a held part", status="occupied_or_uncertain", **metrics)
    dark = ((first < 110) & visible).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    integer_positions = np.clip(np.round(positions).astype(int), [0, 0], np.asarray(first.shape[::-1])-1)
    # Region labels need a modest edge margin to include smooth silhouette
    # corners without substituting unrelated nearby background features.
    regions = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < 180:
            continue
        region = cv2.dilate((labels == index).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        in_region = region[integer_positions[:, 1], integer_positions[:, 0]]
        track_count = int(np.count_nonzero(in_region))
        region_distances = distances[in_region]
        # Global corner ranking can starve a small smooth tube between many
        # high-contrast bin edges. Obtain independent local tracks there; never
        # substitute the motion of a different, better textured component.
        if track_count < 5:
            local_features = cv2.goodFeaturesToTrack(first, 80, .001, 2,
                           mask=(region & (tracking_mask > 0)).astype(np.uint8)*255)
            if local_features is not None:
                local_after, good_a, local_error = cv2.calcOpticalFlowPyrLK(
                    first, second, local_features, None, winSize=(31, 31), maxLevel=4)
                if local_after is not None:
                    local_back, good_b, _ = cv2.calcOpticalFlowPyrLK(
                        second, first, local_after, None, winSize=(31, 31), maxLevel=4)
                    if local_back is not None:
                        local_valid = (good_a.ravel().astype(bool) & good_b.ravel().astype(bool)
                                       & (local_error.ravel() < 20)
                                       & (np.linalg.norm((local_features-local_back).reshape(-1, 2), axis=1) < 1))
                        local_distance = np.linalg.norm((local_after-local_features).reshape(-1, 2), axis=1)[local_valid]
                        if len(local_distance) >= 5:
                            region_distances = local_distance
                            track_count = len(local_distance)
        moving_fraction = float(np.mean(region_distances > 5)) if track_count else 0.
        stationary_count = int(np.count_nonzero(region_distances < 2.5))
        regions.append(dict(pixels=area, tracks=track_count, moving_fraction=moving_fraction,
                            stationary_tracks=stationary_count,
                            bbox_pixels=stats[index, :4].astype(int).tolist()))
        if stationary_count >= 3:
            metrics["dark_regions"] = regions
            return _result(False, "A dark region has camera-stationary feature evidence", status="occupied_or_uncertain", **metrics)
        if track_count < 5 or moving_fraction < .9:
            metrics["dark_regions"] = regions
            return _result(False, "A dark grasp-space region lacks sufficient moving evidence", **metrics)
    metrics["dark_regions"] = regions
    if np.count_nonzero(stationary) > max(2, int(.01*len(distances))):
        return _result(False, "Unresolved stationary grasp-space features remain", **metrics)
    return _result(True, "Visible grasp-space features move with the scene; no stationary part established",
                   status="visually_empty_from_motion", **metrics)
