"""Offline, JSON-native target geometry; this module never contacts hardware."""
from __future__ import annotations

import math
import numpy as np

from pick_place_3d import camera_transform, matrix, pose_to_matrix, transform_points


def _get(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _native(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_native(v) for v in value]
    return value


def build_target_report(config, observation, match, plan=None, *, planning_error=None,
                        direction_evidence=None, clearance_above_target_mm=35,
                        image_path=None):
    """Describe a match in base and current TCP axes, even if planning failed.

    ``grasp_origin_tcp_axes_mm`` translates the origin to the current physical
    contact estimate; axes remain the acquisition TCP axes, never model axes.
    A supplied plan remains a planner result, not proof of physical execution.
    """
    observed = dict(observation)
    assumed_frame = not observed.get("camera_frame")
    if assumed_frame:
        observed["camera_frame"] = config["camera"].get("point_cloud_frame")
    base_tcp = pose_to_matrix(observed["tcp_pose"])
    base_camera = camera_transform(config, observed)
    base_model = base_camera @ matrix(_get(match, "camera_T_model"), "camera_T_model")
    tcp_model = np.linalg.inv(base_tcp) @ base_model
    offset = np.asarray(config["gripper"]["tcp_to_grasp_center_mm"], dtype=float)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("Invalid TCP-to-contact offset")
    model = config["model"]
    shaft_range = model["shaft_z_range_mm"]
    middle = float(sum(shaft_range) / 2)
    points = {"center": [0, 0, middle],
              "large_head": [0, 0, float(model.get("large_end_z_mm", 0))],
              "small_end": [0, 0, float(shaft_range[1])]}
    target = {}
    for name, point in points.items():
        tcp_point = transform_points([point], tcp_model)[0]
        target[name] = dict(base_mm=transform_points([point], base_model)[0],
                            tcp_origin_mm=tcp_point,
                            grasp_origin_tcp_axes_mm=tcp_point-offset)
    axis = tcp_model[:3, 2]
    head_direction = -axis
    center_base = target["center"]["base_mm"]
    contact_base = transform_points([offset], base_tcp)[0]
    tilt = math.degrees(math.acos(float(np.clip(-base_tcp[2, 2], -1, 1))))
    executable = bool(plan is not None and planning_error is None
                      and plan.get("poses") and plan.get("route_checks"))
    descent = None
    if plan and "prepick" in plan.get("poses", {}) and "pick" in plan["poses"]:
        start = transform_points([offset], pose_to_matrix(plan["poses"]["prepick"]))[0]
        finish = transform_points([offset], pose_to_matrix(plan["poses"]["pick"]))[0]
        descent = dict(depth_mm=float(start[2]-finish[2]),
                       horizontal_delta_mm=(finish-start)[:2],
                       base_vertical=bool(np.linalg.norm((finish-start)[:2]) < 1e-5))
    report = dict(schema_version=1, executable=executable,
                  execution_status="planner_checked" if executable else "geometry_only",
                  planning_error=str(planning_error) if planning_error is not None else None,
                  image_path=str(image_path) if image_path else None,
                  coordinate_convention=dict(length_unit="mm", angle_unit="deg",
                      base="robot base at acquisition", tcp="active TCP at acquisition",
                      grasp_origin="estimated current contact center; axes remain current TCP axes",
                      camera_frame=observed.get("camera_frame"),
                      camera_frame_assumed_from_config=assumed_frame),
                  cluster_index=_get(match, "cluster_index"), score=_get(match, "score"),
                  target=target, current_contact_base_mm=contact_base,
                  head_relative_target_center_tcp_axes_mm=head_direction*(middle-float(model.get("large_end_z_mm", 0))),
                  shaft_axis_tcp=axis, shaft_axis_base=base_model[:3, 2],
                  toward_large_head_tcp=head_direction,
                  shaft_angles_tcp_deg=dict(yaw=math.degrees(math.atan2(axis[1], axis[0])),
                      elevation=math.degrees(math.atan2(axis[2], math.hypot(axis[0], axis[1]))),
                      to_positive_axes=np.degrees(np.arccos(np.clip(axis, -1, 1)))),
                  direction_evidence=direction_evidence,
                  current_flange_tilt_deg=tilt,
                  tilt_definition="TCP/flange +Z relative to base downward -Z",
                  flange_tilt_candidates=[dict(tilt_deg=t, executable=False,
                      validation="requires IK and route checks") for t in (30, 45, 60)],
                  downward_depth_from_current_contact_mm=float(contact_base[2]-center_base[2]),
                  suggested_vertical_clearance_mm=float(clearance_above_target_mm),
                  planned_descent=descent,
                  gripper=dict(close_command=config["gripper"].get("close_command_mm"),
                      command_is_measured_gap=False, actual_jaw_gap_mm=None,
                      contact_offset_is_estimate=True),
                  motion=dict(poses=plan.get("poses"), joints=plan.get("joints"),
                      motion_modes=plan.get("motion_modes"), route_checks=plan.get("route_checks"),
                      movement_strategy=plan.get("movement_strategy")) if plan else None)
    return _native(report)
