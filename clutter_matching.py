"""Local CAD hypotheses for touching parts and occlusion-aware validation."""
from __future__ import annotations

import math
import time

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

import pick_place_3d as vision


class DepthVisibility:
    """Measured foreground may occlude CAD; missing depth never proves occlusion."""
    def __init__(self, scene, config):
        projection = config["camera"].get("depth_projection")
        self.depth = None
        self.margin = max(4.0, 2 * float(config["matching"]["icp_distance_mm"]))
        if projection is None:
            return
        from scipy.ndimage import minimum_filter

        self.width, self.height = map(int, projection["image_size"])
        self.intrinsic = np.asarray(projection["matrix"], dtype=float)
        self.extrinsic = np.asarray(projection["color_T_camera"], dtype=float)
        self.distortion = np.asarray(projection.get("distortion", [0] * 5), dtype=float)
        points = np.asarray(scene.points)
        pixels, depths, valid = self.project(points)
        buffer = np.full(self.width * self.height, np.inf, dtype=float)
        indexes = pixels[valid, 1] * self.width + pixels[valid, 0]
        np.minimum.at(buffer, indexes, depths[valid])
        # Only bridge one pixel around an actual measured ray.
        self.depth = minimum_filter(buffer.reshape(self.height, self.width), size=3)

    def project(self, points):
        import cv2

        points = vision.transform_points(points, self.extrinsic)
        pixels, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), self.intrinsic, self.distortion)
        pixels = np.rint(pixels.reshape(-1, 2)).astype(np.int64)
        valid = ((points[:, 2] > 0) & (pixels[:, 0] >= 0) & (pixels[:, 0] < self.width) &
                 (pixels[:, 1] >= 0) & (pixels[:, 1] < self.height))
        return pixels, points[:, 2], valid

    def unoccluded(self, aligned, indexes):
        if self.depth is None:
            return indexes
        pixels, depths, valid = self.project(aligned[indexes])
        measured = np.full(len(indexes), np.inf)
        measured[valid] = self.depth[pixels[valid, 1], pixels[valid, 0]]
        # Unknown/out-of-frame points remain unsupported, not excused.
        occluded = np.isfinite(measured) & (measured < depths - self.margin)
        return indexes[~occluded]


_MODEL_CACHE = []


class PreparedModel:
    def __init__(self, model, config):
        self.model = model
        self.voxel = float(config["matching"]["model_voxel_mm"])
        self.down = model.voxel_down_sample(self.voxel)
        self.down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel * 3, max_nn=40))
        self.points = np.asarray(self.down.points)
        self.center = self.points.mean(axis=0)
        self.axis = vision._normalize(np.asarray(config["model"]["axis_model"], dtype=float))
        self.length = float(np.ptp(self.points @ self.axis))
        self.feature = None

    def features(self):
        if self.feature is None:
            self.feature = o3d.pipelines.registration.compute_fpfh_feature(
                self.down, o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel * 6, max_nn=100))
        return self.down, self.feature


def prepared_model(model, config):
    voxel = float(config["matching"]["model_voxel_mm"])
    for cached in _MODEL_CACHE:
        if cached.model is model and cached.voxel == voxel:
            return cached
    prepared = PreparedModel(model, config)
    _MODEL_CACHE.append(prepared)
    del _MODEL_CACHE[:-2]
    return prepared


def cylinder_hypotheses(prepared, observed, config, scene_tree=None):
    """Known shaft radius plus surface normals seed poses inside merged piles."""
    from scipy.optimize import least_squares

    points = np.asarray(observed.points)
    normals = np.asarray(observed.normals)
    radius = float(config["model"]["grasp_width_mm"]) / 2
    rng = np.random.default_rng(7)
    pairs = rng.integers(0, len(points), (4096, 2))
    p, q = points[pairs[:,0]], points[pairs[:,1]]
    n, m = normals[pairs[:,0]], normals[pairs[:,1]]
    axes = np.cross(n, m)
    lengths = np.linalg.norm(axes, axis=1)
    axes /= np.maximum(lengths[:,None], 1e-9)
    centers = p - radius * n
    difference = q - radius * m - centers
    along = np.einsum("ij,ij->i", difference, axes)
    error = np.linalg.norm(difference - axes * along[:,None], axis=1)
    good = (lengths > .35) & (np.abs(along) > 6) & (error < 2)
    indexes = np.flatnonzero(good)
    indexes = indexes[np.argsort(error[indexes])[:96]]
    if not len(indexes):
        return []
    lines = []
    for index in indexes:
        axis, center = axes[index], centers[index]
        difference = points - center
        radial = difference - (difference @ axis)[:,None] * axis
        radial_length = np.linalg.norm(radial, axis=1)
        inliers = ((np.abs(radial_length - radius) < 1.5) & (np.abs(normals @ axis) < .3) &
                   (np.einsum("ij,ij->i", normals, radial) > .5 * radius))
        count = int(np.count_nonzero(inliers))
        if count < 60 or np.ptp(points[inliers] @ axis) < 16:
            continue
        lines.append((count, axis, center, inliers))
    lines.sort(key=lambda row: row[0], reverse=True)
    selected = []
    hypotheses = []
    target_tree = scene_tree or cKDTree(points)
    for _, axis, center, inliers in lines:
        if any(abs(axis @ a) > .97 and np.linalg.norm(np.cross(center-c, a)) < 4 for a,c in selected):
            continue
        selected.append((axis, center))
        body, body_normals = points[inliers], normals[inliers]
        origin = body.mean(axis=0)
        helper = np.array([1.,0,0]) if abs(axis[0]) < .8 else np.array([0.,1,0])
        u = vision._normalize(np.cross(axis, helper)); v = np.cross(axis, u)
        def unpack(parameters):
            direction = vision._normalize(axis + u * parameters[0] + v * parameters[1])
            return direction, origin + u * parameters[2] + v * parameters[3]
        def residual(parameters):
            direction, location = unpack(parameters)
            delta = body - location
            radial = delta - (delta @ direction)[:,None] * direction
            return np.r_[np.linalg.norm(radial,axis=1) - radius, body_normals @ direction * radius * .3]
        fit = least_squares(residual, [0,0,(center-origin)@u,(center-origin)@v],
                            loss="soft_l1", f_scale=.7, max_nfev=30)
        axis, center = unpack(fit.x)
        for sign in (-1, 1):
            direction = axis * sign
            rotation = vision._rotation_between(prepared.axis, direction)
            coordinates = (body - center) @ direction
            lower, upper = np.quantile(coordinates, [.03,.97])
            shaft_lower, shaft_upper = config["model"]["shaft_z_range_mm"]
            offset_lower, offset_upper = upper - shaft_upper - 3, lower - shaft_lower + 3
            if offset_lower > offset_upper:
                continue
            rough = np.eye(4); rough[:3,:3] = rotation
            rough[:3,3] = center + direction * ((offset_lower + offset_upper) / 2)
            aligned = vision.transform_points(prepared.points, rough)
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(aligned))
            _, visible = cloud.hidden_point_removal([0,0,0], np.linalg.norm(aligned,axis=1).max()*100)
            visible = np.asarray(visible)[::2]
            model_points = prepared.points[visible]
            head_bounds = config["model"]["head_z_range_mm"]
            head = (model_points[:,2] >= head_bounds[0]) & (model_points[:,2] <= head_bounds[1])
            shaft = (model_points[:,2] >= shaft_lower) & (model_points[:,2] <= shaft_upper)
            if np.count_nonzero(head) < 5 or np.count_nonzero(shaft) < 20:
                continue
            count = min(9, max(3, int(math.ceil((offset_upper-offset_lower)/3))+1))
            for offset in np.linspace(offset_lower, offset_upper, count):
                pose = rough.copy(); pose[:3,3] = center + direction * offset
                distance = target_tree.query(vision.transform_points(model_points, pose))[0]
                supported = distance <= float(config["matching"]["icp_distance_mm"])
                score = .5 * supported[head].mean() + .5 * supported[shaft].mean()
                hypotheses.append((float(score), pose))
        if len(selected) >= 2:
            break
    hypotheses.sort(key=lambda row: row[0], reverse=True)
    return [pose for score,pose in hypotheses[:4] if score > .35]


def refine_pose(prepared, initial, config, scene, scene_tree):
    """Reassign and refit against this frame; a previous pose alone is no result."""
    coarse_distance = max(5.0, float(config["matching"]["ransac_distance_mm"]))
    aligned = vision.transform_points(prepared.points, initial)
    center = aligned.mean(axis=0)
    indexes = scene_tree.query_ball_point(center, np.linalg.norm(aligned-center,axis=1).max()+coarse_distance)
    context = np.asarray(scene.points)[indexes]
    nearby = context[cKDTree(aligned).query(context)[0] < coarse_distance]
    if len(nearby) < 60:
        return None
    assigned = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(nearby))
    assigned = assigned.voxel_down_sample(prepared.voxel)
    assigned.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=prepared.voxel*3,max_nn=40))
    fit = o3d.pipelines.registration.registration_icp(
        prepared.down, assigned, coarse_distance, initial,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(
            o3d.pipelines.registration.TukeyLoss(k=float(config["matching"]["icp_distance_mm"]))),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40))
    return fit, assigned


def register_model(model, target, config, scene_cloud=None, local_window=False):
    """Use useful axial shifts instead of repeated spins of an axial CAD part."""
    prepared = prepared_model(model, config)
    voxel = prepared.voxel
    observed = target.voxel_down_sample(voxel)
    points = np.asarray(observed.points)
    if len(points) < 20:
        return vision.RegistrationResult(0.0, float("inf"), np.eye(4))
    observed.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=40))
    observed.orient_normals_towards_camera_location([0,0,0])
    scene_tree = cKDTree(np.asarray(scene_cloud.points)) if scene_cloud is not None else None
    cylinder_seeds = cylinder_hypotheses(prepared, observed, config, scene_tree)
    center = points.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov(points.T))
    axes = [vectors[:, -1]]
    if values[-1] < 2.5 * max(values[-2], 1e-9):
        axes.append(vectors[:, -2])
    coarse_distance = max(5.0, float(config["matching"]["ransac_distance_mm"]))
    fine_distance = float(config["matching"]["icp_distance_mm"])
    proposals = []
    for axis in (() if local_window else axes):
        for sign in (-1, 1):
            rotation = vision._rotation_between(prepared.axis, axis * sign)
            rough = np.eye(4)
            rough[:3, :3] = rotation
            rough[:3, 3] = center - rotation @ prepared.center
            aligned = vision.transform_points(prepared.points, rough)
            temporary = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(aligned))
            _, visible = temporary.hidden_point_removal([0, 0, 0], np.linalg.norm(aligned, axis=1).max() * 100)
            source_center = prepared.points[visible].mean(axis=0) if visible else prepared.center
            for offset in (0, -0.2, 0.2):
                initial = rough.copy()
                initial[:3, 3] = center - rotation @ source_center + axis * offset * prepared.length
                coarse = o3d.pipelines.registration.registration_icp(
                    observed, prepared.down, coarse_distance, np.linalg.inv(initial),
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=25))
                quality = coarse.fitness - 0.08 * coarse.inlier_rmse / fine_distance
                proposals.append((quality, coarse.transformation))
    proposals.sort(key=lambda row: row[0], reverse=True)
    refined = []
    for initial in cylinder_seeds:
        # Fit only the surface assigned by this geometric hypothesis. A nearby
        # second tube must not pull ICP toward the centroid of a merged window.
        assigned_fit = refine_pose(prepared, initial, config,
                                   scene_cloud if scene_cloud is not None else observed,
                                   scene_tree if scene_tree is not None else cKDTree(points))
        if assigned_fit is None:
            continue
        fit, assigned = assigned_fit
        validation = vision.validate_model_support(model, assigned, fit.transformation, config)
        support = min(validation.get(key,0) for key in ("visible_support","head_support","shaft_support"))
        rank = (bool(validation["accepted"]), support, fit.fitness - .08*fit.inlier_rmse/fine_distance)
        refined.append((rank, vision.RegistrationResult(float(fit.fitness),float(fit.inlier_rmse),fit.transformation)))
    if local_window:
        # A mixed window is deliberately not a single-object PCA/FPFH target.
        # Return only a cylinder-seeded pose, which still needs CAD validation.
        return (max(refined, key=lambda row: row[0])[1] if refined else
                vision.RegistrationResult(0.0, float("inf"), np.eye(4)))
    for _, initial in proposals[:4]:
        fit = o3d.pipelines.registration.registration_icp(
            observed, prepared.down, fine_distance, initial,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
        transform = np.linalg.inv(fit.transformation)
        validation = vision.validate_model_support(model, target, transform, config)
        support = min(validation.get(key, 0) for key in ("visible_support", "head_support", "shaft_support"))
        rank = (bool(validation["accepted"]), support, fit.fitness - .08 * fit.inlier_rmse / fine_distance)
        refined.append((rank, vision.RegistrationResult(float(fit.fitness), float(fit.inlier_rmse), transform)))
    refined.sort(key=lambda row: row[0], reverse=True)
    if refined[0][0][0]:
        return refined[0][1]
    # FPFH remains a fallback for fragments with an ambiguous principal axis.
    source_down, source_feature = prepared.features()
    _, target_feature = vision._features(observed, voxel)
    coarse = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, observed, source_feature, target_feature, True, coarse_distance,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(.85),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(coarse_distance)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(12000, .999))
    fit = o3d.pipelines.registration.registration_icp(
        prepared.down, observed, fine_distance, coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
    validation = vision.validate_model_support(model, target, fit.transformation, config)
    support = min(validation.get(key, 0) for key in ("visible_support", "head_support", "shaft_support"))
    rank = (bool(validation["accepted"]), support, fit.fitness - .08 * fit.inlier_rmse / fine_distance)
    refined.append((rank, vision.RegistrationResult(float(fit.fitness), float(fit.inlier_rmse), fit.transformation)))
    return max(refined, key=lambda row: row[0])[1]


def local_candidates(model, clusters, config):
    """Keep original clusters, then recover model-sized windows from merged ones."""
    extent = model.get_axis_aligned_bounding_box().get_extent()
    radius = float(np.linalg.norm(extent) * .5)
    minimum = int(config["matching"]["min_cluster_points"])
    tolerance = float(config["matching"]["extent_tolerance"])
    candidates = []
    for index, cloud in enumerate(clusters):
        if vision._extent_is_plausible(model, cloud, tolerance):
            candidates.append((index, cloud, False))
    if not clusters:
        return candidates
    all_points = np.vstack([np.asarray(c.points) for c in clusters])
    tree = cKDTree(all_points)
    seeds = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(all_points)).voxel_down_sample(22)
    centers = np.asarray(seeds.points)
    centers = centers[np.argsort(centers[:, 2], kind="stable")]
    chosen = []
    budget = int(config["matching"].get("local_candidate_budget", 24))
    for center in centers:
        if chosen and np.min(np.linalg.norm(np.asarray(chosen) - center, axis=1)) < 24:
            continue
        points = all_points[tree.query_ball_point(center, radius)]
        if len(points) < minimum:
            continue
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        # A plane-only window cannot establish a hollow flanged tube.
        eigen = np.linalg.eigvalsh(np.cov(points.T))
        if eigen[0] < 1.0:
            continue
        candidates.append((-1, cloud, True))
        chosen.append(center)
        if len(chosen) >= budget:
            break
    return candidates


def same_instance(first, second, model, config):
    """Axial symmetry must not turn one physical tube into multiple results."""
    a, b = np.asarray(first.camera_T_model), np.asarray(second.camera_T_model)
    center = np.asarray(config["model"]["grasp_point_mm"])
    ca, cb = vision.transform_points([center], a)[0], vision.transform_points([center], b)[0]
    if np.linalg.norm(ca - cb) > 12:
        return False
    axis = np.asarray(config["model"]["axis_model"])
    if abs(float((a[:3,:3] @ axis) @ (b[:3,:3] @ axis))) < math.cos(math.radians(20)):
        return False
    # Heads at opposite ends remain different hypotheses; suppress by overlap.
    points = np.asarray(model.points)[::4]
    pa, pb = vision.transform_points(points, a), vision.transform_points(points, b)
    return bool(np.mean(cKDTree(pb).query(pa)[0] < 3) > .50)


def nearest_foreign_clearance(aligned, scene_tree, grasp, tolerance):
    """Exact nearest foreign point without testing every scene point per pose."""
    model_tree = cKDTree(aligned)
    count = len(scene_tree.data)
    k = min(256, count)
    while k:
        distances, indexes = scene_tree.query(grasp, k=k)
        indexes = np.atleast_1d(indexes)
        foreign = model_tree.query(scene_tree.data[indexes])[0] > tolerance
        if np.any(foreign):
            return float(np.min(np.atleast_1d(distances)[foreign]))
        if k == count:
            break
        k = min(count, k * 2)
    return 100.0


def validate_shaft_curvature(instance, transform, config):
    """A flat tool/bin edge must not explain a cylindrical shaft silhouette."""
    points = vision.transform_points(np.asarray(instance.points), np.linalg.inv(transform))
    lower, upper = config["model"]["shaft_z_range_mm"]
    radius = float(config["model"]["grasp_width_mm"]) / 2
    radial = np.linalg.norm(points[:,:2],axis=1)
    shaft = ((points[:,2] > lower+2) & (points[:,2] < upper-2) &
             (np.abs(radial-radius) <= float(config["matching"]["icp_distance_mm"])))
    body = points[shaft]
    if len(body) < 60 or np.ptp(body[:,2]) < 14:
        return {"curvature_accepted":False,"reason":"too little measured shaft curvature"}
    plane_rmse = float(np.sqrt(max(0,np.linalg.eigvalsh(np.cov(body.T))[0])))
    cylinder_rmse = float(np.sqrt(np.mean((radial[shaft]-radius)**2)))
    accepted = cylinder_rmse <= max(.5,.95*plane_rmse)
    return {"curvature_accepted":bool(accepted),"shaft_plane_rmse_mm":plane_rmse,
            "shaft_cylinder_rmse_mm":cylinder_rmse,
            "reason":"shaft curvature supported" if accepted else "shaft evidence is better explained by a plane"}


def find_matches(model, scene, clusters, config, previous_matches=None):
    started = time.monotonic()
    diagnostics = config["_matching_diagnostics"] = []
    base_count = len(clusters)
    candidates = [(index,cloud,local,None) for index,cloud,local in local_candidates(model,list(clusters),config)]
    previous_matches = list(previous_matches or [])
    if previous_matches:
        local = [c for c in candidates if c[2]]
        cursor = int(config.get("_search_cursor", 0))
        budget = min(len(local), int(config["matching"].get("live_search_budget", 6)))
        selected = [local[(cursor+i) % len(local)] for i in range(budget)] if local else []
        # Revisit isolated clusters periodically while rotating local searches.
        if cursor % max(1,len(local)) == 0:
            selected = [c for c in candidates if not c[2]] + selected
        config["_search_cursor"] = cursor + budget
        candidates = [(-2,None,True,vision.matrix(
            item["camera_T_model"] if isinstance(item,dict) else item.camera_T_model,"previous pose"))
                      for item in previous_matches] + selected
    visibility = DepthVisibility(scene, config)
    scene_points = np.asarray(scene.points)
    scene_tree = cKDTree(scene_points)
    minimum = int(config["matching"]["min_cluster_points"])
    tolerance = float(config["matching"]["icp_distance_mm"])
    accepted = []
    fitted = 0
    for number, (source_index, candidate, local, initial) in enumerate(candidates):
        detail = {"candidate_index": number, "source_cluster_index": source_index,
                  "local_window": local, "previous_pose_refit": initial is not None}
        try:
            if initial is not None:
                assigned_fit = refine_pose(prepared_model(model,config),initial,config,scene,scene_tree)
                if assigned_fit is None:
                    diagnostics.append({**detail,"accepted":False,"reason":"previous pose has no current depth support"})
                    continue
                fit, _assigned = assigned_fit
            else:
                fit = vision.register_model(model, candidate, config, scene_cloud=scene, local_window=local)
        except (RuntimeError,ValueError,np.linalg.LinAlgError) as exc:
            diagnostics.append({**detail,"accepted":False,"reason":f"registration failed: {exc}"})
            continue
        fitted += 1
        if fit.fitness <= 0:
            diagnostics.append({**detail, "accepted": False, "reason": "no cylinder-supported hypothesis"})
            continue
        transform = np.asarray(fit.transformation)
        aligned = vision.transform_points(np.asarray(model.points), transform)
        # Re-segment only model-supported points from the complete scene. This
        # recovers fragments and allows several instances in one DBSCAN cluster.
        center = aligned.mean(axis=0)
        nearby_indexes = np.asarray(scene_tree.query_ball_point(center, np.linalg.norm(aligned - center, axis=1).max() + tolerance))
        nearby = scene_points[nearby_indexes]
        distance, _ = cKDTree(aligned).query(nearby)
        supported = nearby[distance <= tolerance]
        detail["supported_points"] = len(supported)
        if len(supported) < minimum:
            diagnostics.append({**detail, "accepted": False, "reason": "too few model-supported points"})
            continue
        instance = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(supported))
        validation = vision.validate_model_support(model, instance, transform, config, visibility=visibility)
        detail.update(validation)
        if not validation["accepted"]:
            diagnostics.append(detail)
            continue
        curvature = validate_shaft_curvature(instance,transform,config)
        detail.update(curvature)
        if not curvature["curvature_accepted"]:
            detail["accepted"] = False
            diagnostics.append(detail)
            continue
        grasp = vision.transform_points([config["model"]["grasp_point_mm"]], transform)[0]
        clearance = nearest_foreign_clearance(aligned, scene_tree, grasp, tolerance)
        # Keep the original score threshold. Use measured surface support,
        # rather than the arbitrary amount of background in a local window.
        fitness = float(validation["visible_support"])
        rmse = float(np.sqrt(np.mean(distance[distance <= tolerance] ** 2)))
        score = (.78 * fitness + .17 * max(0, 1 - rmse / tolerance) + .05 * min(clearance / 30, 1))
        detail.update(score=score, rmse_mm=rmse)
        if score < float(config["matching"]["minimum_score"]):
            detail.update(accepted=False, reason="score below threshold")
            diagnostics.append(detail)
            continue
        match = vision.MatchResult(-1, score, fitness, rmse, transform.tolist(), len(supported), clearance,
                                    observation=config.get("_observation"))
        accepted.append((match, instance, detail, frozenset(nearby_indexes[distance <= tolerance])))
        diagnostics.append(detail)
    accepted.sort(key=lambda row: row[0].score, reverse=True)
    unique = []
    for match, instance, detail, assigned in accepted:
        if any(same_instance(match, other[0], model, config) or
               len(assigned & other[3]) > .5 * min(len(assigned), len(other[3])) for other in unique):
            detail.update(accepted=False, reason="duplicate pose suppressed")
            continue
        match.cluster_index = len(clusters)
        clusters.append(instance)
        unique.append((match, instance, detail, assigned))
    config["_matching_statistics"] = {"segmented_clusters": base_count, "candidate_regions": len(candidates),
                                      "registered_candidates": fitted, "validated_instances": len(unique),
                                      "previous_pose_hints": len(previous_matches),
                                      "elapsed_s": time.monotonic() - started}
    if not unique and previous_matches:
        # A moved camera or cleared bin must discard all hints and search anew.
        result = find_matches(model,scene,clusters,config,previous_matches=None)
        config["_matching_statistics"]["tracking_fallback"] = True
        return result
    return [row[0] for row in unique]
