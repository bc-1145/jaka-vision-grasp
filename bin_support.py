"""Measured bin references and flange direction without assuming a visible floor.

A bin's largest plane is often a wall, and a clutter surface is not its floor.
If the floor cannot be established, direction still requires measured wide
flange geometry with shaft-aligned planar normals. No synthetic floor is used.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path

import numpy as np
import pick_place_3d as vision

ROOT=Path(__file__).resolve().parent


def _get(value,key,default=None):
    return value.get(key,default) if isinstance(value,dict) else getattr(value,key,default)


@lru_cache(maxsize=4)
def _vertices(path,mtime):
    return np.asarray(vision.load_fcstd_mesh(Path(path)).vertices).copy()


def _model(config):
    path=(ROOT/config["model"]["path"]).resolve()
    vertices=_vertices(str(path),path.stat().st_mtime_ns)
    return vertices,float(config["model"]["grasp_width_mm"])/2,float(np.max(np.linalg.norm(vertices[:,:2],axis=1))),float(np.max(vertices[:,2]))


def prepare_bin_reference(scene_camera,base_T_camera,config,matches=()):
    """Prepare current metric points/normals plus an honest support report.

    Return points/normals are NumPy arrays for subsequent direction checks;
    only ``report`` is intended for JSON logging. No scene points are removed
    from the planner input by this function.
    """
    import open3d as o3d
    points=np.asarray(scene_camera.points if hasattr(scene_camera,"points") else scene_camera,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3:
        raise ValueError("Expected camera-frame XYZ points")
    points=points[np.isfinite(points).all(axis=1)&(points[:,2]>0)]
    if len(points)<500:
        raise ValueError("Too little measured bin geometry")
    transform=vision.matrix(base_T_camera,"base_T_camera")
    world=vision.transform_points(points,transform)
    cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(world))
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4,max_nn=40))
    normals=np.asarray(cloud.normals).copy()
    vertices,_,_,_=_model(config)
    target_points=[]
    for match in matches:
        if float(_get(match,"score",0))>=.78 and float(_get(match,"clearance_mm",0))>=10:
            matrix=transform@vision.matrix(_get(match,"camera_T_model"),"camera_T_model")
            target_points.append(vision.transform_points(vertices,matrix))
    reduced=cloud.voxel_down_sample(1.5)
    planes=[]
    accepted=[]
    # Candidate extraction is bounded. A plane is accepted only when its
    # own local normals, spatial extent and observed target heights agree.
    for _ in range(6):
        if len(reduced.points)<500:
            break
        o3d.utility.random.seed(7)
        coefficients,indexes=reduced.segment_plane(1.5,3,1200)
        coefficients=np.asarray(coefficients,float)
        coefficients/=np.linalg.norm(coefficients[:3])
        if coefficients[2]<0:
            coefficients=-coefficients
        support=np.asarray(reduced.points)[indexes]
        local_normals=np.asarray(reduced.normals)[indexes]
        tilt=float(np.degrees(np.arccos(np.clip(coefficients[2],-1,1))))
        extent=np.percentile(support[:,:2],95,axis=0)-np.percentile(support[:,:2],5,axis=0)
        normal_fraction=float(np.mean(np.abs(local_normals@coefficients[:3])>np.cos(np.deg2rad(12))))
        below=min((float(np.min(p@coefficients[:3]+coefficients[3])) for p in target_points),default=None)
        measured=dict(plane_base=coefficients.tolist(),support_points=len(indexes),normal_tilt_deg=tilt,
                      inlier_xy_extent_mm=extent.tolist(),planar_normal_fraction=normal_fraction,
                      minimum_target_clearance_mm=below)
        measured["accepted"]=bool(len(indexes)>=500 and tilt<=8 and np.min(extent)>=60
                                   and np.prod(extent)>=5000 and normal_fraction>=.8
                                   and below is not None and below>=-2.5 and below<120)
        planes.append(measured)
        if measured["accepted"]:
            accepted.append(measured)
        reduced=reduced.select_by_index(indexes,invert=True)
    support=max(accepted,key=lambda item:item["support_points"]) if accepted else None
    report=dict(support_observed=support is not None,plane_base=support["plane_base"] if support else None,
                reference_kind="measured_bin_support" if support else "measured_flange_normals_without_floor",
                reason="Measured horizontal support lies below accepted CADs" if support else
                       "No defensible floor plane; use independent flange normals and keep full scene clearance checks",
                measured_points=len(world),normal_neighborhood_mm=4,plane_candidates=planes,
                scene_geometry_removed_for_planning=False)
    return dict(points=world,normals=normals,report=report)


def bin_flange_direction_evidence(reference,base_T_model,config):
    """Require real flange-width points whose normals match an end face."""
    _,radius,flange_radius,length=_model(config)
    base=np.asarray(reference["points"],float)
    normals=np.asarray(reference["normals"],float)
    matrix=vision.matrix(base_T_model,"base_T_model")
    local=vision.transform_points(base,np.linalg.inv(matrix))
    radial=np.linalg.norm(local[:,:2],axis=1)
    plane=reference["report"].get("plane_base")
    axis_vertical=abs(float(matrix[2,2]))
    common=dict(reference_kind=reference["report"]["reference_kind"],
                support_observed=reference["report"]["support_observed"],
                axis_vertical_alignment=axis_vertical,normal_alignment_threshold=.75,
                scene_clearance_still_required=True)
    if plane is None and axis_vertical>.65:
        return dict(accepted=False,status="unknown",reason="A near-vertical shaft needs a measured support to distinguish floor from flange",**common)
    above=np.ones(len(base),dtype=bool)
    if plane is not None:
        plane=np.asarray(plane,float)
        above=base@plane[:3]+plane[3]>(6 if axis_vertical<.5 else 1.5)
    annulus=above&(radial>radius+1.5)&(radial<flange_radius+2.5)
    head_z=float(config["model"]["head_z_range_mm"][1])
    head=annulus&(local[:,2]>-1.5)&(local[:,2]<head_z+1.5)
    reverse=annulus&(local[:,2]>length-head_z-1.5)&(local[:,2]<length+1.5)
    alignment=np.abs(normals@matrix[:3,2])
    planar=alignment>.75
    head_count=int(np.count_nonzero(head&planar))
    reverse_count=int(np.count_nonzero(reverse&planar))
    fraction=float(head_count/max(1,np.count_nonzero(head)))
    ratio=float((head_count+1)/(reverse_count+1))
    angles=np.arctan2(local[head&planar,1],local[head&planar,0])
    bands=np.histogram(angles,bins=np.linspace(-np.pi,np.pi,9))[0]
    common.update(head_annulus_points=head_count,reverse_annulus_points=reverse_count,
                  direction_ratio=ratio,head_planar_fraction=fraction,
                  head_angular_band_points=bands.tolist(),head_raw_annulus_points=int(np.count_nonzero(head)))
    if reverse_count>=30 and ratio<.5:
        return dict(accepted=False,status="reversed",reason="Measured planar wide end is opposite to CAD",**common)
    if head_count<30 or fraction<.2 or ratio<2 or np.count_nonzero(bands>=3)<2:
        return dict(accepted=False,status="unknown",reason="Wide-end geometry and independent end-face normals are insufficient",**common)
    return dict(accepted=True,status="verified",reason="Measured wide flange and shaft-aligned planar face agree with CAD direction",**common)
