"""Table-route regressions; synthetic kinematics only, no hardware access."""
import copy
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from diagnostics import stand_table_part as table


class CurvedRobot:
    """MoveJ bows downward; a controller MoveL follows a flat Cartesian line."""
    def __init__(self):
        self.inverse_poses=[]

    def inverse(self,pose,reference):
        self.inverse_poses.append(np.asarray(pose).copy())
        q=np.asarray(pose,dtype=float)/10
        q[2]+=(200*np.sin(np.pi*q[0]/20))/10
        return q

    def forward(self,joints):
        q=np.asarray(joints,dtype=float)
        p=q*10
        p[2]-=200*np.sin(np.pi*q[0]/20)
        return p


class TableRouteTests(unittest.TestCase):
    def setUp(self):
        self.config={"model":{"shaft_z_range_mm":[3.6,51.2],"head_z_range_mm":[0,3.2]},
                     "gripper":{"tcp_to_grasp_center_mm":[-22.33,-1.08,166.42],
                                "jaw_axis_tcp":[0,1,0]},
                     "camera":{"mount":"eye_in_hand","tcp_T_camera":np.eye(4).tolist()},
                     "robot":{"table_observation_pose":{"tcp_pose":[0,0,100,0,0,0]}}}
        model=np.eye(4)
        model[:3,:3]=Rotation.from_euler("y",90,degrees=True).as_matrix()
        model[:3,3]=[100,-400,100]
        self.target={"base_T_model":model.tolist()}

    def test_scaled_and_downward_plane_is_metric_and_upward(self):
        np.testing.assert_allclose(table.normalized_plane({"plane_base":[0,0,-2,130]}),[0,0,1,-65])
        self.assertEqual(table.table_height_at({"plane_base":[0,0,2,-130]},[20,-100]),65)

    def test_place_height_uses_current_grasp_xy_not_stale_height(self):
        plane={"plane_base":[-.01,.02,1,-60],"height_at_grasp_xy_mm":999}
        _,_,placed=table.make_poses(self.target,plane,self.config,45,0,1)
        coefficients=table.normalized_plane(plane)
        self.assertAlmostEqual(float(placed[:3,3]@coefficients[:3]+coefficients[3]),3,places=6)
        np.testing.assert_allclose(placed[:3,2],coefficients[:3],atol=1e-8)

    def test_upright_target_is_rejected_without_nan_route(self):
        upright=np.eye(4); upright[:3,3]=[100,-400,100]
        with self.assertRaisesRegex(ValueError,"already be upright"):
            table.make_poses({"base_T_model":upright.tolist()}, {"plane_base":[0,0,1,-60]},self.config,45,0,1)

    def test_invalid_planes_are_rejected(self):
        for plane in ([0,0,0,1],[1,0,0,1],[0,0,float("nan"),1]):
            with self.subTest(plane=plane),self.assertRaises(ValueError):
                table.normalized_plane({"plane_base":plane})

    def test_pose_generation_does_not_modify_target_matrix(self):
        original=copy.deepcopy(self.target)
        table.make_poses(self.target,{"plane_base":[0,0,1,-60]},self.config,45,0,1)
        self.assertEqual(self.target,original)

    def test_placement_transfer_keeps_held_pose_then_rotates_at_new_center(self):
        plane={"plane_base":[-.01,.02,1,-60]}
        original,original_held,_=table.make_poses(self.target,plane,self.config,45,0,1)
        self.assertNotIn("transfer",original)
        config=copy.deepcopy(self.config)
        config["robot"]["table_placement_xy_mm"]=[43,-537]
        poses,held,placed=table.make_poses(self.target,plane,config,45,0,1)
        np.testing.assert_allclose(poses["pick"],original["pick"])
        np.testing.assert_allclose(poses["lift"],original["lift"])
        np.testing.assert_allclose(held,original_held)
        names=list(poses)
        self.assertEqual(names[names.index("lift")+1],"transfer")
        self.assertEqual(names[names.index("transfer")+1],"turn_1")
        grasp=table.vision.transform_points([[0,0,table.side.shaft_middle_model_z(config)]],
                                            np.asarray(self.target["base_T_model"]))[0]
        expected=np.array([43,-537,grasp[2]+90])
        offset=config["gripper"]["tcp_to_grasp_center_mm"]
        for name in ("transfer","turn_1","turn_2","turn_3","turn_4","turn_5","turned"):
            transform=table.vision.pose_to_matrix(poses[name])
            center=table.vision.transform_points([offset],transform)[0]
            np.testing.assert_allclose(center,expected,atol=1e-7)
        np.testing.assert_allclose(table.vision.pose_to_matrix(poses["transfer"])[:3,:3],
                                   table.vision.pose_to_matrix(poses["pick"])[:3,:3])
        coefficients=table.normalized_plane(plane)
        np.testing.assert_allclose(placed[:2,3],np.array([43,-537])+coefficients[:2]*3,atol=1e-7)
        self.assertAlmostEqual(float(placed[:3,3]@coefficients[:3]+coefficients[3]),3,places=6)

    def test_invalid_placement_xy_is_rejected(self):
        for xy in ([1],[1,2,3],[float("nan"),2]):
            config=copy.deepcopy(self.config)
            config["robot"]["table_placement_xy_mm"]=xy
            with self.subTest(xy=xy),self.assertRaisesRegex(ValueError,"placement requires"):
                table.make_poses(self.target,{"plane_base":[0,0,1,-60]},config,45,0,1)

    def _route(self,include_linear_return):
        robot=CurvedRobot()
        initial=[0,0,100,0,0,0]
        state={"actual_position":initial,"joint_actual_position":robot.inverse(initial,np.zeros(6)).tolist()}
        poses={"prepick":[200,0,100,0,0,0]}
        if include_linear_return:
            poses["retreat"]=initial
        candidate={"poses":poses,"joints":{name:robot.inverse(pose,np.zeros(6)).tolist() for name,pose in poses.items()},
                   "held_model":np.eye(4).tolist()}
        config=copy.deepcopy(self.config)
        config["gripper"]["tcp_to_grasp_center_mm"]=[0,0,0]
        robot.inverse_poses=[]
        return robot,candidate,state,config

    def test_cartesian_segments_do_not_use_joint_interpolation(self):
        robot,candidate,state,config=self._route(True)
        result=table.validate_route(robot,candidate,state,{"plane_base":[0,0,1,0]},config,np.array([[0,0,0]]))
        self.assertEqual([s["mode"] for s in result["segments"]],["MoveL","MoveL","MoveJ"])
        self.assertGreater(len(robot.inverse_poses),100)
        self.assertTrue(result["return_to_overview_checked"])
        self.assertFalse(result["full_collision_checked"])
        self.assertGreaterEqual(min(s["minimum_flange_table_mm"] for s in result["segments"]),99.9)

    def test_return_joint_sweep_is_checked(self):
        robot,candidate,state,config=self._route(False)
        with self.assertRaisesRegex(RuntimeError,"return_overview"):
            table.validate_route(robot,candidate,state,{"plane_base":[0,0,1,0]},config,np.array([[0,0,0]]))

    def test_transfer_uses_cartesian_path_and_checks_held_cad(self):
        robot,candidate,state,config=self._route(True)
        candidate["poses"]={"transfer":candidate["poses"]["prepick"],"retreat":candidate["poses"]["retreat"]}
        candidate["joints"]={"transfer":candidate["joints"]["prepick"],"retreat":candidate["joints"]["retreat"]}
        checks=table.validate_route(robot,candidate,state,{"plane_base":[0,0,1,0]},config,np.array([[0,0,0]]))
        self.assertEqual(checks["segments"][0]["name"],"transfer")
        self.assertEqual(checks["segments"][0]["mode"],"MoveL")
        self.assertGreaterEqual(checks["segments"][0]["minimum_held_cad_table_mm"],99.9)
        with self.assertRaisesRegex(RuntimeError,"Held CAD below table at transfer"):
            table.validate_route(robot,candidate,state,{"plane_base":[0,0,1,0]},config,np.array([[0,0,-101]]))

    def test_stale_camera_or_tool_binding_is_rejected(self):
        saved={"planning_binding":table.planning_binding(self.config)}
        table.verify_plan_binding(saved,self.config)
        changed=copy.deepcopy(self.config)
        changed["camera"]["tcp_T_camera"][0][3]=5
        with self.assertRaisesRegex(RuntimeError,"geometry changed"):
            table.verify_plan_binding(saved,changed)

    def test_changed_table_placement_invalidates_binding(self):
        saved={"planning_binding":table.planning_binding(self.config)}
        changed=copy.deepcopy(self.config)
        changed["robot"]["table_placement_xy_mm"]=[43,-537]
        with self.assertRaisesRegex(RuntimeError,"placement"):
            table.verify_plan_binding(saved,changed)


if __name__=="__main__":
    unittest.main()
