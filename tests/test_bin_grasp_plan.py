"""Bin route planning tests use synthetic kinematics and measured geometry."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

import bin_grasp_plan as planner


class FakeRobot:
    def __init__(self,pose,unreachable_x=None,move_on_second_snapshot=False):
        self.pose=np.asarray(pose,dtype=float)
        self.unreachable_x=unreachable_x
        self.move_on_second_snapshot=move_on_second_snapshot
        self.snapshots=0
        self.inverse_poses=[]
        self.forward_joints=[]

    def motion_snapshot(self):
        self.snapshots+=1
        pose=self.pose.copy()
        if self.move_on_second_snapshot and self.snapshots>1:
            pose[0]+=1
        return dict(actual_position=pose.tolist(),joint_actual_position=(pose/10).tolist(),
                    inpos=True,queue=0,active_queue=0,collision=0,soft_limit=0,estop=0,paused=False)

    def inverse(self,pose,reference):
        pose=np.asarray(pose,dtype=float)
        self.inverse_poses.append(pose.copy())
        if self.unreachable_x is not None and pose[0]>self.unreachable_x:
            raise RuntimeError("Synthetic unreachable X")
        return pose/10

    def forward(self,joints):
        self.forward_joints.append(np.asarray(joints).copy())
        return np.asarray(joints)*10


class BinGraspPlanTests(unittest.TestCase):
    def setUp(self):
        self.observation=dict(created_unix=123,tcp_pose=[0,0,400,180,0,0],tool_id=0,user_id=0,
                              camera_frame="opencv_optical")
        self.config={"camera":{"mount":"eye_in_hand","tcp_T_camera":np.eye(4).tolist(),
                               "calibration_tool_id":0,"point_cloud_frame":"opencv_optical"},
                     "gripper":{"tcp_to_grasp_center_mm":[-22.33,-1.08,166.42],"jaw_axis_tcp":[0,1,0]},
                     "model":{"path":"螺栓.FCStd","shaft_z_range_mm":[3.6,51.2],
                              "head_z_range_mm":[0,3.2],"large_end_z_mm":0},
                     "matching":{"model_sample_points":100,"ransac_distance_mm":4.5,"icp_distance_mm":2.5}}
        angle=np.linspace(0,2*np.pi,30,endpoint=False)
        self.model=np.vstack([np.column_stack((10.5*np.cos(angle),10.5*np.sin(angle),np.full(30,z)))
                              for z in (0,10,27.4,40,51.2)])
        self.base_camera=planner.vision.camera_transform(self.config,self.observation)
        self.model_patch=patch.object(planner,"_model_points",return_value=self.model)
        self.model_patch.start()
        self.addCleanup(self.model_patch.stop)

    def match(self,index=1,score=.85,x=100,clearance=20):
        base=np.eye(4)
        base[:3,:3]=Rotation.from_euler("y",90,degrees=True).as_matrix()
        base[:3,3]=[x,-300,50]
        camera=np.linalg.inv(self.base_camera)@base
        return SimpleNamespace(cluster_index=index,score=score,clearance_mm=clearance,
                               camera_T_model=camera.tolist(),observation=copy.deepcopy(self.observation))

    def scene(self,matches,extra_base=None):
        points=[planner.vision.transform_points(self.model,np.asarray(m.camera_T_model)) for m in matches]
        foreign=np.column_stack((np.linspace(900,1100,120),np.full(120,1000),np.full(120,-1000)))
        if extra_base is not None:
            foreign=np.vstack((foreign,extra_base))
        points.append(planner.vision.transform_points(foreign,np.linalg.inv(self.base_camera)))
        return np.vstack(points)

    def test_only_allowed_actual_flange_tilts_and_true_jaw_axis(self):
        reference=planner.vision.pose_to_matrix(self.observation["tcp_pose"])[:3,:3]
        candidates=planner._orientation_candidates(np.array([1.,0,0]),reference,0,self.config)
        self.assertEqual(set(c[1] for c in candidates),{30,45,60})
        for _,tilt,_,tool,rotation in candidates:
            measured=np.degrees(np.arccos(np.clip((rotation@[0,0,1])@[0,0,-1],-1,1)))
            self.assertAlmostEqual(measured,tilt)
            self.assertAlmostEqual(float((rotation@[0,1,0])@np.array([1.,0,0])),0)
            np.testing.assert_allclose(rotation@[0,0,1],tool,atol=1e-8)

    def test_full_offset_middle_grasp_and_reverse_approach_lifts(self):
        match=self.match()
        robot=FakeRobot(self.observation["tcp_pose"])
        result=planner.plan_bin_grasp(robot,self.config,self.observation,[match],self.scene([match]))
        self.assertIn(result["tilt_deg"],(30,45,60))
        self.assertAlmostEqual(result["grasp_model_z_mm"],27.4)
        offset=self.config["gripper"]["tcp_to_grasp_center_mm"]
        center=planner.vision.transform_points([offset],planner.vision.pose_to_matrix(result["poses"]["pick"]))[0]
        np.testing.assert_allclose(center,result["grasp_base"],atol=1e-8)
        delta=np.asarray(result["poses"]["lift"][:3])-result["poses"]["lift_first"][:3]
        self.assertAlmostEqual(np.linalg.norm(delta),30)
        np.testing.assert_allclose(delta,-np.asarray(result["approach_axis"])*30,atol=1e-8)
        np.testing.assert_allclose(result["poses"]["lift"][3:],result["poses"]["pick"][3:])
        self.assertEqual([s["mode"] for s in result["route_checks"]["segments"]],
                         ["MoveL","MoveJ","MoveL","MoveL","MoveL","MoveL"])
        self.assertGreater(result["statistics"]["clearance_samples"],40)
        self.assertFalse(result["route_checks"]["full_collision_checked"])
        self.assertFalse(result["motion_sent"])
        np.testing.assert_allclose(result["poses"]["retreat"][:3],[0,0,440],atol=1e-8)
        self.assertEqual(result["observation"],self.observation)

    def test_horizontal_alignment_precedes_vertical_descent_and_lift(self):
        match=self.match()
        result=planner.plan_bin_grasp(FakeRobot(self.observation["tcp_pose"]),self.config,
                                     self.observation,[match],self.scene([match]))
        self.assertEqual(list(result["poses"]),
                         ["retreat","approach","prepick","pick","lift_first","lift"])
        offset=self.config["gripper"]["tcp_to_grasp_center_mm"]
        centers={name:planner.vision.transform_points([offset],planner.vision.pose_to_matrix(pose))[0]
                 for name,pose in result["poses"].items()}
        np.testing.assert_allclose(centers["approach"][:2],centers["retreat"][:2],atol=1e-8)
        self.assertAlmostEqual(centers["approach"][2],centers["prepick"][2])
        self.assertGreater(np.linalg.norm(centers["prepick"][:2]-centers["approach"][:2]),1)
        self.assertGreaterEqual(centers["approach"][2],centers["pick"][2]+70-1e-8)
        for name in ("prepick","lift_first","lift"):
            np.testing.assert_allclose(centers[name][:2],centers["pick"][:2],atol=1e-8)
        self.assertGreater(centers["prepick"][2],centers["pick"][2])
        self.assertAlmostEqual(centers["lift_first"][2]-centers["pick"][2],40)
        self.assertAlmostEqual(centers["lift"][2]-centers["pick"][2],70)
        for name in ("prepick","pick","lift_first","lift"):
            np.testing.assert_allclose(result["poses"][name][3:],result["poses"]["approach"][3:],atol=1e-8)
        approach=next(s for s in result["route_checks"]["segments"] if s["name"]=="approach")
        self.assertGreaterEqual(approach["minimum_contact_base_z_mm"],
                                result["orientation_minimum_contact_z_mm"]-.25)

    def test_highest_score_reachable_candidate_is_selected(self):
        unreachable=self.match(1,.9,x=600)
        reachable=self.match(2,.85,x=100)
        result=planner.plan_bin_grasp(FakeRobot(self.observation["tcp_pose"],unreachable_x=400),
                                     self.config,self.observation,[reachable,unreachable],self.scene([unreachable,reachable]))
        self.assertEqual(result["selected_cluster_index"],2)
        self.assertEqual(result["statistics"]["targets_tested"],2)

    def test_low_score_or_clearance_cannot_be_used(self):
        for score,clearance in ((.77,20),(.85,9),(.85,float("nan"))):
            match=self.match(score=score,clearance=clearance)
            with self.subTest(score=score,clearance=clearance),self.assertRaisesRegex(RuntimeError,"No bin candidate"):
                planner.plan_bin_grasp(FakeRobot(self.observation["tcp_pose"]),self.config,
                                       self.observation,[match],self.scene([match]))

    def test_foreign_depth_on_initial_tool_axis_prevents_route(self):
        match=self.match()
        tcp=planner.vision.pose_to_matrix(self.observation["tcp_pose"])
        center=planner.vision.transform_points([self.config["gripper"]["tcp_to_grasp_center_mm"]],tcp)[0]
        extra=center+np.column_stack((np.linspace(-.5,.5,100),np.zeros(100),np.zeros(100)))
        with self.assertRaisesRegex(RuntimeError,"Foreign-point clearance at retreat"):
            planner.plan_bin_grasp(FakeRobot(self.observation["tcp_pose"]),self.config,
                                   self.observation,[match],self.scene([match],extra))

    def test_mismatched_scan_observation_is_rejected(self):
        match=self.match()
        match.observation["created_unix"]+=1
        with self.assertRaisesRegex(RuntimeError,"do not belong"):
            planner.plan_bin_grasp(FakeRobot(self.observation["tcp_pose"]),self.config,
                                   self.observation,[match],self.scene([match]))

    def test_movement_during_planning_is_rejected(self):
        match=self.match()
        with self.assertRaisesRegex(RuntimeError,"moved during"):
            planner.plan_bin_grasp(FakeRobot(self.observation["tcp_pose"],move_on_second_snapshot=True),
                                   self.config,self.observation,[match],self.scene([match]))

    def test_no_foreign_geometry_is_not_treated_as_clearance(self):
        match=self.match()
        only_target=planner.vision.transform_points(self.model,np.asarray(match.camera_T_model))
        with self.assertRaisesRegex(RuntimeError,"Too little non-target"):
            planner.plan_bin_grasp(FakeRobot(self.observation["tcp_pose"]),self.config,
                                   self.observation,[match],only_target)

    def test_verified_empty_inspection_does_not_add_duplicate_upward_retreat(self):
        class LiftLimitedRobot(FakeRobot):
            def inverse(self,pose,reference):
                if np.asarray(pose)[2]>400.01:
                    raise RuntimeError("Synthetic lift joint limit")
                return super().inverse(pose,reference)
        match=self.match()
        # Without a completed inspection the required 40 mm retreat exceeds
        # this synthetic installation's limit; no movement can be planned.
        with self.assertRaisesRegex(RuntimeError,"lift joint limit"):
            planner.plan_bin_grasp(LiftLimitedRobot(self.observation["tcp_pose"]),self.config,
                                   self.observation,[match],self.scene([match]))
        config=copy.deepcopy(self.config)
        config["_bin_inspection_retreat_completed"]=True
        robot=LiftLimitedRobot(self.observation["tcp_pose"])
        result=planner.plan_bin_grasp(robot,config,self.observation,[match],self.scene([match]))
        np.testing.assert_allclose(result["poses"]["retreat"],self.observation["tcp_pose"],atol=1e-8)
        self.assertEqual(result["initial_retreat_distance_mm"],0)
        self.assertTrue(result["inspection_retreat_reused"])
        self.assertTrue(all(pose[2]<=400.01 for pose in robot.inverse_poses))
        # Reusing an already performed inspection does not skip depth/IK/FK
        # checking: the current tool pose still has a sampled route segment.
        self.assertEqual(result["route_checks"]["segments"][0]["name"],"retreat")
        self.assertGreaterEqual(result["route_checks"]["segments"][0]["samples"],8)

    def test_only_explicit_completed_inspection_flag_can_reuse_retreat(self):
        match=self.match()
        config=copy.deepcopy(self.config)
        config["_bin_inspection_retreat_completed"]="true"
        result=planner.plan_bin_grasp(FakeRobot(self.observation["tcp_pose"]),config,
                                     self.observation,[match],self.scene([match]))
        self.assertEqual(result["initial_retreat_distance_mm"],40)
        self.assertFalse(result["inspection_retreat_reused"])


if __name__=="__main__":
    unittest.main()
