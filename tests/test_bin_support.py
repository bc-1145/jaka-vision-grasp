"""Bin-reference and wide-end direction tests; no hardware or camera access."""
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation

import bin_support as evidence


class BinSupportTests(unittest.TestCase):
    def setUp(self):
        self.config={"model":{"path":"unused.FCStd","grasp_width_mm":21,
                              "head_z_range_mm":[0,3.2]}}
        angles=np.linspace(0,2*np.pi,64,endpoint=False)
        self.ring=np.column_stack((14*np.cos(angles),14*np.sin(angles),np.zeros(64)))
        self.vertices=np.vstack((self.ring,np.array([[0,0,51.2]])))
        self.mock_model=patch.object(evidence,"_model",return_value=(self.vertices,10.5,17,51.2))
        self.mock_model.start()
        self.addCleanup(self.mock_model.stop)
        self.model=np.eye(4)
        self.model[:3,:3]=Rotation.from_euler("y",90,degrees=True).as_matrix()
        self.model[:3,3]=[200,0,100]

    def reference(self,head=True,reverse=False,face_normals=True):
        values=[]
        if head:
            values.append(self.ring)
        if reverse:
            points=self.ring.copy(); points[:,2]=51.2
            values.append(points)
        local=np.vstack(values)
        points=evidence.vision.transform_points(local,self.model)
        normal=self.model[:3,2] if face_normals else np.array([0.,0,1])
        return {"points":points,"normals":np.tile(normal,(len(points),1)),
                "report":{"support_observed":False,"plane_base":None,
                          "reference_kind":"measured_flange_normals_without_floor"}}

    def test_wide_end_normals_can_establish_direction_without_visible_floor(self):
        result=evidence.bin_flange_direction_evidence(self.reference(),self.model,self.config)
        self.assertTrue(result["accepted"])
        self.assertFalse(result["support_observed"])
        self.assertTrue(result["scene_clearance_still_required"])

    def test_floor_like_wide_ring_cannot_establish_flange(self):
        result=evidence.bin_flange_direction_evidence(self.reference(face_normals=False),self.model,self.config)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["head_annulus_points"],0)

    def test_equal_wide_ends_remain_ambiguous(self):
        result=evidence.bin_flange_direction_evidence(self.reference(reverse=True),self.model,self.config)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["direction_ratio"],1)

    def test_reverse_wide_face_is_reported_not_silently_flipped(self):
        result=evidence.bin_flange_direction_evidence(self.reference(head=False,reverse=True),self.model,self.config)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["status"],"reversed")

    def test_near_vertical_axis_requires_real_support(self):
        upright=np.eye(4); upright[:3,3]=[200,0,100]
        ref=self.reference()
        ref["points"]=evidence.vision.transform_points(self.ring,upright)
        ref["normals"]=np.tile([0,0,1],(len(self.ring),1))
        result=evidence.bin_flange_direction_evidence(ref,upright,self.config)
        self.assertFalse(result["accepted"])
        self.assertIn("near-vertical",result["reason"])

    def test_sparse_arc_is_not_sufficient_face_evidence(self):
        ref=self.reference()
        ref["points"]=ref["points"][:10]
        ref["normals"]=ref["normals"][:10]
        self.assertFalse(evidence.bin_flange_direction_evidence(ref,self.model,self.config)["accepted"])

    def _floor(self,z):
        x,y=np.meshgrid(np.linspace(0,200,61),np.linspace(-100,100,61))
        return np.column_stack((x.ravel(),y.ravel(),np.full(x.size,z)))

    def _match(self,z=125):
        transform=self.model.copy(); transform[:3,3]=[100,0,z]
        return dict(score=.85,clearance_mm=15,camera_T_model=transform.tolist())

    def test_largest_vertical_wall_is_not_assumed_to_be_floor(self):
        y,z=np.meshgrid(np.linspace(-100,100,101),np.linspace(100,300,101))
        wall=np.column_stack((np.full(y.size,-50),y.ravel(),z.ravel()))
        cloud=np.vstack((wall,self._floor(100)))
        result=evidence.prepare_bin_reference(cloud,np.eye(4),self.config,[self._match()])
        report=result["report"]
        self.assertTrue(report["support_observed"])
        self.assertGreater(report["plane_candidates"][0]["normal_tilt_deg"],80)
        self.assertFalse(report["plane_candidates"][0]["accepted"])
        np.testing.assert_allclose(report["plane_base"],[0,0,1,-100],atol=.01)
        self.assertFalse(report["scene_geometry_removed_for_planning"])

    def test_horizontal_plane_cutting_cad_is_rejected_as_support(self):
        result=evidence.prepare_bin_reference(self._floor(140),np.eye(4),self.config,[self._match()])
        self.assertFalse(result["report"]["support_observed"])
        self.assertIsNone(result["report"]["plane_base"])
        self.assertLess(result["report"]["plane_candidates"][0]["minimum_target_clearance_mm"],-10)


if __name__=="__main__":
    unittest.main()
