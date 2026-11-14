import unittest
from camera_cleanup import CameraResource


class Native:
    def __init__(self, owner):
        self.owner=owner
    def Terminate(self):
        self.owner.events.append("native_terminate")
        self.owner.released=True


class Wrapper:
    def __init__(self, fail=None):
        self.events=[]
        self.fail=fail
        self.released=False
        self._stream=Native(self)
    def init(self):
        self.events.append("init")
        if self.fail=="init": raise RuntimeError("init failed")
    def start(self):
        self.events.append("start")
        if self.fail=="start": raise RuntimeError("start failed")
    def stop(self):
        self.events.append("stop")
        if self.fail=="stop": raise RuntimeError("stop failed")
    def terminate(self):
        self.events.append("wrapper_terminate")
        self.stop()
        self._stream.Terminate()


class CameraCleanupTests(unittest.TestCase):
    def test_started_uses_one_wrapper_termination_and_one_stop(self):
        resource=Wrapper(); lifecycle=CameraResource(resource)
        lifecycle.init(); lifecycle.start()
        self.assertEqual(lifecycle.close(),[])
        lifecycle.close()
        self.assertEqual(resource.events,["init","start","wrapper_terminate","stop","native_terminate"])
    def test_wrapper_stop_failure_still_releases_native(self):
        resource=Wrapper("stop"); lifecycle=CameraResource(resource)
        lifecycle.init(); lifecycle.start()
        self.assertEqual(lifecycle.close(),["stop failed"])
        self.assertTrue(resource.released)
        self.assertEqual(resource.events.count("stop"),1)
    def test_failed_initialization_or_start_skips_wrapper_stop(self):
        for failure in ("init","start"):
            with self.subTest(failure=failure):
                resource=Wrapper(failure); lifecycle=CameraResource(resource)
                with self.assertRaises(RuntimeError):
                    lifecycle.init(); lifecycle.start()
                lifecycle.close()
                self.assertTrue(resource.released)
                self.assertNotIn("stop",resource.events)
                self.assertNotIn("wrapper_terminate",resource.events)
    def test_uninitialized_resource_has_no_sdk_calls(self):
        resource=Wrapper(); CameraResource(resource).close()
        self.assertEqual(resource.events,[])
    def test_initialized_unstarted_resource_uses_native_termination(self):
        resource=Wrapper(); lifecycle=CameraResource(resource)
        lifecycle.init(); lifecycle.close()
        self.assertEqual(resource.events,["init","native_terminate"])


if __name__=="__main__": unittest.main()
