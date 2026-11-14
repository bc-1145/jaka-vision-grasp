import copy
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock

from auto_bin_cycle import AutomaticBinCycle, compare_coordinates, retry_blocker


def result(success=False, **extra):
    return dict(completed=success, physical_grasp_confirmed=success,
                actuator_write_attempted=True, final_state={"queue":0, "collision":0}, **extra)


class CycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name)/"cycle"
        self.runners, self.calls, self.home_calls = [], [], []
        self.results = [result(True)]
        self.errors = []
        self.empty = Mock(return_value={"accepted":True})

    def runner(self, config, settings, output, execute=False):
        index = len(self.runners)
        report = copy.deepcopy(self.results[index])
        obj = Mock(report=report)
        def run(*args):
            self.calls.append(args)
            config["private_marker"] = True
            if index < len(self.errors) and self.errors[index]:
                raise self.errors[index]
            return report
        obj.run_bin.side_effect = run
        self.runners.append(obj)
        return obj

    def home(self, *args, **kwargs):
        self.home_calls.append((args,kwargs))
        return Mock(run=Mock(return_value={"completed":True, "holding":None,"holding_status":"unknown"}))

    def cycle(self, **kwargs):
        return AutomaticBinCycle({}, {}, self.output, runner_factory=self.runner,
                   return_factory=self.home, empty_verifier=self.empty, **kwargs)

    def test_success_returns_once_and_keeps_closed(self):
        report = self.cycle(execute=True).run(True, "reference")
        self.assertTrue(report["completed"])
        self.assertIsNone(report["holding"])
        self.assertEqual(len(self.home_calls),1)
        self.assertFalse(report["released"])
        self.empty.assert_not_called()

    def test_failed_attempt_fresh_empty_then_new_run(self):
        self.results = [result(), result(True)]
        self.errors = [RuntimeError("No visible held part")]
        report = self.cycle(execute=True).run(True,"reference")
        self.assertTrue(report["completed"])
        self.assertEqual(self.calls,[(True,"reference"),(False,None)])
        self.assertEqual(len(self.home_calls),2)
        self.empty.assert_called_once()
        self.assertIsNotNone(self.runners[1].previous_report)

    def test_unknown_empty_never_retries(self):
        self.results = [result()]
        self.empty.return_value = {"accepted":False}
        report = self.cycle(execute=True).run()
        self.assertFalse(report["completed"])
        self.assertEqual(len(self.runners),1)
        self.assertEqual(len(self.home_calls),1)

    def test_write_uncertain_returns_but_never_retries(self):
        self.results = [result()]
        self.errors = [TimeoutError("write")]
        self.cycle(execute=True).run()
        self.assertEqual(len(self.home_calls),1)
        self.empty.assert_not_called()

    def test_fault_never_retries(self):
        self.results = [result()]
        self.results[0]["final_state"]["collision"] = 1
        self.cycle(execute=True).run()
        self.empty.assert_not_called()

    def test_no_actuator_attempt_never_returns(self):
        self.results = [result()]
        self.results[0]["actuator_write_attempted"] = False
        self.cycle(execute=True).run()
        self.assertEqual(self.home_calls,[])

    def test_read_only_never_returns_or_retries(self):
        report = self.cycle().run()
        self.assertTrue(report["ready"])
        self.assertEqual(self.home_calls,[])
        self.empty.assert_not_called()

    def test_home_failure_stops(self):
        home = Mock(return_value=Mock(run=Mock(side_effect=RuntimeError("home failed"))))
        cycle = self.cycle(execute=True)
        cycle.return_factory = home
        report = cycle.run()
        self.assertFalse(report["completed"])
        self.empty.assert_not_called()

    def test_limits(self):
        for value in (0,4,True,1.5):
            with self.assertRaises(ValueError): self.cycle(max_attempts=value)

    def test_coordinate_audit_only_same_target(self):
        now = time.time()
        old = {"grasp_base":[0,0,0],"shaft_axis_base":[1,0,0],"observation":{"created_unix":now-1}}
        new = dict(old,grasp_base=[1,2,3], observation={"created_unix":now})
        audit = compare_coordinates({"plan":old},new)
        self.assertEqual(audit["relative_coordinate_correction_mm"],[1,2,3])
        self.assertFalse(audit["global_calibration_updated"])
        new["shaft_axis_base"] = [-1,0,0]
        self.assertIsNone(compare_coordinates({"plan":old},new)["relative_coordinate_correction_mm"])

    def test_old_observation_rejected(self):
        old = {"grasp_base":[0,0,0],"shaft_axis_base":[1,0,0],"observation":{"created_unix":time.time()}}
        with self.assertRaises(RuntimeError): compare_coordinates({"plan":old},old)

    def test_stop_error_blocks_retry(self):
        self.assertIsNotNone(retry_blocker(result(stop_error="unknown")))


if __name__ == "__main__": unittest.main()
