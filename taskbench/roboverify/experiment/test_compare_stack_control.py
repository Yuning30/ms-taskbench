"""Measurement correctness and parity with uninstrumented controller execution."""

import unittest
from contextlib import nullcontext

import numpy as np

from taskbench.roboverify.api.control import PrimitiveController
from taskbench.roboverify.api.test_control import ServoEnvironment
from taskbench.roboverify.experiment.compare_stack_control import (
    measure,
    previous_controls,
    run_trial,
    segment_distance,
    summarize,
)
from taskbench.roboverify.util.on import using_block_length


class ComparisonTests(unittest.TestCase):
    def test_finite_segment_counts_overshoot_and_zero_length(self):
        np.testing.assert_allclose(
            segment_distance(
                [[0, 0, 0], [0.5, 0.2, 0], [1.3, 0, 0]], [0, 0, 0], [1, 0, 0]
            ),
            [0, 0.2, 0.3],
        )
        np.testing.assert_allclose(
            segment_distance([[1, 2, 2]], [0, 0, 0], [0, 0, 0]), [3]
        )

    def test_scaling_removes_ideal_servo_bend_without_changing_measurement(self):
        rows_by_mode = []
        original = PrimitiveController.move
        for previous in (True, False):
            with previous_controls() if previous else nullcontext():
                plain, measured = ServoEnvironment(), ServoEnvironment()
                target = [0.4, 0.175, 0.6]
                PrimitiveController(plain, []).move(target, phase="approach")
                rows = []
                with measure(rows):
                    PrimitiveController(measured, []).move(target, phase="approach")
                np.testing.assert_array_equal(measured.actions, plain.actions)
                np.testing.assert_array_equal(measured.obs, plain.obs)
                rows_by_mode.append(rows[0])
        self.assertIs(PrimitiveController.move, original)
        self.assertGreater(rows_by_mode[0]["path_mm"], 20)
        self.assertLess(rows_by_mode[1]["path_mm"], 1e-10)

    def test_payload_metrics_use_cube_destination_and_track_attachment(self):
        class PayloadServo(ServoEnvironment):
            def step(self, action):
                before = self.obs[:3].copy()
                super().step(action)
                self.obs[10:13] += self.obs[:3] - before

        env, rows = PayloadServo(), []
        target = env.obs[10:13] + [0.1, 0.025, 0.05]
        with measure(rows):
            self.assertTrue(PrimitiveController(env, []).move(target, payload_id=0))
        row = rows[0]
        self.assertLess(row["path_mm"], 1e-10)
        self.assertLess(row["tcp_path_mm"], 1e-10)
        self.assertLess(row["attachment_change_mm"], 1e-10)
        self.assertLessEqual(row["endpoint_mm"], 2)
        self.assertFalse(np.allclose(row["target"], row["tcp_target"]))

    def test_zero_step_and_failed_phase_remain_visible(self):
        env, rows = ServoEnvironment(stalled=True), []
        with measure(rows):
            controller = PrimitiveController(env, [], limit=2)
            self.assertTrue(controller.move(env.obs[:3]))
            self.assertFalse(controller.move(env.obs[:3] + [0.1, 0, 0]))
        self.assertEqual(rows[0]["steps"], 0)
        self.assertEqual(rows[0]["path_mm"], 0)
        self.assertFalse(rows[1]["converged"])
        self.assertEqual(rows[1]["steps"], 2)

    def test_summary_keeps_failures_and_identifies_regression_seed(self):
        base = dict(
            num_blocks=3,
            seed=9,
            phases=[],
            instruction_steps=[],
            actions=0,
            final_tower_xy_mm=None,
            reason="",
            status="valid",
        )
        records = [
            dict(base, mode="previous", valid=True),
            dict(
                base, mode="updated", valid=False, reason="timeout", status="incomplete"
            ),
        ]
        result = summarize(records)["3"]
        self.assertEqual(result["regression_seeds"], [9])
        self.assertEqual(result["updated"]["executions"], 1)
        self.assertEqual(result["updated"]["failures"][0]["reason"], "timeout")


class PandaComparisonTests(unittest.TestCase):
    def test_measurement_preserves_restored_panda_execution(self):
        with using_block_length(0.04):
            for mode in ("previous", "updated"):
                with self.subTest(mode=mode):
                    initial, _ = run_trial(3, 42, mode, instrument=False)
                    self.assertTrue(initial.snapshots)
                    plain, plain_record = run_trial(
                        3, 42, mode, initial.snapshots[0], instrument=False
                    )
                    measured, measured_record = run_trial(
                        3, 42, mode, initial.snapshots[0]
                    )
                    np.testing.assert_allclose(
                        plain.actions[0], measured.actions[0], atol=1e-7, rtol=0
                    )
                    np.testing.assert_allclose(
                        plain.states, measured.states, atol=1e-7, rtol=0
                    )
                    self.assertEqual(plain.events, measured.events)
                    self.assertEqual(plain_record["valid"], measured_record["valid"])
                    self.assertTrue(measured_record["valid"], measured_record["reason"])
                    self.assertEqual(len(measured_record["phases"]), 12)


if __name__ == "__main__":
    unittest.main()
