"""Regression checks for boundaries introduced by the ManiSkill migration."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import z3

from taskbench.roboverify.api.control import PrimitiveController
from taskbench.roboverify.backend import BACKEND_ID, StackBackend
from taskbench.roboverify.cfg.collection import validate_trace
from taskbench.roboverify.cfg.demos import DemoTrace
from taskbench.roboverify.cfg.reset import Snapshot
from taskbench.roboverify.cfg.straightline import SegmentRollout, postcondition_reached
from taskbench.roboverify.util import on
from taskbench.roboverify.verification_lib.counterexamples import stacks_to_positions


class BackendContractTests(unittest.TestCase):
    def test_block_dimensions_are_scoped_and_numeric_matches_symbolic(self):
        self.assertEqual(on.get_block_length(), 0.05)
        for length in (0.04, 0.05):
            with on.using_block_length(length):
                a, b = (0, 0, length), (length / 2 - 0.001, 0, 0)
                self.assertTrue(on.on(a, b))
                solver = z3.Solver()
                solver.add(z3.Not(on.z3_on(*map(z3.RealVal, a), *map(z3.RealVal, b))))
                self.assertEqual(solver.check(), z3.unsat)
                positions = stacks_to_positions([["a", "b"]], table_height=0)
                self.assertAlmostEqual(positions["b"][2] - positions["a"][2], length)
        self.assertEqual(on.get_block_length(), 0.05)

    def test_action_scaling_and_gripper_hold_are_explicit(self):
        backend = object.__new__(StackBackend)
        backend.simulator = Mock()
        backend.simulator.step.return_value = (None, 0, False, False, {})
        backend._get_obs = lambda: np.zeros(43)
        backend.gripper_command = 1.0
        backend.step([1, -1, 0.5, -0.2])
        np.testing.assert_allclose(
            backend.simulator.step.call_args.args[0], [0.5, -0.5, 0.25, 0, 0, 0, -1]
        )
        backend.step([0, 0, 0, 0])
        self.assertEqual(backend.simulator.step.call_args.args[0][-1], -1)
        backend.step([0, 0, 0, 0.2])
        self.assertEqual(backend.simulator.step.call_args.args[0][-1], 1)

    def test_payload_target_is_not_assumed_equal_to_tcp(self):
        observation = np.zeros(43)
        observation[:3] = [0.004, 0, 0.002]
        env = SimpleNamespace(held_box_id=0)
        env.env = env
        env._get_obs = lambda: observation
        env.flatten_observation = np.asarray
        controller = PrimitiveController(env, [])
        controller.move = Mock(return_value=True)
        controller.move_relative([0, 0, 0], [0, 0, 0.04])
        self.assertEqual(controller.move.call_args.kwargs["payload_id"], 0)
        np.testing.assert_allclose(controller.move.call_args.args[0], [0, 0, 0.04])

    def test_replay_restores_instruction_state_without_resetting_physics(self):
        backend = object.__new__(StackBackend)
        snapshot = Snapshot(
            np.zeros(1),
            {
                "adapter/gripper": np.asarray(-1.0),
                "adapter/held": np.asarray(2),
            },
            {"b": 0, "b_prime": 2},
        )
        backend.restore_runtime_state(snapshot)
        self.assertEqual(backend.held_box_id, 2)
        self.assertEqual(backend.gripper_command, -1)
        self.assertEqual(backend.symbolic_name_to_box_id, snapshot.bindings)
        self.assertIsNot(backend.symbolic_name_to_box_id, snapshot.bindings)

    def test_physical_failure_cannot_become_a_valid_demo(self):
        trace = DemoTrace(
            (np.zeros(43),),
            metadata={
                "status": "completed",
                "backend": BACKEND_ID,
                "simulator_success": False,
            },
        )
        self.assertFalse(validate_trace(trace))
        self.assertEqual(trace.metadata["status"], "invalid")

    def test_failed_rollout_cannot_receive_postcondition_credit(self):
        trajectory = SegmentRollout()
        trajectory.execution_failed = True
        self.assertFalse(postcondition_reached(trajectory, None, None))

    def test_legacy_solver_dispatch_preserves_proof_failure(self):
        from taskbench.solvers.program_synthesis import ProgramSynthesisSolver

        solver = ProgramSynthesisSolver(demos="demo.npz", mode="full", smoke=True)
        with patch(
            "taskbench.roboverify.entry.synthesize_cfg.main", return_value=2
        ) as run:
            result = solver.solve(seed=7)
        self.assertFalse(result.success)
        self.assertEqual(result.verification_status, "unverified")
        self.assertFalse(solver.requires_env)
        self.assertIn("--smoke", run.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
