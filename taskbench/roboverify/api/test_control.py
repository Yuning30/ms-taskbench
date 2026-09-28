"""Controller convergence, bounded execution, and numeric/named parity."""

import hashlib
import json
import unittest
from copy import deepcopy
from dataclasses import replace

import numpy as np

from taskbench.roboverify.api.control import (
    ControlConfig,
    PrimitiveController,
    get_move_action,
)
from taskbench.roboverify.api.instructions import (
    Move,
    Pick,
    PrimitiveExecutionError,
    Release,
)
from taskbench.roboverify.api.program import Program
from taskbench.roboverify.api.runtime import execute_instruction
from taskbench.roboverify.cfg.collection import validate_trace
from taskbench.roboverify.cfg.physical import name_operands
from taskbench.roboverify.cfg.program_source import (
    describe_program,
    program_fingerprint,
)
from taskbench.roboverify.util.actions import bound_delta_action


class ServoEnvironment:
    """Deterministic servo; tests termination independently of MuJoCo contacts."""

    def __init__(self, *, stalled=False):
        self.env = self
        self.symbolic_name_to_box_id = {"base": 0, "other": 1}
        self.obs = np.zeros(43)
        self.obs[:3] = [0.1, 0.1, 0.6]
        self.obs[3:5] = 0.04
        self.obs[10:13] = [0, 0, 0.425]
        self.obs[22:25] = [0.2, 0, 0.425]
        self.actions, self.stalled = [], stalled

    def _get_obs(self):
        return self.obs.copy()

    def flatten_observation(self, obs):
        return obs

    def step(self, action):
        self.actions.append(action.copy())
        if not self.stalled:
            self.obs[:3] += 0.05 * np.clip(action[:3], -1, 1)
            self.obs[3:5] = np.clip(self.obs[3:5] + action[3] * 0.02, 0.01, 0.04)

    def render(self):
        return self.obs[:3].copy()


def placement(control):
    return [
        Pick(1, control=control),
        Move(1, 1, 0, target_offset=[0, 0, 0.20], control=control),
        Move(0, 0, 0, target_offset=[0, 0, 0.20], control=control),
        Move(0, 0, 0, target_offset=[0, 0, 0.05], control=control),
        Release(1, target_z=0.15, control=control),
    ]


class ControlTests(unittest.TestCase):
    def test_action_is_proportional_and_has_explicit_gripper_command(self):
        np.testing.assert_allclose(
            get_move_action([1, 2, 3], [1.1, 1.8, 3.3], gain=2, close_gripper=True),
            [0.2, -0.4, 0.6, -0.2],
        )
        np.testing.assert_array_equal(
            get_move_action([0, 0, 0], [0, 0, 0]), np.zeros(4)
        )
        with self.assertRaises(TypeError):
            get_move_action([0, 0, 0], [0, 0, 0], atol=0.1)

    def test_large_commands_keep_direction_and_gripper_strength(self):
        for error in ([0.2, 0.05, 0], [-0.2, 0.05, -0.1], [0, 0, 0.2]):
            raw = 20 * np.asarray(error)
            for closed in (False, True):
                action = get_move_action([0, 0, 0], error, close_gripper=closed)
                self.assertLessEqual(np.max(np.abs(action[:3])), 1)
                np.testing.assert_allclose(np.cross(raw, action[:3]), 0, atol=1e-15)
                np.testing.assert_allclose(action[:3], raw / np.max(np.abs(raw)))
                self.assertEqual(action[3], -0.2 if closed else 0)

    def test_vertical_only_ignores_xy_before_scaling(self):
        env = ServoEnvironment()
        target = env.obs[:3] + [10, 10, 0.1]
        controller = PrimitiveController(env, [])
        self.assertTrue(controller.move(target, vertical_only=True))
        self.assertEqual(controller.steps, 2)
        np.testing.assert_allclose(np.array(env.actions)[:, :3], [[0, 0, 1], [0, 0, 1]])

    def test_action_bounds_are_idempotent_and_do_not_mutate_input(self):
        raw = np.array([4, 1, -2, -3])
        bounded = bound_delta_action(raw)
        np.testing.assert_array_equal(raw, [4, 1, -2, -3])
        np.testing.assert_array_equal(bounded, [1, 0.25, -0.5, -1])
        np.testing.assert_array_equal(bound_delta_action(bounded), bounded)
        for invalid in ([1, 2, 3], [1, 2, 3, np.nan], [np.inf, 0, 0, 0]):
            with self.assertRaises(ValueError):
                bound_delta_action(invalid)

    def test_vertical_payload_motion_uses_payload_position_before_scaling(self):
        env = ServoEnvironment()
        env.obs[:3] = [20, 20, 20]
        controller = PrimitiveController(env, [], limit=1)
        target = controller.box_position(0) + [10, 10, 0.01]
        controller.move(target, vertical_only=True, payload_id=0)
        np.testing.assert_allclose(env.actions[0], [0, 0, 0.2, -0.2])

    def test_position_tolerance_controls_termination(self):
        counts = []
        for tolerance in (0.02, 0.002):
            env, trajectory = ServoEnvironment(), []
            controller = PrimitiveController(
                env,
                trajectory,
                control=ControlConfig(position_tolerance=tolerance, gain=10),
            )
            target = env.obs[:3] + [0.1, 0.1, 0.1]
            self.assertTrue(controller.move(target))
            self.assertLessEqual(np.linalg.norm(env.obs[:3] - target), tolerance)
            self.assertEqual(len(trajectory), controller.steps)
            np.testing.assert_array_equal(trajectory[-1], env.obs)
            counts.append(controller.steps)
        self.assertGreater(counts[1], counts[0])

    def test_exact_boundary_and_zero_budget_do_not_step(self):
        env = ServoEnvironment()
        controller = PrimitiveController(
            env, [], limit=0, control=ControlConfig(position_tolerance=0.125)
        )
        self.assertTrue(controller.move(env.obs[:3] + [0, 0, 0.125]))
        self.assertEqual(env.actions, [])
        self.assertFalse(controller.move(env.obs[:3] + [0, 0, 0.25]))
        self.assertFalse(controller.result.converged)
        self.assertEqual(controller.result.steps, 0)

    def test_step_budget_covers_entire_pick_and_reports_failed_phase(self):
        env = ServoEnvironment()
        instruction = Pick(
            1, limit=8, control=ControlConfig(position_tolerance=0.005, gain=10)
        )
        with self.assertRaises(PrimitiveExecutionError):
            instruction.eval(env, [])
        self.assertEqual(len(env.actions), 8)
        self.assertFalse(instruction.last_control_result.converged)
        self.assertEqual(instruction.last_control_result.phase, "descend")

    def test_stalled_controller_and_runtime_event_preserve_failure(self):
        env, events = ServoEnvironment(stalled=True), []
        instruction = Move(0, 0, 0, limit=3)
        with self.assertRaises(PrimitiveExecutionError):
            execute_instruction(
                instruction, env, [env.obs.copy()], path="0", on_event=events.append
            )
        self.assertEqual(len(env.actions), 3)
        self.assertEqual(instruction.last_control_result.steps, 3)
        self.assertFalse(instruction.last_control_result.converged)
        self.assertGreater(instruction.last_control_result.position_error, 0)
        self.assertFalse(any(e["kind"] == "instruction_end" for e in events))

    def test_release_opens_before_vertical_retreat(self):
        env = ServoEnvironment()
        env.obs[3:5] = 0.02
        instruction = Release(0, target_z=0.3)
        instruction.eval(env, [])
        self.assertTrue(instruction.last_control_result.converged)
        split = next(i for i, a in enumerate(env.actions) if a[2] != 0)
        self.assertGreater(split, 0)
        self.assertTrue(
            all(a[3] > 0 and np.all(a[:3] == 0) for a in env.actions[:split])
        )
        self.assertTrue(
            all(a[3] == 0 and np.all(a[:2] == 0) for a in env.actions[split:])
        )

    def test_release_corrects_opening_and_motion_drift_to_original_xy(self):
        class DriftingServo(ServoEnvironment):
            def step(self, action):
                super().step(action)
                self.obs[:2] += [0.003, -0.002] if action[3] > 0 else [0.0005, -0.0003]

        env = DriftingServo()
        env.obs[3:5] = 0.02
        target = np.r_[env.obs[:2].copy(), env.obs[12] + 0.3]
        instruction = Release(0, target_z=0.3)
        instruction.eval(env, [])
        self.assertTrue(instruction.last_control_result.converged)
        self.assertLessEqual(np.linalg.norm(env.obs[:3] - target), 0.002)
        first_retreat = next(a for a in env.actions if a[2] != 0)
        self.assertLess(first_retreat[0], 0)
        self.assertGreater(first_retreat[1], 0)
        self.assertAlmostEqual(
            instruction.last_control_result.position_error,
            np.linalg.norm(env.obs[:3] - target),
        )

    def test_release_rejects_xy_error_even_when_z_has_converged(self):
        class StalledXYServo(ServoEnvironment):
            def step(self, action):
                xy = self.obs[:2].copy()
                super().step(action)
                self.obs[:2] = xy + ([0.003, 0] if action[3] > 0 else [0, 0])

        env = StalledXYServo()
        env.obs[3:5] = 0.02
        instruction = Release(0, target_z=env.obs[2] - env.obs[12], limit=5)
        with self.assertRaises(PrimitiveExecutionError):
            instruction.eval(env, [])
        self.assertFalse(instruction.last_control_result.converged)
        self.assertEqual(instruction.last_control_result.steps, 5)
        self.assertEqual(instruction.last_control_result.phase, "retreat")
        self.assertGreater(instruction.last_control_result.position_error, 0.002)

    def test_default_pick_tolerance_is_shared_by_numeric_and_named_forms(self):
        numeric = Pick(1)
        named = name_operands(numeric, {1: "other"})
        self.assertEqual(numeric.control.position_tolerance, 0.002)
        self.assertEqual(named.control, numeric.control)

    def test_named_conversion_preserves_controls_actions_observations_and_frames(self):
        config = ControlConfig(position_tolerance=0.003, gain=8)
        numeric_env, named_env = ServoEnvironment(), ServoEnvironment()
        for numeric in placement(config):
            named = name_operands(numeric, {0: "base", 1: "other"})
            self.assertEqual(named.control, config)
            numeric_states, named_states = [], []
            numeric_images = numeric.eval(numeric_env, numeric_states, True)
            named_images = named.eval(named_env, named_states, True)
            np.testing.assert_array_equal(numeric_env.actions, named_env.actions)
            np.testing.assert_array_equal(numeric_states, named_states)
            np.testing.assert_array_equal(numeric_images, named_images)
            self.assertEqual(numeric.last_control_result, named.last_control_result)

    def test_rendering_does_not_change_execution(self):
        instructions = placement(ControlConfig())
        envs = [ServoEnvironment(), ServoEnvironment()]
        for env, render in zip(envs, (False, True)):
            for inst in deepcopy(instructions):
                inst.eval(env, [], render)
        np.testing.assert_array_equal(envs[0].actions, envs[1].actions)
        np.testing.assert_array_equal(envs[0].obs, envs[1].obs)

    def test_control_settings_are_validated_and_part_of_fingerprint(self):
        for field in (
            "position_tolerance",
            "gain",
            "gripper_threshold",
            "gripper_tolerance",
        ):
            for invalid in (0, -1, float("nan"), float("inf")):
                with (
                    self.subTest(field=field, invalid=invalid),
                    self.assertRaises(ValueError),
                ):
                    ControlConfig(**{field: invalid})
        original = Program(1, [Move()])
        changed = deepcopy(original)
        changed.instructions[0].control = replace(
            original.instructions[0].control, position_tolerance=0.004
        )
        self.assertNotEqual(program_fingerprint(original), program_fingerprint(changed))
        self.assertEqual(
            describe_program(changed)[0]["control"]["position_tolerance"], 0.004
        )
        self.assertEqual(
            describe_program(changed)[0]["action_scaling"], "uniform-xyz-v1"
        )
        # The old descriptor omitted action semantics and must have a different ID.
        old_description = deepcopy(describe_program(changed))
        old_description[0].pop("action_scaling")
        old_fingerprint = hashlib.sha256(
            json.dumps(old_description, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        self.assertNotEqual(program_fingerprint(changed), old_fingerprint)
        # Runtime diagnostics must not change executable identity.
        before = program_fingerprint(changed)
        changed.instructions[0].eval(ServoEnvironment(), [])
        self.assertEqual(program_fingerprint(changed), before)

    def test_unconverged_control_cannot_be_an_accepted_demo(self):
        from taskbench.roboverify.backend import BACKEND_ID
        from taskbench.roboverify.cfg.demos import DemoTrace

        trace = DemoTrace(
            (np.zeros(43),),
            metadata={
                "status": "completed",
                "backend": BACKEND_ID,
                "simulator_success": True,
            },
        )
        trace.events = (
            {
                "kind": "instruction_end",
                "path": "0",
                "control": {"converged": False, "steps": 50, "phase": "move"},
            },
        )
        self.assertFalse(validate_trace(trace))
        self.assertEqual(trace.metadata["status"], "incomplete")
        self.assertIn("step limit", trace.metadata["reason"])


if __name__ == "__main__":
    unittest.main()
