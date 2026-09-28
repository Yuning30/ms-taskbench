"""Planner selection, bounded execution, payload geometry, and failure reporting."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from taskbench.roboverify.api.control import PrimitiveController
from taskbench.roboverify.api.instructions import Move, PrimitiveExecutionError
from taskbench.roboverify.api.test_control import ServoEnvironment
from taskbench.roboverify.backend import StackBackend
from taskbench.roboverify.cfg.reset import Snapshot
from taskbench.roboverify.entry.collect_demos import build_parser


class JointEnvironment(ServoEnvironment):
    """Ideal joint executor whose first three coordinates represent TCP XYZ."""

    move_controller = "planner"
    planner_step_limit = 200
    gripper_command = 1.0
    held_box_id = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.plan_move = Mock(side_effect=self._plan)

    def _plan(self, target):
        points = np.linspace(self.obs[:3], target, 3)
        return np.c_[points, np.zeros((3, 4))]

    def hold_action(self, *, opened):
        return np.r_[self.obs[:3], np.zeros(4), 1.0 if opened else -1.0]

    def step(self, action):
        self.actions.append(action.copy())
        self.gripper_command = action[-1]
        if not self.stalled:
            displacement = action[:3] - self.obs[:3]
            self.obs[:3] = action[:3]
            if self.held_box_id is not None:
                offset = 10 + 12 * self.held_box_id
                self.obs[offset : offset + 3] += displacement
            self.obs[3:5] = 0.04 if action[-1] > 0 else 0.01


class PlannerControlTests(unittest.TestCase):
    def test_payload_target_and_every_joint_step_are_recorded(self):
        env = JointEnvironment()
        env.held_box_id = 0
        offset = env.obs[:3] - env.obs[10:13]
        target = env.obs[10:13] + [0.1, -0.1, 0.05]
        trajectory = []
        controller = PrimitiveController(env, trajectory, render=True)
        self.assertTrue(controller.move(target, payload_id=0))
        np.testing.assert_allclose(env.plan_move.call_args.args[0], target + offset)
        np.testing.assert_allclose(env.obs[10:13], target)
        self.assertEqual(len(trajectory), len(env.actions))
        self.assertEqual(len(controller.images), len(env.actions))
        self.assertEqual(controller.steps, 3)
        self.assertEqual(np.asarray(env.actions).shape, (3, 8))
        np.testing.assert_array_equal(np.asarray(env.actions)[:, -1], -1)

    def test_plan_failure_raises_without_delta_fallback(self):
        env = JointEnvironment()
        env.plan_move.side_effect = None
        env.plan_move.return_value = None
        instruction = Move(0, 0, 0, target_offset=[0, 0, 0.2])
        with self.assertRaises(PrimitiveExecutionError):
            instruction.eval(env, [])
        self.assertEqual(env.actions, [])
        self.assertEqual(
            instruction.last_control_result.failure_reason, "motion_plan_failed"
        )

    def test_plan_success_requires_measured_convergence(self):
        env = JointEnvironment(stalled=True)
        controller = PrimitiveController(env, [], limit=5)
        self.assertFalse(controller.move(env.obs[:3] + [0.1, 0, 0]))
        self.assertEqual(controller.steps, 5)
        self.assertGreater(controller.result.position_error, 0.002)
        # After following the trajectory, hold its final joint target.
        np.testing.assert_array_equal(env.actions[-1], env.actions[-2])

    def test_budget_covers_path_and_gripper_together(self):
        env = JointEnvironment()
        controller = PrimitiveController(env, [], limit=2)
        self.assertFalse(controller.move(env.obs[:3] + [0.1, 0, 0]))
        self.assertFalse(controller.gripper(opened=True))
        self.assertEqual(len(env.actions), 2)

    def test_planner_instruction_uses_the_explicit_planner_budget(self):
        env = JointEnvironment(stalled=True)
        env.planner_step_limit = 5
        instruction = Move(0, 0, 0, limit=1, target_offset=[0, 0, 0.2])
        with self.assertRaises(PrimitiveExecutionError):
            instruction.eval(env, [])
        self.assertEqual(len(env.actions), 5)
        self.assertEqual(instruction.last_control_result.steps, 5)

    def test_zero_budget_and_already_reached_target_do_not_plan(self):
        env = JointEnvironment()
        controller = PrimitiveController(env, [], limit=0)
        self.assertTrue(controller.move(env.obs[:3]))
        self.assertFalse(controller.move(env.obs[:3] + [0.1, 0, 0]))
        env.plan_move.assert_not_called()
        self.assertEqual(env.actions, [])

    def test_vertical_payload_move_preserves_tcp_xy(self):
        env = JointEnvironment()
        env.held_box_id = 0
        tcp = env.obs[:3].copy()
        controller = PrimitiveController(env, [])
        self.assertTrue(
            controller.move(
                env.obs[10:13] + [10, 10, 0.1], payload_id=0, vertical_only=True
            )
        )
        np.testing.assert_allclose(env.plan_move.call_args.args[0], tcp + [0, 0, 0.1])

    def test_release_preserves_xy_reference_and_uses_joint_gripper_hold(self):
        env = JointEnvironment()
        env.held_box_id = 0
        env.obs[3:5] = 0.02
        tcp = env.obs[:3].copy()
        target = np.r_[tcp[:2], env.obs[10:13][2] + 0.3]
        controller = PrimitiveController(env, [])
        self.assertTrue(controller.release(0, 0.3))
        self.assertIsNone(env.held_box_id)
        np.testing.assert_allclose(env.actions[0][:3], tcp)
        np.testing.assert_allclose(env.obs[:3], target)
        self.assertTrue(all(action[-1] == 1 for action in env.actions))

    def test_joint_commands_are_not_cartesian_scaled(self):
        env = object.__new__(StackBackend)
        env.move_controller = "planner"
        env.simulator = Mock()
        env.simulator.step.return_value = (None, 0, False, False, {})
        env._get_obs = lambda: np.zeros(43)
        command = [0, 0.4, 0, -2, 0, 2.3, 0.8, -1]
        env.step(command)
        np.testing.assert_allclose(env.simulator.step.call_args.args[0], command)
        self.assertEqual(env.gripper_command, -1)
        for invalid in ([0, 0, 0, 0], [np.nan] * 8, [0] * 7 + [2]):
            with self.assertRaises(ValueError):
                env.step(invalid)

    def test_retry_keeps_target_orientation_and_never_steps_physics(self):
        env = object.__new__(StackBackend)
        env.move_controller = "planner"
        env.planner = Mock()
        env.simulator = Mock()
        env.gripper_command = -1.0
        quaternion = np.array([[0.0, 1.0, 0.0, 0.0]])
        env.raw = SimpleNamespace(
            agent=SimpleNamespace(
                tcp=SimpleNamespace(pose=SimpleNamespace(q=quaternion))
            )
        )
        target = [0.1, -0.1, 0.2]
        path = np.zeros((2, 7))
        with (
            patch("taskbench.skills.robot_config.get_robot_config"),
            patch(
                "taskbench.skills.motion.move_to_pose",
                side_effect=[None, {"position": path}],
            ) as plan,
        ):
            np.testing.assert_array_equal(env.plan_move(target), path)
        self.assertEqual(
            [call.kwargs["qpos_step"] for call in plan.call_args_list], [0.01, 0.005]
        )
        for call in plan.call_args_list:
            np.testing.assert_allclose(call.args[2].p, target)
            np.testing.assert_allclose(call.args[2].q, quaternion[0])
            self.assertTrue(call.kwargs["dry_run"])
        env.simulator.step.assert_not_called()
        with (
            patch("taskbench.skills.robot_config.get_robot_config"),
            patch(
                "taskbench.skills.motion.move_to_pose",
                return_value=None,
            ) as plan,
        ):
            self.assertIsNone(env.plan_move(target))
            self.assertEqual(plan.call_count, 2)

    def test_controller_selection_is_validated_and_forwarded_by_hydra(self):
        parser = build_parser()
        args = parser.parse_args(["--program", "example:build_program"])
        self.assertEqual(args.move_controller, "delta")
        args = parser.parse_args(
            ["--program", "example:build_program", "--move-controller", "planner"]
        )
        self.assertEqual(args.move_controller, "planner")
        with self.assertRaises(ValueError):
            StackBackend(move_controller="unknown")
        with self.assertRaises(ValueError):
            StackBackend(move_controller="planner", planner_step_limit=0)
        from taskbench.solvers.program_synthesis import ProgramSynthesisSolver

        solver = ProgramSynthesisSolver(demos="demo.npz", move_controller="planner")
        with patch(
            "taskbench.roboverify.entry.synthesize_cfg.main", return_value=2
        ) as run:
            solver.solve(seed=0)
        argv = run.call_args.args[0]
        self.assertEqual(argv[argv.index("--move-controller") + 1], "planner")

    def test_mismatched_snapshot_and_archive_rejected_before_execution(self):
        env = object.__new__(StackBackend)
        env.move_controller = "delta"
        snapshot = Snapshot(np.zeros(1), {"adapter/move_controller": np.asarray(1)}, {})
        with self.assertRaisesRegex(ValueError, "Snapshot move controller"):
            env.restore(snapshot)
        from taskbench.roboverify.entry.synthesize_cfg import _run

        args = SimpleNamespace(task="stack", demos="unused", move_controller="delta")
        trace = SimpleNamespace(metadata={"move_controller": "planner"})
        with patch(
            "taskbench.roboverify.entry.synthesize_cfg.load_traces",
            return_value=[trace],
        ):
            with self.assertRaisesRegex(ValueError, "Demonstration move controller"):
                _run(args, Mock())


if __name__ == "__main__":
    unittest.main()
