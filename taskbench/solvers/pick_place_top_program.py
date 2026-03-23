"""Two-cube pick-and-place (cube_0 -> cube_1 top) using Program abstraction."""

from __future__ import annotations

import logging

from omegaconf import DictConfig

from taskbench.programs.executor import execute_program
from taskbench.programs.ir import Instruction, Program
from taskbench.skills.context import SkillContext
from taskbench.solver import BaseSolver, SolverResult, register_solver

logger = logging.getLogger("taskbench.solvers.pick_place_top_program")


@register_solver("pick_place_top_program")
class PickPlaceTopProgramSolver(BaseSolver):
    """Pick `cube_0`, then place it on top of `cube_1`."""

    def solve(self, env, seed=None, cfg: DictConfig | None = None) -> SolverResult:
        if cfg is None:
            raise ValueError("PickPlaceTopProgramSolver requires cfg (DictConfig).")

        # Planner motion assumes joint-position control.
        assert env.unwrapped.control_mode in [
            "pd_joint_pos",
            "pd_joint_pos_vel",
            "pd_ee_delta_pose",
            "pd_ee_delta_pos",
        ], (
            f"Unsupported control mode: {env.unwrapped.control_mode}"
        )

        base_name = "cube_0"
        target_name = "cube_1"
        lift_height = 0.13
        place_offsets = [-0.001154974102973938, 0.0009374544024467468, 0.05267960578203201]
        retract_height = 0.15
        logger.info("Using hardcoded values: lift_height=%.6f offsets=%s retract_height=%.6f", lift_height, place_offsets, retract_height)

        program = Program(
            instructions=[
                Instruction(
                    "pick",
                    {
                        "obj_name": base_name,
                        "lift_height": lift_height,
                    },
                ),
                Instruction(
                    "place",
                    {
                        "target_cube": target_name,
                        "offsets": place_offsets,
                        "retract_height": retract_height,
                    },
                ),
            ]
        )
        ctx = SkillContext(env)
        ctx.reset(seed=seed)
        exec_result = execute_program(ctx, program)

        return SolverResult(
            success=exec_result.success,
            reward=exec_result.reward,
            elapsed_steps=exec_result.steps,
            info=exec_result.info,
            failure_reason=exec_result.failure_reason,
        )

