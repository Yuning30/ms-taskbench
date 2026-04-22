"""Execute the canonical Build2D linked-list program in Build2D-v1."""

from __future__ import annotations

import logging

from omegaconf import DictConfig

from taskbench.programs.build2d_dsl import canonical_build2d_program
from taskbench.skills.context import SkillContext
from taskbench.solver import BaseSolver, SolverResult, register_solver

logger = logging.getLogger("taskbench.solvers.build2d_program")


class _SkillRuntime:
    """Bridge DSL `put_block(node)` to pick/place skills."""

    def __init__(self, ctx: SkillContext, block_order: list[str], pick_lift_height: float):
        self.ctx = ctx
        self.block_order = block_order
        self.pick_lift_height = pick_lift_height
        self.next_block_idx = 0
        self.steps = 0
        self.last_failure_reason: str | None = None

    def put_block(self, node) -> None:
        if self.next_block_idx >= len(self.block_order):
            raise RuntimeError("No more source blocks available for put_block().")
        block_name = self.block_order[self.next_block_idx]
        self.next_block_idx += 1

        pick_result = self.ctx.pick(block_name, lift_height=self.pick_lift_height)
        self.steps += 1
        if not pick_result.success:
            self.last_failure_reason = pick_result.failure_reason or "pick_failed"
            raise RuntimeError(self.last_failure_reason)

        place_result = self.ctx.place(([node.x, node.y, node.z], pick_result.lift_pose.q))
        self.steps += 1
        if not place_result.success:
            self.last_failure_reason = place_result.failure_reason or "place_failed"
            raise RuntimeError(self.last_failure_reason)


@register_solver("build2d_program")
class Build2DProgramSolver(BaseSolver):
    """Run build2d(h) where h is env's linked-list head."""

    def __init__(self, **kwargs):
        self.pick_lift_height = float(kwargs.get("pick_lift_height", 0.12))

    def solve(self, env, seed=None, cfg: DictConfig | None = None) -> SolverResult:
        if cfg is None:
            raise ValueError("Build2DProgramSolver requires cfg (DictConfig).")

        raw = env.unwrapped
        if not hasattr(raw, "get_grid_head"):
            raise ValueError("Environment must provide get_grid_head() for build2d_program.")
        if not hasattr(raw, "get_available_block_names"):
            raise ValueError(
                "Environment must provide get_available_block_names() for build2d_program."
            )

        ctx = SkillContext(env)
        ctx.reset(seed=seed)

        program = canonical_build2d_program()
        runtime = _SkillRuntime(
            ctx=ctx,
            block_order=list(raw.get_available_block_names()),
            pick_lift_height=self.pick_lift_height,
        )

        try:
            program.eval(raw.get_grid_head(), runtime)
        except Exception as exc:
            reason = runtime.last_failure_reason or str(exc)
            logger.warning("Build2D program execution failed: %s", reason)
            return SolverResult(
                success=False,
                reward=0.0,
                elapsed_steps=runtime.steps,
                info={"program": "canonical_build2d_program"},
                failure_reason=reason,
            )

        info = raw.evaluate()
        success = bool(info["success"].item())
        return SolverResult(
            success=success,
            reward=1.0 if success else 0.0,
            elapsed_steps=runtime.steps,
            info={
                "program": "canonical_build2d_program",
                "num_put_calls": runtime.next_block_idx,
                "env_info": info,
            },
            failure_reason=None if success else "grid_not_completed",
        )

