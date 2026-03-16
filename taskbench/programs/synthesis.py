"""Utility for evaluating a high-level program in a fresh environment."""

from __future__ import annotations

from omegaconf import DictConfig

from taskbench.envs.factory import make_single_env
from taskbench.programs.executor import ExecutionResult, execute_program
from taskbench.programs.ir import Program
from taskbench.skills.context import SkillContext


def evaluate_program(cfg: DictConfig, program: Program, *, skip_steps: int = 0) -> ExecutionResult:
    """Run a single program once in a fresh environment and return its outcome.

    This is a low-level building block for any synthesis algorithm: you can call
    it repeatedly with different programs and plug the results into e.g. random
    search, evolutionary methods, or gradient-free optimizers.
    """
    env = make_single_env(cfg.env)
    try:
        # Buffer to capture all env transitions, including those from skills
        # (via SkillContext.step_callback) and from explicit skip steps.
        trajectory = []

        def _record_step(obs, rew, term, trunc, info):
            trajectory.append((obs, rew, term, trunc, info))

        ctx = SkillContext(env, step_callback=_record_step)
        ctx.reset(seed=cfg.seed)
        return execute_program(ctx, program, step_buffer=trajectory, skip_steps=skip_steps)
    finally:
        env.close()

