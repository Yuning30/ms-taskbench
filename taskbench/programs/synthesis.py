"""Utility for evaluating a high-level program in a fresh environment."""

from __future__ import annotations

import numpy as np
from omegaconf import DictConfig

from taskbench.envs.factory import make_single_env
from taskbench.programs.executor import ExecutionResult, execute_program
from taskbench.programs.ir import Program
from taskbench.skills.context import SkillContext


def _extract_state_positions(raw, objects: dict[str, object]) -> np.ndarray:
    """Extract low-dim state: [tcp_pos, sorted object positions]."""
    tcp_pos = np.asarray(raw.agent.tcp.pose.p, dtype=np.float64).flatten()[:3]
    parts = [tcp_pos]
    for name in sorted(objects.keys()):
        pos = np.asarray(objects[name].pose.p, dtype=np.float64).flatten()[:3]
        parts.append(pos)
    return np.concatenate(parts, axis=0)


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
        state_positions = []

        def _record_step(*args):
            # Skill callbacks in this codebase can be invoked either with
            # full transition tuples (obs, rew, term, trunc, info) or with
            # no arguments from low-level motion-following loops.
            if len(args) == 5:
                trajectory.append(args)
            # Capture low-dimensional state at every callback invocation.
            state_positions.append(_extract_state_positions(env.unwrapped, ctx.objects))

        ctx = SkillContext(env, step_callback=_record_step)
        ctx.reset(seed=cfg.seed)
        # Include initial state before any action.
        state_positions.append(_extract_state_positions(env.unwrapped, ctx.objects))
        result = execute_program(ctx, program, step_buffer=trajectory, skip_steps=skip_steps)
        result.info["state_positions"] = np.asarray(state_positions, dtype=np.float64)
        return result
    finally:
        env.close()

