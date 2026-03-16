from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from taskbench.programs.ir import Instruction, Program
from taskbench.skills.context import SkillContext


@dataclass
class ExecutionResult:
    """Outcome of executing a Program once in an environment.

    ``trajectory`` stores the sequence of environment transitions observed while
    the program was running. Each element is a 5-tuple:

        (obs, reward, terminated, truncated, info)

    This is populated in two ways:
    - For skill instructions, via ``SkillContext``'s ``step_callback``.
    - For ``skip`` instructions, via direct calls to ``env.step()``.
    """

    success: bool
    reward: float = 0.0
    steps: int = 0
    info: dict[str, Any] = field(default_factory=dict)
    failure_reason: Optional[str] = None
    trajectory: list[tuple[Any, Any, Any, Any, dict[str, Any]]] = field(default_factory=list)


def _run_instruction(ctx: SkillContext, instr: Instruction):
    """Dispatch a single instruction to the appropriate SkillContext method."""
    if instr.op == "skip":
        # No-op: used to reserve slots in a program that can later be
        # rewritten by a synthesis algorithm. Returns None to signal that
        # no env interaction occurred.
        return None
    if instr.op == "pick":
        return ctx.pick(**instr.args)
    if instr.op == "place":
        pos = instr.args.get("pos")
        quat = instr.args.get("quat")
        target_pose = (pos, quat)
        return ctx.place(target_pose, **{k: v for k, v in instr.args.items() if k not in {"pos", "quat"}})
    if instr.op == "move":
        pos = instr.args.get("pos")
        quat = instr.args.get("quat")
        target_pose = (pos, quat)
        return ctx.move(target_pose, **{k: v for k, v in instr.args.items() if k not in {"pos", "quat"}})
    if instr.op == "push":
        approach = instr.args.get("approach")
        push_pose = instr.args.get("push_pose")
        return ctx.push(approach, push_pose, **{k: v for k, v in instr.args.items() if k not in {"approach", "push_pose"}})

    raise ValueError(f"Unknown instruction op: {instr.op!r}")


def execute_program(
    ctx: SkillContext,
    program: Program,
    *,
    reward_fn: Optional[Callable[[dict[str, Any]], float]] = None,
    step_buffer: Optional[list[tuple[Any, Any, Any, Any, dict[str, Any]]]] = None,
    skip_steps: int = 0,
) -> ExecutionResult:
    """Execute a high-level program using the provided SkillContext.

    Args:
        ctx: Active SkillContext (env must already be reset).
        program: Program to execute.
        reward_fn: Optional function mapping the env's final ``evaluate()``
            info dict to a scalar reward. If omitted, success is used as
            a 0/1 reward.
    """
    if step_buffer is None:
        step_buffer = []

    total_steps = 0
    cumulative_reward = 0.0

    for instr in program.instructions:
        if instr.op == "skip" and skip_steps > 0:
            # Advance the environment with a zero action for ``skip_steps``.
            env = ctx.env
            zero_action = env.action_space.sample() * 0
            for _ in range(skip_steps):
                obs, rew, term, trunc, info = env.step(zero_action)
                step_buffer.append((obs, rew, term, trunc, info))
                total_steps += 1
                if hasattr(rew, "shape"):
                    cumulative_reward += float(rew[0])
                else:
                    cumulative_reward += float(rew)
            continue

        result = _run_instruction(ctx, instr)

        # Skip instructions with zero skip_steps do nothing and do not contribute
        # steps or reward.
        if result is None:
            continue

        # Each skill returns a SkillResult with optional last (obs, rew, term, trunc, info)
        if result.step_result is not None:
            _obs, rew, term, trunc, _info = result.step_result
            if hasattr(rew, "shape"):
                rew_scalar = float(rew[0])
            else:
                rew_scalar = float(rew)
            cumulative_reward += rew_scalar
            total_steps += 1
        else:
            rew_scalar = 0.0

        if not result.success:
            return ExecutionResult(
                success=False,
                reward=cumulative_reward,
                steps=total_steps,
                info={},
                failure_reason=result.failure_reason,
                trajectory=step_buffer,
            )

    raw = ctx.env.unwrapped
    info = raw.evaluate()
    success = bool(info["success"].item())

    if reward_fn is not None:
        reward = float(reward_fn(info))
    else:
        # Fall back to cumulative_reward if no reward_fn is provided.
        reward = cumulative_reward if cumulative_reward != 0.0 else (1.0 if success else 0.0)

    return ExecutionResult(
        success=success,
        reward=reward,
        steps=total_steps,
        info={"env_info": info},
        failure_reason=None,
        trajectory=step_buffer,
    )


