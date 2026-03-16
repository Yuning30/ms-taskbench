from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
import random
from typing import Any

from omegaconf import DictConfig

from taskbench.programs.executor import execute_program
from taskbench.programs.ir import Instruction, Program
from taskbench.programs.mcmc import mcmc_optimize_program, sample_proportional, swap_two_elements_maybe_same
from taskbench.programs.synthesis import evaluate_program
from taskbench.solver import BaseSolver, SolverResult, register_solver
from taskbench.skills.context import SkillContext


@register_solver("program_synthesis")
class ProgramSynthesisSolver(BaseSolver):
    """Example solver that delegates to a program synthesis routine.

    The solver:
    - builds an initial Program
    - calls ``optimize_program`` to refine it (by default a no-op)
    - executes the resulting Program once to produce a SolverResult
    """

    def __init__(self, **kwargs):
        # Accept arbitrary kwargs so Hydra config can pass through fields
        # without breaking solver construction.
        self.kwargs = kwargs
        # Length of the initial program (number of instruction slots).
        # This can be overridden via Hydra:
        #   run.solver_kwargs.program_length=5
        self.program_length: int = int(kwargs.get("program_length", 3))
        # Number of env steps to advance for each SKIP instruction.
        #   run.solver_kwargs.skip_steps=5
        self.skip_steps: int = int(kwargs.get("skip_steps", 0))

    def _build_initial_program(self, cfg: DictConfig | None) -> Program:
        """Construct an initial program of the requested length.

        All instruction slots are initialized to a ``skip`` op, which is a
        no-op in the executor. A synthesis algorithm can then rewrite these
        instructions in-place (changing ``op`` and ``args``) without having
        to change the length.
        """
        instructions = [Instruction("skip", {}) for _ in range(self.program_length)]
        return Program(instructions=instructions)

    # --- MCMC helpers -----------------------------------------------------

    def _mutate_program(self, current: Program) -> tuple[Program, bool]:
        """Mutate a program in a simple, generic way.

        This mirrors the external MCMC structure:
        - mutate opcode
        - mutate operand (args)
        - swap two instructions
        """
        # Weights for mutation types: opcode, operand, swap
        pc = 1  # opcode
        po = 1  # operand
        ps = 1  # swap
        which = sample_proportional([pc, po, ps])

        new_program = deepcopy(current)
        ops = ["skip", "pick", "move", "place", "push"]

        # 0: mutate opcode
        if which == 0:
            if not new_program.instructions:
                return new_program, False
            idx = int(self.kwargs.get("rng_opcode_idx", 0)) if False else None  # placeholder for future control
            if idx is None:
                idx = random.randint(0, len(new_program.instructions) - 1)
            instr = new_program.instructions[idx]
            old_op = instr.op
            new_op = old_op
            # Ensure we actually change the opcode if possible.
            candidates = [o for o in ops if o != old_op]
            if not candidates:
                return new_program, False
            new_op = random.choice(candidates)
            instr.op = new_op
            # Reset args when opcode changes.
            instr.args = {}
            return new_program, True

        # 1: mutate operand (one arg value)
        if which == 1:
            if not new_program.instructions:
                return new_program, False
            idx = random.randint(0, len(new_program.instructions) - 1)
            instr = new_program.instructions[idx]
            if not instr.args:
                return new_program, False

            key = random.choice(list(instr.args.keys()))
            val = instr.args[key]

            changed = False
            if isinstance(val, (int, float)):
                # Small Gaussian perturbation for numeric args.
                new_val = float(val) + float(random.gauss(0.0, 0.05))
                instr.args[key] = new_val
                changed = True
            elif isinstance(val, list) and all(isinstance(x, (int, float)) for x in val):
                # Element-wise perturbation for vector args (e.g., pos).
                new_list = [float(x) + float(random.gauss(0.0, 0.02)) for x in val]
                instr.args[key] = new_list
                changed = True

            return new_program, changed

        # 2: swap two instructions
        if which == 2:
            if not new_program.instructions:
                return new_program, False
            swap_two_elements_maybe_same(new_program.instructions)
            return new_program, True

        return new_program, False

    def _cost_from_execution(self, exec_result) -> float:
        """Map an ExecutionResult to a scalar cost (higher is better).

        This is intentionally simple and can be replaced with a more
        sophisticated objective that uses ``trajectory``.
        """
        base = 1.0 if exec_result.success else 0.0
        # Mild penalty on length to encourage shorter solutions.
        length_penalty = 0.01 * float(exec_result.steps)
        return base - length_penalty

    def solve(self, env, seed=None, cfg=None) -> SolverResult:
        if cfg is None:
            raise ValueError("ProgramSynthesisSolver requires cfg (DictConfig) to run.")

        if seed is None:
            seed = cfg.seed

        # Build initial program and run an MCMC optimization loop over programs.
        initial_program = self._build_initial_program(cfg)

        def _cost_fn(p: Program) -> float:
            exec_res = evaluate_program(cfg, p, skip_steps=self.skip_steps)
            return self._cost_from_execution(exec_res)

        def _mutate(p: Program) -> tuple[Program, bool]:
            return self._mutate_program(p)

        _samples, best_program = mcmc_optimize_program(
            initial_program,
            cost_fn=_cost_fn,
            mutate_fn=_mutate,
            iters=int(self.kwargs.get("mcmc_iters", 10)),
        )

        # Execute the final program once to produce solver metrics.
        ctx = SkillContext(env)
        ctx.reset(seed=seed)
        # For online execution we also record the trajectory; we treat skip
        # instructions the same way as in offline evaluation.
        trajectory = []

        def _record_step(obs, rew, term, trunc, info):
            trajectory.append((obs, rew, term, trunc, info))

        ctx.step_callback = _record_step  # type: ignore[attr-defined]
        exec_result = execute_program(ctx, best_program, step_buffer=trajectory)

        info = {
            "program": [asdict(instr) for instr in best_program.instructions],
            **exec_result.info,
        }

        return SolverResult(
            success=exec_result.success,
            reward=exec_result.reward,
            elapsed_steps=exec_result.steps,
            info=info,
            failure_reason=exec_result.failure_reason,
        )

