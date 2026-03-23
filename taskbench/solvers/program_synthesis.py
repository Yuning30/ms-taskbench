from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
import math
import random
from typing import Any

from omegaconf import DictConfig

from taskbench.programs.executor import execute_program
from taskbench.programs.ir import Instruction, Program
from taskbench.programs.cem import cem_optimize
from taskbench.programs.params import apply_float_parameters, extract_float_parameters
from taskbench.programs.mcmc import sample_proportional, swap_two_elements_maybe_same
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
        # Cache number of cubes for categorical mutations like obj_name.
        # Updated in solve() from cfg.env.num_cubes when available.
        self.num_cubes: int = int(kwargs.get("num_cubes", 3))
        # MCMC iterations over program structure proposals.
        self.mcmc_iters: int = int(kwargs.get("mcmc_iters", 10))

        # CEM hyperparameters for optimizing continuous float parameters
        # embedded in Program instruction arguments.
        self.cem_iters: int = int(kwargs.get("cem_iters", 10))
        self.cem_N: int = int(kwargs.get("cem_N", 32))
        self.cem_K: int = int(kwargs.get("cem_K", 4))
        self.cem_init_std: float = float(kwargs.get("cem_init_std", 0.1))

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
        # Keep the initial implementation focused on pick/move/place.
        ops = ["skip", "pick", "move", "place"]

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

            # Initialize args for new op so execution doesn't crash.
            # Values here are deliberately simple; CEM can optimize the
            # float-valued subset of these later.
            if new_op == "skip":
                instr.args = {}
            elif new_op == "pick":
                instr.args = {
                    "obj_name": "cube_1",
                    "lift_height": 0.1,  # float (trainable)
                    "verify_grasp": True,  # bool (non-trainable)
                }
            elif new_op == "move":
                instr.args = {
                    "pos": [0.0, 0.0, 0.3],  # floats (trainable)
                    "quat": [1.0, 0.0, 0.0, 0.0],  # floats (trainable)
                    "gripper_open": True,  # bool (non-trainable)
                    "monitor_contacts": True,  # bool (non-trainable)
                }
            elif new_op == "place":
                instr.args = {
                    "pos": [0.0, 0.0, 0.1],  # floats (trainable)
                    "quat": [1.0, 0.0, 0.0, 0.0],  # floats (trainable)
                    "settling_steps": 10,  # int (non-trainable)
                    "retract_height": None,  # optional
                }
            else:
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
            # Only mutate categorical/discrete args here.
            # Continuous float-valued parameters are optimized exclusively by CEM.
            if isinstance(val, str):
                # For pick, mutate which object to grasp.
                if key == "obj_name":
                    # Candidate cube names: cube_0 .. cube_{num_cubes-1}
                    candidates = [f"cube_{i}" for i in range(self.num_cubes)]
                    if len(candidates) > 1:
                        other = random.choice([c for c in candidates if c != val] or candidates)
                        instr.args[key] = other
                        changed = True
            # Otherwise: leave it unchanged (including float / list-of-float).

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

    def _normalize_quaternions(self, program: Program) -> None:
        """Normalize any instruction `args["quat"]` to unit length."""
        for instr in program.instructions:
            q = instr.args.get("quat")
            if not (isinstance(q, list) and len(q) == 4):
                continue
            norm = math.sqrt(sum(float(v) * float(v) for v in q))
            if norm > 1e-8:
                instr.args["quat"] = [float(v) / norm for v in q]

    def _optimize_program_float_params_via_cem(
        self,
        cfg: DictConfig,
        program: Program,
        *,
        cost_fn,
    ) -> tuple[Program, float]:
        """Run CEM to optimize float-valued parameters inside `program`.

        Float parameters are extracted from instruction `args` by recursively
        collecting `float` / `np.floating` values; non-floats (bool/int/None)
        are ignored.
        """
        x0, refs = extract_float_parameters(program)
        if x0.shape[0] == 0 or self.cem_iters <= 0:
            return program, cost_fn(program)

        # Work on a single base copy so we don't repeatedly deepcopy in f().
        base_program = deepcopy(program)

        def f(x):
            apply_float_parameters(base_program, refs, x)
            exec_res = evaluate_program(cfg, base_program, skip_steps=self.skip_steps)
            return self._cost_from_execution(exec_res)

        best_score, best_x = cem_optimize(
            f,
            dim=int(x0.shape[0]),
            iterations=self.cem_iters,
            N=self.cem_N,
            K=self.cem_K,
            init_mu=x0,
            init_std=self.cem_init_std,
        )

        # Apply best parameters back onto the original program object.
        apply_float_parameters(program, refs, best_x)
        self._normalize_quaternions(program)
        return program, float(best_score)

    def solve(self, env, seed=None, cfg=None) -> SolverResult:
        if cfg is None:
            raise ValueError("ProgramSynthesisSolver requires cfg (DictConfig) to run.")

        if seed is None:
            seed = cfg.seed

        # Update cube count for categorical mutations.
        try:
            self.num_cubes = int(cfg.env.num_cubes)
        except Exception:
            pass

        # Build initial program and run an MCMC optimization loop over programs.
        initial_program = self._build_initial_program(cfg)

        def _cost_fn(p: Program) -> float:
            exec_res = evaluate_program(cfg, p, skip_steps=self.skip_steps)
            return self._cost_from_execution(exec_res)

        # --- MCMC with inner CEM optimization (like external mcmc.py) ----
        current_program = deepcopy(initial_program)
        current_cost = _cost_fn(current_program)

        best_program = deepcopy(current_program)
        best_cost = float(current_cost)

        for _ in range(self.mcmc_iters):
            proposed_program, changed = self._mutate_program(current_program)
            if not changed:
                continue

            # Optimize continuous float parameters for the proposed structure.
            proposed_program, proposed_cost = self._optimize_program_float_params_via_cem(
                cfg,
                proposed_program,
                cost_fn=_cost_fn,
            )

            # Metropolis acceptance ratio; higher is better.
            acceptance_ratio = math.exp(proposed_cost - current_cost)
            if random.random() < acceptance_ratio:
                current_program = proposed_program
                current_cost = proposed_cost

            if proposed_cost > best_cost:
                best_cost = proposed_cost
                best_program = deepcopy(proposed_program)

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

