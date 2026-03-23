from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
import json
import logging
import math
import multiprocessing as mp
from pathlib import Path
import random
from typing import Any

import h5py
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.stats import gaussian_kde

from taskbench.programs.ir import Instruction, Program
from taskbench.programs.cem import cem_optimize
from taskbench.programs.params import apply_float_parameters, extract_float_parameters
from taskbench.programs.mcmc import sample_proportional, swap_two_elements_maybe_same
from taskbench.programs.synthesis import evaluate_program
from taskbench.solver import BaseSolver, SolverResult, register_solver

logger = logging.getLogger("taskbench.solvers.program_synthesis")
_EXPERT_STATE_CACHE: dict[tuple[str, int], np.ndarray] = {}


def _load_expert_states_for_seed(seed: int, demo_dir: str) -> np.ndarray:
    """Load expert low-dim states: [tcp_pos, sorted object positions]."""
    key = (demo_dir, int(seed))
    if key in _EXPERT_STATE_CACHE:
        return _EXPERT_STATE_CACHE[key]

    path = Path(demo_dir) / f"episode_seed{int(seed)}.hdf5"
    if not path.exists():
        raise FileNotFoundError(f"Missing demonstration file for seed {seed}: {path}")

    with h5py.File(path, "r") as f:
        tcp = np.asarray(f["robot/tcp_pos"], dtype=np.float64)
        obj_names = sorted(list(f["objects"].keys()))
        obj_pos = [np.asarray(f[f"objects/{name}/pos"], dtype=np.float64) for name in obj_names]
        expert = np.concatenate([tcp, *obj_pos], axis=1)

    _EXPERT_STATE_CACHE[key] = expert
    return expert


def _kl_divergence_kde(states_p: np.ndarray, states_q: np.ndarray, *, n_samples: int = 10000) -> float:
    """Estimate KL(P || Q) where P=candidate and Q=expert occupancy."""
    if states_p.ndim != 2 or states_q.ndim != 2:
        raise ValueError("states must be 2D arrays")
    if states_p.shape[1] != states_q.shape[1]:
        raise ValueError(
            f"State dim mismatch: candidate={states_p.shape[1]} expert={states_q.shape[1]}"
        )
    if len(states_p) < 2 or len(states_q) < 2:
        return 1e6

    n = min(int(n_samples), len(states_p))
    idx = np.random.choice(len(states_p), size=n, replace=(n > len(states_p)))
    samples = states_p[idx]

    eps = 1e-10
    try:
        states_p_noisy = states_p + 1e-6 * np.random.randn(*states_p.shape)
        kde_p = gaussian_kde(states_p_noisy.T)
        kde_q = gaussian_kde(states_q.T)
        p_vals = np.clip(kde_p(samples.T), eps, None)
        q_vals = np.clip(kde_q(samples.T), eps, None)
        kl = float(np.mean(np.log(p_vals / q_vals)))
        if not np.isfinite(kl):
            return 1e6
        return kl
    except Exception:
        # KDE can fail on degenerate / near-constant trajectories.
        return 1e6


def _evaluate_program_for_seed(
    cfg_yaml: str,
    program: Program,
    skip_steps: int,
    eval_seed: int,
) -> dict[str, Any]:
    """Evaluate one program on one seed in an isolated process."""
    cfg = OmegaConf.create(cfg_yaml)
    OmegaConf.update(cfg, "seed", int(eval_seed))
    exec_res = evaluate_program(cfg, program, skip_steps=skip_steps)
    cand_states = np.asarray(exec_res.info.get("state_positions", []), dtype=np.float64)
    return {
        "seed": int(eval_seed),
        "success": bool(exec_res.success),
        "reward": float(exec_res.reward),
        "steps": int(exec_res.steps),
        "failure_reason": exec_res.failure_reason,
        "state_positions": cand_states,
    }


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
        _num_workers = kwargs.get("num_workers", None)
        self.num_workers: int | None = None if _num_workers is None else int(_num_workers)
        self.eval_seeds_raw = kwargs.get("eval_seeds", kwargs.get("seeds"))
        self.demo_dir: str = str(kwargs.get("demo_dir", "data/success"))
        self.kl_samples: int = int(kwargs.get("kl_samples", 10000))

    def _parse_eval_seeds(self, cfg: DictConfig, fallback_seed: int) -> list[int]:
        """Resolve evaluation seeds from solver kwargs or fallback to one seed."""
        raw = self.eval_seeds_raw
        if raw is None:
            return [int(fallback_seed)]

        if isinstance(raw, str):
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            return [int(p) for p in parts]
        if isinstance(raw, (list, tuple)):
            return [int(x) for x in raw]
        # OmegaConf ListConfig or other sequence-like
        try:
            return [int(x) for x in list(raw)]
        except Exception as exc:
            raise ValueError(f"Invalid eval_seeds value: {raw!r}") from exc

    def _evaluate_program_across_seeds(
        self,
        *,
        cfg_yaml: str,
        program: Program,
        eval_seeds: list[int],
        pool: mp.pool.Pool,
    ) -> list[dict[str, Any]]:
        jobs = [(cfg_yaml, program, self.skip_steps, int(s)) for s in eval_seeds]
        return pool.starmap(_evaluate_program_for_seed, jobs)

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

    def _score_from_rollouts(
        self,
        per_seed_results: list[dict[str, Any]],
        expert_states_total: np.ndarray,
    ) -> tuple[float, float]:
        """Compute one global KL over concatenated states across all seeds."""
        cand_chunks = [
            np.asarray(r.get("state_positions", []), dtype=np.float64)
            for r in per_seed_results
            if np.asarray(r.get("state_positions", [])).ndim == 2
            and np.asarray(r.get("state_positions", [])).shape[0] > 0
        ]
        if not cand_chunks:
            return 1e6, -1e6
        cand_total = np.concatenate(cand_chunks, axis=0)
        total_kl = _kl_divergence_kde(
            cand_total,
            expert_states_total,
            n_samples=self.kl_samples,
        )
        return float(total_kl), float(-total_kl)

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
            return cost_fn(base_program)

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

        eval_seeds = self._parse_eval_seeds(cfg, int(seed))
        cfg_yaml = OmegaConf.to_yaml(cfg, resolve=True)
        workers = self.num_workers or len(eval_seeds)
        workers = max(1, min(int(workers), len(eval_seeds)))
        expert_states_total = np.concatenate(
            [_load_expert_states_for_seed(s, self.demo_dir) for s in eval_seeds],
            axis=0,
        )
        candidate_log_path = Path("outputs") / (
            f"program_synthesis_candidates_seed{int(seed)}_"
            f"n{len(eval_seeds)}.jsonl"
        )
        candidate_log_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_log_rows: list[dict[str, Any]] = []
        logger.info(
            "Program synthesis started: mcmc_iters=%d, eval_seeds=%s, workers=%d",
            self.mcmc_iters,
            eval_seeds,
            workers,
        )

        # Build initial program and run an MCMC optimization loop over programs.
        initial_program = self._build_initial_program(cfg)

        mp_ctx = mp.get_context("spawn")
        with mp_ctx.Pool(processes=workers) as pool:
            def _cost_fn(p: Program) -> float:
                per_seed = self._evaluate_program_across_seeds(
                    cfg_yaml=cfg_yaml,
                    program=p,
                    eval_seeds=eval_seeds,
                    pool=pool,
                )
                _kl, score = self._score_from_rollouts(per_seed, expert_states_total)
                return float(score)

            # --- MCMC with inner CEM optimization (like external mcmc.py) ----
            current_program = deepcopy(initial_program)
            current_cost = _cost_fn(current_program)
            logger.info(
                "Initial program: %s",
                json.dumps([asdict(instr) for instr in current_program.instructions]),
            )
            candidate_log_rows.append(
                {
                    "phase": "initial",
                    "iter": -1,
                    "accepted": True,
                    "current_cost": float(current_cost),
                    "best_cost": float(current_cost),
                    "program": [asdict(instr) for instr in current_program.instructions],
                }
            )
            logger.info("Initial program cost: %.6f", float(current_cost))

            best_program = deepcopy(current_program)
            best_cost = float(current_cost)

            for it in range(self.mcmc_iters):
                proposed_program, changed = self._mutate_program(current_program)
                if not changed:
                    candidate_log_rows.append(
                        {
                            "phase": "mcmc",
                            "iter": it,
                            "changed": False,
                            "accepted": False,
                            "current_cost": float(current_cost),
                            "best_cost": float(best_cost),
                            "program": [asdict(instr) for instr in current_program.instructions],
                        }
                    )
                    logger.info(
                        "[MCMC %d/%d] no-op mutation, current=%.6f best=%.6f",
                        it + 1,
                        self.mcmc_iters,
                        float(current_cost),
                        float(best_cost),
                    )
                    logger.info(
                        "[MCMC %d/%d] current program: %s",
                        it + 1,
                        self.mcmc_iters,
                        json.dumps([asdict(instr) for instr in current_program.instructions]),
                    )
                    continue

                # Optimize continuous float parameters for the proposed structure.
                proposed_program, proposed_cost = self._optimize_program_float_params_via_cem(
                    proposed_program,
                    cost_fn=_cost_fn,
                )

                # Metropolis acceptance ratio; higher is better.
                acceptance_ratio = math.exp(proposed_cost - current_cost)
                u = random.random()
                accepted = u < acceptance_ratio
                if accepted:
                    current_program = proposed_program
                    current_cost = proposed_cost

                if proposed_cost > best_cost:
                    best_cost = proposed_cost
                    best_program = deepcopy(proposed_program)
                    logger.info(
                        "[MCMC %d/%d] new best cost: %.6f",
                        it + 1,
                        self.mcmc_iters,
                        float(best_cost),
                    )

                candidate_log_rows.append(
                    {
                        "phase": "mcmc",
                        "iter": it,
                        "changed": True,
                        "accepted": bool(accepted),
                        "u": float(u),
                        "acceptance_ratio": float(acceptance_ratio),
                        "proposed_cost": float(proposed_cost),
                        "current_cost_after": float(current_cost),
                        "best_cost": float(best_cost),
                        "program": [asdict(instr) for instr in proposed_program.instructions],
                    }
                )
                logger.info(
                    "[MCMC %d/%d] proposed=%.6f ratio=%.4f u=%.4f accepted=%s current=%.6f best=%.6f",
                    it + 1,
                    self.mcmc_iters,
                    float(proposed_cost),
                    float(acceptance_ratio),
                    float(u),
                    accepted,
                    float(current_cost),
                    float(best_cost),
                )
                logger.info(
                    "[MCMC %d/%d] proposed program: %s",
                    it + 1,
                    self.mcmc_iters,
                    json.dumps([asdict(instr) for instr in proposed_program.instructions]),
                )
                logger.info(
                    "[MCMC %d/%d] current program after accept/reject: %s",
                    it + 1,
                    self.mcmc_iters,
                    json.dumps([asdict(instr) for instr in current_program.instructions]),
                )

            final_per_seed = self._evaluate_program_across_seeds(
                cfg_yaml=cfg_yaml,
                program=best_program,
                eval_seeds=eval_seeds,
                pool=pool,
            )
            final_kl, final_score = self._score_from_rollouts(
                final_per_seed,
                expert_states_total,
            )

        success_rate = float(
            sum(1 for r in final_per_seed if r["success"]) / max(1, len(final_per_seed))
        )
        mean_reward = float(sum(r["reward"] for r in final_per_seed) / max(1, len(final_per_seed)))
        mean_steps = float(sum(r["steps"] for r in final_per_seed) / max(1, len(final_per_seed)))
        for r in final_per_seed:
            # Keep info JSON-serializable.
            if "state_positions" in r:
                r["num_state_rows"] = int(np.asarray(r["state_positions"]).shape[0])
                r.pop("state_positions", None)
        failed = [r for r in final_per_seed if not r["success"]]

        info = {
            "program": [asdict(instr) for instr in best_program.instructions],
            "eval_seeds": eval_seeds,
            "num_eval_seeds": len(eval_seeds),
            "success_rate": success_rate,
            "mean_reward": mean_reward,
            "mean_steps": mean_steps,
            "total_kl": float(final_kl),
            "total_score": float(final_score),
            "per_seed_results": final_per_seed,
            "candidate_log_path": str(candidate_log_path),
        }
        with candidate_log_path.open("w", encoding="utf-8") as f:
            for row in candidate_log_rows:
                f.write(json.dumps(row) + "\n")
        logger.info(
            "Saved %d candidate programs to %s",
            len(candidate_log_rows),
            candidate_log_path,
        )
        logger.info(
            "Best program: %s",
            json.dumps([asdict(instr) for instr in best_program.instructions]),
        )
        # Keep the caller env in a valid post-solve state for run.py's
        # zero-action settle loop.
        env.reset(seed=int(eval_seeds[0]))

        return SolverResult(
            success=(success_rate == 1.0),
            reward=float(final_score),
            elapsed_steps=int(round(mean_steps)),
            info=info,
            failure_reason=None if not failed else failed[0]["failure_reason"],
        )

