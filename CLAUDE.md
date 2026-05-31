# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

This project uses **uv** for dependency management. Always use `uv run` (not `source .venv/bin/activate && python`) and `uv pip` (not bare `pip`) for all operations.

```bash
# Create and sync the venv (installs all deps from pyproject.toml + uv.lock)
uv sync

# Or with dev extras (includes black, isort)
uv sync --extra dev

# Install additional packages (ALWAYS use uv pip, never bare pip)
uv pip install <package>
```

## Running

Entry point is Hydra-based. Solver selection uses Hydra config groups (`solver=name`). **Always use `uv run`.**

```bash
# Default run (random solver, 16 envs, 100 episodes)
uv run python -m taskbench.run

# Cube stacking solver (env requirements set by configs/solver/stack_cubes.yaml)
uv run python -m taskbench.run solver=stack_cubes

# Common overrides
uv run python -m taskbench.run env.record_video=true run.num_episodes=10
uv run python -m taskbench.run seed=123 logging.use_wandb=true
```

Hydra writes timestamped output dirs under `outputs/`. Videos go to `videos/`.

## Formatting

```bash
uv run black taskbench/
uv run isort taskbench/
```

No tests exist in this repo.

## Architecture

**taskbench** is a robotics research testbed for evaluating solvers on ManiSkill3 manipulation tasks. See `docs/architecture.md` for full documentation.

### Core Flow

`taskbench/run.py` is the entry point (`@hydra.main`). It dispatches based on `cfg.run.solver`:
- `"random"` → `run_random()` — vectorized env (`ManiSkillVectorEnv`), samples random actions
- Any other value → `run_solver()` — auto-discovers solver via `@register_solver`, creates a single env, runs episode loop

### Solver System (pluggable, zero-touch)

Solvers self-register via `@register_solver("name")` decorator in `taskbench/solver.py`. Auto-discovery walks `taskbench/solvers/*.py` via `pkgutil`. Adding a new solver requires **no edits to core files**:
1. Create `taskbench/solvers/my_solver.py` with `@register_solver("my_solver")` inheriting `BaseSolver`
2. Create `configs/solver/my_solver.yaml` with env requirements (`# @package _global_`)
3. Run: `python -m taskbench.run solver=my_solver`

### Skill System

Skills are composable objects (`Pick`, `Place`, `Move`, `Push`) in `taskbench/skills/primitives.py`. Use `SkillContext` to eliminate boilerplate:
```python
ctx = SkillContext(env, step_callback=recorder.record)
ctx.reset(seed=42)
ctx.pick("cube_1")
ctx.place(target_pose)
```

Robot-specific constants live in `RobotConfig` (`taskbench/skills/robot_config.py`), not hardcoded. Skills accept `PoseLike` (tuples or `sapien.Pose`) and resolve objects by string name.

### Key Modules

- **`configs/default.yaml`** — Hydra config (YAML-only, no Python dataclasses). Solver configs in `configs/solver/`.
- **`taskbench/envs/factory.py`** — `make_env()` (vectorized) and `make_single_env()` (raw, for motion planner).
- **`taskbench/envs/base.py`** — `TaskEnv` base class with abstract `get_objects()`.
- **`taskbench/solver.py`** — `BaseSolver` ABC, `SolverResult`, `@register_solver`, `discover_solvers()`.
- **`taskbench/skills/robot_config.py`** — `RobotConfig` dataclass + `ROBOT_CONFIGS` registry.
- **`taskbench/skills/context.py`** — `SkillContext` — bundles env + planner + objects + skills.
- **`taskbench/skills/motion.py`** — Low-level mplib helpers: `setup_planner()`, `move_to_pose()` (straight-line screw interpolation, no RRT), `build_action()`, `PoseLike`.
- **`taskbench/skills/mplib_primitives.py`** — mplib-backed `Pick` / `Place` / `Move` / `Push`. Kept available; not the default.
- **`taskbench/skills/curobo_planner.py`** / **`taskbench/skills/curobo_motion.py`** / **`taskbench/skills/curobo_primitives.py`** — cuRobo motion backend. `SkillContext` binds to these by default; the mplib variants stay importable.
- **`taskbench/recorder.py`** — `StateRecorder` for capturing simulation state to HDF5.
- **`taskbench/logger.py`** — Optional WandB logging wrapper.

### Critical Constraints

#### Shared
- **SAPIEN poses are batched**: Even with `num_envs=1`, pose tensors have shape `(1, 3)` / `(1, 4)` — must `.flatten()` before use.
- **numpy < 2.0** required by mplib 0.2.1.

#### mplib backend
- **mplib 0.2.x API**: Uses `mplib.pymp.Pose` objects (not numpy arrays) for `set_base_pose()`, `plan_screw()`, etc.
- **Motion planner requires**: `num_envs=1`, `sim_backend="cpu"`, `pd_joint_pos` control mode, no `ManiSkillVectorEnv` wrapper.
- **Video recording with planner**: Must use `save_on_reset=False` on `RecordEpisode` and call `env.flush_video()` manually.

#### cuRobo backend (default for `SkillContext`)
- **Optional dependency**: install with `uv sync --extra curobo`. Requires a Vulkan-capable NVIDIA GPU.
- **Control mode**: cuRobo's trajectory follower emits 7-dim joint targets, so the env must use `pd_joint_pos_vel` (preferred — enables velocity feedforward) or `pd_joint_pos`. The legacy `pd_ee_delta_pose` raises `ValueError: pd_ee_delta_pose expects 6-dim delta action, got 7`. The default solver configs still set `pd_ee_delta_pose`; override on the command line: `env.control_mode=pd_joint_pos_vel`.
- **Warmup cost**: first `SkillContext.reset()` takes ~5–10s to build cuRobo. Reused across episodes inside the same `SkillContext`.
- **Stack release pose**: when placing a cube on top of another cube, the solver MUST compute the TCP target from the *actual* TCP and held-cube positions (see `taskbench/solvers/stack_n_cubes.py`), not from `pick_result.lift_pose`. cuRobo has ~5mm tracking error on the lift; using `lift_pose` directly produces a placement target ~6mm too low, which intersects the underlying cube and topples the tower.
- **Retract**: `CuroboPlace` re-syncs the cuRobo collision scene with no exclusion *after* settle and *before* retract so the just-placed cube is treated as an obstacle (otherwise the lifted hand can curl laterally and brush the cube off).

#### SAPIEN + multi-process (`program_synthesis` solver)
- SAPIEN initializes a Vulkan instance even when `obs_mode="state"` because `PandaWristCam` mounts a camera. When N spawn workers init concurrently, expect one of: `Failed to find a supported physical device "cuda:0"`, `vk::PhysicalDevice::createDeviceUnique: ErrorInitializationFailed`, or `CUDA error: an illegal memory access was encountered` on the first real rollout.
- `ProgramSynthesisSolver` works around this with:
  1. **Per-worker env+SkillContext cache** (`_WORKER_ENV`, `_WORKER_CTX`) — env is built once per worker process and reused via `env.reset(seed=…)` across cost-fn calls, so the cuRobo warmup is paid once per worker, not once per cost-fn.
  2. **`multiprocessing.Manager().Lock()`** passed to the Pool initializer, which serializes the cuRobo first-init across workers (only one builds cuRobo at a time).
- **Known unresolved issue — `num_workers > 1` crashes after a few real CEM iters** with `CUDA error: an illegal memory access was encountered`, regardless of GPU VRAM (observed identical failures on 16 GB, 20 GB, and 48 GB cards). The init-Lock fixes the Vulkan startup race, but a separate cross-process cuRobo / CUDA-stream interaction surfaces once thousands of plans flow through 5 workers. Until this is root-caused, `num_workers=1` is the only safe setting (~5× slower than the design intent). A small diagnostic with `cem_iters=2, cem_N=3` will pass with `num_workers=5` and mask the bug — verify with a real-sized CEM (≥ 1 full CEM iter at `cem_N=64`).
- **`task_reward_weight` kwarg**: synthesis cost is `-KL(rollout ‖ expert)` by default. Set `run.solver_kwargs.task_reward_weight=10.0` to add `λ · success_rate` to the score, directly rewarding programs whose env `evaluate()` returns True (closes the proxy-vs-task gap in pure-KL runs).
- **Candidate log filename**: includes both `n{eval_seeds}` and `_tw{weight}` tags so parallel runs with different hyperparams don't clobber each other (`outputs/program_synthesis_candidates_seed{S}_n{N}_tw{W}.jsonl`).

### Demo collection conventions
- `solver=stack_cubes` writes one HDF5 per episode to either `data/success/episode_seed{N}.hdf5` or `data/failure/episode_seed{N}.hdf5`, depending on the solver's *internal* placement check (NOT the env's `evaluate()` — the two can disagree at the edges).
- `ProgramSynthesisSolver` reads expert demos directly from `data/success/episode_seed{N}.hdf5` for the seeds listed in `run.solver_kwargs.eval_seeds`. There is no quality grading or re-evaluation; the directory split is the only filter.
