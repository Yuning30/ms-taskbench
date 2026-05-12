# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## SLURM Defaults

- `working_dir`: `/common/home/st1122/Projects/ms-taskbench`
- `log_dir`: `/common/home/st1122/Projects/ms-taskbench/slurm_logs/`
- `env_activate`: `source /common/home/st1122/Projects/ms-taskbench/.venv/bin/activate`
- `default_mem`: `16G`
- `default_cpus`: `8`
- `default_partition`: `unlimited`

This is a uv project (no conda env). Activate via the venv directly.
For cuRobo, also set `CUDA_HOME=/usr/local/cuda` and prepend
`$CUDA_HOME/bin` to `PATH`.

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
- **`taskbench/skills/primitives.py`** — Composable skill objects with `SkillResult` dataclasses.
- **`taskbench/recorder.py`** — `StateRecorder` for capturing simulation state to HDF5.
- **`taskbench/logger.py`** — Optional WandB logging wrapper.

### Critical Constraints (mplib / ManiSkill)

- **mplib 0.2.x API**: Uses `mplib.pymp.Pose` objects (not numpy arrays) for `set_base_pose()`, `plan_screw()`, etc.
- **SAPIEN poses are batched**: Even with `num_envs=1`, pose tensors have shape `(1, 3)` / `(1, 4)` — must `.flatten()` before passing to mplib.
- **Motion planner requires**: `num_envs=1`, `sim_backend="cpu"`, `pd_joint_pos` control mode, no `ManiSkillVectorEnv` wrapper.
- **Video recording with planner**: Must use `save_on_reset=False` on `RecordEpisode` and call `env.flush_video()` manually.
- **numpy < 2.0** required by mplib 0.2.1.
