# CLAUDE.md

Guidance for working with code in this repository.

## Environment setup

This project uses **uv** for dependency management. Always use `uv run`
for Python commands and `uv pip` for package operations; do not activate the
virtual environment manually or use bare `pip`.

```bash
uv sync                               # Base project
uv sync --extra roboverify            # Synthesis and verification dependencies
uv sync --extra dev --extra roboverify
uv pip install <package>
```

## Current Stack workflow

The canonical synthesis solver is `program_synthesis`. Its self-contained
implementation is in `taskbench/roboverify`; it does not import `~/RoboVerify`.
See [the Stack guide](docs/roboverify.md) for complete commands and proof scope.

```bash
uv run --extra roboverify python -m taskbench.roboverify.entry.collect_demos \
  --program taskbench.roboverify.examples.stack:build_program \
  --num-blocks 3 --seeds 0 42 --max-loop-iterations 2 \
  --move-controller delta --output-dir demos/stack3-delta

uv run --extra roboverify python -m taskbench.run solver=program_synthesis \
  run.solver_kwargs.demos=demos/stack3-delta/demonstrations.npz \
  'run.solver_kwargs.initial_arm=[0,0,0.17]'
```

Use a new collection directory. `initial_arm` is a geometric proof premise,
not a physical robot reset command or a value inferred from the archive.
The solver owns its environments: block count, controller, and planner budget
belong under `run.solver_kwargs`. Select `move_controller=delta|planner`
consistently for collection and synthesis. Planner mode currently has an
unresolved mplib numerical termination failure on some seeds; see the guide.

RoboVerify resets use Panda, 14 cm minimum block-center separation along X or Y,
and 50 initial settling steps. Final success uses geometry and release, plus
complete execution and symbolic task predicates. Velocities are diagnostic;
there is no stability-duration requirement.

## Other entry points

Solver selection uses Hydra config groups (`solver=name`).

```bash
# Random baseline on StackNCube; its reward must be sparse or none
uv run python -m taskbench.run env.reward_mode=sparse run.num_episodes=10

# Shared skills and HDF5 recording; default control is pd_ee_delta_pose
uv run python -m taskbench.run solver=stack_cubes \
  env.max_episode_steps=1000 run.num_episodes=10 env.record_video=true

# Shared planner path
uv run python -m taskbench.run solver=stack_cubes \
  env.control_mode=pd_joint_pos env.max_episode_steps=1000 run.num_episodes=1

# Build2D program execution
uv run python -m taskbench.run solver=build2d_program \
  env.extra_kwargs.grid_rows=3 env.extra_kwargs.grid_cols=4
```

Hydra writes configuration/logging output under `outputs/`. Shared Stack
recordings go to `data/` and runner videos to `videos/`. RoboVerify collections
write NPZ traces and optional videos under their collection directory; synthesis
artifacts go to `runs/cfg/`. See [recording formats](docs/demos.md).

## Architecture and extension

See [architecture](docs/architecture.md) and [shared skills](docs/skills.md).

`taskbench/run.py` dispatches random actions to `run_random()`. Other solvers
are auto-discovered through `@register_solver` and called with
`solve(env, seed=None, cfg=None)`. The runner creates a single CPU environment
only when `requires_env=True`. `program_synthesis` sets it to `False`.
Results with `verification_status` retain their formal outcome rather than
being overwritten by environment success.

To add a solver:

1. Create `taskbench/solvers/my_solver.py` with a registered `BaseSolver`
   subclass and the current `solve` signature.
2. Add `configs/solver/my_solver.yaml` with `# @package _global_` and its
   solver/environment settings.
3. Run `uv run python -m taskbench.run solver=my_solver`.

Shared `SkillContext` binds Pick/Place/Move/Push after `reset()`.
Its default feedback path and planner path depend on `env.control_mode`.
RoboVerify has its own Pick/Move/Release runtime; changes to shared skills do
not automatically change that runtime. Both paths reuse low-level mplib helpers.

Key modules:

- `taskbench/solver.py`: solver registry and result contract.
- `taskbench/envs/factory.py`: vectorized and single-environment factories.
- `taskbench/envs/base.py`: `TaskEnv` and object-discovery interface.
- `taskbench/skills/context.py`, `primitives.py`, `motion.py`: shared skills.
- `taskbench/skills/robot_config.py`: robot-specific planner/gripper constants.
- `taskbench/recorder.py`: HDF5 state and skill-call recording.
- `taskbench/programs/`: Build2D program representation and execution.
- `taskbench/solvers/program_synthesis.py`: Hydra-to-RoboVerify wrapper.
- `taskbench/roboverify/backend.py`, `stack_reset.py`: ManiSkill adapter/reset.
- `taskbench/roboverify/entry/`, `cfg/`: collection and synthesis pipeline.
- `taskbench/roboverify/SOURCE.md`: copied-source provenance and adaptations.

## Validation and formatting

```bash
uv run black taskbench/
uv run isort taskbench/
```

RoboVerify uses `unittest` under `taskbench/roboverify/`, plus
`taskbench/envs/test_stack_n_cube.py`. See the
[test command and simulator validation workflow](docs/roboverify.md#tests).
Dated results are in [the validation history](docs/roboverify-validation.md).

## mplib / ManiSkill constraints

- mplib 0.2.x takes `mplib.pymp.Pose` for `set_base_pose()` and `plan_screw()`.
- SAPIEN pose tensors are batched even for one environment; flatten before
  passing positions/quaternions to mplib.
- Planner execution requires one CPU environment without a
  `ManiSkillVectorEnv` wrapper, and joint-position control. Shared helpers also
  support `pd_joint_pos_vel`; RoboVerify planner mode uses `pd_joint_pos`.
- Shared planner video recording uses `RecordEpisode(save_on_reset=False)`
  and explicit `flush_video()`.
- NumPy must stay below 2.0 for mplib 0.2.1.
