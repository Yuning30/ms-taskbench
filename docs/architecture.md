# Taskbench architecture

Taskbench has two manipulation workflows: shared skills used by conventional
solvers, and a RoboVerify Stack backend used for program synthesis and formal
verification. They share ManiSkill and mplib utilities, but have separate
primitive semantics, reset policies, and recording formats.

## Modules

| Path | Responsibility |
| --- | --- |
| `taskbench/run.py` | Hydra dispatch, episode loop, logging, and environment-based video recording |
| `taskbench/solver.py` | `BaseSolver`, `SolverResult`, registration, and discovery |
| `taskbench/envs/` | Custom ManiSkill tasks, object lookup, and environment factories |
| `taskbench/skills/` | Shared `SkillContext`, Pick/Place/Move/Push, robot configuration, and mplib helpers |
| `taskbench/recorder.py` | HDF5 state trajectories and shared-skill calls |
| `taskbench/programs/` | Build2D program representation and execution |
| `taskbench/solvers/` | Registered solver implementations |
| `taskbench/roboverify/` | Copied search/verifier, ManiSkill backend, NPZ traces, and experiment tools |
| `configs/default.yaml` | Base Hydra values |
| `configs/solver/` | Per-solver configuration overrides |

## Runner and solver contract

`uv run python -m taskbench.run solver=name` selects a Hydra config group.
The runner dispatches as follows:

| Solver | Environment creation | Result |
| --- | --- | --- |
| `random` | `make_env()`: vectorized environments | Random-action episode statistics |
| Registered solver with `requires_env=True` | `make_single_env()`: one CPU environment | Solver result, followed by environment evaluation |
| `program_synthesis` (`requires_env=False`) | Solver creates its own RoboVerify environments | Formal verification result |

Registered solvers receive `solve(env, seed=cfg.seed + ep, cfg=cfg)`, with
episode numbers starting at 1. `run.solver_kwargs` is passed to the solver
constructor. Discovery imports modules under `taskbench.solvers` using
`pkgutil.walk_packages`, triggering their `@register_solver` decorators.

For environment-based solvers without a `verification_status`, the runner
performs up to 100 additional zero-action steps, evaluates the environment,
and updates `result.success`. This runner behavior does not apply to RoboVerify
collection or synthesis. Formal results are preserved.

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class SolverResult:
    success: bool
    reward: float = 0.0
    elapsed_steps: int = 0
    info: dict = field(default_factory=dict)
    failure_reason: Optional[str] = None
    verification_status: Optional[str] = None
```

## RoboVerify Stack pipeline

The canonical Hydra solver is `program_synthesis`. Its wrapper translates
`run.solver_kwargs` into the arguments of
`taskbench.roboverify.entry.synthesize_cfg`.

1. Collect executions of a DSL program into validated NPZ traces, including
   simulation snapshots, action prefixes, events, and bindings.
2. Build/refine a relational control-flow graph from demonstrations. ID-first
   search uses MCMC/CEM to search numeric primitives and offsets, then recovers
   loops and named bodies.
3. Execute candidates in the simulator from recorded starts.
4. Learn loop invariants from candidate executions with partition-based
   `InvInference`.
5. Run symbolic checks, including the unbounded proof, and geometric motion
   obligations. Both stages must pass for `verified_model`.

`backend.py` owns one CPU Panda environment. `stack_reset.py` applies the
14 cm initial separation. `move_controller=delta|planner` chooses the motion
implementation inside RoboVerify Pick/Move/Release; the planner path reuses the
shared screw-planning helper. It does not call the shared `SkillContext` skills.

Use `run.solver_kwargs.num_blocks`, `move_controller`, and
`planner_step_limit` for this workflow. Root `env.num_cubes` and
`env.control_mode` do not configure its backend.
[The Stack guide](roboverify.md) documents commands, current planner failures,
and the distinction between model verification and physical validation.

## Environments

Custom tasks implement `TaskEnv.get_objects()`. The
`taskbench.envs.get_objects(env)` helper uses that method or a built-in
StackCube fallback.

| Environment | Objects / structure | Purpose |
| --- | --- | --- |
| `StackCube-v1` | `cube_0 = cubeB`, `cube_1 = cubeA` | Built-in ManiSkill two-cube task |
| `StackNCube-v1` | `cube_0` through `cube_{N-1}` | Uniform 40 mm cubes; green cube 0 is the tower base |
| `StackCubeDistractor-v1` | `cube_0`, `cube_1`, `cube_2` | Two-cube stacking with a distractor |
| `Build2D-v1` | Blocks and linked grid nodes | Place a block at each node target |
| `ShelfEnv-v1` | `cyl_0` through `cyl_19` | Shelf with 19 blue cylinders and one red cylinder |
| `BinWithObjects-v1` | Actors keyed by name | Bin containing primitives and YCB objects |

### Stack initialization and success

The raw Stack environment defaults to `panda_wristcam` and joint-position
initialization noise of 0.02. It samples centers within X = [−0.1, 0.1] m,
Y = [−0.2, 0.2] m, with random yaw. Its placement sampler uses a radius of
about 29.3 mm per cube, targeting roughly 58.6 mm pairwise center clearance.

The RoboVerify backend selects `panda` with zero joint initialization noise
and replaces the XY layout with a seed-local, bounded sampler requiring
`abs(dx) >= 0.14 OR abs(dy) >= 0.14` for every pair. It preserves the workspace
and cube yaws, then settles for 50 steps before recording. This 14 cm setting
is specific to RoboVerify; the shared `stack_cubes` solver uses the raw reset.

Stack success requires cube 0 to be the lowest, consecutive centers in
height order to be within about 33.3 mm horizontally and 40 ± 5 mm vertically,
and no cube to be grasped. `all_static` is diagnostic only. There is no final
velocity gate or required stability-duration window. RoboVerify collection also
checks task predicates and complete primitive/loop execution.

### Factories

`make_env(cfg.env)` creates a `ManiSkillVectorEnv`; multiple environments
use the GPU simulation backend by default. `make_single_env(cfg.env)` creates
one CPU environment without the vector wrapper, for direct planner access.
Extra environment constructor arguments come from `env.extra_kwargs`.

The base YAML sets `normalized_dense`, while StackNCube supports `sparse`
and `none`. Specify a supported reward for the random Stack baseline:

```bash
uv run python -m taskbench.run env.reward_mode=sparse run.num_episodes=10
```

## Shared skills and recording

`SkillContext.reset()` resets the environment, creates a planner, discovers
objects, and binds shared Pick, Place, Move, and Push instances. The environment
control mode determines whether motions use Cartesian feedback or joint-space
planner execution. `stack_cubes` defaults to `pd_ee_delta_pose`.
See [skills](skills.md) for destination semantics and parameter signatures.

[Recording formats](demos.md) are workflow-specific:

- RoboVerify collects NPZ traces consumed by synthesis and snapshot/action replay.
- `StateRecorder` writes HDF5 robot/object state and shared-skill programs.
- `demo_recorder` saves an interactive scene and program to JSON.

## Adding a solver

Create `taskbench/solvers/my_solver.py` with the current signature:

```python
from taskbench.solver import BaseSolver, SolverResult, register_solver

@register_solver("my_solver")
class MySolver(BaseSolver):
    def solve(self, env, seed=None, cfg=None) -> SolverResult:
        env.reset(seed=seed)
        # Execute the task here.
        success = bool(env.unwrapped.evaluate()["success"].item())
        return SolverResult(success=success)
```

This is an extension skeleton; it does not manipulate objects. Add
`configs/solver/my_solver.yaml`:

```yaml
# @package _global_
run:
  solver: my_solver
  num_episodes: 1
env:
  env_id: StackNCube-v1
  control_mode: pd_ee_delta_pose
  num_envs: 1
  reward_mode: sparse
  num_cubes: 3
  max_episode_steps: 1000
```

Run `uv run python -m taskbench.run solver=my_solver`. A solver that owns its
environments should set `requires_env=False` and accept `env=None`.
For a new environment, implement scene construction, reset, evaluation, and
`get_objects()` in a `TaskEnv` subclass, register it with ManiSkill's
`@register_env`, and import the module from `taskbench/envs/__init__.py`.

## Configuration and constraints

Hydra configuration is YAML-based. Override existing fields with
`key=value`; add absent keys with `+key=value`. For example, the supplied
program option is absent from the synthesis YAML and needs
`+run.solver_kwargs.program=module:factory` in verify mode.

- mplib 0.2.x uses `mplib.pymp.Pose`. Flatten batched SAPIEN positions and
  quaternions before passing them to the planner.
- Planner execution needs one CPU environment and joint control
  (`pd_joint_pos`, or `pd_joint_pos_vel` in shared helpers).
- The shared `RecordEpisode` wrapper uses `save_on_reset=False`; the runner
  flushes videos explicitly. RoboVerify collection uses its own video recorder.
- mplib 0.2.1 requires NumPy below 2.0; the uv overrides enforce this.
