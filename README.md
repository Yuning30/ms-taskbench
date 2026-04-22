# taskbench

A robotics research testbed for evaluating solvers on ManiSkill3 manipulation tasks. Provides composable skill primitives (pick, place, push), HDF5 demonstration recording with skill program traces, and a pluggable solver system.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/dhruvmetha/ms-taskbench.git
cd ms-taskbench
uv sync
```

## Tasks

### Build2D

Build an `m x n` 2D grid by placing one block at each linked-list node target.
The env exposes a head node `h` where each node has `.r` (right), `.d` (down),
and world coordinates `(x, y, z)`.

**Solver:** `build2d_program` — executes the canonical nested-loop program:

```python
i <- h
while i != null:
    j <- i
    while j != null:
        put a block on j
        j <- j.r
    i <- i.d
```

```bash
# Run in simulator
uv run python -m taskbench.run solver=build2d_program env.extra_kwargs.grid_rows=3 env.extra_kwargs.grid_cols=4

# Dry-run DSL semantics only (no ManiSkill required)
uv run python -m taskbench.programs.build2d_demo --rows 3 --cols 4
```

### StackNCube

Stack N cubes into a tower. `cube_0` (green) is always the base — the solver picks the remaining cubes in random order and stacks them on top.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `env.num_cubes` | 3 | Number of cubes (2-6) |
| `env.reward_mode` | sparse | `sparse` or `normalized_dense` |

**Solver:** `stack_cubes` — uses sequential pick-place with screw-based motion planning. Automatically records demos to `data/success/` and `data/failure/` in HDF5 format with full state trajectories and skill program traces.

```bash
# Stack 3 cubes (default)
uv run python -m taskbench.run solver=stack_cubes

# Stack 5 cubes, 50 episodes, with video
uv run python -m taskbench.run solver=stack_cubes env.num_cubes=5 run.num_episodes=50 env.record_video=true
```

**Replay:** Re-execute a recorded demo's skill program on a fresh env with the same seed:

```bash
uv run python -m taskbench.run solver=replay run.solver_kwargs.demo_path=data/success/episode_seed45.hdf5
```

## Usage

All commands use `uv run` — no manual venv activation needed.

```bash
# Random baseline (16 parallel envs, 100 episodes)
uv run python -m taskbench.run

# Collect 10 demos with the stack_cubes solver
uv run python -m taskbench.run solver=stack_cubes run.num_episodes=10

# Replay a recorded demo
uv run python -m taskbench.run solver=replay run.solver_kwargs.demo_path=data/success/episode_seed45.hdf5
```

Demos are saved to `data/success/` and `data/failure/`. Videos go to `videos/`.

## Common overrides

```bash
env.num_cubes=N          # number of cubes (default: 3)
env.record_video=true    # save videos
run.num_episodes=N       # number of episodes
seed=123                 # random seed
logging.use_wandb=true   # enable WandB logging
```

## Project structure

```
configs/              # Hydra YAML configs (single source of truth)
  default.yaml        # base config
  solver/             # per-solver overrides
taskbench/
  run.py              # entry point (@hydra.main)
  solver.py           # BaseSolver, @register_solver, auto-discovery
  recorder.py         # HDF5 state + program recorder
  envs/               # ManiSkill3 environments
  skills/             # composable primitives (pick, place, push)
  solvers/            # solver implementations
```

## Adding a new solver

No edits to core files needed — solvers are auto-discovered. Just create two files:

### 1. Solver implementation

`taskbench/solvers/my_stacker.py`:

```python
from taskbench.skills.context import SkillContext
from taskbench.solver import BaseSolver, SolverResult, register_solver

@register_solver("my_stacker")
class MyStackerSolver(BaseSolver):
    def solve(self, env, seed=None) -> SolverResult:
        ctx = SkillContext(env)
        ctx.reset(seed=seed)

        objects = ctx.objects  # {"cube_0": actor, "cube_1": actor, ...}
        raw = env.unwrapped

        # Pick cube_1
        pick_result = ctx.pick("cube_1")
        if not pick_result.success:
            return SolverResult(success=False, failure_reason=pick_result.failure_reason)

        # Place on top of cube_0
        target_pos = objects["cube_0"].pose.p.cpu().numpy().flatten()
        cube_h = (raw.cube_half_size[2] * 2).item()
        target_pos[2] += cube_h
        place_result = ctx.place((target_pos, pick_result.lift_pose.q))

        info = raw.evaluate()
        return SolverResult(success=bool(info["success"].item()))
```

Available skills on `ctx`: `pick(obj_name)`, `place(target_pose)`, `push(approach, push_pose)`, `move(target_pose)`.

### 2. Hydra config

`configs/solver/my_stacker.yaml`:

```yaml
# @package _global_
run:
  solver: my_stacker
env:
  env_id: StackNCube-v1
  control_mode: pd_joint_pos
  num_envs: 1
  reward_mode: sparse
  num_cubes: 3
```

### 3. Run

```bash
uv run python -m taskbench.run solver=my_stacker
```

## Using as a library

You don't have to go through `python -m taskbench.run`. The components are importable directly — write your own scripts for custom workflows like program synthesis, search, or evaluation.

### Core components

| Component | Import | Purpose |
|-----------|--------|---------|
| `SkillContext` | `from taskbench.skills.context import SkillContext` | Main interface — `pick()`, `place()`, `push()`, `move()` |
| `make_single_env` | `from taskbench.envs.factory import make_single_env` | Create a single env for motion planning |
| `StateRecorder` | `from taskbench.recorder import StateRecorder` | Record trajectories + skill programs to HDF5 |
| `RobotConfig` | `from taskbench.skills.robot_config import get_robot_config` | Robot-specific constants |

### Example: custom script

```python
# scripts/my_experiment.py
import gymnasium as gym
from taskbench.skills.context import SkillContext
import taskbench.envs  # register custom envs

env = gym.make(
    "StackNCube-v1",
    num_cubes=3,
    obs_mode="state",
    control_mode="pd_joint_pos",
    reward_mode="sparse",
    num_envs=1,
    sim_backend="cpu",
)

ctx = SkillContext(env)
ctx.reset(seed=42)

# Use skills directly
pick_result = ctx.pick("cube_1")
if pick_result.success:
    target_pos = ctx.objects["cube_0"].pose.p.cpu().numpy().flatten()
    cube_h = (env.unwrapped.cube_half_size[2] * 2).item()
    target_pos[2] += cube_h
    ctx.place((target_pos, pick_result.lift_pose.q))

info = env.unwrapped.evaluate()
print(f"Success: {info['success'].item()}")
env.close()
```

```bash
uv run python scripts/my_experiment.py
```

### Example: adding a new task

Create a new environment and solver in the repo:

```python
# taskbench/envs/my_task.py
from mani_skill.utils.registration import register_env
from taskbench.envs.base import TaskEnv

@register_env("MyTask-v1", max_episode_steps=200)
class MyTaskEnv(TaskEnv):
    def get_objects(self):
        return {"target": self.target_obj}

    def _load_scene(self, options):
        # Build scene actors ...

    def evaluate(self):
        # Return {"success": tensor}
        ...
```

Register it in `taskbench/envs/__init__.py`:

```python
import taskbench.envs.my_task  # noqa: F401
```

Then use it from any script or through the solver system.

## Dev

```bash
uv sync --extra dev
uv run black taskbench/
uv run isort taskbench/
```
