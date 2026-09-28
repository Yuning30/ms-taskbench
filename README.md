# taskbench

A robotics research testbed for ManiSkill3 manipulation tasks. Stack program
synthesis and verification use the RoboVerify implementation in
`taskbench/roboverify`. The repository also provides shared manipulation skills,
HDF5 recording, and a Build2D program executor.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Yuning30/ms-taskbench.git
cd ms-taskbench
uv sync --extra roboverify
```

Use `uv run` for Python commands. `uv sync` installs the base project;
`--extra roboverify` adds the synthesis and verification dependencies.

## Stack: collect, synthesize, and verify

Run the following commands from the repository root in the same terminal.
Delta is currently the more reliable controller.

### 1. Choose the block count and controller

```bash
STACK_BLOCKS=3
STACK_CONTROLLER=delta
STACK_DEMOS="demos/stack${STACK_BLOCKS}-${STACK_CONTROLLER}-$(date +%Y%m%d-%H%M%S)"
```

Set `STACK_BLOCKS=4` for four blocks. To test planner motion, set
`STACK_CONTROLLER=planner` before assigning `STACK_DEMOS` and collecting a new
dataset. Keep the same block count and controller throughout the sequence.

### 2. Collect demonstrations and videos

```bash
uv run --extra roboverify python -m taskbench.roboverify.entry.collect_demos \
  --program taskbench.roboverify.examples.stack:build_program \
  --num-blocks "$STACK_BLOCKS" \
  --seeds 0 42 \
  --max-loop-iterations "$((STACK_BLOCKS - 1))" \
  --move-controller "$STACK_CONTROLLER" \
  --save-video \
  --output-dir "$STACK_DEMOS"
```

For 100 demonstrations, replace `--seeds 0 42` with
`--seed-start 0 --num-trajectories 100`. The 14 cm spacing and current
geometry/release acceptance criteria are applied automatically.

Use a new output directory for each collection; the timestamp above supplies
one. An all-valid batch produces `$STACK_DEMOS/demonstrations.npz`.
A batch containing failures retains individual traces and a `collection.json`
report. Videos are saved under `$STACK_DEMOS/videos/`.
Continue once collection produces `demonstrations.npz`.

### 3. Verify the supplied Stack program

```bash
uv run --extra roboverify python -m taskbench.run \
  solver=program_synthesis \
  run.solver_kwargs.mode=verify \
  +run.solver_kwargs.program=taskbench.roboverify.examples.stack:build_program \
  run.solver_kwargs.num_blocks="$STACK_BLOCKS" \
  run.solver_kwargs.move_controller="$STACK_CONTROLLER" \
  run.solver_kwargs.demos="$STACK_DEMOS/demonstrations.npz" \
  'run.solver_kwargs.initial_arm=[0,0,0.17]' \
  +run.solver_kwargs.verification_timeout_ms=15000 \
  +run.solver_kwargs.motion_timeout_ms=30000
```

This executes the supplied program, learns its loop invariants, and runs the
symbolic and motion verification stages.

### 4. Synthesize a program and verify the result

```bash
uv run --extra roboverify python -m taskbench.run \
  solver=program_synthesis \
  run.solver_kwargs.mode=full \
  run.solver_kwargs.num_blocks="$STACK_BLOCKS" \
  run.solver_kwargs.move_controller="$STACK_CONTROLLER" \
  run.solver_kwargs.demos="$STACK_DEMOS/demonstrations.npz" \
  'run.solver_kwargs.initial_arm=[0,0,0.17]' \
  +run.solver_kwargs.verification_timeout_ms=15000 \
  +run.solver_kwargs.motion_timeout_ms=30000
```

Results go under `runs/cfg/`. Add `run.solver_kwargs.smoke=true` to the full
command for a small diagnostic search; `budget_exhausted` does not indicate a
synthesized or verified solution.

`initial_arm` is an explicit initial TCP position assumed by the geometric proof,
in metres. The example uses a nominal Panda reset TCP. It does not initialize
the physical robot; physical starts come from the demonstrations. See the
[Stack guide](docs/roboverify.md#initial-arm-premise) before choosing that premise
for another dataset.

This solver owns its environments: use `run.solver_kwargs` for block count and
controller selection.

### Current behavior

Current RoboVerify Stack behavior:

- Panda, one CPU simulation, 40 mm cubes, and 14 cm minimum center separation
  along X or Y at reset.
- Final acceptance requires tower geometry, released cubes, complete execution,
  and the task predicates. Linear/angular velocity and a stability-duration
  window are not acceptance requirements.
- `delta` is the default controller. Select `planner` during both collection
  and synthesis to use mplib screw planning with joint-position control.
- Planner collection currently has a numerical planning failure that can stop
  execution abruptly. In the matched 100-seed evaluation it accepted 92/100
  three-block and 90/100 four-block runs; delta accepted 100/100 for each.
- `verified_model` means both formal verification stages passed under their
  model assumptions. Physical collection success alone does not establish it.

The [Stack synthesis and verification guide](docs/roboverify.md) includes
supplied-program verification, controller settings, proof scope, and validation
results. New-program synthesis convergence remains an experimental result;
bounded search checks so far have returned `budget_exhausted`.

## Other workflows

### Shared-skill Stack solver

`stack_cubes` stacks onto the green `cube_0` using the shared
`taskbench.skills` Pick/Place implementation. Its default control mode is
`pd_ee_delta_pose`. It records HDF5 state trajectories and skill calls under
`data/success/` and `data/failure/`.

```bash
uv run python -m taskbench.run solver=stack_cubes \
  env.num_cubes=3 env.max_episode_steps=1000 \
  run.num_episodes=10 env.record_video=true
```

For the shared solver's planner path, set `env.control_mode=pd_joint_pos`.
The [skills guide](docs/skills.md) describes this control-mode dispatch and the
[recording guide](docs/demos.md) covers HDF5 replay and its current limitation.
RoboVerify synthesis consumes its own NPZ traces.

`StackNCube-v1` supports 2–6 cubes and `sparse` or `none` rewards. Its raw
environment reset uses a smaller placement clearance; RoboVerify's 14 cm sampler
is applied by the RoboVerify backend.

### Build2D

Build an `m × n` grid by placing a block at each linked-list node target.
The environment exposes a head `h`; nodes have `.r` (right), `.d` (down),
and world coordinates `(x, y, z)`. `build2d_program` executes:

```text
i <- h
while i != null:
    j <- i
    while j != null:
        put a block on j
        j <- j.r
    i <- i.d
```

```bash
# Execute in the simulator
uv run python -m taskbench.run solver=build2d_program \
  env.extra_kwargs.grid_rows=3 env.extra_kwargs.grid_cols=4

# Check DSL semantics without starting ManiSkill
uv run python -m taskbench.programs.build2d_demo --rows 3 --cols 4
```

### Random baseline

```bash
uv run python -m taskbench.run env.reward_mode=sparse run.num_episodes=10
```

The explicit reward override is needed for the default Stack environment:
the base configuration sets `normalized_dense`, which StackNCube does not support.

## Configuration and extension

Hydra selects a solver with `solver=name`. Existing configuration keys can be
overridden directly; use `+` to add a solver option absent from its YAML.

| Workflow | Task/controller settings | Output |
| --- | --- | --- |
| RoboVerify | `run.solver_kwargs.num_blocks`, `move_controller`, `planner_step_limit` | NPZ collections; synthesis/proof artifacts in `runs/cfg/` |
| Shared-skill Stack | `env.num_cubes`, `env.control_mode` | HDF5 in `data/`; optional videos in `videos/` |
| Build2D | `env.extra_kwargs.grid_rows`, `grid_cols` | Solver result; optional videos in `videos/` |

Add a solver module using `@register_solver("name")` and
`solve(self, env, seed=None, cfg=None)`, then add
`configs/solver/name.yaml`. Solvers are discovered automatically. See the
[architecture guide](docs/architecture.md) for the runner contract and extension
examples.

## Project structure

```text
configs/                 # Base Hydra configuration and solver config group
taskbench/
  run.py                 # Hydra entry point
  solver.py              # Solver registry and result contract
  envs/                  # ManiSkill environments and factories
  skills/                # Shared Pick, Place, Move, Push skills and mplib helpers
  recorder.py            # HDF5 state and skill-call recorder
  programs/              # Build2D program representation and executor
  roboverify/            # Copied synthesis/verification core and ManiSkill adapter
  solvers/               # Registered solvers, including program_synthesis
docs/
  architecture.md        # Dispatch, environments, and extension points
  skills.md              # Shared skill APIs and controller behavior
  demos.md               # NPZ, HDF5, and interactive JSON recording
  roboverify.md          # Stack collection, synthesis, and verification
  roboverify-validation.md # Dated experiment results
```

## Development

```bash
uv sync --extra dev --extra roboverify
uv run black taskbench/
uv run isort taskbench/
```

See the [RoboVerify tests](docs/roboverify.md#tests) for the unit-test command
and simulator validation steps. Provenance is recorded in
[taskbench/roboverify/SOURCE.md](taskbench/roboverify/SOURCE.md).
