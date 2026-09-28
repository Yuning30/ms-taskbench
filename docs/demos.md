# Demonstration recording and replay

Choose the recorder that matches the consuming workflow:

| Workflow | Format | Contents / consumer |
| --- | --- | --- |
| RoboVerify `collect_demos` | NPZ | Observations, actions, simulator snapshots, DSL events/bindings; input to `program_synthesis` |
| Shared `StateRecorder` / `stack_cubes` | HDF5 | Robot/object state and shared-skill calls; read by `solver=replay` |
| Interactive `demo_recorder` | JSON | Scene configuration and a manually selected skill program |

## RoboVerify collections

For three blocks:

```bash
uv run --extra roboverify python -m taskbench.roboverify.entry.collect_demos \
  --program taskbench.roboverify.examples.stack:build_program \
  --num-blocks 3 --seeds 0 42 --max-loop-iterations 2 \
  --move-controller delta --save-video --output-dir demos/stack3-delta
```

For four blocks, use `--num-blocks 4 --max-loop-iterations 3` and a new output
directory. To collect a larger batch, replace `--seeds 0 42` with
`--seed-start 0 --num-trajectories 100`. Each trajectory has a default
60-second wall-clock limit. Use `--move-controller planner` for planner
collection, and keep that selection when loading its data for synthesis.

The collector settles the reset for 50 steps, then executes the supplied
program and records instruction boundaries, loop heads, exits, and bindings.
Acceptance requires complete execution, Stack geometry/release success,
and symbolic task predicates. See [the Stack guide](roboverify.md) for the
14 cm sampler, controllers, and exact acceptance criteria.

Each explicit output directory must be new. Outputs are:

- `collection.json`: requested seeds, configuration, per-seed outcomes, and
  media status.
- `demonstrations.npz`: written only if **every requested trajectory is valid**.
- `diagnostics/seed_NNNN.npz`: when the batch fails, all available traces are
  retained here, including successful trajectories. A failure before any
  observation is available appears only in the report.
- `videos/seed_NNNN.mp4`: optional 20 FPS videos, including failed executions.

A failed batch reports `invalid_demonstrations` and the collector exits with
code 2. Media failures also cause exit code 2; inspect `status` and
`media_status` separately. The collector does not automatically create an
accepted subset archive. The `accepted-demonstrations.npz` files cited in the
planner experiment are separately prepared subsets.

### Reading NPZ traces

```python
from taskbench.roboverify.cfg.recordings import load_traces

traces = load_traces("demos/stack3-delta/demonstrations.npz", require_valid=True)
for trace in traces:
    actions, observation_action_indices = trace.actions
    print(trace.seed, trace.num_blocks, trace.metadata["status"])
    print("initial TCP:", trace.states[0][:3])
    print("observations:", len(trace.states), "actions:", len(actions))
    print("snapshots:", len(trace.snapshots), "events:", len(trace.events))
```

An observation is a compact `13 + 15 * num_blocks` feature vector. Snapshots
add ManiSkill actor/articulation state, controller state, drive targets, elapsed
steps, held-object state, gripper bookkeeping, and bindings. Action indices map
each observation to an action prefix.

Delta archives contain four-component Cartesian/gripper commands. Planner
archives contain eight-component actions: seven joint positions and a gripper
command. Metadata identifies the controller, planner budget, task/program,
block geometry, and predicate tolerance. Synthesis checks these settings.

### Replay during synthesis

The default `--reset-mode replay` restores the initial snapshot, executes the
recorded action prefix for a segment, checks the resulting observation against
the recording at tolerance `1e-5`, and restores the segment's runtime bindings
and held-object bookkeeping. A mismatch rejects the replay.

`--reset-mode reset` directly restores the boundary snapshot instead. Neither
method captures hidden PhysX contact caches, so exact physical replay is not
guaranteed. Successful replay at one boundary does not establish replay at
every boundary. This mechanism is part of RoboVerify; `solver=replay` handles
shared HDF5 programs.

## Recording shared skills

`stack_cubes` automatically writes
`data/success/episode_seedN.hdf5` or `data/failure/episode_seedN.hdf5`.
These outcomes are recorded inside the solver, before the runner's later
environment reevaluation. Use distinct seeds or directories for new datasets;
these filenames are not automatically made unique.

For custom recording, given a single CPU StackNCube environment configured
with `pd_ee_delta_pose`:

```python
from taskbench.recorder import StateRecorder
from taskbench.skills.context import SkillContext

ctx = SkillContext(env)
ctx.reset(seed=42)
recorder = StateRecorder(
    env,
    objects=ctx.objects,
    robot_fields=["qpos", "tcp_pos", "tcp_quat", "gripper_qpos"],
)
ctx.step_callback = recorder.record
ctx._build_skills()  # Rebind skills after assigning the callback.
recorder.record()

pick_kwargs = {"obj_name": "cube_1", "lift_height": 0.13}
recorder.record_skill_call("pick", pick_kwargs)
result = ctx.pick(**pick_kwargs)

if result.success:
    place_kwargs = {
        "target_pose_or_cube": "cube_0",
        "offsets": [0, 0, 0.04],
        "retract_height": 0.2,
    }
    recorder.record_skill_call("place", place_kwargs)
    result = ctx.place(**place_kwargs)

recorder.save(
    "data/custom/episode.hdf5",
    metadata={
        "seed": 42,
        "solver": "custom",
        "success": bool(env.unwrapped.evaluate()["success"].item()),
    },
)
```

The example moves one block. It only builds the complete tower in a two-cube
environment. `record_skill_call()` sets the per-frame label and records the
current frame index; the callback captures a frame after each skill control step.
Pass `hydra_cfg=cfg` to `save()` when a resolved Hydra configuration is
available. Other supported robot fields include `qvel`.

### HDF5 layout

```text
episode.hdf5
├── metadata/          attributes: env_id, control_freq, num_frames,
│                                  supplied metadata, optional hydra_config
├── robot/             selected fields: qpos, qvel, tcp_pos, tcp_quat, gripper_qpos
├── objects/
│   └── cube_N/        pos (frames, 3), quat (frames, 4)
├── skill              per-frame labels
└── program/
    ├── skill          skill names
    ├── args           JSON keyword arguments
    └── start_frame    frame index of each call
```

Panda `qpos` has nine components; joint-state dimensions depend on the robot.

```python
import json
import h5py

with h5py.File("data/custom/episode.hdf5", "r") as f:
    print(dict(f["metadata"].attrs))
    tcp_positions = f["robot/tcp_pos"][:]
    for skill, args, frame in zip(
        f["program/skill"].asstr()[:],
        f["program/args"].asstr()[:],
        f["program/start_frame"][:],
    ):
        print(frame, skill, json.loads(args))
```

### HDF5 program replay

The replay solver resets with the recorded seed and re-executes skill calls.
It does not restore recorded simulator snapshots or replay low-level actions.
When a saved Hydra configuration is present, the runner imports its environment
ID and cube count, but does not automatically restore its control mode. For a
two-cube delta recording using the current skill argument names:

```bash
uv run python -m taskbench.run solver=replay \
  run.solver_kwargs.demo_path=data/custom/episode.hdf5 \
  env.num_cubes=2 env.control_mode=pd_ee_delta_pose env.max_episode_steps=1000
```

**Current limitation:** `stack_cubes` writes Place calls with a
`target_pose` keyword, but the shared Place API expects `target_pose_or_cube`.
The replay solver forwards keywords unchanged, so those calls raise a
`TypeError`. The custom recording example above uses the current API.
This issue does not affect RoboVerify NPZ replay.

The replay implementation also deserializes pose dictionaries and nested pose
arrays; directly applying `json.loads()` and forwarding every value is not
sufficient for arbitrary recorded poses.

## Interactive recording

```bash
uv run python -m taskbench.run solver=demo_recorder env.max_episode_steps=1000
```

This workflow requires a graphical viewer. Use `1` for Pick, `2` for Place,
`3` for Push, `s` to save, `r` to reset, and `q` to quit.
It writes `demo_record.json` containing the scene configuration and selected
program. It does not produce the NPZ execution snapshots required by synthesis.
