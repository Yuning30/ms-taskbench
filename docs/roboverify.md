# RoboVerify on StackNCube

Use `solver=program_synthesis` for Stack synthesis and verification.
The algorithms are copied into `taskbench/roboverify`, with ManiSkill providing
the physical runtime. The package is self-contained; see
[source provenance](../taskbench/roboverify/SOURCE.md).

The current default is **delta control with 14 cm initial block separation**.
Planner motion is selectable, but a numerical mplib termination issue still
causes abrupt failures on some seeds. The matched 100-seed evaluation accepted
100/100 delta runs for both three and four blocks, versus 92/100 and 90/100 with
planner motion. See [dated validation results](roboverify-validation.md).

## Setup and demonstration collection

```bash
uv sync --extra roboverify
```

Collect three-block demonstrations and videos:

```bash
uv run --extra roboverify python -m taskbench.roboverify.entry.collect_demos \
  --program taskbench.roboverify.examples.stack:build_program \
  --num-blocks 3 --seeds 0 42 --max-loop-iterations 2 \
  --move-controller delta --save-video --output-dir demos/stack3-delta
```

For four blocks:

```bash
uv run --extra roboverify python -m taskbench.roboverify.entry.collect_demos \
  --program taskbench.roboverify.examples.stack:build_program \
  --num-blocks 4 --seeds 0 42 --max-loop-iterations 3 \
  --move-controller delta --save-video --output-dir demos/stack4-delta
```

Use a new output directory on every run. Replace `--seeds 0 42` with
`--seed-start 0 --num-trajectories 100` for a consecutive 100-seed batch.
Collection defaults to a 60-second wall-clock limit per trajectory.
On a machine where rendering needs a particular GPU, prefix the command with
`CUDA_VISIBLE_DEVICES=0`.

The supplied relational program starts with cube 0 as the base. Each iteration
chooses another clear block, executes Pick, lifts it, translates above the
current tower, lowers it onto the tower, and executes Release. RoboVerify Pick
closes onto the block; lifting is a separate Move. Moves track the held cube's
center. Release opens the gripper and retreats with a fixed XY reference.

The collector holds the initial TCP for 50 settling steps before recording.
It stores observations, low-level actions, full exposed simulator snapshots,
instruction boundaries, loop heads/exits, and symbolic bindings.

If every requested trace is valid, the batch produces `demonstrations.npz`.
Otherwise it reports `invalid_demonstrations` and retains all available
per-seed traces, including successes, in `diagnostics/`. Inspect
`collection.json` for outcomes and media errors. Optional 20 FPS videos are
written to `videos/seed_NNNN.mp4`.
See [recording and replay](demos.md) for the archive format.

## Initialization and acceptance

The backend uses Panda, one CPU environment, zero robot joint initialization
noise, and uniform 40 mm cubes. Coordinates, offsets, and error bounds use
world-frame metres.

RoboVerify resets require **14 cm minimum block-center separation along X or Y**:

```text
abs(dx) >= 0.14 OR abs(dy) >= 0.14    for every pair of cubes
```

The bounded sampler retains X = [−0.1, 0.1] m, Y = [−0.2, 0.2] m and random
cube yaw. Retries stay within each seed's own RNG stream; they do not advance to
another seed. This physical clearance is separate from the symbolic
`Scattered` predicate's 8 cm threshold. The raw Stack environment used by
shared-skill solvers has its own smaller-clearance reset.

A demonstration is accepted when execution completes within its budgets and
passes both the symbolic task pre/postconditions and StackNCube success:

- Cube 0 is the lowest cube.
- Adjacent centers sorted by height differ horizontally by at most about
  33.3 mm and vertically by 40 ± 5 mm.
- No cube is grasped.

The final state is evaluated when the program finishes. Linear/angular velocity
does not affect acceptance; `all_static` is retained as a diagnostic.
There is no requirement to maintain the final conditions for a duration.
Initial settling and the primitive's physical actions are separate from a
final stability window.

## Choose the move controller

Use `--move-controller delta|planner` in collection and the standalone CLI.
With Hydra, use `run.solver_kwargs.move_controller=delta|planner`.
This selects motion inside all three RoboVerify primitives: Pick, Move, Release.

| Setting | Delta (default) | Planner |
| --- | --- | --- |
| ManiSkill control mode | `pd_ee_delta_pose` | `pd_joint_pos` |
| Motion | Position feedback, gain 20 | mplib `plan_screw()`, fixed TCP orientation per motion |
| Default primitive budget | 50 control steps from the DSL | 200 control steps, including gripper phases |
| Position stopping tolerance | 2 mm | 2 mm measured after executing the plan |
| Recorded action | XYZ delta and gripper | Seven joint positions and gripper |

Delta uniformly scales XYZ when bounding Cartesian commands, preserving their
direction; the gripper command is bounded independently. Carrying moves use
actual held-cube feedback. Release captures its XY reference before opening
and corrects lateral drift during retreat. The 2 mm criterion applies to the
endpoint, not the entire physical path.

Planner mode plans straight translation at fixed TCP orientation and accounts
for the measured cube-to-TCP offset when carrying. It checks measured endpoint
convergence and that the carried cube remains grasped. Planning uses a 0.01 rad
joint integration step, with one retry at 0.005 rad from the same start to the
same destination. It executes at the simulator's normal control timestep.

Table and self-collision checks remain enabled, with only the fixed Panda base's
mounting contact with the table allowed. Loose cubes, the tower, and the held
payload are not registered as planner collision geometry. A rejected plan ends
the primitive; there is no delta fallback or detour search.

**Current planner limitation:** on the 18 failed trials in the broader
evaluation, mplib rejected a tiny final integration increment at both
resolutions, even though collision and joint-limit checks passed. The runtime
therefore stopped before starting the newly planned motion. The integration
does not yet fix this numerical termination rule.

Collect planner demonstrations with:

```bash
uv run --extra roboverify python -m taskbench.roboverify.entry.collect_demos \
  --program taskbench.roboverify.examples.stack:build_program \
  --num-blocks 3 --seeds 0 42 --max-loop-iterations 2 \
  --move-controller planner --planner-step-limit 200 \
  --save-video --output-dir demos/stack3-planner
```

Use the same settings when synthesizing:

```bash
uv run --extra roboverify python -m taskbench.run solver=program_synthesis \
  run.solver_kwargs.num_blocks=3 \
  run.solver_kwargs.move_controller=planner \
  run.solver_kwargs.planner_step_limit=200 \
  run.solver_kwargs.demos=demos/stack3-planner/demonstrations.npz \
  'run.solver_kwargs.initial_arm=[0,0,0.17]'
```

Archives record controller and planner budget; synthesis rejects mismatched
settings and uses the selected backend for candidate execution and action
replay. This solver owns its environments, so `env.control_mode` does not
select its controller.

## Verify a supplied program

```bash
uv run --extra roboverify python -m taskbench.roboverify.entry.synthesize_cfg \
  --mode verify --program taskbench.roboverify.examples.stack:build_program \
  --num-blocks 3 --demos demos/stack3-delta/demonstrations.npz \
  --move-controller delta --max-loop-iterations 2 \
  --verification-timeout-ms 15000 --motion-timeout-ms 30000 \
  --invariant-relations ON_star Higher Scattered equality \
  --supported-towers --table-surface-height 0 \
  --initial-arm 0 0 0.17
```

Verify mode skips initial program search, executes the supplied candidate,
learns invariants with RoboVerify's partition-based `InvInference`, and checks
symbolic and motion obligations. It does not copy invariants from the example
program.

With Hydra, set `run.solver_kwargs.mode=verify` and add
`+run.solver_kwargs.program=taskbench.roboverify.examples.stack:build_program`
to the synthesis command below. Options absent from the YAML, such as
`verification_timeout_ms`, also need the `+` prefix.

### Initial arm premise

`--initial-arm 0 0 0.17`, or
`'run.solver_kwargs.initial_arm=[0,0,0.17]'`, constrains the initial TCP position
in the geometric verification model. It does not move the robot, initialize its
joints, or replace the recorded physical start. Omitting it leaves that initial
model position unconstrained; it is not automatically inferred.

The example is a nominal Panda reset TCP. Inspect the exact recorded initial
positions when deciding what premise a dataset supports:

```python
from taskbench.roboverify.cfg.recordings import load_traces

for trace in load_traces("demos/stack3-delta/demonstrations.npz", require_valid=True):
    print(trace.seed, trace.states[0][:3])
```

A nominal rounded position is a model assumption, not proof that every physical
reset has exactly those coordinates. Candidate execution restores recorded
simulation starts, independently of this symbolic premise.

## Synthesize and verify

```bash
uv run --extra roboverify python -m taskbench.run solver=program_synthesis \
  run.solver_kwargs.mode=full \
  run.solver_kwargs.num_blocks=3 \
  run.solver_kwargs.move_controller=delta \
  run.solver_kwargs.demos=demos/stack3-delta/demonstrations.npz \
  'run.solver_kwargs.initial_arm=[0,0,0.17]'
```

For four blocks, set `run.solver_kwargs.num_blocks=4` and
`run.solver_kwargs.demos=demos/stack4-delta/demonstrations.npz`.
Root `env.num_cubes` does not configure this solver.

The Hydra defaults select ID-first search, five slots, 20 search iterations,
five refinements, supported-tower geometry, and a table surface at Z = 0.
ID-first search mutates numeric primitives and optimizes offsets with CEM,
refines the CFG, recovers repeated flat-loop structure, and searches named loop
bodies. Candidates must pass execution, invariant learning, and both proof
stages.

The standalone CLI exposes the full configuration:

```bash
uv run --extra roboverify python -m taskbench.roboverify.entry.synthesize_cfg \
  --mode full --synthesis-approach id-first --slots 5 \
  --num-blocks 3 --demos demos/stack3-delta/demonstrations.npz \
  --move-controller delta \
  --invariant-relations ON_star Higher Scattered equality \
  --supported-towers --table-surface-height 0 \
  --initial-arm 0 0 0.17
```

Its unqualified defaults differ from Hydra, including relational search,
four slots, and no supported-tower premise. Specify these flags when using the
ID-first workflow above. `--synthesis-approach relational --quotient` selects
the alternative that introduces relational names earlier.

The standalone CLI defaults to seed 0. The Hydra runner passes
`cfg.seed + episode_number`, so its first run uses 43 with the default
`seed=42`. Use `+run.solver_kwargs.seed=0` to explicitly match CLI seed 0.

For a bounded diagnostic run, add `--smoke` or
`run.solver_kwargs.smoke=true`. A `budget_exhausted` result is not a
synthesized/verified solution. Supplied-program verification has passed;
the recorded bounded search checks have exercised real rollouts without
demonstrating synthesis of a new successful program.

## Results, replay, and proof scope

```bash
uv run --extra roboverify python -m taskbench.roboverify.experiment.report \
  --run runs/cfg/latest
```

Run artifacts include configuration, CFGs, candidate programs/traces, learned
invariants, individual obligations, and the final result. `verified_model`
requires both proof stages. Failure, unknown, vacuity, unsupported checks, and
budget exhaustion cannot become verification success. The standalone synthesis
CLI returns a nonzero exit code on failure; the Hydra runner preserves the
formal result in `SolverResult.verification_status`.

The symbolic stage performs finite checks followed by an unbounded proof.
Motion checks use idealized waypoints, supported-tower geometry, and explicit
primitive contracts. They establish partial correctness under those premises;
they do not prove robot-arm collision freedom, physical controller refinement,
grasp reliability, or total termination. This scope also applies to planner
motion.

The adapter supports 2–6 uniform cubes, Panda, and CPU simulation; the broad
collection studies cover three and four cubes. Traces must agree on task,
block geometry, and predicate tolerances as well as controller settings.
The default `Higher` tolerance is 0.001 m.

Replay restores a saved start and executes an action prefix. It rejects
recorded-feature discrepancies above `1e-5`. `--reset-mode reset` instead
restores a boundary snapshot directly. Hidden PhysX contact caches are not
serialized, so successful replay is checked numerically. See
[NPZ replay details](demos.md#replay-during-synthesis).

## Experiment tools

Compare initial spacing with the delta controller:

```bash
uv run --extra roboverify python -m taskbench.roboverify.experiment.compare_stack_spacing \
  --num-blocks 4 --separation-cm 14 --seed-start 0 --num-seeds 100 \
  --output-dir runs/stack-spacing/4b-14cm
```

This uses the production sampler with a spacing override and records settings,
source hashes, all outcomes, sampling effort, and full traces. An all-valid
cohort also writes `demonstrations.npz`. The 14 cm sweep and fresh-seed
confirmation accepted 300/300 distinct layouts for each block count.

Compare the previous and updated **delta** controller implementations:

```bash
uv run --extra roboverify python -m taskbench.roboverify.experiment.compare_stack_control \
  --num-blocks 3 4 --seed-start 0 --num-seeds 100
```

This tool compares uniform scaling and Release feedback from matching serialized
settled starts; it is not a delta-versus-planner comparison. It checks restored
float32 state at tolerance `1e-7`, retains failures, and reports empirical
path/endpoint errors. Hidden PhysX caches are excluded. A newly failing
baseline-successful seed causes exit code 2.

Outputs under `runs/stack-control/latest/artifacts/` include `report.md`,
`summary.json`, `executions.json`, and PNG/SVG plots. Historical experiments
used different acceptance and reset settings; see
[validation history](roboverify-validation.md) when interpreting their rates.

## Tests

```bash
uv run --extra roboverify python - <<'PY'
from pathlib import Path
import unittest

modules = sorted(
    ".".join(p.with_suffix("").parts)
    for p in Path("taskbench/roboverify").rglob("test_*.py")
)
modules.append("taskbench.envs.test_stack_n_cube")
result = unittest.TextTestRunner().run(
    unittest.defaultTestLoader.loadTestsFromNames(modules)
)
raise SystemExit(not result.wasSuccessful())
PY
```

The planner integration passed 237 tests on 2026-09-28. Simulator validation
should additionally collect fresh demonstrations, restore/replay a held-cube
boundary, verify a supplied program, and exercise bounded full search. Unit
tests alone do not establish those outcomes.
