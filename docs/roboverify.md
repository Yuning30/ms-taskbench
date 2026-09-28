# RoboVerify on StackNCube

Use `solver=program_synthesis` to run RoboVerify synthesis and verification on
StackNCube. The algorithms are copied into `taskbench/roboverify`, with ManiSkill
providing the runtime. The source revision and adaptations are recorded in
[`SOURCE.md`](../taskbench/roboverify/SOURCE.md).

## Setup and demonstrations

```bash
uv sync --extra roboverify

uv run --extra roboverify python -m taskbench.roboverify.entry.collect_demos \
  --program taskbench.roboverify.examples.stack:build_program \
  --num-blocks 3 --seeds 0 42 --max-loop-iterations 2 \
  --output-dir demos/stack3-uniform-control
```

The collector executes the supplied relational Stack program with real Panda
Pick/Move/Release control. It holds the initial TCP for 50 settling steps, then
records observations, actions, snapshots, instruction boundaries, loop heads,
normal exits, and symbolic bindings. Accepted demonstrations must pass both
StackNCube's tower-geometry/ungrasped check and RoboVerify's task predicates.
Each output directory must be new. Use multiple distinct seeds when collecting
a training set; failed runs are retained as diagnostics and prevent acceptance
of the requested batch.

Fresh RoboVerify Stack resets use **14 cm minimum block-center separation along
X or Y**: `abs(dx) >= 0.14 OR abs(dy) >= 0.14` for every pair. The bounded sampler
keeps the original workspace and random cube yaw, with retries confined to each
seed's own RNG stream. This physical clearance is separate from the symbolic
Scattered predicate's 8 cm threshold.

StackNCube accepts the final geometry when cube 0 is the lowest cube, adjacent
cube centers sorted by height are within 33.3 mm horizontally and 40 ± 5 mm
vertically, and no cube is grasped. The final state is evaluated when the program
finishes. The `all_static` velocity flag is retained as a diagnostic and does not
affect acceptance.

All integration coordinates, offsets, and error bounds are in world-frame metres.
The physical and formal block length is 0.04 m; there is no coordinate rescaling.

The default delta controller bounds Cartesian actions with uniform XYZ scaling, uses a 2 mm
Pick stopping tolerance, and corrects Release XY drift toward the position
captured before opening. All primitives use gain 20 and 50-step budgets, with
gripper latching, actual Panda grasp checks, and held-cube feedback. The 2 mm
tolerance is an endpoint criterion, not a bound on the entire path.

## Choose the move controller

Set `--move-controller delta` (the default) or `--move-controller planner` for
collection and the standalone synthesis/verification CLI. With Hydra, set
`run.solver_kwargs.move_controller=delta` or `planner`. This selects the motion
implementation inside Pick, Move, and Release; the program's operands, offsets,
14 cm reset spacing, and task acceptance criteria stay the same.

For planner demonstrations:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --extra roboverify python -m taskbench.roboverify.entry.collect_demos \
  --program taskbench.roboverify.examples.stack:build_program \
  --num-blocks 3 --seeds 0 42 --max-loop-iterations 2 \
  --move-controller planner --output-dir demos/stack3-planner
```

Then use the same controller for synthesis:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --extra roboverify python -m taskbench.run \
  solver=program_synthesis \
  run.solver_kwargs.move_controller=planner \
  run.solver_kwargs.demos=demos/stack3-planner/demonstrations.npz \
  'run.solver_kwargs.initial_arm=[0,0,0.17]'
```

Planner mode uses the shared mplib `plan_screw()` helper with `pd_joint_pos`.
Each motion keeps the current TCP orientation and plans straight translation.
Held-cube destinations account for the measured cube-to-TCP offset. Execution
still requires the measured TCP/cube endpoint to reach the 2 mm tolerance and
checks that a carried cube remains grasped. A planning failure stops the
primitive without switching controllers or searching for a detour.

Planner primitives use `--planner-step-limit 200` by default, including gripper
phases, instead of the DSL's delta step budget. Adjust it through
`run.solver_kwargs.planner_step_limit` in Hydra. The planner uses a 0.01 rad joint
integration step for numerical accuracy, retrying once at 0.005 rad if planning
fails, and executes at the normal simulator control timestep. Both attempts use
the same start and destination. It retains table and self-collision checks, allowing only the
fixed Panda base's mounting contact with the table. Loose cubes, the tower, and
the carried cube are not registered as planner collision geometry.

Archives record the selected controller and planner budget. Synthesis checks
these settings and uses them for candidate execution and joint-action replay.
Set controller selection under `run.solver_kwargs`; this solver owns its
environment, so `env.control_mode` does not choose its controller.

## Compare controller paths

```bash
uv run --extra roboverify python -m taskbench.roboverify.experiment.compare_stack_control \
  --num-blocks 3 4 --seed-start 0 --num-seeds 100
```

This serial diagnostic compares the previous controller with the updated one
from matching serialized, settled starts. Restored float32 state is checked at
absolute tolerance `1e-7`, with the maximum discrepancy recorded. Unexposed
PhysX contact caches are not included. Both successful and failed seeds remain
in the report; a newly failing baseline-successful seed produces exit code 2.

Read `runs/stack-control/latest/artifacts/report.md` for the summary,
`summary.json` for distributions and failures, and `executions.json` for sampled
paths and physical success checks. `comparison.png` and `comparison.svg` plot the
distributions. Sampling uses the environment's actual control timestep. Gripper
opening/closing and turns between motion phases are excluded; Release retains
the XY reference captured before opening. Carrying-motion errors measure the
cube against its actual destination, with TCP and attachment drift reported
separately. These are empirical measurements, not certified tracking bounds.

The diagnostic retains historical behavior only inside its scoped comparison
context. It does not write demonstration archives or change the shared skills
in `taskbench/skills/`.

## Compare initial block spacing

```bash
CUDA_VISIBLE_DEVICES=0 uv run --extra roboverify python -m taskbench.roboverify.experiment.compare_stack_spacing \
  --num-blocks 4 --separation-cm 14 --seed-start 0 --num-seeds 100 \
  --output-dir runs/stack-spacing/4b-14cm
```

This experiment collects with configurable pairwise center separation:
`abs(dx) >= separation OR abs(dy) >= separation`. The physical clearance is
independent of the symbolic Scattered predicate. Use the same experiment runner
at 8 cm for its baseline. Every spacing retains the original workspace, cube
rotations, controller, settling, and acceptance criteria. Sampling retries use
each seed's own RNG stream instead of advancing to another requested seed.

Each cohort saves its settings and source hashes, per-seed outcomes, sampling
effort, and full traces. Sampling failures count toward the failure rate. An
all-valid cohort also produces `demonstrations.npz`. The experiment uses the
production sampler, overriding its 14 cm default for each comparison.

The 2026-09-28 sweep used 100 distinct layouts per spacing and block count:

| Minimum center separation along X or Y | 3 blocks | 4 blocks |
| --- | --- | --- |
| 8 cm | 90/100 | 84/100 |
| 10 cm | 98/100 | 95/100 |
| 12 cm | 100/100 | 98/100 |
| 14 cm | 100/100 | 100/100 |
| 16 cm | 100/100 | 100/100 |

At 14 cm, fresh seeds 100–299 also passed 200/200 for each block count, giving
300/300 distinct layouts per task across the sweep and confirmation. The 14 cm
setting is now the default for RoboVerify Stack resets. The 16 cm
condition showed no observed success improvement and required more rejection
sampling in the fixed workspace. This is empirical collection evidence for
three and four cubes, not a robot collision-freedom proof.

Results, source hashes, a failure video, and all traces are in
`runs/stack-spacing-20260928/`. The two fresh-seed all-valid archives are
`confirmation/3b-14cm/demonstrations.npz` and
`confirmation/4b-14cm/demonstrations.npz` under that directory.

## Verify a supplied program first

```bash
uv run --extra roboverify python -m taskbench.roboverify.entry.synthesize_cfg \
  --mode verify --program taskbench.roboverify.examples.stack:build_program \
  --num-blocks 3 --demos demos/stack3-uniform-control/demonstrations.npz \
  --max-loop-iterations 2 \
  --verification-timeout-ms 15000 --motion-timeout-ms 30000 \
  --invariant-relations ON_star Higher Scattered equality \
  --supported-towers --table-surface-height 0 \
  --initial-arm 0 0 0.17
```

`--initial-arm` is an explicit geometric proof premise. The example uses a
nominal Panda reset TCP, not an automatic assertion about every physical reset.
The exact recorded TCP can be read from the first three values of a trace's
initial state. The migration validation used that recorded value. The runtime
always restores the complete saved start for each candidate rather than
regenerating a scene from its seed.

This mode skips initial search, then executes the supplied candidate, learns
loop invariants from that execution, and checks symbolic and motion obligations.
The learner is RoboVerify's partition-based `InvInference` path. Invariants are
not copied from the supplied example program.

## Synthesize and verify

```bash
uv run --extra roboverify python -m taskbench.roboverify.entry.synthesize_cfg \
  --mode full --synthesis-approach id-first \
  --num-blocks 3 --demos demos/stack3-uniform-control/demonstrations.npz \
  --slots 5 --max-loop-iterations 2 \
  --invariant-relations ON_star Higher Scattered equality \
  --supported-towers --table-surface-height 0 \
  --initial-arm 0 0 0.17
```

ID-first search mutates numeric primitives and optimizes their offsets with CEM,
refines the CFG, recovers repeated flat-loop structure, then searches named loop
bodies. Every resulting candidate must pass fresh execution, invariant learning,
and both verification stages. `--synthesis-approach relational --quotient` retains
RoboVerify's alternative of introducing relational names earlier in search.

For a bounded diagnostic run, add `--smoke`. A small budget may finish with
`budget_exhausted`; that is not a synthesized or verified solution. Full synthesis
convergence is an experimental outcome, not guaranteed by the migration.

The equivalent Hydra entry point is:

```bash
uv run --extra roboverify python -m taskbench.run solver=program_synthesis \
  run.solver_kwargs.demos=demos/stack3-uniform-control/demonstrations.npz \
  'run.solver_kwargs.initial_arm=[0,0,0.17]'
```

Synthesis owns its environments; this solver does not create an extra episode
environment. The runner preserves the formal result instead of overwriting it
with a simulator success flag. The standalone CLI exposes the full search and
verification configuration and returns a nonzero exit code on failure.

## Results and proof scope

```bash
uv run --extra roboverify python -m taskbench.roboverify.experiment.report \
  --run runs/cfg/latest
```

Run artifacts include configuration, the CFG, candidate programs and traces,
learned invariants, individual symbolic and motion obligations, and the final
result. `verified_model` requires both proof stages. A failed, unknown, vacuous,
unsupported, or budget-exhausted check cannot become verification success.

The symbolic stage performs finite checks followed by an unbounded proof.
Motion checks use idealized waypoint motion, supported-tower geometry, and
explicit primitive contracts. They do not prove robot-arm collision freedom,
Panda controller refinement, grasp reliability, or total termination. Successful
physical demonstrations are validation evidence, separate from those proofs.

The adapter currently supports 2–6 uniform cubes, Panda, CPU simulation, and
`pd_ee_delta_pose` or planner-driven `pd_joint_pos`. Runtime data are still subject to the copied predicate
assumptions, including scattered initial cubes and a shared height tolerance.
The snapshot API does not expose all internal PhysX caches, so replay is checked
numerically and refused if it differs by more than 1e-5 m in recorded features.

## Tests

The copied core tests and the additional adapter-contract tests use `unittest`:

```bash
uv run --extra roboverify python - <<'PY'
from pathlib import Path
import unittest
modules = sorted('.'.join(p.with_suffix('').parts)
                 for p in Path('taskbench/roboverify').rglob('test_*.py'))
modules.append('taskbench.envs.test_stack_n_cube')
result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromNames(modules))
raise SystemExit(not result.wasSuccessful())
PY
```

Simulator-backed validation should additionally collect a fresh demonstration,
restore/replay a boundary while a cube is held, run supplied-program verification,
and exercise bounded full search. Unit tests alone do not establish these results.

Migration validation on 2026-09-24 passed all 193 tests. A three-cube, seed-42
ManiSkill collection was accepted. Supplied-program verification learned from
fresh candidate execution and passed 12 symbolic checks (finite sizes and the
unbounded proof), plus all 63 noiseless motion obligations. Restoring a held-cube
boundary reproduced recorded positions exactly; action replay differed by at
most 7.3e-6 m and preserved the held object and bindings. A one-iteration MCMC/CEM
diagnostic ran real simulator evaluations and returned `budget_exhausted`;
synthesis of a new successful program has not been demonstrated by this check.

Controller-update validation on 2026-09-25 passed **216 tests**, including
instrumented/uninstrumented Panda rollout parity. Fresh three-cube seeds 0 and
42 were accepted in `demos/stack3-uniform-control/demonstrations.npz`. A new
supplied-program run learned from both traces and passed all 12 symbolic checks,
including the unbounded proof, and all 63 noiseless motion obligations:
`runs/cfg/20260925-042645-f748f42-uniform-control-verify-longer`. One obligation
timed out in an earlier run at 3 seconds; the successful run allowed 30 seconds
per motion obligation and 15 seconds per symbolic check, without changing the
proof premises. The bounded search diagnostic executed MCMC/CEM and returned
`budget_exhausted` in `runs/cfg/20260925-042913-f748f42-uniform-control-search-smoke`.

Action replay is still refused when it exceeds the existing `1e-5` tolerance.
In the new seed-42 trace, the first held-object boundary differed by 11.1 micrometres
and a later boundary by 49.1 micrometres; these are rejected, not silently accepted.
Direct snapshot restore at the first held boundary differed by zero for seed 42
and less than 0.06 micrometres for seed 0. `--reset-mode reset` is the existing
explicit option for restoring snapshots rather than replaying action prefixes;
the default and tolerance remain unchanged. Simulator-internal contact caches
are not captured, so successful replay at one boundary does not establish it
for every boundary.

The paired comparison completed all **400 executions** (seeds 0–99, two controller
versions, three and four cubes), using the velocity-based acceptance criterion.
Results are in
`runs/stack-control/20260925-042428-f748f42-uniform-release-panda/artifacts/report.md`:

| Measurement | Previous | Updated |
| --- | ---: | ---: |
| Three-cube valid programs | 77/100 | 71/100 |
| Four-cube valid programs | 0/100 | 0/100 |
| Three-cube transfer path deviation, P95 | 41.50 mm | 8.30 mm |
| Four-cube transfer path deviation, P95 | 37.72 mm | 8.48 mm |
| Three-cube Release final XY error, P95 | 4.08 mm | 1.85 mm |
| Four-cube Release final XY error, P95 | 3.94 mm | 1.84 mm |

All 18 newly failing baseline-successful three-cube seeds failed only the final
`all_static` check: tower geometry and release checks passed. Twelve other seeds
recovered, giving the net decrease of six successes. Diagnostic holds at seeds
9, 10 and 15 reached physical success after one additional control step, but
these holds were not added to production or counted as paired-run successes.
Four-cube seeds 0 and 42 remained non-static after 20 diagnostic hold steps.
The detailed probes are in `settling-diagnosis.json` beside the report.

Under that criterion, the controller update improved direct-motion agreement
without improving overall task success. Peak Release sideways deviation was
largely unchanged (three-cube P95 4.63 to 4.69 mm). Some runs also failed primitive or loop budgets; failed and
partial executions remain in the report. Endpoint and path distributions include
all observed phases, not just successful programs. No seeds were filtered or
replaced; the sampler used in that experiment could map multiple requested seeds to
the same accepted layout. Those controller-only measurements retained the
velocity-based success criterion, collision masks, replay tolerance, and
instruction budgets.

## Collection with geometry and release acceptance

Fresh collection on 2026-09-25 ran all requested seeds 0–99 for each block count
with velocity flags used only as diagnostics. The reference program, controller,
50-step primitive budgets, and 50 initial settling steps were unchanged. Three
cubes used two loop iterations, and four cubes used three, with a 60-second
trajectory timeout.

| Cubes | Accepted demonstrations | Remaining failures |
| --- | ---: | --- |
| 3 | 96/100 (96%) | 3 loop-limit failures, 1 Pick descent failure |
| 4 | 85/100 (85%) | 9 loop-limit failures, 5 Pick descent failures, 1 Move failure |

Reapplying the velocity thresholds to these same recorded final states would
accept 61/100 and 2/100, respectively. These are fresh collection runs; the earlier
paired controller comparison restored saved starts. The reset sampler produced
66 distinct recorded starting layouts for three cubes and 44 for four cubes.
The reported rates include every requested seed.

The full report, source hashes, and per-seed measurements are in
`runs/stack-geometry-acceptance-20260925/`. The collections are in
`demos/stack3-geometry-acceptance-20260925/` and
`demos/stack4-geometry-acceptance-20260925/`. Both batches contain failures, so their
`diagnostics/` directories retain all 100 individual traces instead of a single
accepted-batch archive. All 10 targeted acceptance/backend tests passed.

Planner-selection validation on 2026-09-28 passed all **237 tests**, including
joint-action archive roundtrip, held-cube snapshot restore/replay, bounded
execution, and controller mismatch rejection. Planner collection accepted seeds
0 and 42 for both three and four cubes. Delta collection on those seeds exactly
matched the previous three-cube actions, observations, and simulator states.
Supplied-program verification through Hydra returned `verified_model`; a bounded
MCMC/CEM search exercised planner rollouts and returned `budget_exhausted`.
Details and archives are in `runs/planner-move-20260928/report.md`.

A broader planner collection on seeds 0–99 accepted **92/100 three-block** and
**90/100 four-block** demonstrations. Initial physical state vectors exactly
matched the earlier 14 cm delta cohort, which accepted 100/100 for both sizes.
All 18 planner failures came from mplib rejecting a tiny final integration step
at both configured resolutions; collision and joint-limit checks passed.
Successful planner runs used about 2.1 times as many simulator steps as delta on
the same seeds. Controller settings stayed fixed throughout the experiment.
The 182 accepted demonstrations, all failed-seed videos, exact rerun-parity
checks, and detailed diagnosis are in `runs/planner-collection-20260928/report.md`.
