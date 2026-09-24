# RoboVerify on StackNCube

The current RoboVerify synthesis and verification algorithms are copied into
`taskbench/roboverify`, with ManiSkill replacing the MuJoCo runtime. The source
revision and adaptations are recorded in
[`SOURCE.md`](../taskbench/roboverify/SOURCE.md).

The migration replaces the obsolete `program_synthesis` solver. Both
`solver=program_synthesis` and `solver=roboverify_stack` now dispatch to the
current pipeline. Build2D is outside this integration.

## Setup and demonstrations

```bash
uv sync --extra roboverify

uv run --extra roboverify python -m taskbench.roboverify.entry.collect_demos \
  --program taskbench.roboverify.examples.stack:build_program \
  --num-blocks 3 --seeds 42 --max-loop-iterations 2 \
  --output-dir data/roboverify/stack3
```

The collector executes the supplied relational Stack program with real Panda
Pick/Move/Release control. It holds the initial TCP for 50 settling steps, then
records observations, actions, snapshots, instruction boundaries, loop heads,
normal exits, and symbolic bindings. Accepted demonstrations must pass both
StackNCube's stable-tower/ungrasped check and RoboVerify's task predicates.
Each output directory must be new. Use multiple distinct seeds when collecting
a training set; failed runs are retained as diagnostics and prevent acceptance
of the requested batch.

Existing HDF5 demonstrations and RoboVerify MuJoCo NPZ files are not compatible:
they do not contain this backend's replay state. Recollect them. All integration
coordinates, offsets, and error bounds are in world-frame metres. The physical
and formal block length is 0.04 m; there is no coordinate rescaling.

## Verify a supplied program first

```bash
uv run --extra roboverify python -m taskbench.roboverify.entry.synthesize_cfg \
  --mode verify --program taskbench.roboverify.examples.stack:build_program \
  --num-blocks 3 --demos data/roboverify/stack3/demonstrations.npz \
  --max-loop-iterations 2 \
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
  --num-blocks 3 --demos data/roboverify/stack3/demonstrations.npz \
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
uv run --extra roboverify python -m taskbench.run solver=roboverify_stack \
  run.solver_kwargs.demos=data/roboverify/stack3/demonstrations.npz \
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
`pd_ee_delta_pose`. Runtime data are still subject to the copied predicate
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
