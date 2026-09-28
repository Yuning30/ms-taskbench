# Source and local adaptations

Copied from the local RoboVerify repository at commit
`dec922349052883970d885600cd853c0e523cc05` on 2026-09-24.
The original implementation is under `roboverify/synthesis/`.
This package is self-contained; it does not import `~/RoboVerify`.

The copied core includes the program IR, relational CFG refinement and flat-loop
quotienting, MCMC/CEM straight-line search, partition-based invariant inference,
symbolic verification, geometric motion verification, and their supporting
predicate and quantifier utilities. Selected upstream regression tests are kept
beside these modules. `source_manifest.json` records the original file hashes.

Controller changes were selectively ported from RoboVerify commit
`fd8dd579796ed31e46416fdf9bcd237f5294e1d0` on 2026-09-25. The manifest's
`updates` entry records the source hashes separately from the original import.
These changes introduce uniform XYZ scaling, 2 mm Pick stopping, fixed-XY/full-3D
Release feedback, action-scaling fingerprints, and adapted controller tests.
The compact Panda comparison tool reuses the upstream finite-segment measurement
method and paired-trial structure, with payload targets and failure handling
adapted to this backend. Fetch collision-mask changes were not ported.

Local changes:

- Imports use `taskbench.roboverify`.
- `backend.py` executes `StackNCube-v1` with one CPU Panda environment.
  No MuJoCo environments, standalone legacy search pipeline, or unrelated
  experiments were copied. `mcmc/synthesis.py` retains only the current mutation
  and RNG helpers, plus the local environment factory.
- Block length is scoped through `util.on.using_block_length`. Reference tests
  retain the upstream 50 mm default; the Stack entry points explicitly use
  40 mm consistently in numerical predicates, BMC, motion checks, counterexample
  geometry, and the supplied program.
- Recording uses ManiSkill actor/articulation state, controller state, drive
  targets, elapsed steps, object bindings, and held-object/gripper bookkeeping.
  Its archive format is distinct from MuJoCo archives. Replay must reproduce
  the segment observation within 10 micrometres; simulator-internal PhysX
  contact caches are not exposed by this snapshot API.
- Primitive control uses Panda grasp/contact feedback and gripper limits.
  Carrying moves track the actual payload center. Controller exhaustion and
  lost grasps stop execution. Failed candidate prefixes can inform search but
  cannot receive successful postcondition credit.
- Pick/Move/Release use 2 mm positional stopping tolerances, gain 20 and shared
  50-step instruction budgets. Commands scale XYZ uniformly at both controller
  and backend boundaries; gripper commands remain independent. Release captures
  its XY target before opening and corrects lateral drift during retreat.
- Collection accepts only complete executions satisfying both StackNCube
  success and the symbolic pre/postconditions. Candidate preparation also
  requires physical success. `stack_reset.py` uses the validated Panda spacing
  sampler with 14 cm block-center separation along X or Y. Bounded retries use
  each seed's own RNG, with no expansion of the workspace. The copied Stack
  `Scattered` predicate keeps its separate 8 cm threshold for 40 mm cubes.
- Runtime execution of legacy `PickPlace` macros is unsupported; the current
  pipeline uses explicit `Pick`, `Move`, and `Release` primitives.

The guarantee remains partial correctness in an idealized model, conditional
on its premises and primitive contracts. Copying the verifier does not prove
physical-controller refinement or termination. See
[`docs/roboverify.md`](../../docs/roboverify.md) for the supported workflow.
