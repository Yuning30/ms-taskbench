# RoboVerify validation history

These are dated experiments, not results rerun on every documentation update.
The [Stack guide](roboverify.md) describes current commands and behavior.
Artifact paths below refer to local experiment output under ignored `runs/`
and `demos/` directories; they are not included in a fresh clone.

## Planner collection (2026-09-28)

The matched seeds 0–99 comparison used 14 cm separation, 50 initial settling
steps, a 60-second trajectory timeout, and two/three loop iterations for
three/four blocks. Planner primitives had a 200-step budget; delta had 50.

| Blocks | Planner accepted | Delta accepted | Planner steps, median | Delta steps on the same successful seeds, median |
| --- | ---: | ---: | ---: | ---: |
| 3 | 92/100 | 100/100 | 299 | 141 |
| 4 | 90/100 | 100/100 | 467.5 | 220.5 |

Every initial physical state vector, including cube yaw and robot joints,
exactly matched the earlier delta trial. Each cohort had 100 distinct layouts,
and program fingerprints matched. No trial exhausted its step or wall-time
budget.

All 18 planner failures came from mplib rejecting a tiny final screw-integration
increment: `flag=True` indicated the final step, but
`norm(delta_twist) < 1e-4` was treated as failure before completion handling.
Both 0.01 and 0.005 rad integration resolutions hit the condition. At rejection,
`collide=False` and `within_joint_limit=True`.

The new motion was never executed because the complete plan was rejected,
which explains the abrupt stop in the videos. This numerical termination issue
remains unresolved in the current integration.

| Blocks | Failed seeds |
| --- | --- |
| 3 | 12, 24, 47, 57, 70, 74, 86, 99 |
| 4 | 12, 22, 24, 52, 57, 59, 74, 86, 95, 99 |

All failures were rerun with video and read-only planner instrumentation.
Actions, observations, raw simulator states, events, and errors exactly matched
the original traces. Diagnostic reruns were not additional success-rate samples.
Successful planner runs used about 2.1 times as many simulator steps as delta.

Local artifacts are under `runs/planner-collection-20260928/`:
`report.md`, `summary.json`, `failure-summary.json`, and `index.html`
for all failed-seed videos. The explicitly prepared
`3-blocks/accepted-demonstrations.npz` and
`4-blocks/accepted-demonstrations.npz` contain 92 and 90 valid traces.
Original rejected batches and all per-seed diagnostics remain intact.

## Initial spacing sweep (2026-09-28)

This sweep used the delta controller and geometry/release acceptance.

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

## Planner integration (2026-09-28)

Planner-selection validation on 2026-09-28 passed all **237 tests**, including
joint-action archive roundtrip, held-cube snapshot restore/replay, bounded
execution, and controller mismatch rejection. Planner collection accepted seeds
0 and 42 for both three and four cubes. Delta collection on those seeds exactly
matched the previous three-cube actions, observations, and simulator states.
Supplied-program verification through Hydra returned `verified_model`; a bounded
MCMC/CEM search exercised planner rollouts and returned `budget_exhausted`.
Details and archives are in `runs/planner-move-20260928/report.md`.

## Initial migration (2026-09-24)

Migration validation on 2026-09-24 passed all 193 tests. A three-cube, seed-42
ManiSkill collection was accepted. Supplied-program verification learned from
fresh candidate execution and passed 12 symbolic checks (finite sizes and the
unbounded proof), plus all 63 noiseless motion obligations. Restoring a held-cube
boundary reproduced recorded positions exactly; action replay differed by at
most 7.3e-6 m and preserved the held object and bindings. A one-iteration MCMC/CEM
diagnostic ran real simulator evaluations and returned `budget_exhausted`;
synthesis of a new successful program has not been demonstrated by this check.

## Delta controller integration (2026-09-25)

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

## Historical velocity-based comparison (2026-09-25)

This experiment used the earlier velocity gate and reset sampler. Its collection
rates do not describe the current 14 cm, geometry-based configuration.

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

## Geometry-based acceptance before 14 cm spacing (2026-09-25)

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
