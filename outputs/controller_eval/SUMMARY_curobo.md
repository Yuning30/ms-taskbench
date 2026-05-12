# cuRobo controller integration — final summary

Same locked 500-scene seed-set as the mplib stages (`outputs/controller_eval/baseline_scenes.parquet`).
All numbers are deterministic (cuRobo `reset_seed()` is called before each plan).

## Headline

| stage | success | random | canonical | templated | mean wall | p95 wall |
|---|---:|---:|---:|---:|---:|---:|
| **v0_baseline** | 271 (54.2%) | 60.9% | 58.0% | 0.0% | 2.62 s | 14.67 s |
| **v6_multiapproach** | 298 (59.6%) | 60.0% | 58.0% | 60.0% | 3.57 s | 18.24 s |
| **c2_curobo_motion** | 338 (67.6%) | 68.6% | 62.0% | 72.0% | 3.66 s | 5.63 s |
| **c3_batched_grasp** | 328 (65.6%) | 70.9% | 68.0% | 24.0% | 2.75 s | 4.41 s |
| **c4_plan_grasp** | 367 (73.4%) | 74.6% | 73.0% | 66.0% | 2.33 s | 3.67 s |
| **c5_relax_expand** | 373 (74.6%) | 76.3% | 73.0% | 66.0% | 2.31 s | 3.63 s |
| **c6_more_seeds** | 373 (74.6%) | 76.6% | 72.0% | 66.0% | 2.33 s | 3.63 s |

## Net change v6 → c5 / c6 (final)
- Total successes: **298 → 373** (+75, +15.0pp)
- Random: **60.0% → 76.3%**
- Canonical: **58.0% → 73.0%**
- Templated: **60.0% → 66.0%**
- Mean wall time: **3.57 s → 2.31 s** (-35%)
- p95 wall time: **18.24 s → 3.63 s** (-80%)

## Per-stage marginal contribution (cuRobo path)

| from → to | Δ success | source of the gain / loss |
|---|---:|---|
| v6_multiapproach → c2_curobo_motion | +40 | c2: route motion through cuRobo (keep yaw search) |
| c2_curobo_motion → c3_batched_grasp | -10 | c3: batched 36-pose goal-set grasp synthesis |
| c3_batched_grasp → c4_plan_grasp | +39 | c4: full three-phase plan_grasp |
| c4_plan_grasp → c5_relax_expand | +6 | c5: relax collision margin + 24 yaws (72 candidates) |
| c5_relax_expand → c6_more_seeds | +0 | c6: double IK/trajopt seeds (no-op) |

## Failure-leg distribution

| stage | grasp_search | reach | grasp_approach | grasp_verify | lift | total fails |
|---|---:|---:|---:|---:|---:|---:|
| v0_baseline | 0 | 38 | 175 | 15 | 1 | 229 |
| v6_multiapproach | 21 | 25 | 125 | 24 | 7 | 202 |
| c2_curobo_motion | 71 | 22 | 62 | 7 | 0 | 162 |
| c3_batched_grasp | 73 | 9 | 83 | 7 | 0 | 172 |
| c4_plan_grasp | 84 | 0 | 0 | 49 | 0 | 133 |
| c5_relax_expand | 81 | 0 | 0 | 46 | 0 | 127 |
| c6_more_seeds | 81 | 0 | 0 | 46 | 0 | 127 |

## Headline numbers vs mplib v6

- **Best controller:** c5 (= c6) — 373/500 = 74.6% success at 2.31 s/sample.
- **+75 successes vs v6** (mplib screw-line, deterministic).
- **35% faster mean wall-clock** than v6.

Failure modes for c5/c6:
- 81 grasp_plan_failed: no candidate of the 72 was kinematically + collision-free reachable.
- 46 grasp_verification_failed: gripper closed but block slipped (physics, not motion planning).
- 0 reach, grasp_approach, or lift failures: when plan_grasp returns a trajectory, execution always succeeds.

## Determinism

- cuRobo's trajopt uses a stateful Halton sample buffer. `CuroboPlanner.reset_seed()` is called before every plan, yielding byte-identical trajectories across runs.
- Smoke + determinism test: `tests/skills/test_curobo_smoke.py::test_plan_pose_deterministic_with_reset_seed`.
- End-to-end test: `tests/skills/test_curobo_planner.py`.

## Files / artifacts

- Code: stages c0–c6 each landed as a single commit; `git log --oneline f9d14ac^..HEAD` covers them.
- Per-stage results: `outputs/controller_eval/c[0-6]_*.parquet`.
- This summary: `outputs/controller_eval/SUMMARY_curobo.{csv,md}`.
- cuRobo install: `/common/home/st1122/Projects/third_party/curobo` (editable). Use the `[curobo]` extra: `uv pip install -e "taskbench[curobo]"`.
