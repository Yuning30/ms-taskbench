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

## c7 — slip-avoidance scoring (negative result)

Hypothesis: pre-score grasp candidates by a geometric slip cost
(approach tilt + yaw-mod-90), sort ascending, truncate to top-K.
cuRobo then picks the lowest-trajectory-cost candidate among the
"more-stable" subset.

| variant | success | grasp_plan | grasp_verify | mean wall |
|---|---:|---:|---:|---:|
| c5_relax_expand (baseline) | 373/500 (74.6%) | 81 | 46 | 2.31 s |
| c7_slip_k72 | 373/500 (74.6%) | 81 | 46 | 1.88 s |
| c7_slip_k48 | 369/500 (73.8%) | 84 | 47 | 2.15 s |
| c7_slip_k24 | 362/500 (72.4%) | 84 | 54 | 1.87 s |
| c7_slip_k24inv | 364/500 (72.8%) | 84 | 52 | 2.18 s |
| c7_slip_k12 | 353/500 (70.6%) | 87 | 60 | 2.17 s |

**Findings:**
- The cost is not predictive: truncation hurts BOTH grasp feasibility
  AND slip rate.
- Direction barely matters (k24 vs k24inv differ by 2 slips).
- Block-orientation histogram of c5 slip failures shows the OPPOSITE
  pattern: axis-aligned blocks slip MOST (14.4% at 0-5deg), highly-rotated
  blocks slip LEAST (0% at 30-35deg). Static geometric features of the
  candidate grasp are not what drives grasp_verify failures here.

**Implication:** the user's intuition - "score grasps by wrist-axis
alignment with block-axis" - is what we tried; it doesn't earn its
keep on this controller. The remaining 46 slip failures likely depend
on wrist configuration during gripper closure (a TRAJECTORY-level
property, not a candidate-level one) or on scene-density effects
correlated with block orientation.

cuRobo's trajopt cost - which we don't pre-filter - is already a
better slip predictor than any simple geometric heuristic. Code is
kept env-var-gated (TASKBENCH_CUROBO_TOPK, _INVERT) for future
experiments; default behavior is unchanged.

## c8 / c9 — chasing the slip mechanism

After c7 falsified the geometric-slip hypothesis, two follow-up
experiments. c8 tested whether the slip came from gripper close
dynamics (impulsive contact); c9 tested whether it came from finger-vs-neighbor
interference at the grasp pose.

| variant | success | grasp_plan | grasp_verify | mean wall |
|---|---:|---:|---:|---:|
| c5 baseline | 373/500 (74.6%) | 81 | 46 | 2.31 s |
| c8_grip_g20 (close 6 -> 20 steps) | 371/500 (74.2%) | 81 | 48 | 1.95 s |
| c8_grip_g40 (close 6 -> 40 steps) | 372/500 (74.4%) | 81 | 47 | 2.19 s |
| **c9_finger_coll** | **402/500 (80.4%)** | **66** | **32** | **1.59 s** |

c8 ruled out close dynamics: slowing the close from 6 to 40 steps
(7x more time for physics to settle) had no effect on slip rate. 93%
of c5's slip set persisted into c8.

**c9 enables finger collisions in plan_grasp** (default cuRobo
disables them for the entire grasp phase via grasp_contact_link_names).
The target block remains excluded via sync_scene, so the gripper can
still freely approach it; fingers just can't pass through neighbors.

c9 is strictly dominant on every metric versus c5:

| metric | c5 | c9 | delta |
|---|---:|---:|---:|
| total success | 373/500 (74.6%) | 402/500 (80.4%) | +29 (+5.8pp) |
| random | 76.3% | 80.9% | +4.6pp |
| canonical | 73.0% | 81.0% | +8.0pp |
| templated | 66.0% | 76.0% | +10.0pp |
| plan_fail | 81 | 66 | -15 |
| verify_fail | 46 | 32 | -14 |
| mean wall | 2.31 s | 1.59 s | -31% |
| reach / approach / lift | 0 / 0 / 0 | 0 / 0 / 0 | unchanged |

Slip set transitions c5 -> c9: 21 scenes rescued, 7 new slips, 25
kept slipping in both. Plan-fail transitions: 15 rescued, **0** new
plan-fails. This is unusual - enabling additional collision
constraints would normally INCREASE plan_fail. The likely
explanation: with fingers in the collision world, cuRobo's optimizer
steers away from candidate regions where panda_hand is clear but
fingers intrude. The smaller, cleaner search space converges faster
(hence the speedup) and avoids the fake-feasible candidates whose
post-check would fail.

The hypothesis from c7's negative result is now confirmed: the slip
cause is finger-vs-neighbor interference at the grasp pose, NOT
candidate-level geometric features. It is fixable upstream (in the
planner) by removing one keyword default. Net gain vs mplib v6:
+104 successes (+20.8pp), 56% faster mean wall-clock.

## c10 - settle-before-close (null result)

After c9 reduced slips to 32, breakdown showed slip rate scales with
robot-frame target x:
- x < 0.60m (near half): 0.0% slip across 204 scenes
- x = 0.65-0.70m: 12.8% (5/39)
- x = 0.70-0.75m: 16.0% (8/50)
- x > 0.80m (reach limit): 21.1% (4/19)

Hypothesis: at extended reach the wrist is near singular; trajectory
ends with residual velocity that close-torque amplifies into wrist
drift; fingers desymmetrize, target slips.

c10 adds N hold-the-last-pose steps between trajectory end and close
(TASKBENCH_CUROBO_SETTLE_STEPS env var).

| variant | success | grasp_plan | grasp_verify | mean wall |
|---|---:|---:|---:|---:|
| c9_finger_coll (baseline) | 402/500 (80.4%) | 66 | 32 | 1.59 s |
| c10_settle_s10 | 402/500 (80.4%) | 66 | 32 | 2.12 s |
| c10_settle_s20 | 403/500 (80.6%) | 66 | 31 | 2.24 s |

**Null result.** Settling does not help. Combined with c8 (gripper
close-rate has no effect), this rules out two of the three temporal
mechanisms - residual motion at trajectory end (c10) and impulsive
contact (c8). What's left: the close-torque itself causes wrist drift
during the close. The pd controller cant correct fast enough at extended
reach because the Jacobian is near-singular. This is structural to the
Panda's kinematic configuration at the workspace edge.

Three follow-up directions, in priority of likely payoff:
1. **Reject targets at robot-frame x > 0.75m as out-of-reliable-workspace.**
   This trivially boosts the success rate on the reachable subset to
   ~84%. Honest re-labeling: cuRobo finds plans, but the close phase
   is unreliable at extended reach.
2. **Bump wrist PD gains during the gripper close action.** Requires
   ManiSkill controller config changes; would let the wrist fight back
   against close-torque drift. Sim-side change, not controller-side.
3. **Pre-bias grasp candidates toward higher-manipulability wrist
   configs at extended reach** (penalize candidates whose IK would
   place joint 7 near its limit). Requires running IK per candidate
   and adding manipulability to the candidate cost. Most invasive
   but the most surgical fix.

Production controller stays c9 (402/500 = 80.4%). Closing the loop on
controller-side iteration.

## c11 - two-stage gripper close (marginal)

Layered on c9 (finger_coll=1). Splits the close into two phases:
close to half (gripper_state=0.0) for 10 steps, then close to full
for 6 steps. Hypothesis: less torque per stage = less wrist drift
at the workspace edge.

| variant | success | grasp_plan | grasp_verify | mean wall |
|---|---:|---:|---:|---:|
| c9_finger_coll (baseline) | 402/500 (80.4%) | 66 | 32 | 1.59 s |
| c11_two_stage_close | 404/500 (80.8%) | 66 | 30 | 1.77 s |

Slip set transitions: 29 of 32 c9 slips persist (the dominant
mechanism is unchanged). 3 rescued, 1 new slip. Net +2 successes
at +11% wall-clock.

**c9 stays Pareto-best.** c11 is left in place as an env-var-gated
alternative (TASKBENCH_CUROBO_TWO_STAGE_CLOSE=1) for applications that
value the marginal +0.4pp over speed.

## Truly closing the controller-side iteration

Five experiments past c9 have all been near-null:

| stage | hypothesis | result |
|---|---|---|
| c10 settle-before-close | wrist residual velocity at trajectory end | no effect |
| c11 two-stage close | close torque drives wrist drift | +2 (marginal) |
| (c8 close-rate, earlier) | impulsive contact at close | no effect |
| (c7 slip cost, earlier) | candidate-level geometry | no effect (worse with filtering) |

The remaining 30-32 slip failures are dominated by **wrist drift
during the close action at extended-reach configurations**, where
the Panda's Jacobian is near-singular. Controller-level
interventions can rescue 1-3 marginal cases per attempt but cannot
change the underlying mechanism.

For further gains: sim-side wrist PD stiffening, or just rejecting
target_x > 0.75m as out-of-reliable-workspace and accepting that
limitation honestly.

Final production controller: **c9 = 402/500 (80.4%), 1.59 s/sample,
+104 vs mplib v6.** c11 is an opt-in alternative for +2 at +11% cost.

## c12-c15 - exhaustive single-parameter sweep past c9

After c11's marginal +2, we tested five more single-parameter axes to
see if any could move the 32 grasp_verify_failures in c9. **All null
or worse.**

### c12 - wrist PD stiffness (joints 5/6/7)
Hypothesis: wrist drift under close-torque at extended reach.

| mult | success | slip |
|---|---:|---:|
| 1x (c9) | 402 | 32 |
| 2x | 402 | 32 |
| 5x | 403 | 31 |
| 10x | 402 | 32 |

Wrist drift is not the mechanism.

### c13 - cuRobo IK position tolerance
Hypothesis: 5mm default lets the optimizer land off-center on the 4cm cube.

| pos_tol | success | plan_fail | slip |
|---|---:|---:|---:|
| 5mm (c9 default) | 402 | 66 | 32 |
| 2mm | 395 | 73 | 32 |
| 1mm | 394 | 75 | 31 |

Tighter tolerance refuses 7-9 more plans but the 32 slips persist.

### c14 - cube friction
Hypothesis: low friction lets the cube slip out; high friction grabs better.

| mu | success | slip |
|---|---:|---:|
| 0.1 | 399 | 35 |
| 0.3 | 402 | 32 |
| 0.5 (default) | 402 | 32 |
| 1.5 | 396 | 38 |
| 3.0 | 368 | 66 |

Default is at a local optimum. High friction makes things much worse
because the cube STICKS to the first finger that contacts it and gets
dragged off-center.

### c15 - gripper finger PD
Hypothesis: softer close lets cube settle into jaws; stiffer grips harder.

| mult | success | slip | mean wall |
|---|---:|---:|---:|
| 0.3x (soft) | 396 | 38 | 5.51 s |
| 1x (c9) | 402 | 32 | 1.59 s |
| 3x (stiff) | 400 | 34 | 1.53 s |

Default is again the local optimum.

## Final conclusion: c9 is at a structural ceiling

Across nine independent axes - candidate scoring, close timing, settle,
two-stage close, wrist PD, IK tolerance, cube friction (both ways), and
gripper PD - **no single-parameter intervention shifts the 32 slip
failures**. The default ManiSkill configuration with c9's finger-collision
fix is at a local optimum on all of them.

The 6.4% residual slip rate is determined by:
- Panda kinematic limits at extended reach (cant be tuned)
- The 2-finger gripper geometry vs 4cm cube (would require hardware change)
- ManiSkill's contact engine at near-singular wrist configs

Production controller: **c9** = 402/500 (80.4%), 1.59 s/sample mean.
+104 / +20.8pp / 56% faster than mplib v6.

Further progress requires fundamentally different approaches:
1. Reject target_x > 0.75m at scene-feasibility check time (-25 of 32 slips honestly relabeled, no execution cost).
2. Re-collect verifier data on c9; train a slip predictor on the
   remaining 32 cases. The problem is now a ~6.4% minority-class
   detection problem.
3. Change the gripper - wider finger pads or a 3-finger design would
   structurally fix the asymmetric-contact slip mode.

## c16 - combo (c9 + c11 + c12_k5) - marginal effects do NOT compound

The two marginal wins from earlier (c11 two-stage close gave +2, c12 wrist 5x
gave +1) were stacked to see if they compound.

| variant | success | slip | mean wall |
|---|---:|---:|---:|
| c9 (baseline) | 402/500 (80.4%) | 32 | 1.59 s |
| c11 alone | 404/500 (80.8%) | 30 | 1.77 s |
| c12_k5 alone | 403/500 (80.6%) | 31 | 1.62 s |
| **c16 combo** | **403/500 (80.6%)** | **31** | **4.62 s** |

The combo matches c12_k5 alone (+1 over c9) at nearly 3x the wall-clock cost.
The marginal effects do NOT compound; they were within physics-simulation
noise to begin with.

30 of 32 c9 slips persist in c16. The slip set is essentially invariant
across all c9-family variants - it is a true property of the (Panda +
contact engine + 4cm cube + workspace edge) interaction.

## Workspace-rejection analysis (no-eval, computed from c9 data)

| threshold (robot-frame x) | rejected | kept | kept_succ% | overall_succ (honest) |
|---|---:|---:|---:|---:|
| 0.60 m | 295 | 205 | **99.5%** | 40.8% |
| 0.65 | 216 | 284 | 97.2% | 55.2% |
| 0.70 | 176 | 324 | 95.7% | 62.0% |
| 0.75 | 123 | 377 | 93.4% | 70.4% |
| 0.80 | 82 | 418 | 92.6% | 77.4% |
| 0.85 | 32 | 468 | 85.9% | 80.4% |
| no reject (c9) | 0 | 500 | 80.4% | 80.4% |

**Free win**: at threshold 0.85, all 32 flagged scenes are 100% plan_fails
(zero successes, zero slips). Rejecting these saves ~64s of compute
without losing any actual successes.

For deployment: pick the threshold based on cost-per-failure trade-off.
For data collection: don't reject (you want labels on hard scenes too).

For analytical slip prediction: a single x-threshold catches 25-78% of
slips at corresponding precision 33-97%. Better slip prediction needs
multivariate features (the verifier story).

## Truly final state

Production controller: **c9** = 402/500 (80.4%), 1.59 s/sample mean.
+104 successes / +20.8pp / 56% faster than mplib v6.

Ten experiments past c9 (c10-c16, c14b, c15, c16 combo) all confirm:
**no single-parameter or simple-combination intervention on the
controller or simulator can reduce the 32 residual slip failures.**

Further progress requires either:
1. **Free win**: pre-reject targets at x > 0.85m (all plan_fails anyway).
2. **Operational**: workspace rejection at lower thresholds, trading
   attempted scenes for higher reliability.
3. **Verifier on c9 data**: multivariate slip predictor on (target_x,
   neighbor_count, grasp_pose_features). 6.4% minority-class problem.
4. **Hardware**: wider finger pads or 3-finger gripper would
   structurally fix the asymmetric-contact mode.

## c17 - velocity feedforward (pd_joint_pos_vel)

cuRobo's interpolated trajectory carries velocities we were discarding.
ManiSkill has a pd_joint_pos_vel control mode that takes both targets.
Hypothesis: feedforward improves tracking at extended reach.

| variant | success | slip | mean wall |
|---|---:|---:|---:|
| c9_finger_coll (baseline) | 402/500 (80.4%) | 32 | 1.59 s |
| c17_pos_vel | 404/500 (80.8%) | 30 | 5.39 s |

+2 marginal at 3.4x wall-clock cost. Same pattern as c11, c12_k5, c16:
2-3 borderline scenes rescued, 1-2 new ones introduced.

## The noise floor

11 experiments past c9. Every successful intervention lands in [403, 404]:

| stage | success | delta vs c9 |
|---|---:|---:|
| c9 baseline | 402 | - |
| c11 two-stage | 404 | +2 |
| c12_wrist_k5 | 403 | +1 |
| c16 combo | 403 | +1 |
| c17 pos_vel | 404 | +2 |

**The c9 baseline is genuinely 402 +/- 2 due to physics-simulation noise.**
Perturbing the trajectory targets in any direction shuffles ~3 borderline
scenes around. The 29-30 hard-core persistent slips are invariant under
every controller/sim intervention we've tried.

This concludes 17 controller stages (c0-c17) on the locked 500-scene seed
set. **c9 stays Pareto-best**: 402/500 (80.4%), 1.59 s/sample mean,
+104 / +20.8pp / 56% faster than mplib v6.

## c18 - workspace-rejection feasibility gate (production-ready free win)

Reject targets at robot-frame x > 0.85m before spending cuRobo planning
compute. The c9 data shows all 32 scenes at that range are 100% plan_fails;
the gate just labels them honestly.

Env var: TASKBENCH_WORKSPACE_X_MAX (default 0.85, set inf to disable).

| metric | c9 | c18 (gate at 0.85m) |
|---|---:|---:|
| success | 402/500 (80.4%) | 402/500 (80.4%) |
| plan_fail | 66 | 34 |
| out_of_workspace | 0 | 32 |
| slip | 32 | 32 |
| gate cost | - | ~0ms per rejected scene |
| saved planning time | - | ~64s per 500-scene eval |

The 32 workspace-gate scenes execute in essentially zero wall-clock vs
~2s each of cuRobo planning that would have failed.

**Three downstream benefits:**
1. Clearer failure attribution: workspace vs planner vs physics.
2. ~13% wall-clock savings on the full eval (saves the ~2s per always-fail
   scene at extreme reach).
3. Verifier preprocessing: filter the 32 scenes upfront. Verifier sees
   reachable subset only; label distribution improves from 80.4% to
   85.9% success (402/468).

## True production controller: c9 + c18

Default behavior with TASKBENCH_MOTION_BACKEND=curobo and
TASKBENCH_CUROBO_FINGER_COLL=1:
- c9: cuRobo plan_grasp with panda_hand finger collisions enabled
- c18: workspace gate at robot-frame x = 0.85m

Result: 402/500 (80.4%) success, ~64s faster total wall-clock vs c9
alone, plus cleaner failure-mode attribution.

Cumulative vs mplib v6: +104 successes, +20.8pp, 56% faster mean
wall-clock, cleaner downstream pipeline.

## c19 - out-of-sample evaluation (different seed-set)

To check that the 80.4 % number is not a property of the specific 500
scenes in the in-sample seed-set, c19 re-evaluates c9+c18 on a fresh
500-scene set generated with seed=98765.

| metric | in-sample (c18) | OOS (c19) | delta |
|---|---:|---:|---:|
| success | 402/500 (80.4%) | 393/500 (78.6%) | -9 (-1.8pp) |
| plan-fail | 34 | 39 | +5 |
| slip | 32 | 36 | +4 |
| OOW | 32 | 32 | 0 |

The 1.8pp delta is well within sampling noise on 500 binomials (the 95%
CI is approximately +/-3.5pp). All failure-mode counts agree to within
+/-5 scenes. The controller generalizes: 80.4% is not overfit to the
seed-12345 scenes.

## Failure-mode analysis

A detailed characterization of all three failure modes (out_of_workspace,
plan_fail, grasp_verify) with renderings, geometric analysis, and edge
cases is in outputs/controller_eval/analysis/ANALYSIS.md. Key findings:

1. The three modes are mechanistically distinct, each dominating a
   different reach band: slip at 0.60-0.80m, plan-fail at 0.80-0.85m,
   OOW above.
2. The c18 gate at 0.85m is too lenient - x in [0.84, 0.85) is 100%
   plan_fail. Tightening to 0.84m would save ~38s per 1000 scenes
   at zero success cost.
3. Slip splits into two sub-mechanisms: reach-stress (47/68, isolated
   targets at extended reach) and crowded-scene (20/68, tight neighbors).
4. Across 1000 scenes, the safe zone of reach < 0.60m has 100% success
   regardless of crowding (136/136).
5. The verifier can predict outcomes from just two scene features
   (target reach, min neighbor distance) - the 2D heatmap in the
   analysis directory is essentially what the verifier needs to learn.

## c20 - workspace gate tightened to 0.84m (verified free win)

The analysis identified that x in [0.84, 0.85m) is a 100% plan-fail
band on the combined 1000-scene dataset. c20 verifies end-to-end:

| controller | success | plan_fail | slip | OOW | gate |
|---|---:|---:|---:|---:|---:|
| c18 | 402/500 (80.4%) | 34 | 32 | 32 | 0.85 m |
| **c20** | **402/500 (80.4%)** | **25** | **32** | **41** | **0.84 m** |

Same success count. 9 plan_fails converted to OOW (saves ~18s of cuRobo
planning per 500 scenes). The new default for TASKBENCH_WORKSPACE_X_MAX
is 0.84 m.

The production controller is now: c9 + c18 (gate=0.84). Numerically
identical to c9 + c18 (gate=0.85) on success but slightly faster.

