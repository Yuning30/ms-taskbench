# Controller-improvement plan — final summary

Locked seed-set: 500 scenes (350 random / 100 canonical / 50 templated), seed=12345.
All numbers are on the **same** scene set so stages are directly comparable.
Each stage is also a single git commit with a determinism receipt that passes.

## Cumulative impact (v0 → v6)

| stage | success | random | canonical | templated | mean wall | p95 wall | notes |
|---|---:|---:|---:|---:|---:|---:|---|
| **v0_baseline** | 271 (54.2%) | 60.9% | 58.0% | 0.0% | 2.62 s | 14.27 s | baseline: TCP-y closing, robot_init_qpos_noise=0.02 |
| **v1_determinism** | 287 (57.4%) | 56.6% | 59.0% | 60.0% | 4.28 s | 25.32 s | + fixed [0,1,0] closing, qpos_noise=0 -> DETERMINISTIC |
| **v2_blockobs** | 283 (56.6%) | 56.3% | 56.0% | 60.0% | 3.34 s | 17.91 s | + free blocks registered as planner obstacles |
| **v3_yaw12** | 298 (59.6%) | 60.0% | 58.0% | 60.0% | 3.71 s | 19.76 s | + 12-angle yaw search (was 6) |
| **v4b_standoff** | 297 (59.4%) | 59.7% | 58.0% | 60.0% | 3.49 s | 17.16 s | + adaptive pre-grasp standoff |
| **v5_twostage_lift** | 297 (59.4%) | 60.0% | 57.0% | 60.0% | 3.44 s | 17.22 s | + two-stage lift with -x nudge |
| **v6_multiapproach** | 298 (59.6%) | 60.0% | 58.0% | 60.0% | 3.57 s | 18.02 s | + 3-approach grasp synthesis (=current) |

## Net change v0 → v6
- Total successes: **271 → 298** (+27, +5.4pp)
- Random: **60.9% → 60.0%**
- Canonical: **58.0% → 58.0%**
- Templated: **0.0% → 60.0%**
- Mean wall time: **2.62s → 3.57s**
- p95 wall time: **14.27s → 18.02s**

## Per-stage marginal contribution (deltas in successes)

| from → to | Δ success | source of the gain |
|---|---:|---|
| v0_baseline → v1_determinism | +16 | + fixed [0,1,0] closing, qpos_noise=0 -> DETERMINISTIC |
| v1_determinism → v2_blockobs | -4 | + free blocks registered as planner obstacles |
| v2_blockobs → v3_yaw12 | +15 | + 12-angle yaw search (was 6) |
| v3_yaw12 → v4b_standoff | -1 | + adaptive pre-grasp standoff |
| v4b_standoff → v5_twostage_lift | +0 | + two-stage lift with -x nudge |
| v5_twostage_lift → v6_multiapproach | +1 | + 3-approach grasp synthesis (=current) |

## Numbers to bring to the cuRobo comparison

- **Controller:** screw-line + 3-approach × 12-yaw search + obstacle-aware planner.
- **Determinism:** byte-identical outputs across two runs (`tests/skills/test_pick_determinism_smoke.py`).
- **Seed-set:** `outputs/controller_eval/baseline_scenes.parquet` (500 scenes).
- **Success rate:** 298/500 = 59.6% (per-source: random 60.0%, canonical 58.0%, templated 60.0%).
- **Wall-clock:** mean 3.57 s, p95 18.02 s, max 52.66 s per Pick.
- **Failure distribution:** grasp_approach 125, reach 25, grasp_search 21, grasp_verify 24, lift 7 (out of 202 failures).

## Files / artifacts

- Code: stages 1-6 each landed as a single commit; `git log --oneline 971f7ab^..HEAD` covers them.
- Per-stage results: `outputs/controller_eval/v0_baseline.parquet` ... `v6_multiapproach.parquet`.
- Per-row logs: `outputs/controller_eval/v*.log`.
- This summary: `outputs/controller_eval/SUMMARY.{csv,md}`.
- Determinism test: `tests/skills/test_pick_determinism_smoke.py` (pytest -m slow).
