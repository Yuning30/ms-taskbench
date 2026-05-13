"""Evaluate a Pick controller version on a fixed locked seed-set of scenes.

The first time it runs, a 500-scene seed-set (350 random / 100 canonical /
50 templated) is built and persisted at --scenes. All subsequent controller
versions are evaluated on the *same* scenes so comparisons are apples-to-apples.

Usage:
    uv run python scripts/eval_controller.py \\
        --scenes outputs/controller_eval/baseline_scenes.parquet \\
        --out    outputs/controller_eval/v0_baseline.parquet \\
        --version v0_baseline
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import taskbench.envs  # noqa: F401 — env registration
from taskbench.data.runner import run_pick_sample
from taskbench.data.sample import samples_to_table
from taskbench.data.scene_specs import (
    SceneSpec,
    canonical_program_snapshot_specs,
    random_scene_spec,
    templated_scene_specs,
)
from taskbench.skills.context import SkillContext


logger = logging.getLogger("eval_controller")

_POSE_COMPS = ["x", "y", "z", "qw", "qx", "qy", "qz"]


# ---------------------------------------------------------------------------
# Seed-set construction (deterministic, replayable from a single seed)
# ---------------------------------------------------------------------------

def build_baseline_seed_set(
    *,
    seed: int = 12345,
    n_random: int = 350,
    n_canonical: int = 100,
    n_templated: int = 50,
    grid_rows: int = 3,
    grid_cols: int = 3,
) -> list[SceneSpec]:
    rng = np.random.default_rng(seed)
    specs: list[SceneSpec] = []

    for _ in range(n_random):
        s = int(rng.integers(0, 2**31 - 1))
        specs.append(random_scene_spec(seed=s, grid_rows=grid_rows, grid_cols=grid_cols))

    while sum(1 for x in specs if x.source == "canonical") < n_canonical:
        s = int(rng.integers(0, 2**31 - 1))
        snaps = canonical_program_snapshot_specs(seed=s, grid_rows=grid_rows, grid_cols=grid_cols)
        specs.append(snaps[int(rng.integers(0, len(snaps)))])

    while sum(1 for x in specs if x.source == "templated") < n_templated:
        s = int(rng.integers(0, 2**31 - 1))
        tpl = templated_scene_specs(seed=s, grid_rows=grid_rows, grid_cols=grid_cols)
        specs.append(tpl[int(rng.integers(0, len(tpl)))])

    return specs


def specs_to_table(specs: list[SceneSpec]) -> pa.Table:
    rows = []
    for i, spec in enumerate(specs):
        n = spec.block_poses.shape[0]
        row = {
            "spec_id": f"spec_{i:05d}",
            "source": spec.source,
            "seed": int(spec.seed),
            "grid_rows": int(spec.grid_rows),
            "grid_cols": int(spec.grid_cols),
            "target_idx": int(spec.target_idx),
        }
        for j in range(n):
            present = bool(spec.block_mask[j])
            row[f"block_{j}_present"] = present
            for k, c in enumerate(_POSE_COMPS):
                row[f"block_{j}_{c}"] = float(spec.block_poses[j, k]) if present else None
        rows.append(row)
    return pa.Table.from_pylist(rows)


def table_to_specs(table: pa.Table) -> list[SceneSpec]:
    specs: list[SceneSpec] = []
    for r in table.to_pylist():
        n = int(r["grid_rows"]) * int(r["grid_cols"])
        poses = np.zeros((n, 7), dtype=np.float64)
        mask = np.zeros(n, dtype=bool)
        for j in range(n):
            present = bool(r[f"block_{j}_present"])
            mask[j] = present
            if present:
                for k, c in enumerate(_POSE_COMPS):
                    poses[j, k] = float(r[f"block_{j}_{c}"])
        specs.append(SceneSpec(
            seed=int(r["seed"]),
            grid_rows=int(r["grid_rows"]),
            grid_cols=int(r["grid_cols"]),
            block_poses=poses,
            block_mask=mask,
            target_idx=int(r["target_idx"]),
            source=str(r["source"]),
        ))
    return specs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", type=Path, required=True,
                   help="Parquet with locked seed-set scenes. Generated if missing.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output parquet for per-sample controller results.")
    p.add_argument("--version", type=str, required=True,
                   help="Controller version tag (used in scene_id and log lines).")
    p.add_argument("--grid-rows", type=int, default=3)
    p.add_argument("--grid-cols", type=int, default=3)
    p.add_argument("--seed-set-seed", type=int, default=12345)
    p.add_argument("--limit", type=int, default=None,
                   help="If set, only evaluate the first N scenes (smoke test).")
    return p


def main():
    args = _build_parser().parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stdout)
    args.scenes.parent.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.scenes.exists():
        logger.info("Loading existing seed-set: %s", args.scenes)
        specs = table_to_specs(pq.read_table(args.scenes))
    else:
        logger.info("Building new seed-set (seed=%d) -> %s", args.seed_set_seed, args.scenes)
        specs = build_baseline_seed_set(
            seed=args.seed_set_seed,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
        )
        pq.write_table(specs_to_table(specs), args.scenes)
        logger.info("  saved %d scenes (random=%d  canonical=%d  templated=%d)",
                    len(specs),
                    sum(1 for s in specs if s.source == "random"),
                    sum(1 for s in specs if s.source == "canonical"),
                    sum(1 for s in specs if s.source == "templated"))

    if args.limit is not None:
        specs = specs[: args.limit]
        logger.info("--limit set: evaluating first %d scenes only", len(specs))

    import os as _os
    control_mode = _os.environ.get("TASKBENCH_CONTROL_MODE", "pd_joint_pos")
    env = gym.make(
        "Build2D-v1",
        num_envs=1,
        sim_backend="cpu",
        render_backend="cpu",
        control_mode=control_mode,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
    )
    logger.info("Control mode: %s", control_mode)
    ctx = SkillContext(env)

    t_start = time.perf_counter()
    samples = []
    n_succ = 0
    for i, spec in enumerate(specs):
        sample = run_pick_sample(env, ctx, spec,
                                  scene_id=f"{args.version}_{i:05d}")
        samples.append(sample)
        n_succ += int(sample.success)
        if (i + 1) % 25 == 0 or (i + 1) == len(specs):
            elapsed = time.perf_counter() - t_start
            logger.info("  %3d/%d  succ=%d (%.1f%%)  elapsed=%.0fs  avg=%.2fs/sample",
                        i + 1, len(specs), n_succ, 100 * n_succ / (i + 1),
                        elapsed, elapsed / (i + 1))
    env.close()

    pq.write_table(samples_to_table(samples), args.out)
    elapsed = time.perf_counter() - t_start
    logger.info("Wrote %s (%d samples, succ=%d/%d, %.1fs total)",
                args.out, len(samples), n_succ, len(samples), elapsed)


if __name__ == "__main__":
    main()
