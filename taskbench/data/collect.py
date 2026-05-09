"""Pick-feasibility data collection CLI.

Mixes 70% stratified-random / 20% canonical-program-snapshot / 10% templated
hard-case scenes, executes Pick for real on each, writes parquet shards.

Run: uv run python -m taskbench.data.collect --n 1000 --out outputs/pick_feasibility/run0
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import random
import subprocess
import time
from pathlib import Path
from typing import Iterator

import gymnasium as gym
import numpy as np
import pyarrow.parquet as pq

import taskbench.envs  # noqa: F401
from taskbench.data.runner import run_pick_sample
from taskbench.data.sample import samples_to_table
from taskbench.data.scene_specs import (
    SceneSpec,
    canonical_program_snapshot_specs,
    random_scene_spec,
    templated_scene_specs,
)
from taskbench.skills.context import SkillContext

logger = logging.getLogger("taskbench.data.collect")

MIX = {"random": 0.70, "canonical": 0.20, "templated": 0.10}


def _yield_specs(
    *,
    seed: int,
    grid_rows: int,
    grid_cols: int,
    n_target: int,
) -> Iterator[SceneSpec]:
    """Yield exactly n_target SceneSpecs drawn from the 70/20/10 mix.

    Within each batch we precompute templated + a chunk of canonical specs
    once and then top up with random specs for diversity.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # Pre-generate enough canonical + templated specs to satisfy the mix.
    n_canon_target = int(n_target * MIX["canonical"])
    n_templ_target = int(n_target * MIX["templated"])

    canonical_pool: list[SceneSpec] = []
    while len(canonical_pool) < n_canon_target:
        s = int(np_rng.integers(0, 2**31 - 1))
        canonical_pool.extend(
            canonical_program_snapshot_specs(seed=s, grid_rows=grid_rows, grid_cols=grid_cols)
        )
    rng.shuffle(canonical_pool)
    canonical_pool = canonical_pool[:n_canon_target]

    templated_pool: list[SceneSpec] = []
    while len(templated_pool) < n_templ_target:
        s = int(np_rng.integers(0, 2**31 - 1))
        templated_pool.extend(
            templated_scene_specs(seed=s, grid_rows=grid_rows, grid_cols=grid_cols)
        )
    rng.shuffle(templated_pool)
    templated_pool = templated_pool[:n_templ_target]

    queue = []
    queue.extend(canonical_pool)
    queue.extend(templated_pool)
    rng.shuffle(queue)

    n_random = n_target - len(queue)
    for i in range(n_random):
        s = int(np_rng.integers(0, 2**31 - 1))
        queue.append(random_scene_spec(seed=s, grid_rows=grid_rows, grid_cols=grid_cols))
    rng.shuffle(queue)

    yield from queue


def _existing_shards(out_dir: Path) -> tuple[int, int]:
    """Return (next_shard_idx, n_samples_already_written).

    Corrupt/truncated shards are logged, deleted, and skipped so a
    SLURM preemption that leaves a half-written file does not abort a resume.
    """
    shards = sorted(out_dir.glob("shard_*.parquet"))
    valid_shards = []
    n = 0
    for s in shards:
        try:
            meta = pq.read_metadata(s)
            n += meta.num_rows
            valid_shards.append(s)
        except Exception as exc:
            logger.warning("Corrupt shard %s removed: %s", s.name, exc)
            try:
                s.unlink()
            except OSError as oex:
                logger.warning("  also failed to unlink: %s", oex)
    if not valid_shards:
        return 0, n
    last = int(valid_shards[-1].stem.split("_")[1])
    return last + 1, n


def _build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the collect CLI."""
    p = argparse.ArgumentParser(
        description="Pick-feasibility data collection."
    )
    p.add_argument("--n", type=int, required=True, help="Total samples to collect.")
    p.add_argument("--grid-rows", type=int, default=3)
    p.add_argument("--grid-cols", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True, help="Output dir for shards.")
    p.add_argument("--shard-size", type=int, default=500,
                   help="Samples per parquet shard.")
    p.add_argument("--resume", action="store_true",
                   help="Skip already-written samples (counted via shards).")
    p.add_argument("--task-id", type=int, default=None,
                   help="Optional integer task id; when set, scene_ids are prefixed t{task_id:04d}_.")
    return p


def _build_run_meta(args) -> dict:
    """Return the run_meta dict for args, including git hash and start time."""
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        git_hash = "unknown"

    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")

    return {
        "n_target": args.n,
        "grid_rows": args.grid_rows,
        "grid_cols": args.grid_cols,
        "seed": args.seed,
        "shard_size": args.shard_size,
        "mix": MIX,
        "task_id": args.task_id,
        "git_hash": git_hash,
        "started_at": started_at,
    }


def main():
    args = _build_parser().parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    next_idx, written = _existing_shards(args.out) if args.resume else (0, 0)
    if args.resume:
        logger.info("Resuming: %d samples already written, next shard idx %d", written, next_idx)

    remaining = max(0, args.n - written)
    if remaining == 0:
        logger.info("Nothing to do.")
        return

    # Save run metadata once per run.
    meta_path = args.out / "run_meta.json"
    if not meta_path.exists():
        meta_path.write_text(json.dumps(_build_run_meta(args), indent=2))

    env = gym.make(
        "Build2D-v1",
        num_envs=1,
        sim_backend="cpu",
        render_backend="cpu",
        control_mode="pd_joint_pos",
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
    )
    ctx = SkillContext(env)

    spec_iter = _yield_specs(
        seed=args.seed + written,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        n_target=remaining,
    )

    buf = []
    shard_idx = next_idx
    t_start = time.perf_counter()
    for i, spec in enumerate(spec_iter, start=written):
        scene_id = (
            f"t{args.task_id:04d}_s{i:08d}"
            if args.task_id is not None
            else f"s{i:08d}"
        )
        try:
            sample = run_pick_sample(env, ctx, spec, scene_id=scene_id)
        except Exception as exc:
            logger.exception("Sample %s failed hard, skipping: %s", scene_id, exc)
            continue
        buf.append(sample)

        if len(buf) >= args.shard_size:
            shard_path = args.out / f"shard_{shard_idx:05d}.parquet"
            pq.write_table(samples_to_table(buf), shard_path)
            elapsed = time.perf_counter() - t_start
            done = i + 1
            rate = done / elapsed if elapsed > 0 else 0
            logger.info("Wrote %s (%d rows). %d/%d done. rate=%.2f/s",
                        shard_path.name, len(buf), done, args.n, rate)
            buf = []
            shard_idx += 1

    if buf:
        shard_path = args.out / f"shard_{shard_idx:05d}.parquet"
        pq.write_table(samples_to_table(buf), shard_path)
        logger.info("Wrote final %s (%d rows).", shard_path.name, len(buf))

    env.close()


if __name__ == "__main__":
    main()
