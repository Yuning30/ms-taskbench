"""Benchmark cold per-sample pick-feasibility timing.

Run: uv run python -m taskbench.data.benchmark --n 20 --grid-rows 3 --grid-cols 3
"""

from __future__ import annotations

import argparse
import logging
import statistics
import time

import gymnasium as gym

import taskbench.envs  # noqa: F401 — register env
from taskbench.data.runner import run_pick_sample
from taskbench.data.scene_specs import random_scene_spec
from taskbench.skills.context import SkillContext

logger = logging.getLogger("taskbench.data.benchmark")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20, help="Number of samples to time.")
    p.add_argument("--grid-rows", type=int, default=3)
    p.add_argument("--grid-cols", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    print(f"Building env (grid {args.grid_rows}x{args.grid_cols}) ...")
    env_t0 = time.perf_counter()
    env = gym.make(
        "Build2D-v1",
        num_envs=1,
        sim_backend="cpu",
        control_mode="pd_joint_pos",
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
    )
    ctx = SkillContext(env)
    env_setup_s = time.perf_counter() - env_t0
    print(f"Env setup: {env_setup_s:.1f} s")

    times: list[float] = []
    failures = 0
    for i in range(args.n):
        spec = random_scene_spec(
            seed=args.seed + i,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
        )
        sample = run_pick_sample(env, ctx, spec, scene_id=f"bench-{i:04d}")
        times.append(sample.wall_time_s)
        if not sample.success:
            failures += 1
        print(f"  [{i+1:>3}/{args.n}] success={sample.success} "
              f"reason={sample.failure_reason} time={sample.wall_time_s:.2f}s")

    env.close()

    times.sort()
    mean = statistics.mean(times)
    median = statistics.median(times)
    p95 = times[int(0.95 * len(times)) - 1] if len(times) >= 20 else times[-1]
    print()
    print(f"=== Benchmark (n={args.n}) ===")
    print(f"  env_setup_s: {env_setup_s:.1f}")
    print(f"  mean:   {mean:.2f} s")
    print(f"  median: {median:.2f} s")
    print(f"  p95:    {p95:.2f} s")
    print(f"  min:    {min(times):.2f} s")
    print(f"  max:    {max(times):.2f} s")
    print(f"  failure_rate: {failures}/{args.n} ({100.0*failures/args.n:.1f}%)")
    print()
    proj_500k_single = mean * 500_000 / 3600
    proj_500k_100x = mean * 500_000 / (3600 * 100)
    print(f"  Projected 500K single-process: {proj_500k_single:.0f} h")
    print(f"  Projected 500K with 100 tasks: {proj_500k_100x:.0f} h")


if __name__ == "__main__":
    main()
