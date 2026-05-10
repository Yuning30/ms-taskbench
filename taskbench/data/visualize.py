"""Reproduce dataset rows with video + snapshot recording for presentations.

Selects representative rows by category (easy/medium/hard success, plus per-leg
failures), reconstructs each scene from its parquet row, replays ctx.pick(...)
with RecordEpisode wrapping the env, and emits MP4 videos + PNG snapshots +
manifest.csv mapping each artifact back to the dataset row.

Run:
    uv run python -m taskbench.data.visualize \
      --dataset '/common/users/shared/pracsys/.../task_*/shard_*.parquet' \
      --out /common/users/shared/pracsys/.../visualizations \
      --n-per-cat 2
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import random
import signal
import threading
from pathlib import Path
from typing import Callable


class _PickTimeout(Exception):
    pass


def _alarm_supported() -> bool:
    return threading.current_thread() is threading.main_thread() and hasattr(signal, "SIGALRM")


def _alarm_handler(signum, frame):
    raise _PickTimeout()

import gymnasium as gym
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sapien
from PIL import Image

from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.wrappers import RecordEpisode

import taskbench.envs  # noqa: F401 — register env
import taskbench.envs.build2d as _build2d
from taskbench.envs import get_objects
from taskbench.skills.context import SkillContext
from taskbench.skills.motion import setup_planner


def _topdown_render_camera(self):
    """Override Build2DEnv's render camera to view the workspace from the side
    opposite the robot (the +X end of the table), looking back toward the robot.

    Default camera sits behind the robot, so the arm blocks the manipulation
    site during pick. Placing the camera on the far side of the grid keeps the
    target slots in the foreground, the source blocks visible, and the robot
    in the background — without flipping the world-up axis (so the image is
    not upside down).
    """
    # Eye at (0.5, 0, 0.5), looking at origin (the workspace center) with
    # world +Z as up. Forward = (-1, 0, -1)/sqrt(2); right = (0, 1, 0);
    # up = (-1, 0, 1)/sqrt(2). The rotation that maps default camera frame
    # (+X forward, +Y left, +Z up) onto these axes is a 180-deg rotation about
    # the unit axis (-sin(22.5deg), 0, cos(22.5deg)) ≈ (-0.3827, 0, 0.9239).
    pose = sapien.Pose(
        p=[0.5, 0.0, 0.5],
        q=[0.0, -0.3826834, 0.0, 0.9238795],  # [w, x, y, z]
    )
    return CameraConfig("render_camera", pose, 768, 768, 1.0, 0.01, 100)


_build2d.Build2DEnv._default_human_render_camera_configs = property(_topdown_render_camera)

logger = logging.getLogger("taskbench.data.visualize")


# Each predicate takes a row dict and returns True if the row fits the category.
CATEGORY_QUERIES: dict[str, Callable[[dict], bool]] = {
    "easy_success": lambda r: (
        r["source"] == "random"
        and r["success"]
        and r["wall_time_s"] < 2.0
    ),
    "medium_success": lambda r: (
        r["source"] == "canonical"
        and r["success"]
        and r["wall_time_s"] >= 1.0
    ),
    "hard_success": lambda r: (
        r["source"] == "templated"
        and r["success"]
    ),
    "fail_reach": lambda r: r["failed_leg"] == "reach",
    "fail_grasp_approach": lambda r: r["failed_leg"] == "grasp_approach",
    "fail_grasp_verify": lambda r: r["failed_leg"] == "grasp_verify",
    "fail_lift": lambda r: r["failed_leg"] == "lift",
    "fail_timeout": lambda r: r["failed_leg"] == "timeout",
}


def _reconstruct_block_poses(row: dict, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild (N, 7) poses + (N,) mask from flattened row columns."""
    poses = np.zeros((n, 7), dtype=np.float64)
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        present = bool(row.get(f"block_{i}_present", False))
        if present:
            poses[i] = [
                row[f"block_{i}_x"], row[f"block_{i}_y"], row[f"block_{i}_z"],
                row[f"block_{i}_qw"], row[f"block_{i}_qx"],
                row[f"block_{i}_qy"], row[f"block_{i}_qz"],
            ]
            mask[i] = True
    return poses, mask


def _select_rows(dataset_glob: str, n_per_cat: int, seed: int) -> dict[str, list[dict]]:
    """Read shards once and pick n_per_cat representative rows per category."""
    files = sorted(glob.glob(dataset_glob))
    if not files:
        raise FileNotFoundError(f"no shards matched: {dataset_glob}")
    logger.info("reading %d shards ...", len(files))
    table = pa.concat_tables([pq.read_table(f) for f in files])
    logger.info("loaded %d rows; selecting per category ...", table.num_rows)

    rng = random.Random(seed)
    selected: dict[str, list[dict]] = {}
    for cat, predicate in CATEGORY_QUERIES.items():
        # Convert column-by-column to a streamable iterator without dumping all
        # 500K dicts into memory at once.
        matches: list[dict] = []
        for batch in table.to_batches(max_chunksize=10000):
            for row in batch.to_pylist():
                if predicate(row):
                    matches.append(row)
                    if len(matches) >= 200:  # cap; we only need a few samples
                        break
            if len(matches) >= 200:
                break
        if matches:
            picks = rng.sample(matches, min(n_per_cat, len(matches)))
            selected[cat] = picks
            logger.info("  %s: %d candidates -> picked %d", cat, len(matches), len(picks))
        else:
            selected[cat] = []
            logger.info("  %s: 0 candidates", cat)
    return selected


def _save_snapshot(env, path: Path) -> bool:
    """Render one RGB frame and save to PNG. Returns True on success."""
    try:
        frame = env.render()
    except Exception as exc:
        logger.warning("env.render() failed: %s", exc)
        return False
    try:
        if hasattr(frame, "cpu"):
            frame = frame.cpu().numpy()
        frame = np.asarray(frame)
    except Exception:
        return False
    if frame.ndim == 4:  # batch dim
        frame = frame[0]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    Image.fromarray(frame).save(path)
    return True


def _render_one(out_dir: Path, category: str, row: dict, grid_rows: int, grid_cols: int,
                pick_timeout_s: float = 60.0):
    """Reproduce one row and save snapshot + video. Returns manifest dict."""
    cat_dir = out_dir / category
    cat_dir.mkdir(parents=True, exist_ok=True)
    scene_id = row["scene_id"]
    n = grid_rows * grid_cols

    poses, mask = _reconstruct_block_poses(row, n)
    overrides = poses.astype(np.float32, copy=True)
    for i in range(n):
        if not bool(mask[i]):
            overrides[i, :3] = (0.0, 0.0, -1.0)
            overrides[i, 3:] = (1.0, 0.0, 0.0, 0.0)

    env = gym.make(
        "Build2D-v1",
        num_envs=1,
        sim_backend="cpu",
        render_backend="cpu",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        grid_rows=grid_rows,
        grid_cols=grid_cols,
    )
    env = RecordEpisode(
        env,
        output_dir=str(cat_dir),
        save_trajectory=False,
        save_video=True,
        save_on_reset=False,
        record_reward=False,
        video_fps=30,
    )

    snapshot_path = cat_dir / f"snapshot_{scene_id}.png"
    video_basename = f"video_{scene_id}"

    try:
        env.reset(seed=row["seed"], options={"block_overrides": overrides})
        snapshot_ok = _save_snapshot(env, snapshot_path)

        # Build skill context manually (avoid runner's double-reset).
        ctx = SkillContext(env)
        ctx.planner = setup_planner(env, ctx.robot_config)
        ctx.objects = get_objects(env)
        ctx._build_skills()

        target_name = f"block_{int(row['target_idx'])}"
        prev_handler = None
        timer_set = False
        if pick_timeout_s and pick_timeout_s > 0 and _alarm_supported():
            prev_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.setitimer(signal.ITIMER_REAL, float(pick_timeout_s))
            timer_set = True
        try:
            ctx.pick(target_name, lift_height=0.12)
        except _PickTimeout:
            logger.info("ctx.pick hit %.1fs timeout (expected for fail_timeout)", pick_timeout_s)
        except Exception as exc:
            logger.info("ctx.pick raised (expected for some failure cases): %s", exc)
        finally:
            if timer_set:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, prev_handler)

        if hasattr(env, "flush_video"):
            try:
                env.flush_video(name=video_basename)
            except Exception as exc:
                logger.warning("flush_video failed: %s", exc)
    finally:
        try:
            env.close()
        except Exception:
            pass

    video_path = cat_dir / f"{video_basename}.mp4"
    return {
        "category": category,
        "scene_id": scene_id,
        "source": row["source"],
        "seed": row["seed"],
        "target_idx": row["target_idx"],
        "success": row["success"],
        "failure_reason": row.get("failure_reason"),
        "failed_leg": row.get("failed_leg"),
        "wall_time_s": round(float(row["wall_time_s"]), 3),
        "snapshot": str(snapshot_path) if snapshot_ok else "",
        "video": str(video_path) if video_path.exists() else "",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True, help="Glob for parquet shards")
    p.add_argument("--out", type=Path, required=True, help="Output dir")
    p.add_argument("--n-per-cat", type=int, default=2, help="Samples per category")
    p.add_argument("--grid-rows", type=int, default=3)
    p.add_argument("--grid-cols", type=int, default=3)
    p.add_argument("--seed", type=int, default=0, help="Selection RNG seed")
    p.add_argument("--pick-timeout-s", type=float, default=60.0,
                   help="Per-pick wall-clock budget; mirrors runner default.")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.out.mkdir(parents=True, exist_ok=True)

    selected = _select_rows(args.dataset, args.n_per_cat, args.seed)

    manifest = []
    for category, rows in selected.items():
        for row in rows:
            logger.info("rendering %s / %s ...", category, row["scene_id"])
            try:
                m = _render_one(args.out, category, row, args.grid_rows, args.grid_cols,
                                pick_timeout_s=args.pick_timeout_s)
                manifest.append(m)
            except Exception as exc:
                logger.exception("  render FAILED: %s", exc)

    if manifest:
        manifest_path = args.out / "manifest.csv"
        keys = list(manifest[0].keys())
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(manifest)
        logger.info("manifest -> %s", manifest_path)

    logger.info("done: %d artifacts under %s", len(manifest), args.out)


if __name__ == "__main__":
    main()
