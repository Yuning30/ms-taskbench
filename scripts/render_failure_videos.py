"""Render MP4 videos of c9+c18 failure modes for the analysis report.

Picks representative scenes from c18_workspace_gate.parquet + c19_oos_seed98765
for each failure mode, replays them with cuRobo + RecordEpisode, and saves
MP4s + still snapshots to outputs/controller_eval/analysis/videos/.

Categories (one scene each, ~6 videos total):
  reach_stress_slip_easy   : isolated target, x ~ 0.65m (start of slip zone)
  reach_stress_slip_hard   : isolated target, x ~ 0.78m (heart of slip zone)
  crowded_slip             : target with neighbor within 5cm
  plan_fail_canonical      : canonical mid-build, no feasible plan
  success_far_clean        : reach > 0.70m, isolated, succeeded (contrast)
  success_crowded          : crowded scene that succeeded (contrast)

Run via SLURM (needs CUDA + SAPIEN renderer):
  sbatch slurm_logs/<ts>_render_videos.sh
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pyarrow.parquet as pq
import sapien
from PIL import Image

from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.wrappers import RecordEpisode

# Ensure cuRobo path is enabled before importing taskbench modules that read
# the env var at import time (e.g. for wrist-PD patches).
os.environ.setdefault("TASKBENCH_MOTION_BACKEND", "curobo")
os.environ.setdefault("TASKBENCH_CUROBO_FINGER_COLL", "1")
os.environ.setdefault("TASKBENCH_WORKSPACE_X_MAX", "0.85")

import taskbench.envs  # noqa: F401 — registers Build2D-v1
import taskbench.envs.build2d as _build2d
from taskbench.envs import get_objects
from taskbench.skills.context import SkillContext
from taskbench.skills.curobo_planner import CuroboPlanner
from taskbench.skills.motion import setup_planner


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("render")


# Override the env camera to a useful angled view (same as visualize.py).
def _angled_render_camera(self):
    pose = sapien.Pose(
        p=[0.5, 0.0, 0.5],
        q=[0.0, -0.3826834, 0.0, 0.9238795],
    )
    return CameraConfig("render_camera", pose, 768, 768, 1.0, 0.01, 100)

_build2d.Build2DEnv._default_human_render_camera_configs = property(_angled_render_camera)


ROBOT_BASE_X = -0.615
PARQUETS = [
    "/common/home/st1122/Projects/ms-taskbench/outputs/controller_eval/c18_workspace_gate.parquet",
    "/common/home/st1122/Projects/ms-taskbench/outputs/controller_eval/c19_oos_seed98765.parquet",
]
OUT_DIR = Path("/common/home/st1122/Projects/ms-taskbench/outputs/controller_eval/analysis/videos")


def _load_all_rows():
    rows = []
    for p in PARQUETS:
        t = pq.read_table(p)
        for i in range(len(t)):
            r = {col: t.column(col)[i].as_py() for col in t.column_names}
            r["_src"] = Path(p).stem
            rows.append(r)
    return rows


def _enrich(r):
    """Add tx_robot and min_nbr_d to a row."""
    tgt = r["target_idx"]
    r["tx_robot"] = r[f"block_{tgt}_x"] - ROBOT_BASE_X
    tx, ty = r[f"block_{tgt}_x"], r[f"block_{tgt}_y"]
    min_d = float("inf")
    for j in range(9):
        if j == tgt:
            continue
        if not r[f"block_{j}_present"]:
            continue
        if r[f"block_{j}_z"] < 0:
            continue
        ox, oy = r[f"block_{j}_x"], r[f"block_{j}_y"]
        d = ((ox - tx) ** 2 + (oy - ty) ** 2) ** 0.5
        if d < min_d:
            min_d = d
    r["min_nbr_d"] = min_d
    return r


def pick_scenes(rows):
    """Return one row per category by priority-order matching."""
    rows = [_enrich(r) for r in rows]

    def select(predicate, sort_key=None):
        matches = [r for r in rows if predicate(r)]
        if not matches:
            return None
        if sort_key:
            matches.sort(key=sort_key)
        return matches[0]

    categories = {
        "reach_stress_slip_easy": select(
            lambda r: (r["failure_reason"] == "grasp_verification_failed"
                       and r["tx_robot"] < 0.72 and r["min_nbr_d"] > 0.10),
            sort_key=lambda r: r["tx_robot"],
        ),
        "reach_stress_slip_hard": select(
            lambda r: (r["failure_reason"] == "grasp_verification_failed"
                       and r["tx_robot"] > 0.74 and r["min_nbr_d"] > 0.10),
            sort_key=lambda r: -r["tx_robot"],
        ),
        "crowded_slip": select(
            lambda r: (r["failure_reason"] == "grasp_verification_failed"
                       and r["min_nbr_d"] < 0.06),
            sort_key=lambda r: r["min_nbr_d"],
        ),
        "plan_fail_canonical": select(
            lambda r: (r["failure_reason"] == "grasp_plan_failed"
                       and r["source"] == "canonical"),
            sort_key=lambda r: -r["min_nbr_d"] if np.isfinite(r["min_nbr_d"]) else 0,
        ),
        "success_far_clean": select(
            lambda r: (r["success"] and r["tx_robot"] > 0.70
                       and r["min_nbr_d"] > 0.10),
            sort_key=lambda r: -r["tx_robot"],
        ),
        "success_crowded": select(
            lambda r: r["success"] and r["min_nbr_d"] < 0.06,
            sort_key=lambda r: r["min_nbr_d"],
        ),
    }
    return categories


def _save_snapshot(env, path: Path) -> bool:
    try:
        frame = env.render()
        if hasattr(frame, "cpu"):
            frame = frame.cpu().numpy()
        frame = np.asarray(frame)
        if frame.ndim == 4:
            frame = frame[0]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        Image.fromarray(frame).save(path)
        return True
    except Exception as exc:
        log.warning("snapshot failed for %s: %s", path.name, exc)
        return False


def reconstruct_overrides(row):
    n = 9  # 3x3
    poses = np.zeros((n, 7), dtype=np.float64)
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        if row[f"block_{i}_present"]:
            poses[i] = [
                row[f"block_{i}_x"], row[f"block_{i}_y"], row[f"block_{i}_z"],
                row[f"block_{i}_qw"], row[f"block_{i}_qx"],
                row[f"block_{i}_qy"], row[f"block_{i}_qz"],
            ]
            mask[i] = True
    overrides = poses.astype(np.float32, copy=True)
    for i in range(n):
        if not mask[i]:
            overrides[i, :3] = (0.0, 0.0, -1.0)
            overrides[i, 3:] = (1.0, 0.0, 0.0, 0.0)
    return overrides


def render_one(category: str, row: dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cat_dir = OUT_DIR / category
    cat_dir.mkdir(exist_ok=True)
    scene_id = row["scene_id"]

    overrides = reconstruct_overrides(row)
    env = gym.make(
        "Build2D-v1",
        num_envs=1,
        sim_backend="cpu",
        render_backend="gpu",  # need vulkan for render frames
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        grid_rows=3,
        grid_cols=3,
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

    snapshot_path = cat_dir / f"{scene_id}_snapshot.png"

    env.reset(seed=row["seed"], options={"block_overrides": overrides})
    _save_snapshot(env, snapshot_path)

    # Manually build SkillContext (we already reset). Mirror what reset() does.
    ctx = SkillContext(env)
    ctx.planner = setup_planner(env, ctx.robot_config)
    # cuRobo path is gated by env var; build planner directly.
    ctx.curobo_planner = CuroboPlanner(env)
    ctx.objects = get_objects(env)
    ctx._build_skills()

    target_name = f"block_{int(row['target_idx'])}"
    try:
        result = ctx.pick(target_name, lift_height=0.12)
        log.info("%s [%s]: success=%s reason=%s",
                 category, scene_id, result.success, result.failure_reason)
    except Exception as exc:
        log.warning("%s [%s] raised: %s", category, scene_id, exc)
        result = None

    env.flush_video(name=scene_id, save=True)
    env.close()


def main():
    rows = _load_all_rows()
    log.info("loaded %d scenes total from %d files", len(rows), len(PARQUETS))
    cats = pick_scenes(rows)
    for cat, row in cats.items():
        if row is None:
            log.warning("no scene picked for %s", cat)
            continue
        log.info("[%s] picked %s (src=%s, tx_robot=%.3f, min_nbr_d=%.3f, "
                 "success=%s, reason=%s)",
                 cat, row["scene_id"], row["_src"], row["tx_robot"],
                 row["min_nbr_d"], row["success"], row["failure_reason"])
        render_one(cat, row)


if __name__ == "__main__":
    main()
