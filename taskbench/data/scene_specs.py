"""Scene specifications and generators for pick-feasibility data collection.

A SceneSpec captures everything needed to reset Build2DEnv into a specific
configuration: grid size, block poses, and (optionally) a target index.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Workspace bounds on the table (x, y) in meters. Conservative box that's
# inside the Panda's reachable area.
WORKSPACE_X = (-0.26, 0.26)
WORKSPACE_Y = (-0.26, 0.26)

# Block half-extent matches Build2DEnv (`half_size=0.02`).
BLOCK_HALF = 0.02
BLOCK_TOP_Z = 0.02

# Minimum xy separation between blocks (2 * half + slack).
MIN_SEP = 0.041


@dataclass
class SceneSpec:
    """A reset-able scene description.

    block_poses: (N, 7) array of [x, y, z, qw, qx, qy, qz].
    block_mask:  (N,) bool. Real blocks where True; padded slots otherwise.
    target_idx:  optional index into block_poses for the pick target.
    source:      provenance tag — "random" / "canonical" / "templated".
    """
    seed: int
    grid_rows: int
    grid_cols: int
    block_poses: np.ndarray
    block_mask: np.ndarray
    target_idx: Optional[int] = None
    source: str = "random"
    metadata: dict = field(default_factory=dict)


def _sample_non_overlapping_xy(
    rng: np.random.Generator,
    n: int,
    *,
    bounds_x=WORKSPACE_X,
    bounds_y=WORKSPACE_Y,
    min_sep=MIN_SEP,
    max_tries: int = 500,
) -> np.ndarray:
    """Rejection-sample n non-overlapping (x, y) points in the workspace."""
    pts = np.empty((n, 2), dtype=np.float64)
    placed = 0
    for _ in range(max_tries * n):
        if placed >= n:
            break
        x = rng.uniform(*bounds_x)
        y = rng.uniform(*bounds_y)
        if placed == 0:
            pts[placed] = (x, y)
            placed += 1
            continue
        d = np.linalg.norm(pts[:placed] - np.array([x, y]), axis=1)
        if d.min() >= min_sep:
            pts[placed] = (x, y)
            placed += 1
    if placed < n:
        raise RuntimeError(f"Could not place {n} non-overlapping blocks; got {placed}.")
    return pts


def _random_yaw_quat(rng: np.random.Generator) -> np.ndarray:
    """Random yaw rotation as a SAPIEN [w, x, y, z] quaternion."""
    yaw = rng.uniform(-np.pi, np.pi)
    half = yaw / 2.0
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


def random_scene_spec(
    *,
    seed: int,
    grid_rows: int,
    grid_cols: int,
) -> SceneSpec:
    """Stratified-random scene: N = grid_rows*grid_cols blocks scattered on table.

    All blocks are on the table (z = BLOCK_TOP_Z) with random xy + yaw,
    no inter-block overlap.
    """
    rng = np.random.default_rng(seed)
    n = grid_rows * grid_cols

    xy = _sample_non_overlapping_xy(rng, n)
    poses = np.zeros((n, 7), dtype=np.float64)
    poses[:, :2] = xy
    poses[:, 2] = BLOCK_TOP_Z
    for i in range(n):
        poses[i, 3:] = _random_yaw_quat(rng)
    mask = np.ones(n, dtype=bool)

    target_idx = int(rng.integers(0, n))
    return SceneSpec(
        seed=seed,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        block_poses=poses,
        block_mask=mask,
        target_idx=target_idx,
        source="random",
    )
