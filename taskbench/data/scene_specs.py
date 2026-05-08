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


def canonical_program_snapshot_specs(
    *,
    seed: int,
    grid_rows: int,
    grid_cols: int,
    grid_origin_x: float = 0.05,
    grid_spacing: float = 0.07,
) -> list[SceneSpec]:
    """Return one SceneSpec per pick step of the canonical row-major program.

    At step k:
        - the first k blocks are at their grid target xy (placed)
        - the remaining n-k blocks are at their initial randomized positions
        - target_idx points to the block being picked at step k (the (k+1)-th)
    """
    rng = np.random.default_rng(seed)
    n = grid_rows * grid_cols

    # Initial randomized table positions for all blocks.
    init_xy = _sample_non_overlapping_xy(rng, n)
    init_quats = np.stack([_random_yaw_quat(rng) for _ in range(n)])

    # Grid target xy in row-major order, matching Build2DEnv layout.
    grid_origin_y = -((grid_cols - 1) * grid_spacing) / 2.0
    targets_xy = []
    for i in range(grid_rows):
        for j in range(grid_cols):
            targets_xy.append((grid_origin_x + i * grid_spacing,
                               grid_origin_y + j * grid_spacing))
    targets_xy = np.array(targets_xy, dtype=np.float64)

    specs: list[SceneSpec] = []
    for k in range(n):
        poses = np.zeros((n, 7), dtype=np.float64)
        poses[:, 2] = BLOCK_TOP_Z
        # First k blocks: placed at grid targets, identity orientation.
        for placed_idx in range(k):
            poses[placed_idx, :2] = targets_xy[placed_idx]
            poses[placed_idx, 3] = 1.0  # qw
        # Remaining blocks: at initial randomized positions, random yaw.
        for j in range(k, n):
            poses[j, :2] = init_xy[j]
            poses[j, 3:] = init_quats[j]
        mask = np.ones(n, dtype=bool)
        specs.append(SceneSpec(
            seed=seed,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            block_poses=poses,
            block_mask=mask,
            target_idx=k,
            source="canonical",
            metadata={"step": k},
        ))
    return specs


def _spec_from_xy(
    *, seed, grid_rows, grid_cols, xys, target_idx, archetype, sweep_step
) -> SceneSpec:
    """Create a SceneSpec from xy positions, padding to grid size."""
    n_total = grid_rows * grid_cols
    # Silently truncate xys if more objects than grid slots.
    xys = xys[:n_total]
    n_real = len(xys)
    poses = np.zeros((n_total, 7), dtype=np.float64)
    mask = np.zeros(n_total, dtype=bool)
    for i, (x, y) in enumerate(xys):
        poses[i, :2] = (x, y)
        poses[i, 2] = BLOCK_TOP_Z
        poses[i, 3] = 1.0
        mask[i] = True
    return SceneSpec(
        seed=seed,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        block_poses=poses,
        block_mask=mask,
        target_idx=target_idx,
        source="templated",
        metadata={"archetype": archetype, "sweep_step": sweep_step,
                  "n_real": n_real},
    )


def templated_scene_specs(
    *,
    seed: int,
    grid_rows: int,
    grid_cols: int,
) -> list[SceneSpec]:
    """Hand-designed scene archetypes with parametric sweeps.

    Each archetype tests a specific feasibility boundary. Returned specs share
    the SceneSpec contract (padded to grid_rows*grid_cols block slots).
    """
    rng = np.random.default_rng(seed)
    specs: list[SceneSpec] = []

    # 1. adjacent_obstacle: target at origin, obstacle along +x at varying d.
    for k, d in enumerate(np.linspace(0.045, 0.12, num=12)):
        xys = [(0.0, 0.0), (float(d), 0.0)]
        specs.append(_spec_from_xy(
            seed=seed, grid_rows=grid_rows, grid_cols=grid_cols,
            xys=xys, target_idx=0, archetype="adjacent_obstacle", sweep_step=k,
        ))

    # 2. edge_target: target near +x workspace edge at varying d.
    for k, x in enumerate(np.linspace(0.18, 0.255, num=10)):
        xys = [(float(x), 0.0)]
        specs.append(_spec_from_xy(
            seed=seed, grid_rows=grid_rows, grid_cols=grid_cols,
            xys=xys, target_idx=0, archetype="edge_target", sweep_step=k,
        ))

    # 3. ringed_target: target at origin, 4 obstacles in a ring at radius r.
    for k, r in enumerate(np.linspace(0.045, 0.10, num=10)):
        ring = [(float(r) * np.cos(a), float(r) * np.sin(a))
                for a in np.linspace(0, 2 * np.pi, num=4, endpoint=False)]
        xys = [(0.0, 0.0), *ring]
        specs.append(_spec_from_xy(
            seed=seed, grid_rows=grid_rows, grid_cols=grid_cols,
            xys=xys, target_idx=0, archetype="ringed_target", sweep_step=k,
        ))

    # Touch rng so determinism path is exercised (future archetypes may use it).
    _ = rng.random()
    return specs
