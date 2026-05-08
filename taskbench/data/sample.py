"""Schema for a single pick-feasibility data sample."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pyarrow as pa


@dataclass
class PickFeasibilitySample:
    """One labeled (scene, target) → success row."""

    scene_id: str
    seed: int
    source: str  # "random" | "canonical" | "templated"
    grid_rows: int
    grid_cols: int
    block_poses: np.ndarray  # (N_max, 7) [x, y, z, qw, qx, qy, qz]
    block_mask: np.ndarray   # (N_max,) bool — which slots are real blocks
    target_idx: int          # index into block_poses
    robot_qpos: np.ndarray   # (D,) joint positions at sample-call time
    success: bool
    failure_reason: Optional[str]
    failed_leg: Optional[str]
    final_object_pose: Optional[np.ndarray]  # (7,) or None
    wall_time_s: float

    def to_row(self) -> dict:
        row: dict = {
            "scene_id": self.scene_id,
            "seed": self.seed,
            "source": self.source,
            "grid_rows": self.grid_rows,
            "grid_cols": self.grid_cols,
            "target_idx": self.target_idx,
            "success": self.success,
            "failure_reason": self.failure_reason,
            "failed_leg": self.failed_leg,
            "wall_time_s": self.wall_time_s,
        }
        comps = ["x", "y", "z", "qw", "qx", "qy", "qz"]
        n = self.block_poses.shape[0]
        for i in range(n):
            present = bool(self.block_mask[i])
            for j, c in enumerate(comps):
                row[f"block_{i}_{c}"] = float(self.block_poses[i, j]) if present else None
            row[f"block_{i}_present"] = present
        if self.final_object_pose is not None:
            for j, c in enumerate(comps):
                row[f"final_{c}"] = float(self.final_object_pose[j])
        else:
            for c in comps:
                row[f"final_{c}"] = None
        for j, v in enumerate(self.robot_qpos):
            row[f"qpos_{j}"] = float(v)
        return row


def samples_to_table(samples: list[PickFeasibilitySample]) -> pa.Table:
    """Convert samples to a pyarrow Table for parquet writing."""
    rows = [s.to_row() for s in samples]
    return pa.Table.from_pylist(rows)
