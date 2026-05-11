"""Pick-feasibility verifier dataset.

Builds canonical train/val/eval splits from the collected parquet shards and
exposes a torch Dataset that returns 90-dim feature vectors.

Feature layout (concatenation):
    block_poses:    9 * 7 = 63   [x,y,z,qw,qx,qy,qz per block, NaN -> 0]
    block_mask:     9            [1 if present else 0]
    target_onehot:  9            [1 at target_idx, else 0]
    robot_qpos:     9            [Panda 7 arm + 2 gripper joints]
                    --
                    90
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


# Assumed grid for v1; if other sizes appear in the data they're rejected so we
# can revisit the fixed feature dim.
_N_BLOCKS = 9
_QPOS_DIM = 9
FEATURE_DIM = _N_BLOCKS * 7 + _N_BLOCKS + _N_BLOCKS + _QPOS_DIM  # 90


_POSE_COMPS = ["x", "y", "z", "qw", "qx", "qy", "qz"]


def _load_all_shards(source_dir: Path) -> pa.Table:
    """Read every shard_*.parquet under source_dir/task_*/ into a single Table."""
    shards = sorted(source_dir.glob("task_*/shard_*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No shards found under {source_dir}/task_*/")
    return pa.concat_tables([pq.read_table(s) for s in shards])


def _stratified_split(
    table: pa.Table,
    *,
    train_n: int,
    val_n: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Stratified random split by (source, success).

    Within each (source, success) bucket, samples are randomly partitioned into
    train/val/eval at proportions train_n/val_n/rest of the bucket. Globally
    the totals add up close to train_n / val_n / (total - train_n - val_n);
    rounding differences land in eval.
    """
    n = table.num_rows
    sources = np.asarray(table.column("source").to_pylist())
    successes = np.asarray(table.column("success").to_pylist(), dtype=bool)
    rng = np.random.default_rng(seed)

    p_train = train_n / n
    p_val = val_n / n

    assignment = np.empty(n, dtype=object)
    for src in np.unique(sources):
        for ok in (True, False):
            idxs = np.where((sources == src) & (successes == ok))[0]
            rng.shuffle(idxs)
            k_train = int(round(len(idxs) * p_train))
            k_val = int(round(len(idxs) * p_val))
            assignment[idxs[:k_train]] = "train"
            assignment[idxs[k_train : k_train + k_val]] = "val"
            assignment[idxs[k_train + k_val :]] = "eval"

    return {
        "train": np.where(assignment == "train")[0],
        "val": np.where(assignment == "val")[0],
        "eval": np.where(assignment == "eval")[0],
    }


def build_splits(
    *,
    source_dir: Path,
    out_dir: Path,
    train_n: int = 50_000,
    val_n: int = 10_000,
    seed: int = 0,
    rebuild: bool = False,
) -> dict[str, Path]:
    """Materialize canonical train/val/eval parquet splits under out_dir.

    Returns mapping from split name to parquet path. No-op if all three exist
    and rebuild=False.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: out_dir / f"{name}.parquet" for name in ("train", "val", "eval")}
    if not rebuild and all(p.exists() for p in paths.values()):
        return paths

    table = _load_all_shards(source_dir)
    splits = _stratified_split(table, train_n=train_n, val_n=val_n, seed=seed)
    for name, idxs in splits.items():
        pq.write_table(table.take(pa.array(idxs)), paths[name])

    meta = {
        "source_dir": str(source_dir),
        "total_rows": table.num_rows,
        "seed": seed,
        "train_n_target": train_n,
        "val_n_target": val_n,
        "counts": {name: int(len(idx)) for name, idx in splits.items()},
        "feature_dim": FEATURE_DIM,
        "n_blocks": _N_BLOCKS,
        "qpos_dim": _QPOS_DIM,
    }
    # Per-split source × success breakdown for sanity.
    sources = np.asarray(table.column("source").to_pylist())
    successes = np.asarray(table.column("success").to_pylist(), dtype=bool)
    for name, idxs in splits.items():
        meta.setdefault("breakdown", {})[name] = {}
        for src in np.unique(sources):
            sub = idxs[sources[idxs] == src]
            meta["breakdown"][name][src] = {
                "total": int(len(sub)),
                "success": int(successes[sub].sum()),
            }
    (out_dir / "split_meta.json").write_text(json.dumps(meta, indent=2))
    return paths


def build_calibration_split(
    *,
    eval_path: Path,
    out_dir: Path,
    calib_n: int = 10_000,
    seed: int = 42,
    rebuild: bool = False,
) -> dict[str, Path]:
    """Carve a stratified-by-source calibration slice off the eval split.

    Reads eval.parquet, samples calib_n rows stratified by source, writes:
        out_dir/calib.parquet         (calib_n rows)
        out_dir/eval_holdout.parquet  (rest)
    Leaves eval.parquet untouched.
    """
    calib_p = out_dir / "calib.parquet"
    holdout_p = out_dir / "eval_holdout.parquet"
    if not rebuild and calib_p.exists() and holdout_p.exists():
        return {"calib": calib_p, "eval_holdout": holdout_p}

    table = pq.read_table(eval_path)
    n = table.num_rows
    sources = np.asarray(table.column("source").to_pylist())
    rng = np.random.default_rng(seed)

    p_calib = calib_n / n
    calib_idx_parts = []
    for src in np.unique(sources):
        idxs = np.where(sources == src)[0]
        rng.shuffle(idxs)
        k = int(round(len(idxs) * p_calib))
        calib_idx_parts.append(idxs[:k])
    calib_idx = np.sort(np.concatenate(calib_idx_parts))
    holdout_idx = np.setdiff1d(np.arange(n), calib_idx, assume_unique=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.take(pa.array(calib_idx)), calib_p)
    pq.write_table(table.take(pa.array(holdout_idx)), holdout_p)
    return {"calib": calib_p, "eval_holdout": holdout_p}


@dataclass
class _Features:
    X: torch.Tensor  # (N, FEATURE_DIM) float32
    y: torch.Tensor  # (N,) float32 in {0, 1}
    source: list[str]
    scene_id: list[str]


def _extract_features(table: pa.Table) -> _Features:
    """Build the 90-dim feature tensor from a parquet table."""
    n = table.num_rows
    if n == 0:
        return _Features(
            X=torch.zeros((0, FEATURE_DIM), dtype=torch.float32),
            y=torch.zeros((0,), dtype=torch.float32),
            source=[],
            scene_id=[],
        )

    grid_rows = np.asarray(table.column("grid_rows").to_pylist())
    grid_cols = np.asarray(table.column("grid_cols").to_pylist())
    if not (np.all(grid_rows == 3) and np.all(grid_cols == 3)):
        raise ValueError(
            "Verifier v1 assumes a 3x3 grid throughout; got mixed grid sizes."
        )

    X = np.zeros((n, FEATURE_DIM), dtype=np.float32)

    # Block poses (63) + mask (9)
    pose_offset = 0
    mask_offset = _N_BLOCKS * 7
    for i in range(_N_BLOCKS):
        present = np.asarray(table.column(f"block_{i}_present").to_pylist(), dtype=bool)
        X[:, mask_offset + i] = present.astype(np.float32)
        for j, c in enumerate(_POSE_COMPS):
            col = table.column(f"block_{i}_{c}").to_numpy(zero_copy_only=False)
            col = np.where(np.equal(col, None), 0.0, col).astype(np.float32)
            col = np.nan_to_num(col, nan=0.0, posinf=0.0, neginf=0.0)
            X[:, pose_offset + i * 7 + j] = col

    # Target one-hot (9)
    target_offset = _N_BLOCKS * 7 + _N_BLOCKS
    target_idx = np.asarray(table.column("target_idx").to_pylist(), dtype=np.int64)
    X[np.arange(n), target_offset + target_idx] = 1.0

    # qpos (9)
    qpos_offset = target_offset + _N_BLOCKS
    for j in range(_QPOS_DIM):
        col = table.column(f"qpos_{j}").to_numpy(zero_copy_only=False).astype(np.float32)
        X[:, qpos_offset + j] = col

    y = np.asarray(table.column("success").to_pylist(), dtype=np.float32)

    return _Features(
        X=torch.from_numpy(X),
        y=torch.from_numpy(y),
        source=table.column("source").to_pylist(),
        scene_id=table.column("scene_id").to_pylist(),
    )


class PickFeasibilityDataset(Dataset):
    """Loads a single split parquet and exposes (X, y) tensors.

    Features are pre-extracted into memory at construction time. For 50K
    samples × 90 floats this is ~18 MB; fits easily.
    """

    def __init__(self, split_path: Path):
        table = pq.read_table(split_path)
        feats = _extract_features(table)
        self.X = feats.X
        self.y = feats.y
        self.source = feats.source
        self.scene_id = feats.scene_id

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[i], self.y[i]
