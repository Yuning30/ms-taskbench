"""Unit tests for the verifier dataset module.

Tests use synthetic parquet shards so they don't need the real 500K dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from taskbench.verifier.dataset import (
    FEATURE_DIM,
    PickFeasibilityDataset,
    _extract_features,
    _stratified_split,
    build_splits,
)


def _make_row(
    scene_id: str,
    seed: int,
    source: str,
    target_idx: int,
    success: bool,
    block_xy: list[tuple[float, float]] | None = None,
    qpos: list[float] | None = None,
):
    """Construct a single parquet-compatible dict matching the dataset schema."""
    if block_xy is None:
        block_xy = [(0.01 * i, 0.02 * i) for i in range(9)]
    if qpos is None:
        qpos = [0.1 * i for i in range(9)]
    row = {
        "scene_id": scene_id,
        "seed": seed,
        "source": source,
        "grid_rows": 3,
        "grid_cols": 3,
        "target_idx": target_idx,
        "success": success,
        "failure_reason": None if success else "grasp_plan_failed",
        "failed_leg": None if success else "grasp_search",
        "wall_time_s": 1.23,
    }
    for i, (x, y) in enumerate(block_xy):
        row[f"block_{i}_x"] = x
        row[f"block_{i}_y"] = y
        row[f"block_{i}_z"] = 0.02
        row[f"block_{i}_qw"] = 1.0
        row[f"block_{i}_qx"] = 0.0
        row[f"block_{i}_qy"] = 0.0
        row[f"block_{i}_qz"] = 0.0
        row[f"block_{i}_present"] = True
    for c in ["x", "y", "z", "qw", "qx", "qy", "qz"]:
        row[f"final_{c}"] = None
    for j, q in enumerate(qpos):
        row[f"qpos_{j}"] = q
    return row


def _write_shard(rows: list[dict], path: Path) -> None:
    table = pa.Table.from_pylist(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def test_feature_dim_constant():
    assert FEATURE_DIM == 90


def test_extract_features_shape_and_target_onehot():
    rows = [_make_row(f"s{i:04d}", seed=i, source="random", target_idx=i % 9,
                       success=(i % 2 == 0)) for i in range(5)]
    table = pa.Table.from_pylist(rows)
    feats = _extract_features(table)
    assert feats.X.shape == (5, FEATURE_DIM)
    assert feats.y.shape == (5,)
    # Each row: exactly one 1.0 in the target-onehot block.
    target_block = feats.X[:, 63 + 9 : 63 + 9 + 9]
    assert (target_block.sum(dim=1) == 1.0).all()
    for i in range(5):
        assert target_block[i, i % 9].item() == 1.0


def test_extract_features_mask_and_pose_fill():
    """Blocks with present=False should have mask=0 and pose values zeroed."""
    row = _make_row("s0", seed=0, source="templated", target_idx=0, success=True)
    # Mark block_3 as absent and null its pose columns.
    row["block_3_present"] = False
    for c in ["x", "y", "z", "qw", "qx", "qy", "qz"]:
        row[f"block_3_{c}"] = None
    table = pa.Table.from_pylist([row])
    feats = _extract_features(table)
    # Mask offset is 63 (after the 63 pose floats).
    assert feats.X[0, 63 + 3].item() == 0.0  # block_3 mask off
    # Block 3's 7 pose floats live at 3*7=21 .. 28.
    assert feats.X[0, 21:28].abs().sum().item() == 0.0


def test_extract_features_qpos_tail():
    row = _make_row("s0", seed=0, source="random", target_idx=0, success=True,
                    qpos=[0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3])
    feats = _extract_features(pa.Table.from_pylist([row]))
    # qpos is the last 9 features.
    qpos_block = feats.X[0, FEATURE_DIM - 9 :].tolist()
    assert qpos_block == pytest.approx([0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3])


def test_stratified_split_counts_balanced():
    n = 1000
    sources = np.array(["random"] * 700 + ["canonical"] * 200 + ["templated"] * 100)
    successes = np.array([i % 2 == 0 for i in range(n)])
    rows = [
        _make_row(f"s{i:04d}", seed=i, source=sources[i], target_idx=i % 9,
                  success=bool(successes[i]))
        for i in range(n)
    ]
    table = pa.Table.from_pylist(rows)
    splits = _stratified_split(table, train_n=100, val_n=50, seed=0)
    assert abs(len(splits["train"]) - 100) <= 5
    assert abs(len(splits["val"]) - 50) <= 5
    assert len(splits["train"]) + len(splits["val"]) + len(splits["eval"]) == n
    # No overlap.
    all_idx = np.concatenate(list(splits.values()))
    assert len(np.unique(all_idx)) == n


def test_build_splits_creates_three_parquets(tmp_path: Path):
    rows = [_make_row(f"s{i:04d}", seed=i, source="random", target_idx=i % 9,
                       success=(i % 2 == 0)) for i in range(200)]
    _write_shard(rows, tmp_path / "src" / "task_0000" / "shard_00000.parquet")
    paths = build_splits(source_dir=tmp_path / "src", out_dir=tmp_path / "exp",
                          train_n=50, val_n=20, seed=0)
    for name in ("train", "val", "eval"):
        assert paths[name].exists()
    assert (tmp_path / "exp" / "split_meta.json").exists()


def test_pick_feasibility_dataset_roundtrip(tmp_path: Path):
    rows = [_make_row(f"s{i:04d}", seed=i, source="random", target_idx=i % 9,
                       success=(i % 2 == 0)) for i in range(40)]
    _write_shard(rows, tmp_path / "src" / "task_0000" / "shard_00000.parquet")
    paths = build_splits(source_dir=tmp_path / "src", out_dir=tmp_path / "exp",
                          train_n=20, val_n=10, seed=0)
    ds = PickFeasibilityDataset(paths["train"])
    assert len(ds) > 0
    x, y = ds[0]
    assert x.shape == (FEATURE_DIM,)
    assert y.ndim == 0
    assert y.item() in (0.0, 1.0)
