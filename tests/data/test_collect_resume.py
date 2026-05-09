"""Unit tests for collect.py resume / shard-integrity logic."""

from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq
import pytest

from taskbench.data.collect import _existing_shards
from taskbench.data.sample import PickFeasibilitySample, samples_to_table


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sample(scene_id: str) -> PickFeasibilitySample:
    return PickFeasibilitySample(
        scene_id=scene_id,
        seed=0,
        source="random",
        grid_rows=1,
        grid_cols=1,
        block_poses=np.array([[0.0, 0.0, 0.02, 1.0, 0.0, 0.0, 0.0]]),
        block_mask=np.array([True]),
        target_idx=0,
        robot_qpos=np.zeros(9),
        success=True,
        failure_reason=None,
        failed_leg=None,
        final_object_pose=np.array([0.0, 0.0, 0.12, 1.0, 0.0, 0.0, 0.0]),
        wall_time_s=1.0,
    )


# ---------------------------------------------------------------------------
# Fix 2: corrupt shard resilience
# ---------------------------------------------------------------------------

def test_existing_shards_skips_corrupt_file(tmp_path):
    """_existing_shards should delete corrupt shards, count only valid rows."""
    # Write a valid shard with 5 rows.
    valid_samples = [_make_sample(f"s{i:08d}") for i in range(5)]
    valid_shard = tmp_path / "shard_00000.parquet"
    pq.write_table(samples_to_table(valid_samples), valid_shard)

    # Write a corrupt shard.
    corrupt_shard = tmp_path / "shard_00001.parquet"
    corrupt_shard.write_bytes(b"not a parquet file at all")

    next_idx, n = _existing_shards(tmp_path)

    # Only valid shard (index 0) survived → next_idx = 0+1 = 1, n = 5.
    assert (next_idx, n) == (1, 5)
    # Corrupt file must have been deleted.
    assert not corrupt_shard.exists()


def test_existing_shards_empty_dir(tmp_path):
    assert _existing_shards(tmp_path) == (0, 0)


def test_existing_shards_all_valid(tmp_path):
    """Two valid shards: 3 + 4 rows; next_idx = 2, n = 7."""
    for shard_i, count in enumerate([3, 4]):
        samples = [_make_sample(f"s{j:08d}") for j in range(count)]
        pq.write_table(samples_to_table(samples), tmp_path / f"shard_{shard_i:05d}.parquet")

    next_idx, n = _existing_shards(tmp_path)
    assert next_idx == 2
    assert n == 7


def test_existing_shards_all_corrupt(tmp_path):
    """If all shards are corrupt they are removed; returns (0, 0)."""
    for i in range(3):
        (tmp_path / f"shard_{i:05d}.parquet").write_bytes(b"garbage")

    next_idx, n = _existing_shards(tmp_path)
    assert (next_idx, n) == (0, 0)
    remaining = list(tmp_path.glob("shard_*.parquet"))
    assert remaining == []
