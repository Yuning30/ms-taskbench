import numpy as np

from taskbench.data.sample import PickFeasibilitySample, samples_to_table


def _make_sample(**overrides):
    base = dict(
        scene_id="scene-0001",
        seed=42,
        source="random",
        grid_rows=3,
        grid_cols=3,
        block_poses=np.tile(np.array([0.0, 0.0, 0.02, 1, 0, 0, 0]), (9, 1)),
        block_mask=np.ones(9, dtype=bool),
        target_idx=2,
        robot_qpos=np.zeros(9),
        success=True,
        failure_reason=None,
        failed_leg=None,
        final_object_pose=np.array([0.0, 0.0, 0.12, 1, 0, 0, 0]),
        wall_time_s=4.2,
    )
    base.update(overrides)
    return PickFeasibilitySample(**base)


def test_to_row_keys_are_flat():
    s = _make_sample()
    row = s.to_row()
    # Per-block columns are flattened with index suffix.
    assert "block_0_x" in row and "block_0_qw" in row
    assert "block_8_y" in row
    assert row["target_idx"] == 2
    assert row["success"] is True


def test_to_row_handles_failure():
    s = _make_sample(success=False, failure_reason="grasp_plan_failed",
                     failed_leg="grasp_search", final_object_pose=None)
    row = s.to_row()
    assert row["success"] is False
    assert row["failure_reason"] == "grasp_plan_failed"
    assert row["final_x"] is None  # serialised as null


def test_samples_to_table_roundtrips_via_parquet(tmp_path):
    samples = [_make_sample(scene_id=f"scene-{i:04d}") for i in range(3)]
    table = samples_to_table(samples)
    assert table.num_rows == 3
    out = tmp_path / "shard.parquet"
    import pyarrow.parquet as pq
    pq.write_table(table, out)
    back = pq.read_table(out)
    assert back.num_rows == 3
    assert back.column("scene_id").to_pylist() == ["scene-0000", "scene-0001", "scene-0002"]
