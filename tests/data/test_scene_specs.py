import numpy as np

from taskbench.data.scene_specs import (
    SceneSpec,
    random_scene_spec,
)


def test_random_scene_is_deterministic_given_seed():
    a = random_scene_spec(seed=42, grid_rows=3, grid_cols=3)
    b = random_scene_spec(seed=42, grid_rows=3, grid_cols=3)
    assert np.allclose(a.block_poses, b.block_poses)
    assert a.block_mask.tolist() == b.block_mask.tolist()


def test_random_scene_no_overlap():
    spec = random_scene_spec(seed=7, grid_rows=3, grid_cols=4)
    poses = spec.block_poses[spec.block_mask]
    # 2*half_size = 0.04; require at least 0.041 separation in xy
    xy = poses[:, :2]
    dists = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    assert dists.min() >= 0.041, f"min sep {dists.min()}"


def test_random_scene_within_workspace():
    spec = random_scene_spec(seed=1, grid_rows=2, grid_cols=2)
    poses = spec.block_poses[spec.block_mask]
    assert ((poses[:, 0] >= -0.30) & (poses[:, 0] <= 0.30)).all()
    assert ((poses[:, 1] >= -0.30) & (poses[:, 1] <= 0.30)).all()


def test_random_scene_block_count_matches_grid_when_unspecified():
    spec = random_scene_spec(seed=3, grid_rows=3, grid_cols=4)
    # By default exactly grid_rows*grid_cols movable blocks exist on the table.
    assert spec.block_mask.sum() == 12


def test_scene_spec_target_idx_default_is_none():
    spec = SceneSpec(
        seed=0, grid_rows=3, grid_cols=3,
        block_poses=np.zeros((9, 7)), block_mask=np.zeros(9, dtype=bool),
    )
    assert spec.target_idx is None
