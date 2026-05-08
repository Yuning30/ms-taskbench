"""Smoke tests for Build2DEnv block_overrides reset hook."""

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

import taskbench.envs  # noqa: F401 — register env


@pytest.mark.slow
def test_block_overrides_set_poses():
    env = gym.make(
        "Build2D-v1",
        num_envs=1,
        sim_backend="cpu",
        control_mode="pd_joint_pos",
        grid_rows=2,
        grid_cols=2,
    )
    overrides = np.zeros((4, 7), dtype=np.float32)
    overrides[:, 2] = 0.02
    overrides[:, 3] = 1.0
    overrides[0, :2] = (0.10, 0.05)
    overrides[1, :2] = (-0.05, 0.10)
    overrides[2, :2] = (0.15, -0.10)
    overrides[3, :2] = (-0.10, -0.05)
    env.reset(seed=0, options={"block_overrides": overrides})
    raw = env.unwrapped
    actual = np.stack([b.pose.p[0].cpu().numpy() for b in raw.blocks])
    np.testing.assert_allclose(actual[:, :2], overrides[:, :2], atol=1e-3)
    env.close()
