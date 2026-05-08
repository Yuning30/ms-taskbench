import pytest
import numpy as np

gym = pytest.importorskip("gymnasium")

import taskbench.envs  # noqa: F401
from taskbench.data.runner import run_pick_sample
from taskbench.data.scene_specs import random_scene_spec
from taskbench.skills.context import SkillContext


@pytest.mark.slow
def test_run_pick_sample_returns_filled_record():
    env = gym.make(
        "Build2D-v1",
        num_envs=1,
        sim_backend="cpu",
        control_mode="pd_joint_pos",
        grid_rows=2,
        grid_cols=2,
    )
    ctx = SkillContext(env)
    spec = random_scene_spec(seed=0, grid_rows=2, grid_cols=2)
    sample = run_pick_sample(env, ctx, spec, scene_id="test-0001")
    assert sample.scene_id == "test-0001"
    assert sample.target_idx == spec.target_idx
    assert sample.wall_time_s > 0
    assert isinstance(sample.success, bool)
    if not sample.success:
        assert sample.failure_reason is not None
        assert sample.failed_leg is not None
    env.close()
