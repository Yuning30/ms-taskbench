"""Determinism receipt for Pick: same (scene, target) -> same outcome.

After Stage 1 of the controller-improvement plan, Pick should be a pure
deterministic function of the scene + target_idx (no dependence on the
robot's pre-call orientation, no qpos init noise). This test runs Pick twice
on the same SceneSpec via run_pick_sample and asserts the outcomes match.

Marked smoke (slow) since it spins up the full ManiSkill stack.
"""

from __future__ import annotations

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

import taskbench.envs  # noqa: F401 — env registration
from taskbench.data.runner import run_pick_sample
from taskbench.data.scene_specs import random_scene_spec
from taskbench.skills.context import SkillContext


def _run_once(spec, scene_id="det-test"):
    env = gym.make(
        "Build2D-v1",
        num_envs=1,
        sim_backend="cpu",
        render_backend="cpu",
        control_mode="pd_joint_pos",
        grid_rows=spec.grid_rows,
        grid_cols=spec.grid_cols,
    )
    ctx = SkillContext(env)
    sample = run_pick_sample(env, ctx, spec, scene_id=scene_id)
    env.close()
    return sample


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 17])
def test_pick_is_deterministic_two_runs(seed):
    spec = random_scene_spec(seed=seed, grid_rows=3, grid_cols=3)
    a = _run_once(spec)
    b = _run_once(spec)

    assert a.success == b.success, (
        f"Pick gave different success across runs: a={a.success} b={b.success}"
    )
    assert a.failure_reason == b.failure_reason, (
        f"failure_reason mismatch: a={a.failure_reason!r} b={b.failure_reason!r}"
    )
    assert a.failed_leg == b.failed_leg
    # If successful, the final object pose must match to fp tolerance.
    if a.success:
        assert a.final_object_pose is not None and b.final_object_pose is not None
        np.testing.assert_allclose(a.final_object_pose, b.final_object_pose, atol=1e-3)
