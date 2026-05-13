"""Determinism receipt for the cuRobo Pick path (c9 + c18).

Mirrors test_pick_determinism_smoke.py but runs under
TASKBENCH_MOTION_BACKEND=curobo with c9 (finger collisions) and c18
(workspace gate) enabled. Same (scene, target) must yield the same outcome
across two runs - this is the verifier's binding invariant.

Marked slow because:
  - Builds the cuRobo MotionPlanner (CUDA-graph warmup ~10s).
  - Requires a CUDA GPU.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("cuRobo requires CUDA", allow_module_level=True)

gym = pytest.importorskip("gymnasium")
pytest.importorskip("curobo.motion_planner")

import taskbench.envs  # noqa: F401 — env registration
from taskbench.data.runner import run_pick_sample
from taskbench.data.scene_specs import random_scene_spec
from taskbench.skills.context import SkillContext


def _run_once(spec, scene_id="det-curobo-test"):
    # SkillContext reads TASKBENCH_MOTION_BACKEND at construction time; we
    # set it (and the c9 / c18 flags) here so the test is self-contained.
    os.environ["TASKBENCH_MOTION_BACKEND"] = "curobo"
    os.environ["TASKBENCH_CUROBO_FINGER_COLL"] = "1"
    os.environ.setdefault("TASKBENCH_WORKSPACE_X_MAX", "0.84")

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
def test_curobo_pick_is_deterministic_two_runs(seed):
    spec = random_scene_spec(seed=seed, grid_rows=3, grid_cols=3)
    a = _run_once(spec)
    b = _run_once(spec)

    assert a.success == b.success, (
        f"cuRobo Pick gave different success across runs "
        f"(seed={seed}): a={a.success} b={b.success}"
    )
    assert a.failure_reason == b.failure_reason, (
        f"failure_reason mismatch (seed={seed}): "
        f"a={a.failure_reason!r} b={b.failure_reason!r}"
    )
    assert a.failed_leg == b.failed_leg
    if a.success:
        assert a.final_object_pose is not None and b.final_object_pose is not None
        np.testing.assert_allclose(
            a.final_object_pose, b.final_object_pose, atol=1e-3,
        )
