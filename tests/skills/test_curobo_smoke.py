"""Smoke + determinism check for cuRobo v2 motion planning.

Requires the [curobo] extra and a CUDA GPU. Marked `slow` because the first
warmup pre-compiles CUDA graphs.

Determinism is the critical property for our verifier pipeline: identical
(scene, target) -> identical pick outcome. cuRobo's trajectory optimization
uses a stateful Halton sample generator, so we must call planner.reset_seed()
before each plan to get the same trajectory.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("cuRobo requires CUDA", allow_module_level=True)

curobo = pytest.importorskip("curobo.motion_planner")
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState


def _build_planner():
    cfg = MotionPlannerCfg.create(robot="franka.yml", scene_model="collision_test.yml")
    p = MotionPlanner(cfg)
    p.warmup(enable_graph=True, num_warmup_iterations=5)
    return p


@pytest.fixture(scope="module")
def planner():
    return _build_planner()


@pytest.mark.slow
def test_plan_pose_runs(planner):
    q_start = JointState.from_position(
        planner.default_joint_state.position.unsqueeze(0),
        joint_names=planner.joint_names,
    )
    goal = GoalToolPose(
        tool_frames=planner.tool_frames,
        position=torch.tensor([[[[[0.5, 0.0, 0.3]]]]], device="cuda", dtype=torch.float32),
        quaternion=torch.tensor([[[[[1.0, 0.0, 0.0, 0.0]]]]], device="cuda", dtype=torch.float32),
    )
    planner.reset_seed()
    res = planner.plan_pose(goal, q_start)
    assert res is not None and res.success.any(), "plan_pose did not return a successful plan"


@pytest.mark.slow
def test_plan_pose_deterministic_with_reset_seed(planner):
    q_start = JointState.from_position(
        planner.default_joint_state.position.unsqueeze(0),
        joint_names=planner.joint_names,
    )
    goal = GoalToolPose(
        tool_frames=planner.tool_frames,
        position=torch.tensor([[[[[0.5, 0.0, 0.3]]]]], device="cuda", dtype=torch.float32),
        quaternion=torch.tensor([[[[[1.0, 0.0, 0.0, 0.0]]]]], device="cuda", dtype=torch.float32),
    )

    def run_once():
        planner.reset_seed()
        res = planner.plan_pose(goal, q_start)
        return res.get_interpolated_plan().position.detach().cpu().numpy().copy()

    a = run_once()
    b = run_once()
    assert a.shape == b.shape
    np.testing.assert_array_equal(a, b)
