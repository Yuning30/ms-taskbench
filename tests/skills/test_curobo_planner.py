"""End-to-end test for the CuroboPlanner wrapper on Build2D-v1."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("cuRobo requires CUDA", allow_module_level=True)

gym = pytest.importorskip("gymnasium")
pytest.importorskip("curobo.motion_planner")

import taskbench.envs  # noqa: F401 — env registration
from taskbench.skills.curobo_planner import CuroboPlanner, tcp_world_to_hand_robot


def _make_env():
    return gym.make(
        "Build2D-v1",
        num_envs=1,
        sim_backend="cpu",
        render_backend="cpu",
        control_mode="pd_joint_pos",
        grid_rows=3,
        grid_cols=3,
    )


@pytest.mark.slow
def test_tcp_world_to_hand_robot_round_trip():
    base = np.array([-0.615, 0.0, 0.0])
    tcp_pos = np.array([0.0, 0.0, 0.2])
    tcp_quat = np.array([0.0, 1.0, 0.0, 0.0])  # 180-deg around x: hand points down
    hand_pos, hand_quat = tcp_world_to_hand_robot(tcp_pos, tcp_quat, base)
    # With hand quat = [0,1,0,0], local +z maps to world -z, so hand sits above TCP by 0.1034.
    expected_world = tcp_pos + np.array([0.0, 0.0, 0.1034])
    expected_robot = expected_world - base
    np.testing.assert_allclose(hand_pos, expected_robot, atol=1e-6)
    np.testing.assert_allclose(hand_quat, tcp_quat, atol=1e-6)


@pytest.mark.slow
def test_plan_to_tcp_pose_reaches_target():
    env = _make_env()
    env.reset(seed=0)
    planner = CuroboPlanner(env)

    raw = env.unwrapped
    qpos = raw.agent.robot.get_qpos().cpu().numpy().flatten()
    arm_q = qpos[:7]

    # Target: a TCP pose well above the table, gripper pointing down
    target_pos = np.array([0.1, 0.0, 0.30])
    target_quat = np.array([0.0, 1.0, 0.0, 0.0])
    result = planner.plan_to_tcp_pose(target_pos, target_quat, arm_q)
    assert result is not None, "plan_to_tcp_pose returned None"

    interp = result.get_interpolated_plan()
    final_q_full = interp.position.detach().cpu().numpy().squeeze()[-1]
    # Trajectory dimensions = arm (7) + grippers (2 if active); FK wants 7-DoF.
    final_q = final_q_full[: len(planner.joint_names)]
    from curobo.types import JointState
    js = JointState.from_position(
        torch.tensor(final_q, device="cuda", dtype=torch.float32).unsqueeze(0),
        joint_names=planner.joint_names,
    )
    state = planner.planner.kinematics.compute_kinematics(js)
    hand_pos_robot = state.tool_poses.position.detach().cpu().numpy().flatten()[:3]
    hand_pos_world = hand_pos_robot + np.array([-0.615, 0.0, 0.0])

    # Compute expected panda_hand position from target TCP pose
    expected_hand_pos, _ = tcp_world_to_hand_robot(
        target_pos, target_quat, np.array([-0.615, 0.0, 0.0]),
    )
    expected_hand_world = expected_hand_pos + np.array([-0.615, 0.0, 0.0])

    diff = np.abs(hand_pos_world - expected_hand_world)
    assert float(diff.max()) < 0.01, f"hand reached {hand_pos_world}, expected {expected_hand_world}, diff {diff}"
    env.close()


@pytest.mark.slow
def test_sync_scene_then_plan():
    env = _make_env()
    env.reset(seed=0)
    planner = CuroboPlanner(env)

    raw = env.unwrapped
    objects = raw.get_objects()
    planner.sync_scene(objects, exclude="block_4")
    # Re-sync to exercise the update path.
    planner.sync_scene(objects, exclude="block_4")

    arm_q = raw.agent.robot.get_qpos().cpu().numpy().flatten()[:7]
    target_pos = np.array([0.0, 0.1, 0.25])
    target_quat = np.array([0.0, 1.0, 0.0, 0.0])
    result = planner.plan_to_tcp_pose(target_pos, target_quat, arm_q)
    assert result is not None
    env.close()
