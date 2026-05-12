"""Execute cuRobo-planned trajectories in ManiSkill, matching ``move_to_pose``'s contract.

``move_to_pose`` (mplib) returns:
  - None on plan failure
  - a result dict on dry_run success
  - (obs, reward, terminated, truncated, info) on real-run success

We mirror that here so callers can swap planners without restructuring.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import sapien

from taskbench.skills.motion import build_action
from taskbench.skills.robot_config import RobotConfig

logger = logging.getLogger(__name__)


def _read_arm_qpos(env, n_arm: int = 7) -> np.ndarray:
    qpos = env.unwrapped.agent.robot.get_qpos().cpu().numpy().flatten()
    return qpos[:n_arm].astype(np.float64)


def move_to_pose_curobo(
    env,
    curobo_planner,
    pose: sapien.Pose,
    gripper_state,
    robot_config: RobotConfig,
    *,
    dry_run: bool = False,
    monitor_contacts: bool = False,
    step_callback=None,
    refine_steps: int = 0,
):
    """Plan and (optionally) execute a cuRobo trajectory to ``pose``.

    ``pose`` is a sapien.Pose giving the **TCP world pose** (matches the
    convention used elsewhere in primitives.py). cuRobo's tool frame is
    panda_hand; CuroboPlanner handles the conversion.

    Returns:
        - None on planning failure
        - a placeholder dict ``{"status": "Success"}`` on dry_run success
        - (obs, reward, terminated, truncated, info) on execution success
    """
    p = np.asarray(pose.p, dtype=np.float64).flatten()[:3]
    q = np.asarray(pose.q, dtype=np.float64).flatten()[:4]
    arm_q = _read_arm_qpos(env)

    result = curobo_planner.plan_to_tcp_pose(p, q, arm_q)
    if result is None:
        logger.debug("cuRobo plan failed for pose p=%s q=%s", p, q)
        return None
    if dry_run:
        return {"status": "Success"}

    return _follow_curobo_plan(
        env, result, gripper_state, robot_config,
        monitor_contacts=monitor_contacts,
        step_callback=step_callback,
        refine_steps=refine_steps,
        n_arm=len(curobo_planner.joint_names),
    )


def _follow_curobo_plan(
    env, plan_result, gripper_state, robot_config: RobotConfig,
    *, monitor_contacts: bool, step_callback, refine_steps: int, n_arm: int,
):
    """Step the env once per interpolated waypoint, sending arm qpos + gripper."""
    interp = plan_result.get_interpolated_plan()
    # Shape: (batch, T, dof). Squeeze leading batch dim.
    positions = interp.position.detach().cpu().numpy()
    if positions.ndim == 3:
        positions = positions[0]
    elif positions.ndim == 4:
        positions = positions[0, 0]
    n_step = positions.shape[0]
    last = None
    for i in range(n_step + refine_steps):
        idx = min(i, n_step - 1)
        qpos_arm = positions[idx, :n_arm]
        action = build_action(env, qpos_arm, gripper_state)
        obs, reward, terminated, truncated, info = env.step(action)
        last = (obs, reward, terminated, truncated, info)
        if step_callback is not None:
            step_callback()
        if monitor_contacts:
            from taskbench.skills.motion import _get_gripper_contacts  # local import
            for contact, finger, other in _get_gripper_contacts(env, robot_config):
                force = sum(np.linalg.norm(pt.impulse) for pt in contact.points) / env.unwrapped.control_timestep
                if force > 0.01:
                    logger.warning(
                        "Collision at step %d/%d: %s -> %s (%.2f N), aborting",
                        i, n_step, finger, other, force,
                    )
                    return None
    return last
