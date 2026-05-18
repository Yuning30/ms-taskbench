"""Naive Cartesian pick / place policy (ported style: P gain + discrete gripper).

This mirrors a common flat-observation controller pattern, but takes explicit
TCP position, finger ``qpos`` (last two arm joints), and block world position.

Intended for ``pd_ee_delta_pos`` (and compatible with ``pd_ee_delta_pose`` if
only the first three deltas are used by the caller). See
``run_naive_pick_place_loop`` for stepping the env.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from taskbench.skills.motion import build_action
from taskbench.skills.robot_config import RobotConfig

logger = logging.getLogger("taskbench.skills.naive_ee_control")


@dataclass
class NaiveEEParams:
    """Tunable thresholds for :func:`get_pick_control_naive`."""

    workspace_height: float = 0.1
    dist_atol: float = 2e-2
    other_atol: float = 1e-3
    gain: float = 10.0
    gripper_sum_threshold: float = 0.026
    relative_grasp_position: tuple[float, float, float] = (0.0, 0.0, -0.02)
    max_pos_delta: float = 0.25
    max_steps: int = 4000


def _as_vec3(x) -> np.ndarray:
    v = np.asarray(x, dtype=np.float64).flatten()[:3]
    if v.shape[0] != 3:
        raise ValueError("expected length-3 vector")
    return v


def get_move_action(
    observation: np.ndarray,
    target_position,
    *,
    gain: float = 10.0,
    close_gripper: bool = False,
    gripper_close_cmd: float = -0.2,
) -> np.ndarray:
    """P controller toward ``target_position`` in world frame (TCP xyz).

    ``observation`` must be at least length 3 (current TCP xyz).
    Returns length-4 ``[dx, dy, dz, gripper_cmd]`` (same convention as the
    reference snippet).
    """
    current_position = np.asarray(observation[:3], dtype=np.float64).flatten()[:3]
    target_position = _as_vec3(target_position)
    action = gain * np.subtract(target_position, current_position)
    if close_gripper:
        gripper_action = float(gripper_close_cmd)
    else:
        gripper_action = 0.0
    return np.hstack((action, gripper_action))


def block_inside_grippers(
    obs: np.ndarray,
    relative_grasp_position,
    block_position,
    *,
    atol: float = 1e-3,
) -> bool:
    gripper_position = np.asarray(obs[:3], dtype=np.float64).flatten()[:3]
    block_position = _as_vec3(block_position)
    rel = np.asarray(relative_grasp_position, dtype=np.float64).flatten()[:3]
    relative_position = np.subtract(gripper_position, block_position)
    return float(np.linalg.norm(relative_position - rel)) < float(atol)


def grippers_are_closed(obs: np.ndarray, *, atol: float = 1e-3, threshold: float = 0.026) -> bool:
    gripper_state = np.asarray(obs[3:5], dtype=np.float64).flatten()[:2]
    s = float(np.sum(gripper_state))
    t = 2.0 * float(threshold)
    return abs(s - t) < float(atol) or (s - t) < 0.0


def grippers_are_open(obs: np.ndarray, *, atol: float = 1e-3, threshold: float = 0.026) -> bool:
    gripper_state = np.asarray(obs[3:5], dtype=np.float64).flatten()[:2]
    s = float(np.sum(gripper_state))
    t = 2.0 * float(threshold)
    return s > t + float(atol)


def block_is_grasped(
    obs: np.ndarray,
    relative_grasp_position,
    block_position,
    *,
    dist_atol: float = 3e-2,
    other_atol: float = 1e-3,
    threshold: float = 0.026,
) -> bool:
    return block_inside_grippers(
        obs, relative_grasp_position, block_position, atol=dist_atol
    ) and grippers_are_closed(obs, atol=other_atol, threshold=threshold)


def get_pick_control_naive(
    obs: np.ndarray,
    block_position,
    place_position,
    *,
    params: Optional[NaiveEEParams] = None,
    release: bool = True,
    last_block: bool = False,
    debug: bool = False,
) -> tuple[np.ndarray, bool]:
    """Return ``(action4, done)`` matching the reference state machine.

    ``obs`` is shape ``(5,)``: ``[tcp_x, tcp_y, tcp_z, finger_q0, finger_q1]``.
    """
    if params is None:
        params = NaiveEEParams()

    dist_atol = params.dist_atol
    other_atol = params.other_atol
    gain = params.gain
    workspace_height = params.workspace_height
    relative_grasp_position = params.relative_grasp_position

    gripper_position = np.asarray(obs[:3], dtype=np.float64).flatten()[:3]
    block_position = _as_vec3(block_position)
    place_position = _as_vec3(place_position)

    if np.linalg.norm(block_position - place_position) < dist_atol:
        if debug:
            logger.info("block already at target (no-op)")
        return np.array([0.0, 0.0, 0.0, 0.0]), True

    if np.linalg.norm(block_position - place_position) < dist_atol and block_is_grasped(
        obs,
        relative_grasp_position,
        block_position,
        dist_atol=dist_atol,
        other_atol=other_atol,
        threshold=params.gripper_sum_threshold,
    ):
        if not release:
            if debug:
                logger.info("reach target (hold)")
            return np.array([0.0, 0.0, 0.0, -1.0]), True

        if not grippers_are_open(obs, atol=other_atol, threshold=params.gripper_sum_threshold):
            if debug:
                logger.info("reach target -> open gripper")
            return np.array([0.0, 0.0, 0.0, 0.2]), False

        target_position = copy.deepcopy(block_position)
        target_position[2] += workspace_height * (2 if last_block else 1)
        if gripper_position[2] < target_position[2]:
            if debug:
                logger.info("reach target -> lift up")
            return np.array([0.0, 0.0, 0.5, 0.0]), False

        if debug:
            logger.info("released and lifted")
        return np.array([0.0, 0.0, 0.0, 0.2]), True

    if block_is_grasped(
        obs,
        relative_grasp_position,
        block_position,
        dist_atol=dist_atol,
        other_atol=other_atol,
        threshold=params.gripper_sum_threshold,
    ):
        target_position = copy.deepcopy(place_position)
        if debug:
            logger.info("Move to target")
        return get_move_action(obs, target_position, close_gripper=True, gain=gain), False

    if block_inside_grippers(
        obs,
        relative_grasp_position,
        block_position,
        atol=dist_atol,
    ):
        if debug:
            logger.info("Close the grippers")
        return np.array([0.0, 0.0, 0.0, -1.0]), False

    if np.linalg.norm(gripper_position[:2] - block_position[:2]) < dist_atol:
        if not grippers_are_open(obs, atol=other_atol, threshold=params.gripper_sum_threshold):
            if debug:
                logger.info("Open the grippers")
            return np.array([0.0, 0.0, 0.0, 0.2]), False

        target_position = np.add(block_position, relative_grasp_position)
        if debug:
            logger.info("Move down to grasp")
        return get_move_action(obs, target_position, gain=gain), False

    if not grippers_are_open(obs, atol=other_atol, threshold=params.gripper_sum_threshold):
        if debug:
            logger.info("Open the grippers")
        return np.array([0.0, 0.0, 0.0, 0.2]), False

    wh = float(workspace_height)
    if place_position[2] > block_position[2]:
        wh *= 2.0
    target_position = np.add(block_position, relative_grasp_position)
    target_position[2] += wh
    if debug:
        logger.info("Move to above the block")
    return get_move_action(obs, target_position, gain=gain), False


def _pack_obs(env) -> np.ndarray:
    raw = env.unwrapped
    tcp = np.asarray(raw.agent.tcp.pose.p, dtype=np.float64).flatten()[:3]
    gq = np.asarray(raw.agent.robot.get_qpos(), dtype=np.float64).flatten()[-2:]
    return np.hstack([tcp, gq])


def block_world_xyz(actor) -> np.ndarray:
    p = actor.pose.p
    try:
        p = p.cpu().numpy()
    except Exception:
        p = np.asarray(p)
    return np.asarray(p, dtype=np.float64).flatten()[:3]


def _map_gripper_cmd(cmd: float, rc: RobotConfig, last_g: float) -> float:
    """Map reference-style gripper channel to ManiSkill action values."""
    if cmd > 0.05:
        return float(rc.gripper_open)
    if cmd < -0.05:
        return float(rc.gripper_closed)
    return float(last_g)


def run_naive_pick_place_loop(
    env,
    block_actor,
    place_position,
    rc: RobotConfig,
    *,
    params: Optional[NaiveEEParams] = None,
    release: bool = True,
    last_block: bool = False,
    step_callback: Optional[Callable] = None,
    debug: bool = False,
) -> tuple[bool, int]:
    """Run :func:`get_pick_control_naive` until ``done`` or ``max_steps``.

    Returns ``(success, num_steps)``. Clips translation deltas to
    ``params.max_pos_delta``.
    """
    if params is None:
        params = NaiveEEParams()

    cm = env.unwrapped.control_mode
    if cm not in ("pd_ee_delta_pos", "pd_ee_delta_pose"):
        raise ValueError(
            f"run_naive_pick_place_loop requires pd_ee_delta_pos or pd_ee_delta_pose, got {cm!r}"
        )

    place_position = _as_vec3(place_position)
    last_g = float(rc.gripper_open)
    steps = 0

    for _ in range(params.max_steps):
        obs = _pack_obs(env)
        block_p = block_world_xyz(block_actor)
        act4, done = get_pick_control_naive(
            obs,
            block_p,
            place_position,
            params=params,
            release=release,
            last_block=last_block,
            debug=debug,
        )
        d = np.asarray(act4[:3], dtype=np.float64).flatten()
        d = np.clip(d, -params.max_pos_delta, params.max_pos_delta)
        g = _map_gripper_cmd(float(act4[3]), rc, last_g)
        last_g = g

        if cm == "pd_ee_delta_pose":
            action = build_action(
                env,
                np.array([d[0], d[1], d[2], 0.0, 0.0, 0.0], dtype=np.float64),
                g,
            )
        else:
            action = build_action(env, d, g)

        import pdb; pdb.set_trace()
        env.step(action)
        steps += 1
        if step_callback is not None:
            step_callback()
        if done:
            return True, steps

    logger.warning("run_naive_pick_place_loop: max_steps=%d exceeded", params.max_steps)
    return False, steps


def run_naive_move_to_position_loop(
    env,
    target_position,
    rc: RobotConfig,
    *,
    gain: float = 10.0,
    close_gripper: bool = False,
    tol: float = 8e-3,
    max_steps: int = 200,
    max_delta: float = 0.25,
    step_callback: Optional[Callable] = None,
) -> tuple[bool, int]:
    """P servo TCP xyz to ``target_position``; gripper held at ``close_gripper`` intent."""
    cm = env.unwrapped.control_mode
    if cm not in ("pd_ee_delta_pos", "pd_ee_delta_pose"):
        raise ValueError(
            f"run_naive_move_to_position_loop requires pd_ee_delta_pos or pd_ee_delta_pose, got {cm!r}"
        )

    target_position = _as_vec3(target_position)
    g_cmd = float(rc.gripper_closed if close_gripper else rc.gripper_open)

    for i in range(max_steps):
        obs = _pack_obs(env)
        tcp = obs[:3]
        if float(np.linalg.norm(tcp - target_position)) <= tol:
            return True, i
        act = get_move_action(
            obs,
            target_position,
            gain=gain,
            close_gripper=close_gripper,
        )
        d = np.clip(np.asarray(act[:3], dtype=np.float64), -max_delta, max_delta)
        if cm == "pd_ee_delta_pose":
            action = build_action(
                env,
                np.array([d[0], d[1], d[2], 0.0, 0.0, 0.0], dtype=np.float64),
                g_cmd,
            )
        else:
            action = build_action(env, d, g_cmd)
        import pdb; pdb.set_trace()
        env.step(action)
        if step_callback is not None:
            step_callback()

    return False, max_steps
