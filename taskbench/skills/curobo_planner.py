"""cuRobo v2 motion planner wrapper aligned to the ManiSkill Panda.

The ManiSkill `PandaWristCam` agent and cuRobo's built-in `franka.yml` agree
on the 7 actuated arm joints (`panda_joint1`..`panda_joint7`) and panda_hand FK
to float32 precision. They differ in two static transforms that this wrapper
absorbs:

1. **Robot base offset**: ManiSkill places the robot base at world
   ``(-0.615, 0, 0)``; cuRobo plans in robot frame.
2. **TCP offset**: ManiSkill's TCP frame ``panda_hand_tcp`` is offset by
   ``+0.1034 m`` along the local z-axis of ``panda_hand`` (cuRobo's tool frame).

Callers pass **world-frame TCP poses** (the same coordinate system used
elsewhere in taskbench); we convert internally before handing off to cuRobo.

Determinism: cuRobo's trajopt uses a stateful Halton sequence sampler. Call
:meth:`CuroboPlanner.reset_seed` before each plan to get bit-identical results.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# cuRobo's per-call "Running optimizer ..." log is noisy when we drive
# thousands of plans through it. Silence by default; bump back up for debugging.
logging.getLogger("curobo").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Geometry helpers (quaternion conventions: wxyz, ManiSkill / cuRobo standard)
# ---------------------------------------------------------------------------

def _quat_rotate(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = float(q_wxyz[0]), float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3])
    # Rodrigues form: v + 2 w (q_vec x v) + 2 (q_vec x (q_vec x v))
    qv = np.array([x, y, z], dtype=np.float64)
    t = 2.0 * np.cross(qv, v.astype(np.float64))
    return (v.astype(np.float64) + w * t + np.cross(qv, t)).astype(np.float64)


# Standard Franka panda_hand -> panda_hand_tcp offset along hand local z.
PANDA_TCP_Z = 0.1034


def tcp_world_to_hand_robot(
    tcp_pos_world: np.ndarray,
    tcp_quat_world_wxyz: np.ndarray,
    robot_base_pos_world: np.ndarray,
    robot_base_quat_world_wxyz: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0]),
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a TCP target in the world frame into a panda_hand target in the robot frame.

    Assumes the robot base rotation is identity (Build2D-v1 case). The TCP
    offset is removed in the *panda_hand* frame, whose orientation equals the
    TCP orientation since they're linked rigidly.
    """
    if not np.allclose(robot_base_quat_world_wxyz, [1.0, 0.0, 0.0, 0.0], atol=1e-3):
        raise NotImplementedError(
            "CuroboPlanner currently assumes identity robot base rotation; got "
            f"{robot_base_quat_world_wxyz}"
        )
    # panda_hand sits "behind" the TCP along its local +z; back it off.
    hand_offset_world = _quat_rotate(tcp_quat_world_wxyz, np.array([0.0, 0.0, PANDA_TCP_Z]))
    hand_pos_world = np.asarray(tcp_pos_world, dtype=np.float64) - hand_offset_world
    hand_pos_robot = hand_pos_world - np.asarray(robot_base_pos_world, dtype=np.float64)
    return hand_pos_robot.astype(np.float32), np.asarray(tcp_quat_world_wxyz, dtype=np.float32)


# ---------------------------------------------------------------------------
# CuroboPlanner
# ---------------------------------------------------------------------------

class CuroboPlanner:
    """Wrap cuRobo's :class:`MotionPlanner` for our Build2D Pick pipeline."""

    def __init__(
        self,
        env,
        *,
        max_cuboids: int = 16,
        max_goalset: int = 36,
        num_trajopt_seeds: int = 4,
        num_ik_seeds: int = 32,
        random_seed: int = 123,
    ):
        # Lazy-imported so the rest of taskbench is usable without curobo installed.
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

        self._env = env
        self._max_goalset = max_goalset

        raw = env.unwrapped
        agent = raw.agent
        robot = agent.robot
        base_pose = robot.pose
        self._base_pos = np.asarray(base_pose.p.cpu().numpy().flatten()[:3], dtype=np.float64)
        self._base_quat = np.asarray(base_pose.q.cpu().numpy().flatten()[:4], dtype=np.float64)
        logger.info("CuroboPlanner: robot base at %s", self._base_pos.tolist())

        cfg = MotionPlannerCfg.create(
            robot="franka.yml",
            scene_model=None,
            collision_cache={"cuboid": max_cuboids},
            max_goalset=max_goalset,
            num_trajopt_seeds=num_trajopt_seeds,
            num_ik_seeds=num_ik_seeds,
            random_seed=random_seed,
        )
        self._planner = MotionPlanner(cfg)
        self._planner.warmup(enable_graph=True, num_warmup_iterations=5)
        self._registered: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def planner(self):
        return self._planner

    @property
    def joint_names(self) -> List[str]:
        return list(self._planner.joint_names)

    def reset_seed(self) -> None:
        self._planner.reset_seed()

    def sync_scene(self, objects: dict, *, exclude: Optional[str] = None,
                   block_dims: Tuple[float, float, float] = (0.04, 0.04, 0.04)) -> None:
        """Register or update each block in cuRobo's collision world.

        Args:
            objects: mapping ``{name: sapien_object}``.
            exclude: name to skip (the pick target).
            block_dims: dimensions of each block in meters.
        """
        from curobo._src.geom.types import Cuboid

        # Pull world poses, convert to robot frame.
        for name, obj in objects.items():
            if exclude is not None and name == exclude:
                # If previously registered, hide it; otherwise just don't add it.
                if name in self._registered:
                    self._planner.scene_collision_checker.enable_obstacle(name, enable=False)
                continue
            p = obj.pose.p.cpu().numpy().flatten()[:3].astype(np.float64)
            q = obj.pose.q.cpu().numpy().flatten()[:4].astype(np.float64)
            p_robot = p - self._base_pos
            pose_list = [float(p_robot[0]), float(p_robot[1]), float(p_robot[2]),
                         float(q[0]), float(q[1]), float(q[2]), float(q[3])]
            if name in self._registered:
                self._planner.scene_collision_checker.update_obstacle_pose(
                    name, _pose_from_list(pose_list),
                )
                self._planner.scene_collision_checker.enable_obstacle(name, enable=True)
            else:
                cuboid = Cuboid(name=name, pose=pose_list, dims=list(block_dims))
                self._planner.scene_collision_checker.data.add_obstacle(cuboid)
                self._registered.add(name)

    def disable_obstacle(self, name: str) -> None:
        if name in self._registered:
            self._planner.scene_collision_checker.enable_obstacle(name, enable=False)

    def plan_to_tcp_pose(
        self,
        tcp_pos_world: np.ndarray,
        tcp_quat_world_wxyz: np.ndarray,
        current_qpos_arm: np.ndarray,
        *,
        max_attempts: int = 1,
    ):
        """Plan a single trajectory from current arm qpos to a TCP world pose."""
        from curobo.types import GoalToolPose, JointState

        hand_pos, hand_quat = tcp_world_to_hand_robot(
            tcp_pos_world, tcp_quat_world_wxyz, self._base_pos, self._base_quat,
        )
        pos_t = torch.tensor(hand_pos, device="cuda", dtype=torch.float32).view(1, 1, 1, 1, 3)
        quat_t = torch.tensor(hand_quat, device="cuda", dtype=torch.float32).view(1, 1, 1, 1, 4)
        goal = GoalToolPose(
            tool_frames=self._planner.tool_frames,
            position=pos_t,
            quaternion=quat_t,
        )
        q_t = torch.tensor(current_qpos_arm, device="cuda", dtype=torch.float32).unsqueeze(0)
        q_start = JointState.from_position(q_t, joint_names=self._planner.joint_names)
        self.reset_seed()
        result = self._planner.plan_pose(goal, q_start, max_attempts=max_attempts)
        if result is None or not bool(result.success.any()):
            return None
        return result

    def plan_to_tcp_pose_set(
        self,
        candidate_tcp_poses: Sequence[Tuple[np.ndarray, np.ndarray]],
        current_qpos_arm: np.ndarray,
        *,
        max_attempts: int = 1,
    ):
        """Plan from current qpos to *any one* of a goal-set of TCP world poses.

        cuRobo internally batches all candidates, runs IK + trajopt in parallel,
        and returns the trajectory to the lowest-cost reachable candidate.

        Returns a 2-tuple ``(plan_result, selected_index)`` or None on failure.
        """
        from curobo.types import GoalToolPose, JointState

        n = len(candidate_tcp_poses)
        if n == 0:
            return None
        if n > self._max_goalset:
            candidate_tcp_poses = list(candidate_tcp_poses)[: self._max_goalset]
            n = len(candidate_tcp_poses)

        hand_positions = np.zeros((n, 3), dtype=np.float32)
        hand_quats = np.zeros((n, 4), dtype=np.float32)
        for i, (p, q) in enumerate(candidate_tcp_poses):
            hp, hq = tcp_world_to_hand_robot(
                np.asarray(p), np.asarray(q), self._base_pos, self._base_quat,
            )
            hand_positions[i] = hp
            hand_quats[i] = hq
        pos_t = torch.tensor(hand_positions, device="cuda", dtype=torch.float32).view(1, 1, 1, n, 3)
        quat_t = torch.tensor(hand_quats, device="cuda", dtype=torch.float32).view(1, 1, 1, n, 4)
        goal = GoalToolPose(
            tool_frames=self._planner.tool_frames,
            position=pos_t,
            quaternion=quat_t,
        )
        q_t = torch.tensor(current_qpos_arm, device="cuda", dtype=torch.float32).unsqueeze(0)
        q_start = JointState.from_position(q_t, joint_names=self._planner.joint_names)
        self.reset_seed()
        result = self._planner.plan_pose(goal, q_start, max_attempts=max_attempts)
        if result is None or not bool(result.success.any()):
            return None
        idx = None
        if result.goalset_index is not None:
            try:
                idx = int(result.goalset_index.detach().cpu().numpy().flatten()[0])
            except Exception:
                idx = None
        return result, idx

    def plan_grasp_set(
        self,
        candidate_tcp_poses: Sequence[Tuple[np.ndarray, np.ndarray]],
        current_qpos_arm: np.ndarray,
        *,
        grasp_approach_offset: float = 0.10,
        grasp_lift_offset: float = 0.12,
    ):
        """Plan a three-phase grasp using a goal-set of candidate TCP poses.

        Args:
            candidate_tcp_poses: list of (pos_world, quat_world_wxyz) tuples.
            current_qpos_arm: 7-vector arm joint positions.
            grasp_approach_offset: distance to back off along tool z for the
                pre-grasp pose, meters (positive).
            grasp_lift_offset: distance to lift along tool z after grasp.

        Returns:
            GraspPlanResult or None if planning fails.
        """
        from curobo.types import GoalToolPose, JointState

        n = len(candidate_tcp_poses)
        if n == 0:
            return None
        if n > self._max_goalset:
            candidate_tcp_poses = candidate_tcp_poses[: self._max_goalset]
            n = len(candidate_tcp_poses)

        hand_positions = np.zeros((n, 3), dtype=np.float32)
        hand_quats = np.zeros((n, 4), dtype=np.float32)
        for i, (p, q) in enumerate(candidate_tcp_poses):
            hp, hq = tcp_world_to_hand_robot(np.asarray(p), np.asarray(q),
                                             self._base_pos, self._base_quat)
            hand_positions[i] = hp
            hand_quats[i] = hq
        pos_t = torch.tensor(hand_positions, device="cuda", dtype=torch.float32).view(1, 1, 1, n, 3)
        quat_t = torch.tensor(hand_quats, device="cuda", dtype=torch.float32).view(1, 1, 1, n, 4)
        goal = GoalToolPose(
            tool_frames=self._planner.tool_frames,
            position=pos_t,
            quaternion=quat_t,
        )
        q_t = torch.tensor(current_qpos_arm, device="cuda", dtype=torch.float32).unsqueeze(0)
        q_start = JointState.from_position(q_t, joint_names=self._planner.joint_names)
        self.reset_seed()
        result = self._planner.plan_grasp(
            goal,
            q_start,
            grasp_approach_offset=float(grasp_approach_offset),
            grasp_lift_offset=float(grasp_lift_offset),
            plan_approach_to_grasp=True,
            plan_grasp_to_lift=True,
            grasp_lift_in_tool_frame=True,
        )
        if result is None or result.success is None or not bool(result.success.any()):
            return None
        return result


def _pose_from_list(pose_list):
    """Build a cuRobo Pose object from [x,y,z,qw,qx,qy,qz]."""
    from curobo._src.types.pose import Pose as CuPose

    return CuPose.from_list(pose_list)
