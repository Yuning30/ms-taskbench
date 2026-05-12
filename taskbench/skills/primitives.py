"""Reusable manipulation skills as composable objects.

Each skill binds shared context (env, planner, robot_config, step_callback)
at construction, exposing only task-specific parameters in ``__call__``:

    pick = Pick(env, planner, robot_config=rc, objects=objects)
    result = pick("cube_1", lift_height=0.1)

Base class ``Skill`` provides the common interface. All skills return a
dataclass result with ``success`` and ``failure_reason`` fields.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import sapien
from transforms3d.euler import euler2quat, quat2euler

from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)

from taskbench.skills.motion import (
    PoseLike,
    actuate_gripper,
    attach_object,
    detach_object,
    move_to_pose,
    remove_scene_block_obstacles,
    set_scene_block_obstacles,
    to_sapien_pose,
)
from taskbench.skills.robot_config import RobotConfig, get_robot_config

logger = logging.getLogger("taskbench.skills.primitives")


def _axis_aligned_quat_from_tcp(raw) -> list[float]:
    """Return an axis-aligned quaternion based on current TCP orientation.

    We:
    - read the current TCP quaternion ``raw.agent.tcp.pose.q`` (SAPIEN: [w, x, y, z])
    - convert to Euler angles
    - snap each angle to nearest multiple of 90 degrees
    - convert back to quaternion (SAPIEN [w, x, y, z])
    """
    tcp_q = raw.agent.tcp.pose.q
    try:
        tcp_q = tcp_q.cpu().numpy()
    except Exception:
        tcp_q = np.asarray(tcp_q)
    tcp_q = np.asarray(tcp_q, dtype=np.float64).flatten()[:4]

    # transforms3d expects quaternion as [w, x, y, z] here (consistent with euler2quat usage).
    ai, aj, ak = quat2euler(tcp_q)
    step = np.pi / 2.0
    ai_s = float(np.round(ai / step) * step)
    aj_s = float(np.round(aj / step) * step)
    ak_s = float(np.round(ak / step) * step)
    q = euler2quat(ai_s, aj_s, ak_s)
    q = np.asarray(q, dtype=np.float64).flatten()[:4]
    # Numerical safety.
    norm = float(np.linalg.norm(q))
    if norm > 1e-8:
        q = q / norm
    return [float(x) for x in q]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SkillResult:
    """Base result for all skills."""
    success: bool
    failure_reason: Optional[str] = None
    step_result: Optional[tuple] = None


@dataclass
class MoveResult(SkillResult):
    pass


@dataclass
class PickResult(SkillResult):
    grasp_pose: Optional[sapien.Pose] = None
    lift_pose: Optional[sapien.Pose] = None
    obj_size: Optional[np.ndarray] = None


@dataclass
class PlaceResult(SkillResult):
    pass


@dataclass
class PushResult(SkillResult):
    pass


# ---------------------------------------------------------------------------
# Base skill
# ---------------------------------------------------------------------------

class Skill(ABC):
    """Base class for manipulation skills.

    Binds shared context (env, planner, robot_config, objects, step_callback)
    so that ``__call__`` only receives task-specific parameters.

    Args:
        env: Gym env (raw or wrapped).
        planner: mplib.Planner instance.
        robot_config: Robot-specific constants. If None, auto-detected
            from the env's agent.
        objects: Dict mapping string names to scene actors.
            Skills that need actors (e.g. Pick) resolve names through this.
        step_callback: Optional callable invoked after each env.step().
    """

    def __init__(self, env, planner, *, robot_config: Optional[RobotConfig] = None,
                 objects: Optional[dict[str, object]] = None,
                 step_callback: Optional[Callable] = None,
                 curobo_planner=None):
        self.env = env
        self.planner = planner
        self.robot_config = robot_config or get_robot_config(env)
        self.objects = objects or {}
        self.step_callback = step_callback
        # When set, motion routes through cuRobo instead of mplib's plan_screw.
        self.curobo_planner = curobo_planner

    @abstractmethod
    def __call__(self, *args, **kwargs) -> SkillResult:
        ...


# ---------------------------------------------------------------------------
# Move
# ---------------------------------------------------------------------------

class Move(Skill):
    """Move the arm to a target pose.

    Args (at call time):
        target_pose: PoseLike to move the end effector to.
        gripper_open: Gripper state during motion (default True).
        monitor_contacts: Abort on collision during execution (default True).
    """

    def __call__(
        self,
        target_pose_or_cube: PoseLike | str,
        offsets: list[float] | tuple[float, float, float] | None = None,
        *,
        gripper_open=True,
        monitor_contacts=True,
    ) -> MoveResult:
        # New interface: ctx.move(target_cube, [x_offset, y_offset, z_offset])
        # Old interface remains supported: ctx.move(target_pose)
        if offsets is not None:
            if not isinstance(target_pose_or_cube, str):
                raise TypeError("When providing offsets, target must be a cube/object name (str).")
            if target_pose_or_cube not in self.objects:
                raise KeyError(f"Unknown object {target_pose_or_cube!r}. Available: {list(self.objects)}")

            obj = self.objects[target_pose_or_cube]
            # obj.pose.p is typically a torch tensor, but be robust to numpy.
            p = obj.pose.p
            try:
                p = p.cpu().numpy()
            except Exception:
                p = np.asarray(p)
            p = np.asarray(p, dtype=np.float64).flatten()[:3]

            off = np.asarray(offsets, dtype=np.float64).flatten()[:3]
            target_pos = (p + off).tolist()

            # Reuse current TCP orientation for better IK/planning feasibility.
            # SAPIEN convention in this repo is [w, x, y, z].
            tcp_q = self.env.unwrapped.agent.tcp.pose.q
            try:
                tcp_q = tcp_q.cpu().numpy()
            except Exception:
                tcp_q = np.asarray(tcp_q)
            tcp_q = np.asarray(tcp_q, dtype=np.float64).flatten()[:4]
            target_pose = sapien.Pose(target_pos, tcp_q.tolist())
        else:
            target_pose = to_sapien_pose(target_pose_or_cube)
        rc = self.robot_config
        gripper_state = rc.gripper_open if gripper_open else rc.gripper_closed
        if self.curobo_planner is not None:
            from taskbench.skills.curobo_motion import move_to_pose_curobo
            res = move_to_pose_curobo(
                self.env, self.curobo_planner, target_pose, gripper_state, rc,
                monitor_contacts=monitor_contacts,
                step_callback=self.step_callback,
            )
        else:
            res = move_to_pose(self.env, self.planner, target_pose, gripper_state,
                               rc, monitor_contacts=monitor_contacts,
                               step_callback=self.step_callback)
        if res is None:
            return MoveResult(success=False, failure_reason="move_plan_failed")
        return MoveResult(success=True, step_result=res)


# ---------------------------------------------------------------------------
# Pick
# ---------------------------------------------------------------------------

class Pick(Skill):
    """Grasp an object and lift it.

    Internally: compute grasp from OBB, search rotation candidates,
    reach, approach, close gripper, verify grasp, lift.

    Args (at call time):
        obj_name: String name of the object to grasp (resolved via
            ``self.objects``).
        lift_height: Height above grasp pose to lift to (default 0.1m).
        verify_grasp: Check ``agent.is_grasping()`` after closing (default True).
    """

    def __call__(self, obj_name: str, *, lift_height=0.1,
                 verify_grasp=True) -> PickResult:
        if self.curobo_planner is not None:
            return self._call_curobo(obj_name, lift_height=lift_height,
                                     verify_grasp=verify_grasp)
        obj = self.objects[obj_name]
        env, planner, rc = self.env, self.planner, self.robot_config
        raw = env.unwrapped
        move = Move(env, planner, robot_config=rc, step_callback=self.step_callback,
                    curobo_planner=self.curobo_planner)

        # Register every other free block as an mplib collision obstacle so
        # plan_screw rejects paths that would clip them BEFORE the contact
        # monitor catches them mid-execution. Always cleaned up in finally.
        block_obstacles_added = set_scene_block_obstacles(
            env, planner, self.objects, exclude=obj_name,
        )
        try:
            # Compute grasp pose from OBB. target_closing is fixed to the
            # world y-axis so the synthesized grasp is a pure function of
            # (actor.pose, actor.shape) — independent of the robot's
            # current orientation. The approach direction varies across a
            # small fan (straight-down + ±0.1 in x) so we can recover when
            # straight-down is blocked but an angled approach is feasible.
            obb = get_actor_obb(obj)
            obj_size = np.asarray(obb.extents, dtype=np.float64)
            target_closing = np.array([0.0, 1.0, 0.0])

            def _normalized(v):
                v = np.asarray(v, dtype=np.float64)
                return v / np.linalg.norm(v)

            approach_dirs = [
                np.array([0.0, 0.0, -1.0]),
                _normalized([-0.1, 0.0, -1.0]),  # tilted toward the robot
                _normalized([0.1, 0.0, -1.0]),   # tilted away from the robot
            ]

            # 12 yaw rotations around the chosen approach axis, ordered so
            # default wrist orientations still try first.
            _step = np.pi / 6  # 30 degrees
            yaw_angles = np.array([k * _step for k in
                                   [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6]])

            grasp_found = False
            grasp_pose = None
            for approaching in approach_dirs:
                grasp_info = compute_grasp_info_by_obb(
                    obb,
                    approaching=approaching,
                    target_closing=target_closing,
                    depth=rc.finger_length,
                )
                closing, center = grasp_info["closing"], grasp_info["center"]
                base_pose = raw.agent.build_grasp_pose(approaching, closing, center)
                for angle in yaw_angles:
                    delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
                    candidate = base_pose * delta_pose
                    res = move_to_pose(env, planner, candidate, rc.gripper_open, rc,
                                       dry_run=True)
                    if res is None:
                        continue
                    grasp_pose = candidate
                    grasp_found = True
                    break
                if grasp_found:
                    break

            if not grasp_found:
                logger.warning("Failed to find a valid grasp pose")
                return PickResult(success=False, failure_reason="grasp_plan_failed")

            # Adaptive pre-grasp standoff. Find the lowest block top above the
            # target in a vertical column of ~5 cm radius around the target's xy,
            # then back off the gripper to (ceiling - 1 cm) so reach doesn't
            # collide with whatever is sitting directly above the target.
            target_p = obj.pose.p
            try:
                target_p = target_p.cpu().numpy()
            except Exception:
                target_p = np.asarray(target_p)
            target_p = np.asarray(target_p, dtype=np.float64).flatten()[:3]
            column_radius = 0.05
            target_top_z = float(target_p[2] + obj_size[2] / 2.0)
            ceiling = float("inf")
            for other_name, other in self.objects.items():
                if other_name == obj_name:
                    continue
                op = other.pose.p
                try:
                    op = op.cpu().numpy()
                except Exception:
                    op = np.asarray(op)
                op = np.asarray(op, dtype=np.float64).flatten()[:3]
                if op[2] < 0.0:  # parked/padded below table
                    continue
                dxy = float(np.linalg.norm(op[:2] - target_p[:2]))
                if dxy >= column_radius:
                    continue
                other_obb = get_actor_obb(other)
                top_z = float(op[2] + np.asarray(other_obb.extents).max() / 2.0)
                ceiling = min(ceiling, top_z)
            if np.isfinite(ceiling):
                # Set the standoff just under the ceiling; never below 2cm so
                # the gripper has *some* room above the grasp pose. Tight cases
                # are left for the motion planner to reject naturally rather
                # than aborting up front.
                clearance = ceiling - target_top_z
                standoff = float(np.clip(clearance - 0.005, 0.02, 0.12))
            else:
                standoff = 0.05  # default

            # Reach: approach from `standoff` behind grasp pose
            reach_pose = grasp_pose * sapien.Pose([0, 0, -standoff])
            result = move(reach_pose)
            if not result.success:
                return PickResult(success=False, failure_reason="reach_failed")

            # Grasp: move to grasp pose
            result = move(grasp_pose)
            if not result.success:
                return PickResult(success=False, failure_reason="grasp_approach_failed")

            # Close gripper
            actuate_gripper(env, planner, rc.gripper_closed,
                            step_callback=self.step_callback)

            # Verify grasp
            if verify_grasp:
                is_holding = raw.agent.is_grasping(obj)
                if not bool(is_holding.cpu().numpy().item()):
                    logger.warning("Grasp verification failed")
                    return PickResult(success=False,
                                      failure_reason="grasp_verification_failed")

            # Two-stage lift:
            #   A) shift -1cm in world -x (toward the robot) at grasp z, to
            #      clear any block sitting directly above the target.
            #   B) lift vertically to grasp + lift_height.
            # If Phase A fails to plan, fall back to a single-shot vertical
            # lift (the v4 behavior).
            world_nudge = sapien.Pose([-0.01, 0.0, 0.0])
            nudged_pose = world_nudge * grasp_pose
            nudged_res = move(nudged_pose, gripper_open=False, monitor_contacts=False)
            if nudged_res.success:
                lift_pose = world_nudge * sapien.Pose([0, 0, lift_height]) * grasp_pose
            else:
                # Fall back to one-shot vertical lift.
                lift_pose = sapien.Pose([0, 0, lift_height]) * grasp_pose
            result = move(lift_pose, gripper_open=False, monitor_contacts=False)
            if not result.success:
                return PickResult(
                    success=False,
                    failure_reason="lift_failed",
                    grasp_pose=grasp_pose,
                )

            # Tell planner about the held object for collision-aware planning
            attach_object(planner, obj_size)

            return PickResult(
                success=True,
                grasp_pose=grasp_pose,
                lift_pose=lift_pose,
                obj_size=obj_size,
                step_result=result.step_result,
            )
        finally:
            if block_obstacles_added:
                remove_scene_block_obstacles(planner)

    # ------------------------------------------------------------------
    # cuRobo path: single plan_grasp call returns approach/grasp/lift.
    # ------------------------------------------------------------------
    def _call_curobo(self, obj_name: str, *, lift_height: float,
                     verify_grasp: bool) -> PickResult:
        from taskbench.skills.curobo_motion import (
            follow_curobo_joint_trajectory,
        )

        obj = self.objects[obj_name]
        env, rc = self.env, self.robot_config
        raw = env.unwrapped

        self.curobo_planner.sync_scene(self.objects, exclude=obj_name)

        # Build the same 36 candidate grasp poses our other paths use.
        obb = get_actor_obb(obj)
        obj_size = np.asarray(obb.extents, dtype=np.float64)
        target_closing = np.array([0.0, 1.0, 0.0])

        def _normalized(v):
            v = np.asarray(v, dtype=np.float64)
            return v / np.linalg.norm(v)

        approach_dirs = [
            np.array([0.0, 0.0, -1.0]),
            _normalized([-0.1, 0.0, -1.0]),
            _normalized([0.1, 0.0, -1.0]),
        ]
        _step = np.pi / 6
        yaw_angles = np.array([k * _step for k in
                               [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6]])

        candidates: list[sapien.Pose] = []
        for approaching in approach_dirs:
            grasp_info = compute_grasp_info_by_obb(
                obb, approaching=approaching, target_closing=target_closing,
                depth=rc.finger_length,
            )
            closing, center = grasp_info["closing"], grasp_info["center"]
            base_pose = raw.agent.build_grasp_pose(approaching, closing, center)
            for angle in yaw_angles:
                delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
                candidates.append(base_pose * delta_pose)

        # Adaptive standoff (same as mplib path).
        target_p = obj.pose.p
        try:
            target_p = target_p.cpu().numpy()
        except Exception:
            target_p = np.asarray(target_p)
        target_p = np.asarray(target_p, dtype=np.float64).flatten()[:3]
        column_radius = 0.05
        target_top_z = float(target_p[2] + obj_size[2] / 2.0)
        ceiling = float("inf")
        for other_name, other in self.objects.items():
            if other_name == obj_name:
                continue
            op = other.pose.p
            try:
                op = op.cpu().numpy()
            except Exception:
                op = np.asarray(op)
            op = np.asarray(op, dtype=np.float64).flatten()[:3]
            if op[2] < 0.0:
                continue
            dxy = float(np.linalg.norm(op[:2] - target_p[:2]))
            if dxy >= column_radius:
                continue
            other_obb = get_actor_obb(other)
            top_z = float(op[2] + np.asarray(other_obb.extents).max() / 2.0)
            ceiling = min(ceiling, top_z)
        if np.isfinite(ceiling):
            clearance = ceiling - target_top_z
            standoff = float(np.clip(clearance - 0.005, 0.02, 0.12))
        else:
            standoff = 0.05

        cand_pq = [
            (np.asarray(c.p, dtype=np.float64).flatten()[:3],
             np.asarray(c.q, dtype=np.float64).flatten()[:4])
            for c in candidates
        ]
        arm_q = env.unwrapped.agent.robot.get_qpos().cpu().numpy().flatten()[:7]
        result = self.curobo_planner.plan_grasp_set(
            cand_pq, arm_q,
            grasp_approach_offset=standoff,
            grasp_lift_offset=lift_height,
        )
        if result is None:
            return PickResult(success=False, failure_reason="grasp_plan_failed")

        # Recover the selected candidate so PickResult carries the pose.
        grasp_pose = candidates[0]
        if result.goalset_index is not None:
            try:
                idx = int(result.goalset_index.detach().cpu().numpy().flatten()[0])
                if 0 <= idx < len(candidates):
                    grasp_pose = candidates[idx]
            except Exception:
                pass

        # Phase 1: approach -> pre-grasp. Gripper open, no contact monitor
        # (cuRobo's collision world already excludes the target via sync_scene).
        approach_ok = bool(result.approach_success.any()) if result.approach_success is not None else True
        if not approach_ok or result.approach_interpolated_trajectory is None:
            return PickResult(success=False, failure_reason="reach_failed",
                              grasp_pose=grasp_pose)
        follow_curobo_joint_trajectory(
            env, result.approach_interpolated_trajectory,
            gripper_state=rc.gripper_open,
            robot_config=rc,
            last_tstep=result.approach_interpolated_last_tstep,
            step_callback=self.step_callback,
        )

        # Phase 2: pre-grasp -> grasp. Finger collisions are auto-disabled
        # by cuRobo for this segment.
        grasp_ok = bool(result.grasp_success.any()) if result.grasp_success is not None else True
        if not grasp_ok or result.grasp_interpolated_trajectory is None:
            return PickResult(success=False, failure_reason="grasp_approach_failed",
                              grasp_pose=grasp_pose)
        follow_curobo_joint_trajectory(
            env, result.grasp_interpolated_trajectory,
            gripper_state=rc.gripper_open,
            robot_config=rc,
            last_tstep=result.grasp_interpolated_last_tstep,
            step_callback=self.step_callback,
        )

        # Close gripper.
        actuate_gripper(env, self.planner, rc.gripper_closed,
                        step_callback=self.step_callback)

        # Verify grasp.
        if verify_grasp:
            is_holding = raw.agent.is_grasping(obj)
            if not bool(is_holding.cpu().numpy().item()):
                return PickResult(success=False,
                                  failure_reason="grasp_verification_failed",
                                  grasp_pose=grasp_pose)

        # Phase 3: lift.
        lift_ok = bool(result.lift_success.any()) if result.lift_success is not None else True
        if not lift_ok or result.lift_interpolated_trajectory is None:
            return PickResult(success=False, failure_reason="lift_failed",
                              grasp_pose=grasp_pose)
        last = follow_curobo_joint_trajectory(
            env, result.lift_interpolated_trajectory,
            gripper_state=rc.gripper_closed,
            robot_config=rc,
            last_tstep=result.lift_interpolated_last_tstep,
            step_callback=self.step_callback,
        )

        return PickResult(
            success=True,
            grasp_pose=grasp_pose,
            obj_size=obj_size,
            step_result=last,
        )


# ---------------------------------------------------------------------------
# Place
# ---------------------------------------------------------------------------

class Place(Skill):
    """Move to target pose, release the held object, and retract upward.

    Args (at call time):
        target_pose: PoseLike where the gripper moves before releasing.
        settling_steps: Steps to let physics settle after release (default 10).
        retract_height: Absolute Z height to retract to after release.
            If None, retracts 0.1m above the release pose.
    """

    def __call__(
        self,
        target_pose_or_cube: PoseLike | str,
        offsets: list[float] | tuple[float, float, float] | None = None,
        *,
        settling_steps=10,
        retract_height=None,
    ) -> PlaceResult:
        # New interface: ctx.place(target_cube, [x_offset, y_offset, z_offset])
        # Old interface remains supported: ctx.place(target_pose)
        if offsets is not None:
            if not isinstance(target_pose_or_cube, str):
                raise TypeError("When providing offsets, target must be a cube/object name (str).")
            if target_pose_or_cube not in self.objects:
                raise KeyError(f"Unknown object {target_pose_or_cube!r}. Available: {list(self.objects)}")

            obj = self.objects[target_pose_or_cube]
            p = obj.pose.p
            try:
                p = p.cpu().numpy()
            except Exception:
                p = np.asarray(p)
            p = np.asarray(p, dtype=np.float64).flatten()[:3]

            off = np.asarray(offsets, dtype=np.float64).flatten()[:3]
            target_pos = (p + off).tolist()
            # Reuse current TCP orientation in offset mode.
            tcp_q = self.env.unwrapped.agent.tcp.pose.q
            try:
                tcp_q = tcp_q.cpu().numpy()
            except Exception:
                tcp_q = np.asarray(tcp_q)
            tcp_q = np.asarray(tcp_q, dtype=np.float64).flatten()[:4]
            target_pose = sapien.Pose(target_pos, tcp_q.tolist())
        else:
            target_pose = to_sapien_pose(target_pose_or_cube)
        env, planner, rc = self.env, self.planner, self.robot_config
        move = Move(env, planner, robot_config=rc, step_callback=self.step_callback)

        # Move to target pose (contacts off — gripper is holding an object)
        result = move(target_pose, gripper_open=False, monitor_contacts=False)
        if not result.success:
            return PlaceResult(success=False, failure_reason="place_move_failed")

        # Release gripper
        actuate_gripper(env, planner, rc.gripper_open,
                        step_callback=self.step_callback)

        # Object released — remove from planner
        detach_object(planner)

        # Settle
        actuate_gripper(env, planner, rc.gripper_open, steps=settling_steps,
                        step_callback=self.step_callback)

        # Retract: move straight up to clear before next action
        if retract_height is None:
            retract_height = target_pose.p[2] + 0.1
        retract_pose = sapien.Pose(
            [target_pose.p[0], target_pose.p[1], retract_height],
            target_pose.q,
        )
        result = move(retract_pose)
        if not result.success:
            logger.warning("Retract failed, continuing anyway")

        return PlaceResult(success=True, step_result=result.step_result)


# ---------------------------------------------------------------------------
# Push
# ---------------------------------------------------------------------------

class Push(Skill):
    """Lift for clearance, close gripper, approach, sweep, lift, open.

    Args (at call time):
        approach_pose: PoseLike to move to before pushing (no contact).
        push_pose: PoseLike to sweep toward (contact expected).
        clearance_height: Height to lift above current position before
            approaching (default 0.1m).
        lift_height: Height to lift above push_pose after pushing (default 0.1m).
    """

    def __call__(self, approach_pose: PoseLike, push_pose: PoseLike, *,
                 clearance_height=0.1, lift_height=0.1) -> PushResult:
        approach_pose = to_sapien_pose(approach_pose)
        push_pose = to_sapien_pose(push_pose)
        env, planner, rc = self.env, self.planner, self.robot_config
        raw = env.unwrapped
        move = Move(env, planner, robot_config=rc, step_callback=self.step_callback)

        # Lift from current position for clearance
        tcp_pose = raw.agent.tcp.pose
        tcp_p = np.asarray(tcp_pose.p, dtype=np.float64).flatten()[:3]
        tcp_q = np.asarray(tcp_pose.q, dtype=np.float32).flatten()[:4]
        clearance_pose = sapien.Pose(
            np.array([tcp_p[0], tcp_p[1], tcp_p[2] + clearance_height],
                     dtype=np.float32),
            tcp_q,
        )
        result = move(clearance_pose)
        if not result.success:
            return PushResult(success=False, failure_reason="clearance_lift_failed")

        # Close gripper for flat push surface
        actuate_gripper(env, planner, rc.gripper_closed,
                        step_callback=self.step_callback)

        # Approach — closed gripper, contact monitoring on
        result = move(approach_pose, gripper_open=False)
        if not result.success:
            return PushResult(success=False, failure_reason="approach_failed")

        # Sweep — closed gripper, contact monitoring off (contact is intentional)
        result = move(push_pose, gripper_open=False, monitor_contacts=False)
        if not result.success:
            return PushResult(success=False, failure_reason="push_failed")

        # Lift to disengage — gripper stays closed to avoid snagging
        post_lift_pose = sapien.Pose(
            [push_pose.p[0], push_pose.p[1], push_pose.p[2] + lift_height],
            push_pose.q,
        )
        result = move(post_lift_pose, gripper_open=False)
        if not result.success:
            logger.warning("Push lift failed, continuing anyway")

        # Open gripper once clear
        actuate_gripper(env, planner, rc.gripper_open,
                        step_callback=self.step_callback)

        return PushResult(success=True, step_result=result.step_result)
