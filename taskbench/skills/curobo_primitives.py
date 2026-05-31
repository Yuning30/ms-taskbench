"""cuRobo-backed manipulation skills.

These are the production skills bound to ``SkillContext.{pick,place,move,push}``.
They share the API of the mplib variants in :mod:`taskbench.skills.mplib_primitives`
(call sites do not need to change) but route all motion through cuRobo's
trajectory optimizer.

``CuroboPick`` ports the c21 production controller from the ``2dgrid-pick``
branch: 72-candidate goal-set + slip-cost ranking, adaptive standoff,
workspace-rejection gate, finger collisions enabled, two-stage gripper close,
post-lift z-rise verification.

Constructor::

    CuroboPick(env, planner, *, robot_config=rc, objects=objects,
               step_callback=cb, curobo_planner=cup)

``planner`` is the mplib planner (still used for the gripper helpers
``actuate_gripper``); ``curobo_planner`` is the :class:`CuroboPlanner`.

Call::

    pick = ctx.pick                # already bound via SkillContext
    result = pick("cube_1", lift_height=0.1)
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)

from taskbench.skills.base import (
    MoveResult,
    PickResult,
    PlaceResult,
    PushResult,
    Skill,
)
from taskbench.skills.curobo_motion import (
    follow_curobo_joint_trajectory,
    move_to_pose_curobo,
)
from taskbench.skills.curobo_planner import CuroboPlanner
from taskbench.skills.motion import (
    PoseLike,
    actuate_gripper,
    to_sapien_pose,
)
from taskbench.skills.robot_config import RobotConfig, get_robot_config

logger = logging.getLogger("taskbench.skills.curobo_primitives")


# c21 production defaults (from the 2dgrid-pick controller iteration). These
# were originally exposed as TASKBENCH_CUROBO_* env vars; we bake the winning
# values in and re-expose them later if a use case demands it.
_DEFAULT_WORKSPACE_X_MAX = 0.84
_DEFAULT_GRIPPER_STEPS = 25
_DEFAULT_TWO_STAGE_CLOSE = True
_DEFAULT_POSTLIFT_FRAC = 0.6
_DEFAULT_FINGER_COLLISIONS = True


class _CuroboSkill(Skill):
    """Common base: holds the mplib + cuRobo planners side by side.

    The mplib planner is retained only so gripper-actuation helpers like
    :func:`actuate_gripper` (which need a DOF count) keep working without
    changes. All motion goes through ``self.curobo_planner``.
    """

    def __init__(
        self,
        env,
        planner,
        *,
        robot_config: Optional[RobotConfig] = None,
        objects: Optional[dict[str, object]] = None,
        step_callback: Optional[Callable] = None,
        curobo_planner: Optional[CuroboPlanner] = None,
    ):
        super().__init__(env, planner, robot_config=robot_config,
                         objects=objects, step_callback=step_callback)
        if curobo_planner is None:
            raise ValueError(
                f"{type(self).__name__} requires a curobo_planner. "
                "Build a CuroboPlanner and pass it in (SkillContext does this)."
            )
        self.curobo_planner = curobo_planner


# ---------------------------------------------------------------------------
# Move
# ---------------------------------------------------------------------------

class CuroboMove(_CuroboSkill):
    """Move the arm to a target pose using cuRobo.

    Mirrors :class:`MplibMove`'s call API:

        ctx.move(target_pose)
        ctx.move(cube_name, [dx, dy, dz])  # offset from cube
    """

    def __call__(
        self,
        target_pose_or_cube: PoseLike | str,
        offsets: list[float] | tuple[float, float, float] | None = None,
        *,
        gripper_open: bool = True,
        monitor_contacts: bool = True,
    ) -> MoveResult:
        target_pose = _resolve_target(
            self.env, self.objects, target_pose_or_cube, offsets,
        )
        rc = self.robot_config
        gripper_state = rc.gripper_open if gripper_open else rc.gripper_closed
        res = move_to_pose_curobo(
            self.env, self.curobo_planner, target_pose, gripper_state, rc,
            monitor_contacts=monitor_contacts,
            step_callback=self.step_callback,
        )
        if res is None:
            return MoveResult(success=False, failure_reason="move_plan_failed")
        return MoveResult(success=True, step_result=res)


# ---------------------------------------------------------------------------
# Pick (c21 production controller)
# ---------------------------------------------------------------------------

class CuroboPick(_CuroboSkill):
    """Grasp an object and lift it with the cuRobo plan_grasp pipeline.

    Pipeline:
      1. Workspace-rejection gate (target x in robot frame > 0.84 m → reject).
      2. Sync neighbor cubes into cuRobo's collision world; hide the target.
      3. Build 72 candidate TCP grasp poses (3 approach dirs × 24 yaws at 15°).
      4. Rank candidates by slip cost (tilt + yaw-from-face).
      5. Compute adaptive standoff from any neighbor sitting above the target.
      6. cuRobo ``plan_grasp_set`` returns (approach, grasp, lift) trajectories.
      7. Execute approach, then grasp; finger collisions are auto-disabled
         only for the grasp segment.
      8. Two-stage gripper close (half, hold; full, hold).
      9. Verify ``is_grasping`` and that the object's z rose by at least 60%
         of ``lift_height`` after the lift trajectory.
    """

    def __call__(
        self,
        obj_name: str,
        *,
        lift_height: float = 0.1,
        verify_grasp: bool = True,
    ) -> PickResult:
        obj = self.objects[obj_name]
        env, rc = self.env, self.robot_config
        raw = env.unwrapped

        # Workspace gate.
        if self._workspace_reject(obj):
            return PickResult(success=False, failure_reason="out_of_workspace")

        self.curobo_planner.sync_scene(self.objects, exclude=obj_name)

        # Candidate grasp poses with slip-cost ranking.
        candidates, obj_size = self._build_candidates(obj)

        # Adaptive standoff from neighbor ceiling.
        standoff = self._adaptive_standoff(obj_name, obj, obj_size)

        cand_pq = [
            (np.asarray(c.p, dtype=np.float64).flatten()[:3],
             np.asarray(c.q, dtype=np.float64).flatten()[:4])
            for c in candidates
        ]
        arm_q = raw.agent.robot.get_qpos().cpu().numpy().flatten()[:7]

        plan_kwargs = {}
        if _DEFAULT_FINGER_COLLISIONS:
            plan_kwargs["disable_collision_links"] = ["panda_hand", "attached_object"]
        result = self.curobo_planner.plan_grasp_set(
            cand_pq, arm_q,
            grasp_approach_offset=standoff,
            grasp_lift_offset=lift_height,
            **plan_kwargs,
        )
        if result is None:
            return PickResult(success=False, failure_reason="grasp_plan_failed")

        # Recover the selected candidate so PickResult carries the chosen pose.
        grasp_pose = candidates[0]
        if result.goalset_index is not None:
            try:
                idx = int(result.goalset_index.detach().cpu().numpy().flatten()[0])
                if 0 <= idx < len(candidates):
                    grasp_pose = candidates[idx]
            except Exception:
                pass

        # Phase 1: approach -> pre-grasp.
        approach_ok = (bool(result.approach_success.any())
                       if result.approach_success is not None else True)
        if not approach_ok or result.approach_interpolated_trajectory is None:
            return PickResult(success=False, failure_reason="reach_failed",
                              grasp_pose=grasp_pose)
        follow_curobo_joint_trajectory(
            env, result.approach_interpolated_trajectory,
            gripper_state=rc.gripper_open, robot_config=rc,
            last_tstep=result.approach_interpolated_last_tstep,
            step_callback=self.step_callback,
        )

        # Phase 2: pre-grasp -> grasp. cuRobo auto-disables finger collisions here.
        grasp_ok = (bool(result.grasp_success.any())
                    if result.grasp_success is not None else True)
        if not grasp_ok or result.grasp_interpolated_trajectory is None:
            return PickResult(success=False, failure_reason="grasp_approach_failed",
                              grasp_pose=grasp_pose)
        follow_curobo_joint_trajectory(
            env, result.grasp_interpolated_trajectory,
            gripper_state=rc.gripper_open, robot_config=rc,
            last_tstep=result.grasp_interpolated_last_tstep,
            step_callback=self.step_callback,
        )

        # Close gripper.
        if _DEFAULT_TWO_STAGE_CLOSE:
            actuate_gripper(env, self.planner, 0.0, steps=10,
                            step_callback=self.step_callback)
        actuate_gripper(env, self.planner, rc.gripper_closed,
                        steps=_DEFAULT_GRIPPER_STEPS,
                        step_callback=self.step_callback)

        if verify_grasp:
            is_holding = raw.agent.is_grasping(obj)
            if not bool(is_holding.cpu().numpy().item()):
                return PickResult(success=False,
                                  failure_reason="grasp_verification_failed",
                                  grasp_pose=grasp_pose)

        # Snapshot the object's pre-lift z so we can verify it actually moved.
        grasp_z = _obj_z(obj)

        # Phase 3: lift.
        lift_ok = (bool(result.lift_success.any())
                   if result.lift_success is not None else True)
        if not lift_ok or result.lift_interpolated_trajectory is None:
            return PickResult(success=False, failure_reason="lift_failed",
                              grasp_pose=grasp_pose, obj_size=obj_size)
        last = follow_curobo_joint_trajectory(
            env, result.lift_interpolated_trajectory,
            gripper_state=rc.gripper_closed, robot_config=rc,
            last_tstep=result.lift_interpolated_last_tstep,
            step_callback=self.step_callback,
        )

        # Post-lift verification: still grasping AND object z rose enough.
        if verify_grasp:
            post_z = _obj_z(obj)
            min_rise = _DEFAULT_POSTLIFT_FRAC * float(lift_height)
            is_still_holding = bool(raw.agent.is_grasping(obj).cpu().numpy().item())
            if (not is_still_holding) or (post_z - grasp_z < min_rise):
                logger.info(
                    "post-lift slip: holding=%s dz=%.4fm (need>=%.4fm)",
                    is_still_holding, post_z - grasp_z, min_rise,
                )
                return PickResult(success=False, failure_reason="post_lift_slip",
                                  grasp_pose=grasp_pose, obj_size=obj_size)

        # Same lift_pose semantics as MplibPick (target pose, not measured).
        lift_pose = sapien.Pose([0, 0, lift_height]) * grasp_pose
        return PickResult(success=True, grasp_pose=grasp_pose,
                          lift_pose=lift_pose, obj_size=obj_size,
                          step_result=last)

    # ----------------------------------------------------------------------
    # Internals
    # ----------------------------------------------------------------------

    def _workspace_reject(self, obj) -> bool:
        if _DEFAULT_WORKSPACE_X_MAX == float("inf"):
            return False
        p = _obj_pos(obj)
        tx_robot = float(p[0] - self.curobo_planner._base_pos[0])
        if tx_robot > _DEFAULT_WORKSPACE_X_MAX:
            logger.info(
                "workspace gate: target x=%.3fm (robot frame) > %.3fm; rejecting",
                tx_robot, _DEFAULT_WORKSPACE_X_MAX,
            )
            return True
        return False

    def _build_candidates(self, obj):
        """Return a list of candidate TCP poses sorted by slip cost."""
        rc = self.robot_config
        raw = self.env.unwrapped
        obb = get_actor_obb(obj)
        obj_size = np.asarray(obb.extents, dtype=np.float64)
        target_closing = np.array([0.0, 1.0, 0.0])

        approach_dirs = [
            np.array([0.0, 0.0, -1.0]),
            _normalized([-0.1, 0.0, -1.0]),
            _normalized([0.1, 0.0, -1.0]),
        ]
        # 24 yaws at 15°.
        _step = np.pi / 12
        yaw_angles = np.array([k * _step for k in
                               [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6,
                                7, -7, 8, -8, 9, -9, 10, -10, 11, -11, 12, -12]])

        TILT_W = 8.0
        YAW_W = 1.0
        candidates: list[sapien.Pose] = []
        slip_costs: list[float] = []
        for approaching in approach_dirs:
            grasp_info = compute_grasp_info_by_obb(
                obb, approaching=approaching,
                target_closing=target_closing,
                depth=rc.finger_length,
            )
            closing, center = grasp_info["closing"], grasp_info["center"]
            base_pose = raw.agent.build_grasp_pose(approaching, closing, center)
            tilt = float(1.0 - abs(float(approaching[2]) / np.linalg.norm(approaching)))
            for angle in yaw_angles:
                delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
                candidates.append(base_pose * delta_pose)
                yaw_mod = float(angle) % (np.pi / 2)
                yaw_off = min(yaw_mod, np.pi / 2 - yaw_mod) / (np.pi / 4)
                slip_costs.append(TILT_W * tilt + YAW_W * yaw_off)

        order = sorted(range(len(candidates)), key=lambda i: slip_costs[i])
        candidates = [candidates[i] for i in order]
        return candidates, obj_size

    def _adaptive_standoff(self, obj_name: str, obj, obj_size) -> float:
        """Back off to just under the lowest neighbor sitting above the target."""
        target_p = _obj_pos(obj)
        column_radius = 0.05
        target_top_z = float(target_p[2] + obj_size[2] / 2.0)
        ceiling = float("inf")
        for other_name, other in self.objects.items():
            if other_name == obj_name:
                continue
            op = _obj_pos(other)
            if op[2] < 0.0:
                continue
            if float(np.linalg.norm(op[:2] - target_p[:2])) >= column_radius:
                continue
            other_obb = get_actor_obb(other)
            top_z = float(op[2] + np.asarray(other_obb.extents).max() / 2.0)
            ceiling = min(ceiling, top_z)
        if np.isfinite(ceiling):
            clearance = ceiling - target_top_z
            return float(np.clip(clearance - 0.005, 0.02, 0.12))
        return 0.05


# ---------------------------------------------------------------------------
# Place
# ---------------------------------------------------------------------------

class CuroboPlace(_CuroboSkill):
    """Move to target, release the held object, retract upward (cuRobo motion).

    Mirrors :class:`MplibPlace`'s call API:

        ctx.place(target_pose)
        ctx.place(cube_name, [dx, dy, dz])
    """

    def __call__(
        self,
        target_pose_or_cube: PoseLike | str,
        offsets: list[float] | tuple[float, float, float] | None = None,
        *,
        settling_steps: int = 10,
        retract_height: Optional[float] = None,
    ) -> PlaceResult:
        target_pose = _resolve_target(
            self.env, self.objects, target_pose_or_cube, offsets,
        )
        env, rc = self.env, self.robot_config

        # Move to target. Contacts off — gripper is holding an object.
        res = move_to_pose_curobo(
            env, self.curobo_planner, target_pose, rc.gripper_closed, rc,
            monitor_contacts=False, step_callback=self.step_callback,
        )
        if res is None:
            return PlaceResult(success=False, failure_reason="place_move_failed")

        # Release gripper, then settle for ``settling_steps`` extra steps.
        actuate_gripper(env, self.planner, rc.gripper_open,
                        step_callback=self.step_callback)
        actuate_gripper(env, self.planner, rc.gripper_open,
                        steps=settling_steps, step_callback=self.step_callback)

        # Re-sync the scene with no exclusions so the just-placed cube becomes
        # an obstacle again for the retract plan.
        self.curobo_planner.sync_scene(self.objects)

        # Retract: vertical lift above the release pose.
        if retract_height is None:
            retract_height = target_pose.p[2] + 0.1
        retract_pose = sapien.Pose(
            [target_pose.p[0], target_pose.p[1], retract_height],
            target_pose.q,
        )
        retract_res = move_to_pose_curobo(
            env, self.curobo_planner, retract_pose, rc.gripper_open, rc,
            monitor_contacts=False, step_callback=self.step_callback,
        )
        if retract_res is None:
            logger.warning("Retract failed, continuing anyway")
            retract_res = res

        return PlaceResult(success=True, step_result=retract_res)


# ---------------------------------------------------------------------------
# Push
# ---------------------------------------------------------------------------

class CuroboPush(_CuroboSkill):
    """Lift for clearance, close gripper, approach, sweep, lift, open (cuRobo motion)."""

    def __call__(
        self,
        approach_pose: PoseLike,
        push_pose: PoseLike,
        *,
        clearance_height: float = 0.1,
        lift_height: float = 0.1,
    ) -> PushResult:
        approach_pose = to_sapien_pose(approach_pose)
        push_pose = to_sapien_pose(push_pose)
        env, rc = self.env, self.robot_config
        raw = env.unwrapped

        # Clearance lift from current TCP.
        tcp_pose = raw.agent.tcp.pose
        tcp_p = np.asarray(tcp_pose.p, dtype=np.float64).flatten()[:3]
        tcp_q = np.asarray(tcp_pose.q, dtype=np.float64).flatten()[:4]
        clearance_pose = sapien.Pose(
            [tcp_p[0], tcp_p[1], tcp_p[2] + clearance_height], tcp_q,
        )
        res = move_to_pose_curobo(
            env, self.curobo_planner, clearance_pose, rc.gripper_open, rc,
            step_callback=self.step_callback,
        )
        if res is None:
            return PushResult(success=False, failure_reason="clearance_lift_failed")

        # Close gripper for a flat push surface.
        actuate_gripper(env, self.planner, rc.gripper_closed,
                        step_callback=self.step_callback)

        # Approach — closed gripper, contact monitoring on.
        res = move_to_pose_curobo(
            env, self.curobo_planner, approach_pose, rc.gripper_closed, rc,
            monitor_contacts=True, step_callback=self.step_callback,
        )
        if res is None:
            return PushResult(success=False, failure_reason="approach_failed")

        # Sweep — closed gripper, contact monitoring off (contact is intentional).
        res = move_to_pose_curobo(
            env, self.curobo_planner, push_pose, rc.gripper_closed, rc,
            monitor_contacts=False, step_callback=self.step_callback,
        )
        if res is None:
            return PushResult(success=False, failure_reason="push_failed")

        # Lift to disengage — gripper stays closed to avoid snagging.
        post_lift_pose = sapien.Pose(
            [push_pose.p[0], push_pose.p[1], push_pose.p[2] + lift_height],
            push_pose.q,
        )
        lift_res = move_to_pose_curobo(
            env, self.curobo_planner, post_lift_pose, rc.gripper_closed, rc,
            step_callback=self.step_callback,
        )
        if lift_res is None:
            logger.warning("Push lift failed, continuing anyway")
            lift_res = res

        # Open gripper once clear.
        actuate_gripper(env, self.planner, rc.gripper_open,
                        step_callback=self.step_callback)

        return PushResult(success=True, step_result=lift_res)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_target(env, objects, target_pose_or_cube, offsets) -> sapien.Pose:
    """Shared cube-name+offset vs raw-pose resolution for Move/Place."""
    if offsets is None:
        return to_sapien_pose(target_pose_or_cube)
    if not isinstance(target_pose_or_cube, str):
        raise TypeError("When providing offsets, target must be a cube/object name (str).")
    if target_pose_or_cube not in objects:
        raise KeyError(f"Unknown object {target_pose_or_cube!r}. Available: {list(objects)}")
    p = _obj_pos(objects[target_pose_or_cube])
    off = np.asarray(offsets, dtype=np.float64).flatten()[:3]
    target_pos = (p + off).tolist()
    tcp_q = env.unwrapped.agent.tcp.pose.q
    try:
        tcp_q = tcp_q.cpu().numpy()
    except Exception:
        tcp_q = np.asarray(tcp_q)
    tcp_q = np.asarray(tcp_q, dtype=np.float64).flatten()[:4]
    return sapien.Pose(target_pos, tcp_q.tolist())


def _obj_pos(obj) -> np.ndarray:
    p = obj.pose.p
    try:
        p = p.cpu().numpy()
    except Exception:
        p = np.asarray(p)
    return np.asarray(p, dtype=np.float64).flatten()[:3]


def _obj_z(obj) -> float:
    return float(_obj_pos(obj)[2])


def _normalized(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)
