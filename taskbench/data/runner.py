"""Per-sample pick-feasibility executor.

Resets Build2DEnv with a SceneSpec's block overrides, runs ctx.pick(target),
captures rich signals, and returns a PickFeasibilitySample.
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from typing import Optional

import numpy as np

from taskbench.data.sample import PickFeasibilitySample
from taskbench.data.scene_specs import SceneSpec
from taskbench.skills.context import SkillContext

logger = logging.getLogger("taskbench.data.runner")

# Map Pick failure_reason strings to a coarser leg label.
_LEG_BY_REASON = {
    "grasp_plan_failed": "grasp_search",
    "approach_blocked": "approach_blocked",
    "reach_failed": "reach",
    "grasp_approach_failed": "grasp_approach",
    "grasp_verification_failed": "grasp_verify",
    "lift_failed": "lift",
    "post_lift_slip": "post_lift_slip",
    "out_of_workspace": "workspace_gate",
}


# ---------------------------------------------------------------------------
# SIGALRM-based per-sample timeout
# ---------------------------------------------------------------------------

class _PickTimeout(Exception):
    """Raised by the SIGALRM handler when a pick exceeds its budget."""


def _alarm_supported() -> bool:
    """True only when SIGALRM is available AND we are in the main thread."""
    return (
        threading.current_thread() is threading.main_thread()
        and hasattr(signal, "SIGALRM")
    )


def _alarm_handler(signum, frame):  # noqa: ANN001
    raise _PickTimeout()


def _leg_for_reason(reason: Optional[str]) -> Optional[str]:
    if reason is None:
        return None
    return _LEG_BY_REASON.get(reason, "unknown")


def _read_object_pose(obj) -> np.ndarray:
    p = obj.pose.p
    q = obj.pose.q
    try:
        p = p.cpu().numpy()
    except Exception:
        p = np.asarray(p)
    try:
        q = q.cpu().numpy()
    except Exception:
        q = np.asarray(q)
    p = np.asarray(p, dtype=np.float64).flatten()[:3]
    q = np.asarray(q, dtype=np.float64).flatten()[:4]
    return np.concatenate([p, q])


def _read_robot_qpos(env) -> np.ndarray:
    raw = env.unwrapped
    qpos = raw.agent.robot.get_qpos()
    try:
        qpos = qpos.cpu().numpy()
    except Exception:
        qpos = np.asarray(qpos)
    return np.asarray(qpos, dtype=np.float64).flatten()


def run_pick_sample(
    env,
    ctx: SkillContext,
    spec: SceneSpec,
    *,
    scene_id: str,
    pick_lift_height: float = 0.12,
    pick_timeout_s: float = 60.0,
) -> PickFeasibilitySample:
    """Reset env to spec, execute Pick on spec.target_idx, return labeled sample.

    Args:
        pick_timeout_s: Maximum wall-clock seconds allowed for ctx.pick().
            If the pick exceeds this budget (and SIGALRM is available), the
            sample is returned as a failure with failure_reason="timeout".
            Default 60 s. Pass 0 or math.inf to disable.
    """
    if spec.target_idx is None:
        raise ValueError("SceneSpec.target_idx must be set for run_pick_sample.")
    n = spec.grid_rows * spec.grid_cols
    if spec.block_poses.shape[0] != n:
        raise ValueError(
            f"SceneSpec block_poses has {spec.block_poses.shape[0]} rows; "
            f"expected {n} (grid_rows*grid_cols)."
        )

    overrides = spec.block_poses.astype(np.float32, copy=True)
    # Padded slots get parked far below the table to keep them out of the way.
    for i in range(n):
        if not bool(spec.block_mask[i]):
            overrides[i, :3] = (0.0, 0.0, -1.0)
            overrides[i, 3:] = (1.0, 0.0, 0.0, 0.0)

    env.reset(seed=spec.seed, options={"block_overrides": overrides})
    ctx.reset(seed=spec.seed)
    # ctx.reset calls env.reset internally without options; we have to re-apply.
    env.reset(seed=spec.seed, options={"block_overrides": overrides})
    ctx.planner = ctx.planner  # planner is fine; objects unchanged
    ctx.objects = ctx.env.unwrapped.get_objects()
    ctx._build_skills()

    target_name = f"block_{spec.target_idx}"
    target_obj = ctx.objects[target_name]

    # Read the post-settle block poses straight from the env so the recorded
    # features match what the controller actually saw. Padded slots keep the
    # spec's parked-below-table pose.
    actual_poses = spec.block_poses.astype(np.float64, copy=True)
    for i in range(n):
        if not bool(spec.block_mask[i]):
            continue
        actual_poses[i] = _read_object_pose(ctx.objects[f"block_{i}"])

    robot_qpos = _read_robot_qpos(env)

    use_alarm = _alarm_supported() and pick_timeout_s > 0
    t0 = time.perf_counter()
    failure_reason: Optional[str] = None
    failed_leg: Optional[str] = None
    success = False
    try:
        if use_alarm:
            old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.setitimer(signal.ITIMER_REAL, pick_timeout_s)
        try:
            result = ctx.pick(target_name, lift_height=pick_lift_height)
            success = bool(result.success)
            if not success:
                failure_reason = result.failure_reason
                failed_leg = _leg_for_reason(failure_reason)
        except _PickTimeout:
            failure_reason = "timeout"
            failed_leg = "timeout"
            logger.warning("Pick timed out after %.1f s for scene %s", pick_timeout_s, scene_id)
        except Exception as exc:
            failure_reason = f"exception:{type(exc).__name__}:{exc}"
            failed_leg = "exception"
            logger.warning("Pick raised: %s", exc)
        finally:
            if use_alarm:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old_handler)
    except Exception as exc:  # safety net for signal setup errors
        failure_reason = f"exception:{type(exc).__name__}:{exc}"
        failed_leg = "exception"
        logger.warning("Pick raised (outer): %s", exc)
    elapsed = time.perf_counter() - t0

    final_pose = _read_object_pose(target_obj) if success else None

    return PickFeasibilitySample(
        scene_id=scene_id,
        seed=spec.seed,
        source=spec.source,
        grid_rows=spec.grid_rows,
        grid_cols=spec.grid_cols,
        block_poses=actual_poses,
        block_mask=spec.block_mask.astype(bool, copy=True),
        target_idx=int(spec.target_idx),
        robot_qpos=robot_qpos,
        success=success,
        failure_reason=failure_reason,
        failed_leg=failed_leg,
        final_object_pose=final_pose,
        wall_time_s=float(elapsed),
    )
