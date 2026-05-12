"""Skill context — bundles env + planner + objects for skill execution.

Eliminates the boilerplate that every solver repeats::

    ctx = SkillContext(env, step_callback=recorder.record)
    ctx.reset(seed=42)

    ctx.pick("cube_1", lift_height=0.15)
    ctx.place(target_pose)
"""

import os
from typing import Callable, Optional

from taskbench.envs import get_objects
from taskbench.skills.motion import setup_planner
from taskbench.skills.primitives import Move, Pick, Place, Push
from taskbench.skills.robot_config import RobotConfig, get_robot_config

# When set to "curobo", SkillContext builds a CuroboPlanner and routes motion
# through it. Any other value (or unset) uses mplib's plan_screw.
MOTION_BACKEND_ENV_VAR = "TASKBENCH_MOTION_BACKEND"

_NOT_READY_MSG = "SkillContext.reset() must be called before using skills"


class _SkillProxy:
    """Raises a clear error when skills are accessed before reset()."""

    def __call__(self, *args, **kwargs):
        raise RuntimeError(_NOT_READY_MSG)

    def __getattr__(self, name):
        raise RuntimeError(_NOT_READY_MSG)


class SkillContext:
    """Shared context for skill-based solvers.

    Holds the env, motion planner, robot config, object references, and
    pre-bound skill instances. Call ``reset()`` to re-initialize everything
    for a new episode.

    Args:
        env: Gym env (num_envs=1, sim_backend="cpu").
        step_callback: Optional callable invoked after each env.step()
            (e.g. ``recorder.record``).
    """

    def __init__(self, env, *, step_callback: Optional[Callable] = None):
        self.env = env
        self.step_callback = step_callback
        self.robot_config: RobotConfig = get_robot_config(env)
        self.planner = None
        self.curobo_planner = None
        self._use_curobo = os.environ.get(MOTION_BACKEND_ENV_VAR, "").lower() == "curobo"
        self.objects: dict[str, object] = {}

        # Skill instances — populated by reset() → _build_skills()
        _proxy = _SkillProxy()
        self.pick: Pick = _proxy  # type: ignore[assignment]
        self.place: Place = _proxy  # type: ignore[assignment]
        self.push: Push = _proxy  # type: ignore[assignment]
        self.move: Move = _proxy  # type: ignore[assignment]

    def reset(self, seed=None):
        """Reset the env and rebuild planner, objects, and skills."""
        self.env.reset(seed=seed)
        self.planner = setup_planner(self.env, self.robot_config)
        # cuRobo planner is built lazily once (warmup is expensive) and reused
        # across resets; the scene is re-synced inside Pick via sync_scene().
        if self._use_curobo and self.curobo_planner is None:
            from taskbench.skills.curobo_planner import CuroboPlanner
            # c13: optional tighter IK precision for the position/orientation
            # tolerance (defaults 5mm / ~3deg). Tighter = more accurate
            # gripper-to-cube alignment at the grasp pose but may fail more
            # plans at extended reach.
            kw = {}
            if (v := os.environ.get("TASKBENCH_CUROBO_POS_TOL")) is not None:
                kw["position_tolerance"] = float(v)
            if (v := os.environ.get("TASKBENCH_CUROBO_ORI_TOL")) is not None:
                kw["orientation_tolerance"] = float(v)
            self.curobo_planner = CuroboPlanner(self.env, **kw)
        self.objects = get_objects(self.env)
        self._build_skills()

    def _build_skills(self):
        """Create skill instances with current planner/objects."""
        kw = dict(
            robot_config=self.robot_config,
            objects=self.objects,
            step_callback=self.step_callback,
            curobo_planner=self.curobo_planner,
        )
        self.pick = Pick(self.env, self.planner, **kw)
        self.place = Place(self.env, self.planner, **kw)
        self.push = Push(self.env, self.planner, **kw)
        self.move = Move(self.env, self.planner, **kw)
