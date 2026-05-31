"""Skill context — bundles env + planners + objects for skill execution.

Eliminates the boilerplate that every solver repeats::

    ctx = SkillContext(env, step_callback=recorder.record)
    ctx.reset(seed=42)

    ctx.pick("cube_1", lift_height=0.15)
    ctx.place(target_pose)

By default ``ctx.pick``/``place``/``move``/``push`` route through the
cuRobo-backed primitives (:mod:`taskbench.skills.curobo_primitives`). The
mplib variants (:mod:`taskbench.skills.mplib_primitives`) remain importable.

The cuRobo planner warmup is ~5–10 s, so we build it once on first
``reset()`` and reuse the instance across episodes; the per-episode scene is
re-synced inside ``CuroboPick`` via ``sync_scene``.
"""

from typing import Callable, Optional

from taskbench.envs import get_objects
from taskbench.skills.curobo_planner import CuroboPlanner
from taskbench.skills.curobo_primitives import (
    CuroboMove,
    CuroboPick,
    CuroboPlace,
    CuroboPush,
)
from taskbench.skills.motion import setup_planner
from taskbench.skills.robot_config import RobotConfig, get_robot_config

_NOT_READY_MSG = "SkillContext.reset() must be called before using skills"


class _SkillProxy:
    """Raises a clear error when skills are accessed before reset()."""

    def __call__(self, *args, **kwargs):
        raise RuntimeError(_NOT_READY_MSG)

    def __getattr__(self, name):
        raise RuntimeError(_NOT_READY_MSG)


class SkillContext:
    """Shared context for skill-based solvers.

    Holds the env, motion planners (mplib + cuRobo), robot config, object
    references, and pre-bound skill instances. Call ``reset()`` to
    re-initialize everything for a new episode.

    Args:
        env: Gym env (num_envs=1, sim_backend="cpu").
        step_callback: Optional callable invoked after each ``env.step()``
            (e.g. ``recorder.record``).
    """

    def __init__(self, env, *, step_callback: Optional[Callable] = None):
        self.env = env
        self.step_callback = step_callback
        self.robot_config: RobotConfig = get_robot_config(env)
        self.planner = None
        self.curobo_planner: Optional[CuroboPlanner] = None
        self.objects: dict[str, object] = {}

        _proxy = _SkillProxy()
        self.pick: CuroboPick = _proxy  # type: ignore[assignment]
        self.place: CuroboPlace = _proxy  # type: ignore[assignment]
        self.push: CuroboPush = _proxy  # type: ignore[assignment]
        self.move: CuroboMove = _proxy  # type: ignore[assignment]

    def reset(self, seed=None):
        """Reset the env and rebuild planners, objects, and skills."""
        self.env.reset(seed=seed)
        self.planner = setup_planner(self.env, self.robot_config)
        if self.curobo_planner is None:
            self.curobo_planner = CuroboPlanner(self.env)
        self.objects = get_objects(self.env)
        self._build_skills()

    def _build_skills(self):
        """Create cuRobo-backed skill instances with current planners/objects."""
        kw = dict(
            robot_config=self.robot_config,
            objects=self.objects,
            step_callback=self.step_callback,
            curobo_planner=self.curobo_planner,
        )
        self.pick = CuroboPick(self.env, self.planner, **kw)
        self.place = CuroboPlace(self.env, self.planner, **kw)
        self.push = CuroboPush(self.env, self.planner, **kw)
        self.move = CuroboMove(self.env, self.planner, **kw)
