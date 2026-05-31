"""Shared base classes and result dataclasses for manipulation skills.

Both the mplib (``mplib_primitives``) and cuRobo (``curobo_primitives``)
backends import from here so they share a common ``Skill`` ABC and result
types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import sapien
from transforms3d.euler import euler2quat, quat2euler

from taskbench.skills.robot_config import RobotConfig, get_robot_config


def _axis_aligned_quat_from_tcp(raw) -> list[float]:
    """Snap the current TCP orientation to the nearest axis-aligned quaternion.

    Reads ``raw.agent.tcp.pose.q`` (SAPIEN: ``[w, x, y, z]``), converts to
    Euler, rounds each angle to the nearest 90 degrees, converts back.
    """
    tcp_q = raw.agent.tcp.pose.q
    try:
        tcp_q = tcp_q.cpu().numpy()
    except Exception:
        tcp_q = np.asarray(tcp_q)
    tcp_q = np.asarray(tcp_q, dtype=np.float64).flatten()[:4]

    ai, aj, ak = quat2euler(tcp_q)
    step = np.pi / 2.0
    ai_s = float(np.round(ai / step) * step)
    aj_s = float(np.round(aj / step) * step)
    ak_s = float(np.round(ak / step) * step)
    q = euler2quat(ai_s, aj_s, ak_s)
    q = np.asarray(q, dtype=np.float64).flatten()[:4]
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
        planner: mplib.Planner instance. cuRobo-backed skills additionally
            accept ``curobo_planner`` to route motion through cuRobo while
            still using mplib for gripper helpers.
        robot_config: Robot-specific constants. If None, auto-detected from
            the env's agent.
        objects: Dict mapping string names to scene actors. Skills that need
            actors (e.g. Pick) resolve names through this.
        step_callback: Optional callable invoked after each ``env.step()``.
    """

    def __init__(self, env, planner, *, robot_config: Optional[RobotConfig] = None,
                 objects: Optional[dict[str, object]] = None,
                 step_callback: Optional[Callable] = None):
        self.env = env
        self.planner = planner
        self.robot_config = robot_config or get_robot_config(env)
        self.objects = objects or {}
        self.step_callback = step_callback

    @abstractmethod
    def __call__(self, *args, **kwargs) -> SkillResult:
        ...
