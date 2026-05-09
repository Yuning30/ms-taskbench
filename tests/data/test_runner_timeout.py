"""Unit tests for per-sample pick timeout in runner.py.

No real env / ManiSkill loaded — everything is faked.
"""

from __future__ import annotations

import time
import types

import numpy as np
import pytest

from taskbench.data.runner import run_pick_sample
from taskbench.data.scene_specs import SceneSpec


# ---------------------------------------------------------------------------
# Minimal fakes
# ---------------------------------------------------------------------------

class _FakeObject:
    """Minimal object with pose.p / pose.q attributes."""

    class _Pose:
        p = np.array([0.0, 0.0, 0.02])
        q = np.array([1.0, 0.0, 0.0, 0.0])

    pose = _Pose()


class _FakeRobot:
    def get_qpos(self):
        return np.zeros(9)


class _FakeAgent:
    robot = _FakeRobot()


class _FakeUnwrapped:
    agent = _FakeAgent()

    def get_objects(self):
        return {"block_0": _FakeObject()}


class _FakeEnv:
    unwrapped = _FakeUnwrapped()

    def reset(self, *, seed=None, options=None):
        return None, {}


class _SlowCtx:
    """Fake SkillContext whose .pick() blocks longer than the timeout."""

    def __init__(self, sleep_s: float):
        self._sleep_s = sleep_s
        self.objects = {"block_0": _FakeObject()}
        self.planner = None
        self.env = _FakeEnv()

    def reset(self, seed):
        pass

    def _build_skills(self):
        pass

    def pick(self, name, *, lift_height=0.12):
        time.sleep(self._sleep_s)
        # Should not reach here in the timeout case.
        raise AssertionError("pick() completed — timeout did not fire")


def _make_spec() -> SceneSpec:
    n = 1
    poses = np.zeros((n, 7), dtype=np.float64)
    poses[0] = [0.0, 0.0, 0.02, 1.0, 0.0, 0.0, 0.0]
    mask = np.ones(n, dtype=bool)
    return SceneSpec(
        seed=0,
        grid_rows=1,
        grid_cols=1,
        block_poses=poses,
        block_mask=mask,
        target_idx=0,
        source="random",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_timeout_produces_failure_sample(monkeypatch):
    """A pick that sleeps 2 s must be killed within pick_timeout_s=0.2 s."""
    import taskbench.data.runner as runner_mod

    # Patch SkillContext import inside runner so we don't need the real one.
    # (run_pick_sample uses the ctx passed in directly; we only need to patch
    # the module-level name so that _alarm_supported check doesn't see a
    # threading issue — our ctx IS in main thread.)
    env = _FakeEnv()
    ctx = _SlowCtx(sleep_s=2.0)
    spec = _make_spec()

    sample = run_pick_sample(
        env, ctx, spec, scene_id="test-timeout-0001", pick_timeout_s=0.2
    )

    assert sample.success is False
    assert sample.failure_reason == "timeout"
    assert sample.failed_leg == "timeout"
    assert sample.final_object_pose is None
    assert sample.scene_id == "test-timeout-0001"
    # Wall time must be much less than the 2 s sleep.
    assert sample.wall_time_s < 1.0


def test_no_timeout_when_pick_is_fast(monkeypatch):
    """If pick completes quickly, no timeout fires and result is normal."""
    import taskbench.data.runner as runner_mod

    class _FastCtx:
        objects = {"block_0": _FakeObject()}
        planner = None
        env = _FakeEnv()

        def reset(self, seed):
            pass

        def _build_skills(self):
            pass

        def pick(self, name, *, lift_height=0.12):
            # Return a successful result immediately.
            result = types.SimpleNamespace(success=True, failure_reason=None)
            return result

    env = _FakeEnv()
    ctx = _FastCtx()
    spec = _make_spec()

    sample = run_pick_sample(
        env, ctx, spec, scene_id="test-fast-0001", pick_timeout_s=5.0
    )

    assert sample.success is True
    assert sample.failure_reason is None
    assert sample.failed_leg is None
