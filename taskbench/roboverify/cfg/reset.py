"""Faithful segment starts: full simulator snapshots and deterministic replay."""

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Snapshot:
    gt_state: np.ndarray
    arrays: dict
    bindings: dict


def inner_env(env):
    return env.unwrapped if hasattr(env, "unwrapped") else env


def capture(env):
    inner = inner_env(env)
    return inner.capture()


def restore(env, snapshot):
    inner = inner_env(env)
    return inner.restore(snapshot)


@dataclass
class Recording:
    snapshots: list = field(default_factory=list)
    action_indices: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    states: list = field(default_factory=list)
    events: list = field(default_factory=list)


def collect_recording(
    program,
    env,
    *,
    recording=None,
    initial_observation=None,
    max_loop_iterations=None,
    on_frame=None,
):
    """Record the same execution used for learning and optional video rendering.

    A caller-supplied Recording retains the partial execution if the program raises.
    Frames are captured initially and after actions; rendering never adds actions.
    """
    recording = Recording() if recording is None else recording
    original_step = env.step
    invocations = {}

    def step(action):
        result = original_step(action)
        recording.actions.append(np.array(action, copy=True))
        if on_frame is not None:
            on_frame(env)
        return result

    def on_state(obs):
        actual = inner_env(env).flatten_observation(inner_env(env)._get_obs())
        if not np.allclose(obs, actual, atol=1e-8, rtol=0):
            raise ValueError(
                "Recorded observation does not match its simulator snapshot"
            )
        recording.states.append(np.array(obs, copy=True))
        recording.snapshots.append(capture(env))
        recording.action_indices.append(len(recording.actions))
        if len(recording.states) == 1 and on_frame is not None:
            on_frame(env)

    def on_event(event):
        if event["kind"] == "loop_enter":
            invocations[event["path"]] = invocations.get(event["path"], -1) + 1
        event["invocation"] = invocations.get(event["path"])
        recording.events.append(deepcopy(event))

    env.step = step
    try:
        program.eval(
            env,
            on_state=on_state,
            on_event=on_event,
            initial_observation=initial_observation,
            max_loop_iterations=max_loop_iterations,
        )
    finally:
        env.step = original_step
    return tuple(recording.states), recording


def reset_segment(env, segment, *, mode="replay"):
    """Default to replay; callers may opt into measured exact reset."""
    trace = segment.trace
    if mode not in ("reset", "replay"):
        raise ValueError("Unknown segment reset mode")
    if mode == "reset" and trace.snapshots:
        observation = restore(env, trace.snapshots[segment.t_start])
    elif trace.snapshots and trace.actions:
        observation = restore(env, trace.snapshots[0])
        actions, indices = trace.actions
        for action in actions[: indices[segment.t_start]]:
            observation = env.step(np.array(action, copy=True))[0]
        # Symbolic assignments are not actions; reinstate recorded aliases.
        inner_env(env).restore_runtime_state(trace.snapshots[segment.t_start])
    else:
        raise ValueError(
            "Full simulator snapshots and action recordings are required; recollect demonstrations"
        )
    inner_env(env).symbolic_name_to_box_id.update(segment.bindings)
    from taskbench.roboverify.backend import REPLAY_ATOL

    if not np.allclose(
        observation, trace.states[segment.t_start], atol=REPLAY_ATOL, rtol=0
    ):
        raise ValueError(
            "Segment replay/reset did not reproduce its recorded observation"
        )
    return observation
