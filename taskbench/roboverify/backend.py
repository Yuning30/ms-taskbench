"""ManiSkill execution and snapshots for the copied RoboVerify algorithms.

All positions and offsets are world-frame metres. The compact observation keeps
RoboVerify's position slots; unused feature slots are zero, not simulator state.
Full actor, articulation, controller, and drive state is archived separately.
"""

from copy import deepcopy

import numpy as np

from taskbench.roboverify.stack_reset import (
    BLOCK_LENGTH,
    DEFAULT_SEPARATION,
    sample_layout,
)
from taskbench.roboverify.util import on
from taskbench.roboverify.util.actions import bound_delta_action

BACKEND_ID = "maniskill-stack-v1"
REPLAY_ATOL = 1e-5


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.array(value, copy=True)


def flatten_state(state, prefix="state"):
    result = {}
    for name, value in state.items():
        key = f"{prefix}/{name}"
        if isinstance(value, dict):
            result.update(flatten_state(value, key))
        else:
            result[key] = array(value)
    return result


class StackBackend:
    """Single CPU Panda environment exposing RoboVerify's primitive interface."""

    def __init__(self, num_blocks=3, *, separation=DEFAULT_SEPARATION):
        import gymnasium as gym

        import taskbench.envs  # noqa: F401

        if not 2 <= num_blocks <= 6:
            raise ValueError("StackNCube supports 2 through 6 cubes")
        if not np.isclose(on.get_block_length(), BLOCK_LENGTH):
            raise ValueError("Use using_block_length(0.04) with StackNCube")
        self.num_blocks = num_blocks
        self.separation = separation
        self.layout_sampling = {}
        self.simulator = gym.make(
            "StackNCube-v1",
            num_cubes=num_blocks,
            robot_uids="panda",
            robot_init_qpos_noise=0.0,
            obs_mode="state",
            reward_mode="sparse",
            control_mode="pd_ee_delta_pose",
            num_envs=1,
            sim_backend="cpu",
            render_backend="cpu",
            render_mode="rgb_array",
            max_episode_steps=10000,
        )
        self.raw = self.simulator.unwrapped
        self.symbolic_name_to_box_id = {}
        self.gripper_command = 1.0
        self.held_box_id = None
        self.table_surface_height = 0.0
        self.env = self
        self.unwrapped = self
        try:
            self.reset(seed=0)
        except Exception:
            self.close()
            raise

    def reset(self, *, seed=None):
        from mani_skill.utils.structs.pose import Pose

        # Current search deliberately controls numpy's global RNG per demo.
        if seed is None:
            seed = int(np.random.randint(0, 2**31))
        xy, statistics = sample_layout(self.num_blocks, self.separation, seed)
        # Reset physics and retain the original per-seed cube yaw, then install
        # the tested layout before collection's normal 50 settling steps.
        self.simulator.reset(seed=seed)
        for cube, position in zip(self.raw.cubes, xy):
            cube.set_pose(
                Pose.create_from_pq(
                    p=np.r_[position, BLOCK_LENGTH / 2], q=array(cube.pose.q)
                )
            )
        self.gripper_command = 1.0
        self.held_box_id = None
        self.symbolic_name_to_box_id = {}
        self.layout_sampling = dict(
            seed=seed,
            separation_m=self.separation,
            xy=xy.tolist(),
            **statistics,
        )
        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.zeros(13 + 15 * self.num_blocks, dtype=float)
        obs[:3] = array(self.raw.agent.tcp.pose.p).reshape(-1)[:3]
        obs[3:5] = array(self.raw.agent.robot.get_qpos()).reshape(-1)[-2:]
        for i, cube in enumerate(self.raw.cubes):
            obs[10 + 12 * i : 13 + 12 * i] = array(cube.pose.p).reshape(-1)[:3]
        return obs

    @staticmethod
    def flatten_observation(obs):
        return np.asarray(obs, dtype=float).copy()

    def step(self, action):
        action = bound_delta_action(action)
        # RoboVerify's controller assumes 5 cm per bounded Cartesian command;
        # Panda's normalized delta controller uses 10 cm. Both use world/root
        # aligned translation here, with a fixed upright robot base.
        command = np.zeros(7, dtype=np.float32)
        command[:3] = action[:3] * 0.5
        if action[3] > 0.05:
            self.gripper_command = 1.0
        elif action[3] < -0.05:
            self.gripper_command = -1.0
        command[-1] = self.gripper_command
        _, reward, terminated, truncated, info = self.simulator.step(command)
        return self._get_obs(), reward, terminated, truncated, info

    def capture(self):
        from taskbench.roboverify.cfg.reset import Snapshot

        arrays = flatten_state(self.raw.get_state_dict())
        arrays["adapter/gripper"] = np.asarray(self.gripper_command)
        arrays["adapter/held"] = np.asarray(
            -1 if self.held_box_id is None else self.held_box_id
        )
        arrays["adapter/elapsed"] = array(self.raw._elapsed_steps)
        # ManiSkill's get_controller_state() is empty for non-target delta
        # controllers; retain their drive targets and interpolation state too.
        for name, controller in self.raw.agent.controller.controllers.items():
            for field in (
                "_start_qpos",
                "_target_qpos",
                "_step",
                "_step_size",
                "_target_pose",
            ):
                value = getattr(controller, field, None)
                if value is not None:
                    arrays[f"drive/{name}/{field}"] = array(
                        value.raw_pose if hasattr(value, "raw_pose") else value
                    )
        return Snapshot(
            array(self.raw.get_state()).reshape(-1),
            arrays,
            deepcopy(self.symbolic_name_to_box_id),
        )

    def restore(self, snapshot):
        import torch
        from mani_skill.utils.structs.pose import Pose

        state = {}
        for key, value in snapshot.arrays.items():
            if not key.startswith("state/"):
                continue
            names = key.split("/")[1:]
            parent = state
            for name in names[:-1]:
                parent = parent.setdefault(name, {})
            parent[names[-1]] = torch.as_tensor(value.copy(), device=self.raw.device)
        if not {"actors", "articulations"} <= state.keys():
            raise ValueError(
                "Archive has no ManiSkill simulation state; recollect demonstrations"
            )
        self.raw.set_state_dict(state)
        self.raw.agent.set_controller_state(state.get("controller", {}))
        for key, value in snapshot.arrays.items():
            if key.startswith("drive/"):
                _, name, field = key.split("/")
                data = torch.as_tensor(value.copy(), device=self.raw.device)
                if field == "_step":
                    data = int(data.item())
                elif field == "_target_pose":
                    data = Pose.create(data)
                setattr(self.raw.agent.controller.controllers[name], field, data)
        for controller in self.raw.agent.controller.controllers.values():
            if getattr(controller, "_target_qpos", None) is not None:
                controller.set_drive_targets(controller._target_qpos)
        self.raw._elapsed_steps = torch.as_tensor(
            snapshot.arrays["adapter/elapsed"].copy(), device=self.raw.device
        )
        self.restore_runtime_state(snapshot)
        return self._get_obs()

    def restore_runtime_state(self, snapshot):
        """Restore instruction bookkeeping after replaying only physical actions."""
        self.gripper_command = float(snapshot.arrays["adapter/gripper"])
        held = int(snapshot.arrays["adapter/held"])
        self.held_box_id = None if held < 0 else held
        self.symbolic_name_to_box_id = deepcopy(snapshot.bindings)

    def task_success(self):
        return bool(self.raw.evaluate()["success"].item())

    def gripper_ready(self, opened, box_id=None):
        if opened:
            # Panda fingers travel to 40 mm each. A rotated 40 mm cube can
            # already exceed Fetch's 53 mm threshold while still held.
            return float(self._get_obs()[3:5].sum()) >= 0.075
        return box_id is not None and bool(
            self.raw.agent.is_grasping(self.raw.cubes[box_id]).item()
        )

    def render(self, mode="rgb_array"):
        frame = array(self.simulator.render())
        # ManiSkill keeps the environment batch dimension even for one Panda.
        if frame.ndim == 4 and frame.shape[0] == 1:
            frame = frame[0]
        return frame

    def close(self):
        self.simulator.close()
