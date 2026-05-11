"""2D block building environment for ManiSkill3.

Task: place one movable block at every target location of an m x n grid.
The target grid is also exposed as a 2D linked list via right/down pointers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
import sapien
import torch
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SceneConfig, SimConfig
from taskbench.envs.base import TaskEnv


@dataclass
class GridNode:
    """Linked-list cell used by the Build2D DSL runtime."""

    name: str
    x: float
    y: float
    z: float
    r: "GridNode | None" = None
    d: "GridNode | None" = None


@register_env("Build2D-v1", max_episode_steps=300)
class Build2DEnv(TaskEnv):
    """Build an m x n block grid by placing blocks on target markers."""

    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    SUPPORTED_REWARD_MODES = ["sparse", "none"]
    agent: Union[Panda, Fetch]

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0.0,
        grid_rows: int = 3,
        grid_cols: int = 3,
        **kwargs,
    ):
        if grid_rows <= 0 or grid_cols <= 0:
            raise ValueError("grid_rows and grid_cols must be positive integers.")
        self.grid_rows = int(grid_rows)
        self.grid_cols = int(grid_cols)
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            scene_config=SceneConfig(
                solver_position_iterations=20,
                solver_velocity_iterations=4,
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien.Pose(p=[-0.45, 0, 0.42], q=[0.9238795, 0, 0.3826834, 0])
        return CameraConfig("render_camera", pose, 768, 768, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        self.block_half_size = common.to_tensor([0.02, 0.02, 0.02], device=self.device)
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.target_nodes: list[GridNode] = []
        self.target_coords: dict[str, tuple[float, float, float]] = {}
        self.target_markers = []
        self.blocks = []

        spacing = 0.07
        origin_x = 0.05
        origin_y = -((self.grid_cols - 1) * spacing) / 2.0
        target_z = float(self.block_half_size[2].item())

        # Visible target markers and linked-list nodes (right / down).
        node_matrix: list[list[GridNode]] = []
        for i in range(self.grid_rows):
            row: list[GridNode] = []
            for j in range(self.grid_cols):
                x = origin_x + i * spacing
                y = origin_y + j * spacing
                name = f"target_{i}_{j}"
                marker = actors.build_cube(
                    self.scene,
                    half_size=0.012,
                    color=[0.2, 0.6, 1.0, 0.35],
                    name=name,
                    body_type="kinematic",
                    initial_pose=sapien.Pose(p=[x, y, 0.012]),
                )
                self.target_markers.append(marker)

                node = GridNode(name=name, x=float(x), y=float(y), z=float(target_z))
                self.target_nodes.append(node)
                self.target_coords[name] = (node.x, node.y, node.z)
                row.append(node)
            node_matrix.append(row)

        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                if j + 1 < self.grid_cols:
                    node_matrix[i][j].r = node_matrix[i][j + 1]
                if i + 1 < self.grid_rows:
                    node_matrix[i][j].d = node_matrix[i + 1][j]
        self.grid_head = node_matrix[0][0]

        # Movable blocks (one per cell), initialized high and re-positioned on reset.
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                block = actors.build_cube(
                    self.scene,
                    half_size=0.02,
                    color=[0.9, 0.5, 0.1, 1.0],
                    name=f"block_{i}_{j}",
                    initial_pose=sapien.Pose(p=[-0.2, 0, 0.2]),
                )
                self.blocks.append(block)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            overrides = (options or {}).get("block_overrides")
            if overrides is not None:
                arr = np.asarray(overrides, dtype=np.float32)
                if arr.shape != (len(self.blocks), 7):
                    raise ValueError(
                        f"block_overrides shape {arr.shape}; expected "
                        f"({len(self.blocks)}, 7)"
                    )
                for k, block in enumerate(self.blocks):
                    p = torch.tensor(arr[k, :3], device=self.device).expand(b, 3).clone()
                    q = torch.tensor(arr[k, 3:], device=self.device).expand(b, 4).clone()
                    block.set_pose(Pose.create_from_pq(p=p, q=q))
                return

            sampler = randomization.UniformPlacementSampler(
                bounds=[[-0.26, -0.2], [-0.12, 0.2]],
                batch_size=b,
                device=self.device,
            )
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001
            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.02
            for block in self.blocks:
                xy = sampler.sample(radius, 100, verbose=False)
                xyz[:, :2] = xy
                qs = randomization.random_quaternions(
                    b, lock_x=True, lock_y=True, lock_z=False
                )
                block.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

    def get_objects(self) -> dict[str, object]:
        return {f"block_{i}": b for i, b in enumerate(self.blocks)}

    def get_grid_head(self) -> GridNode:
        return self.grid_head

    def get_available_block_names(self) -> list[str]:
        return [f"block_{i}" for i in range(len(self.blocks))]

    def evaluate(self):
        target_xy = torch.tensor(
            [[self.target_nodes[k].x, self.target_nodes[k].y] for k in range(len(self.target_nodes))],
            device=self.device,
            dtype=torch.float32,
        )  # (T, 2)
        target_z = torch.tensor(
            [self.target_nodes[k].z for k in range(len(self.target_nodes))],
            device=self.device,
            dtype=torch.float32,
        )  # (T,)
        pos = torch.stack([b.pose.p for b in self.blocks], dim=1)  # (B, N, 3)
        pos_xy = pos[..., :2]  # (B, N, 2)
        pos_z = pos[..., 2]  # (B, N)

        diffs = pos_xy[:, :, None, :] - target_xy[None, None, :, :]
        dists = torch.linalg.norm(diffs, dim=-1)  # (B, N, T)
        min_dist_per_target, _ = torch.min(dists, dim=1)  # (B, T)
        close_enough = min_dist_per_target <= 0.025

        z_diffs = torch.abs(pos_z[:, :, None] - target_z[None, None, :])  # (B, N, T)
        min_z_diff_per_target, _ = torch.min(z_diffs, dim=1)  # (B, T)
        z_ok = min_z_diff_per_target <= 0.02

        all_targets_occupied = torch.all(close_enough & z_ok, dim=1)
        all_static = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        any_grasped = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        for block in self.blocks:
            all_static &= block.is_static(lin_thresh=1e-2, ang_thresh=0.5)
            any_grasped |= self.agent.is_grasping(block)
        success = all_targets_occupied & all_static & (~any_grasped)
        return {
            "all_targets_occupied": all_targets_occupied,
            "all_static": all_static,
            "any_grasped": any_grasped,
            "success": success.bool(),
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            for idx, block in enumerate(self.blocks):
                obs[f"block_{idx}_pose"] = block.pose.raw_pose
        return obs

    def compute_sparse_reward(self, obs, action, info):
        return info["success"].float()
