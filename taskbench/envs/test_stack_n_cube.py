"""Stack acceptance covers geometry and release across vectorized scenes."""

import unittest
from types import SimpleNamespace

import torch

from taskbench.envs.stack_n_cube import StackNCubeEnv


class Cube:
    def __init__(self, positions, velocities, angular_velocities, grasped):
        self.pose = SimpleNamespace(p=positions)
        self.velocities = velocities
        self.angular_velocities = angular_velocities
        self.grasped = grasped

    def is_static(self, lin_thresh, ang_thresh):
        return (self.velocities.norm(dim=1) <= lin_thresh) & (
            self.angular_velocities.norm(dim=1) <= ang_thresh
        )


class StackAcceptanceTests(unittest.TestCase):
    def check_acceptance(self, num_cubes):
        # Seven independent scenes: moving tower, stationary tower, held tower,
        # wrong base, horizontal gap, vertical gap, and valid reordered cubes.
        positions = torch.zeros((7, num_cubes, 3))
        positions[:, :, 2] = 0.02 + 0.04 * torch.arange(num_cubes)
        positions[3, [0, 1]] = positions[3, [1, 0]].clone()
        positions[4, -1, 0] = 0.04
        positions[5, -1, 2] += 0.01
        positions[6, [1, 2]] = positions[6, [2, 1]].clone()
        velocities = torch.zeros_like(positions)
        angular_velocities = torch.zeros_like(positions)
        velocities[0, :, 2] = 0.1
        angular_velocities[0, :, 2] = 1.0
        grasped = torch.zeros((7, num_cubes), dtype=torch.bool)
        grasped[2, -1] = True
        cubes = [
            Cube(
                positions[:, i],
                velocities[:, i],
                angular_velocities[:, i],
                grasped[:, i],
            )
            for i in range(num_cubes)
        ]
        env = SimpleNamespace(
            cubes=cubes,
            num_cubes=num_cubes,
            num_envs=7,
            device=torch.device("cpu"),
            cube_half_size=torch.tensor([0.02, 0.02, 0.02]),
            agent=SimpleNamespace(is_grasping=lambda cube: cube.grasped),
        )
        result = StackNCubeEnv.evaluate(env)
        self.assertEqual(
            result["success"].tolist(), [True, True, False, False, False, False, True]
        )
        self.assertFalse(result["all_static"][0].item())
        self.assertTrue(result["all_static"][1].item())

    def test_three_cube_acceptance(self):
        self.check_acceptance(3)

    def test_four_cube_acceptance(self):
        self.check_acceptance(4)


if __name__ == "__main__":
    unittest.main()
