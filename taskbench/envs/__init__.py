import os

# c12: optionally stiffen the Panda's wrist joints (5, 6, 7) at PD-controller
# init. The c9 slip set is dominated by wrist drift during gripper close at
# extended-reach configurations where the Jacobian is near-singular; stiffer
# wrist PD lets the controller fight back faster.
#   TASKBENCH_PANDA_WRIST_STIFFNESS_MULT - multiplier on default 1e3 stiffness
#                                          for joints 5,6,7. Default 1.0.
#   TASKBENCH_PANDA_WRIST_DAMPING_MULT   - multiplier on default 1e2 damping
#                                          for joints 5,6,7. Default 1.0.
_wrist_k = float(os.environ.get("TASKBENCH_PANDA_WRIST_STIFFNESS_MULT", "1.0"))
_wrist_d = float(os.environ.get("TASKBENCH_PANDA_WRIST_DAMPING_MULT", "1.0"))
if _wrist_k != 1.0 or _wrist_d != 1.0:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
    # ManiSkill defaults: arm_stiffness=1e3 (scalar), arm_damping=1e2 (scalar).
    # Replace with per-joint arrays so joints 5,6,7 use bumped gains.
    _k_base = 1e3
    _d_base = 1e2
    _stiff = [_k_base] * 4 + [_k_base * _wrist_k] * 3
    _damp = [_d_base] * 4 + [_d_base * _wrist_d] * 3
    Panda.arm_stiffness = _stiff
    Panda.arm_damping = _damp
    PandaWristCam.arm_stiffness = _stiff
    PandaWristCam.arm_damping = _damp

# c15: gripper finger PD gains (defaults 1e3 / 1e2). Multiplies both fingers
# uniformly. Softer = gentler close, less impulse on the cube; stiffer = faster
# close, more squeeze force.
_grip_k = float(os.environ.get("TASKBENCH_PANDA_GRIPPER_STIFFNESS_MULT", "1.0"))
_grip_d = float(os.environ.get("TASKBENCH_PANDA_GRIPPER_DAMPING_MULT", "1.0"))
if _grip_k != 1.0 or _grip_d != 1.0:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
    Panda.gripper_stiffness = 1e3 * _grip_k
    Panda.gripper_damping = 1e2 * _grip_d
    PandaWristCam.gripper_stiffness = 1e3 * _grip_k
    PandaWristCam.gripper_damping = 1e2 * _grip_d

import taskbench.envs.bin_with_objects  # noqa: F401 — triggers env registration
import taskbench.envs.build2d  # noqa: F401 — triggers env registration
import taskbench.envs.shelf_env  # noqa: F401 — triggers env registration
import taskbench.envs.stack_cube_distractor  # noqa: F401 — triggers env registration
import taskbench.envs.stack_n_cube  # noqa: F401 — triggers env registration


def get_objects(env) -> dict[str, object]:
    """Get the name→actor mapping from any supported env.

    Prefers the env's own ``get_objects()`` method. Falls back to
    a convention-based lookup for built-in ManiSkill envs that don't
    implement it (e.g. StackCube-v1).
    """
    raw = env.unwrapped
    if hasattr(raw, "get_objects"):
        return raw.get_objects()

    # Built-in StackCube-v1: cubeB is base (green), cubeA stacks on top (red)
    if hasattr(raw, "cubeA") and hasattr(raw, "cubeB"):
        return {"cube_0": raw.cubeB, "cube_1": raw.cubeA}

    raise NotImplementedError(
        f"No get_objects() on {type(raw).__name__} and no known fallback"
    )
