# Shared manipulation skills

This guide covers `taskbench.skills`, used by `stack_cubes`, `replay`,
`demo_recorder`, and other skill-based solvers. RoboVerify has a separate
Pick/Move/Release runtime; its selectable controller is described in the
[Stack guide](roboverify.md#choose-the-move-controller).

## SkillContext and control selection

```python
from taskbench.skills.context import SkillContext

ctx = SkillContext(env)
ctx.reset(seed=42)
pick_result = ctx.pick("cube_1", lift_height=0.15)
```

Use one CPU environment. `reset()` resets the scene, creates an mplib planner,
discovers objects, and binds new skill instances. `ctx.objects` maps names to
actors. The registered robot configurations currently cover `panda` and
`panda_wristcam`.

Pick, Place, and Move default to `naive=True`. Their behavior depends on the
environment control mode:

| Control mode | Default motion implementation |
| --- | --- |
| `pd_ee_delta_pose`, `pd_ee_delta_pos` | Cartesian position feedback in `naive_ee_control.py` |
| `pd_joint_pos`, `pd_joint_pos_vel` | OBB-based grasp selection and mplib screw planning |

`stack_cubes` defaults to `pd_ee_delta_pose`. Select its planner path with:

```bash
uv run python -m taskbench.run solver=stack_cubes \
  env.control_mode=pd_joint_pos env.max_episode_steps=1000 run.num_episodes=1
```

Setting `naive=False` alone does not select mplib in a delta-control environment:
the lower-level `move_to_pose()` also detects delta control and uses TCP position
feedback, ignoring the requested orientation. Shared skills do not acquire
RoboVerify's 2 mm stopping rule or 14 cm reset spacing through this setting.

A `step_callback` supplied to `SkillContext` is bound to skills at reset.
If it is assigned after reset, call `ctx._build_skills()` to rebind the existing
skills, as the current Stack recorder does. See the
[recording example](demos.md#recording-shared-skills).

## Poses and relative destinations

A `PoseLike` is a `sapien.Pose` or a `(position, quaternion)` tuple.
Positions and offsets are in world-frame metres; quaternions use
`[w, x, y, z]`.

Move and Place also accept an object name and a world-frame XYZ offset. This
form resolves the object's position at call time and retains the current TCP
orientation:

```python
# Move the TCP 15 cm above cube_0, keeping the gripper closed.
move_result = ctx.move("cube_0", [0, 0, 0.15], gripper_open=False)

# With default delta feedback, place the held 40 mm cube onto cube_0.
place_result = ctx.place("cube_0", [0, 0, 0.04], retract_height=0.2)
```

The Place target has different semantics across its two paths: delta feedback
targets the **held object's center**, while the planner path targets the **TCP**.
For a planner placement, account for the measured cube-to-TCP offset when
constructing the target pose. Move always targets the TCP in these shared skills.

## Pick

Signature:

```text
ctx.pick(obj_name, *, lift_height=0.1, verify_grasp=True,
         naive=True, naive_params=None)
```

Pick grasps a named object and lifts it. `lift_height` is the additional height
above the initial object/grasp position.

- Delta feedback uses the pick/place feedback loop with release disabled.
- The planner path computes a grasp from the object's oriented bounding box
  and probes six rotations about Z before approaching, closing, and lifting.
- `verify_grasp=True` checks `agent.is_grasping()`.
- The held box is attached to the shared planner's collision model.
- `naive_params` accepts `NaiveEEParams` from `naive_ee_control.py`.

`PickResult` includes `success`, `failure_reason`, `grasp_pose`,
`lift_pose`, and `obj_size`. In the delta branch the returned poses describe
the post-lift TCP; do not assume `grasp_pose` records the initial contact pose.

## Place

Signature:

```text
ctx.place(target_pose_or_cube, offsets=None, *, settling_steps=10,
          retract_height=None, naive=True, naive_params=None)
```

Place moves the held object to its destination, opens the gripper, detaches the
shared planner's payload, settles, and retracts. The delta path discovers the
held object through `agent.is_grasping()`; it fails if none is grasped.

`settling_steps` controls additional steps after release.
`retract_height` is an absolute world Z coordinate; the default is target
Z + 0.1 m. A failed final retract currently logs a warning and still returns
`PlaceResult(success=True)`. Evaluate the task separately.

Use `target_pose_or_cube` as the keyword when recording or calling this method.

## Move

Signature:

```text
ctx.move(target_pose_or_cube, offsets=None, *, gripper_open=True,
         monitor_contacts=True, naive=True, naive_gain=10.0,
         naive_tol=0.008, naive_max_steps=200)
```

Move targets the TCP. The default delta branch servos its position with the
given gain, tolerance, and step budget. The joint-control path uses
`plan_screw()` and executes its trajectory. Planning failure returns
`MoveResult(success=False)`; there is no RRT detour search.

`monitor_contacts` applies to planner trajectory execution. It checks configured
gripper links for contacts above 0.01 N and aborts on a detected contact.
It is not a whole-scene collision monitor for the delta feedback loop.

## Push

Signature:

```text
ctx.push(approach_pose, push_pose, *, clearance_height=0.1, lift_height=0.1)
```

Push lifts by `clearance_height`, closes the gripper, approaches, sweeps to
`push_pose`, lifts by `lift_height`, and opens. It calls the shared Move
implementation with `naive=False`; the environment control mode still determines
the lower-level controller. Contact monitoring is disabled during the sweep.
A failed final lift logs a warning and does not make the Push result fail.

## Results and lower-level helpers

Every result has `success`, optional `failure_reason`, and optional
`step_result` containing the last Gym step tuple. Pick adds the pose/size fields
above. A skill result describes that skill's execution, not formal verification
or overall task success.

`RobotConfig` stores the move group, finger length, gripper-link names, and
open/closed action values. `SkillContext` selects it from
`env.unwrapped.agent.uid`.

| Function in `taskbench/skills/motion.py` | Purpose |
| --- | --- |
| `setup_planner(env, robot_config)` | Construct mplib planner and table point cloud |
| `move_to_pose(env, planner, pose, gripper_state, robot_config, ...)` | Delta feedback or screw-plan execution; `dry_run=True` returns a plan in joint mode |
| `follow_path(env, result, gripper_state, robot_config, ...)` | Execute a joint trajectory |
| `actuate_gripper(env, planner, gripper_state, steps=6, ...)` | Hold the arm while opening/closing |
| `build_action(env, qpos, gripper_state, qvel=None)` | Format actions for the selected control mode |
| `attach_object(planner, size, pose=None)`, `detach_object(planner)` | Update the held-box collision model |
| `add_collision_boxes(planner, boxes, resolution=0.01)` | Explicitly add obstacle geometry |

The default planner registers a table point cloud, positioned 2 cm below the
table top. Loose blocks and the tower are not automatically added as obstacles.
Shared Pick attaches its held box; RoboVerify's planner backend currently does
not register a held-box collision object.

The shared screw helper defaults to a 0.1 rad joint integration step.
RoboVerify explicitly uses 0.01 rad with one 0.005 rad retry and its own endpoint
checks. That retry does not resolve all numerical planning failures; see the
[planner collection results](roboverify-validation.md#planner-collection-2026-09-28).
