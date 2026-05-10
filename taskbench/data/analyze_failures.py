"""Per-failure explanation generator for the visualization set.

For each failure row referenced in the visualizations manifest, look up the
full parquet row, analyze the scene geometry around the target, and emit a
per-video markdown explanation tailored to the specific failure leg.

Run:
    uv run python -m taskbench.data.analyze_failures \
      --dataset '/common/users/shared/pracsys/.../task_*/shard_*.parquet' \
      --manifest /common/users/shared/pracsys/.../visualizations/manifest.csv \
      --out /common/users/shared/pracsys/.../visualizations/per_video_explanations.md
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


WORKSPACE_X = (-0.26, 0.26)
WORKSPACE_Y = (-0.26, 0.26)
GRIPPER_OPEN_HALF_SPAN = 0.04   # rough — finger separation when open ~ 8cm
NEIGHBOR_BRUSH_RADIUS = 0.06    # if a neighbor is within this xy of target,
                                # the descending fingers may brush it


def _row_block_poses(row: dict, n: int) -> tuple[np.ndarray, np.ndarray]:
    poses = np.zeros((n, 7), dtype=np.float64)
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        if bool(row.get(f"block_{i}_present", False)):
            poses[i] = [
                row[f"block_{i}_x"], row[f"block_{i}_y"], row[f"block_{i}_z"],
                row[f"block_{i}_qw"], row[f"block_{i}_qx"],
                row[f"block_{i}_qy"], row[f"block_{i}_qz"],
            ]
            mask[i] = True
    return poses, mask


def _nearest_neighbors(poses: np.ndarray, mask: np.ndarray, target_idx: int, k: int = 3):
    """Return list of (block_idx, dx, dy, distance) sorted by distance, excluding target."""
    target_xy = poses[target_idx, :2]
    rows = []
    for i in range(len(poses)):
        if not bool(mask[i]) or i == target_idx:
            continue
        dx = poses[i, 0] - target_xy[0]
        dy = poses[i, 1] - target_xy[1]
        d = float(np.hypot(dx, dy))
        rows.append((i, float(dx), float(dy), d))
    rows.sort(key=lambda r: r[3])
    return rows[:k]


def _distance_to_workspace_edges(target_xy: np.ndarray) -> dict[str, float]:
    return {
        "+x": WORKSPACE_X[1] - target_xy[0],
        "-x": target_xy[0] - WORKSPACE_X[0],
        "+y": WORKSPACE_Y[1] - target_xy[1],
        "-y": target_xy[1] - WORKSPACE_Y[0],
    }


def _archetype_from_template(poses, mask, target_idx) -> str:
    """Heuristic to infer templated archetype from the geometry."""
    n_real = int(mask.sum())
    if n_real == 1:
        return "edge_target"
    if n_real == 2:
        return "adjacent_obstacle"
    if n_real == 5:
        return "ringed_target"
    return f"unknown_n_real={n_real}"


def _explain_failure(row: dict, n: int) -> str:
    """Return a markdown paragraph explaining why this specific row failed."""
    poses, mask = _row_block_poses(row, n)
    target_idx = int(row["target_idx"])
    target_xy = poses[target_idx, :2]
    nbrs = _nearest_neighbors(poses, mask, target_idx, k=3)
    edges = _distance_to_workspace_edges(target_xy)
    leg = row["failed_leg"]
    source = row["source"]
    n_real = int(mask.sum())
    wall = float(row["wall_time_s"])

    lines = []
    lines.append(f"**Scene type:** `{source}`"
                 + (f" — likely `{_archetype_from_template(poses, mask, target_idx)}`"
                    if source == "templated" else "")
                 + f", {n_real} real blocks, target at xy = "
                 f"({target_xy[0]:+.3f}, {target_xy[1]:+.3f}).")

    if nbrs:
        nbr_strs = []
        for (i, dx, dy, d) in nbrs[:2]:
            nbr_strs.append(f"block {i} at Δxy=({dx:+.3f}, {dy:+.3f}), {d*100:.1f} cm away")
        lines.append("**Nearest neighbors:** " + "; ".join(nbr_strs) + ".")
    else:
        lines.append("**Nearest neighbors:** none (target is the only block).")

    closest_edge_name = min(edges, key=edges.get)
    closest_edge_d = edges[closest_edge_name]
    lines.append(f"**Workspace clearance:** "
                 f"closest edge is `{closest_edge_name}` at {closest_edge_d*100:.1f} cm; "
                 f"x-range distance to robot side ({WORKSPACE_X[0]}) = {edges['-x']*100:.1f} cm.")

    lines.append(f"**Wall time:** {wall:.2f} s.")
    lines.append("")

    if leg == "reach":
        # Reach failure: either plan_screw failed (target far/unreachable)
        # or contact-monitor tripped during the trajectory.
        if wall < 0.5:
            lines.append(
                "**Why it failed:** wall time is short (<0.5 s), which means "
                "`plan_screw` rejected the trajectory before any execution — "
                "no `env.step` was ever called. The pre-grasp pose is "
                "geometrically unreachable from the current arm configuration "
                "(IK has no solution, or the straight-line screw motion would "
                "violate joint limits or clip the table point cloud). "
            )
            if closest_edge_d < 0.05:
                lines.append(
                    f"The target sits {closest_edge_d*100:.1f} cm from the `{closest_edge_name}` "
                    f"workspace boundary, which puts it near the limit of the Panda's "
                    f"reachable manifold for top-down grasps. "
                )
            else:
                lines.append(
                    "The target is well inside the workspace, so the IK "
                    "infeasibility likely comes from the wrist orientation: the "
                    "OBB-derived closing axis at this object's yaw forces a wrist "
                    "configuration the planner can't reach in a single screw motion."
                )
        else:
            lines.append(
                "**Why it failed:** wall time is non-trivial, indicating "
                "`follow_path` started executing the reach trajectory and was "
                "aborted by the contact monitor (>0.01 N gripper-finger contact "
                "with a non-robot entity). On the way to the pre-grasp pose the "
                "open gripper brushed against a block. "
            )
            if nbrs and nbrs[0][3] < NEIGHBOR_BRUSH_RADIUS * 1.5:
                lines.append(
                    f"The nearest non-target block ({nbrs[0][3]*100:.1f} cm away) "
                    "lies along the approach corridor; even though the planner "
                    "didn't know about it (only the table is in mplib's "
                    "collision world), the simulated arm trajectory passed "
                    "through that volume."
                )
    elif leg == "grasp_approach":
        # Grasp-approach failure dominates. Almost always contact-monitor.
        lines.append(
            "**Why it failed:** the robot reached the pre-grasp pose 5 cm "
            "above the target, then descended along `-Z` toward the grasp "
            "pose. During that 5 cm descent the contact monitor recorded "
            "≥0.01 N on a gripper finger. Because mplib's planner only knows "
            "the table (not the source blocks), `plan_screw` accepted a "
            "trajectory that brushes against another block at execution time."
        )
        if nbrs and nbrs[0][3] < NEIGHBOR_BRUSH_RADIUS:
            (i, dx, dy, d) = nbrs[0]
            lines.append(
                f" The likely culprit is block {i}, only {d*100:.1f} cm away "
                f"in direction (Δx={dx:+.3f}, Δy={dy:+.3f}) — well inside the "
                f"~6 cm finger-clearance envelope for an open gripper, so as "
                f"the fingers descend they make contact with this neighbor "
                f"before reaching the target's bounding box."
            )
        elif nbrs:
            (i, dx, dy, d) = nbrs[0]
            lines.append(
                f" Nearest block is {d*100:.1f} cm away (Δxy={dx:+.3f}, {dy:+.3f}). "
                "Even at this larger separation the failure is consistent with a "
                "near-miss contact: the OBB-derived grasp pose is not perfectly "
                "centered, so a tilted finger or the narrow finger tip nicks the "
                "neighbor or the target's edge during descent."
            )
        else:
            lines.append(
                " No neighbors are nearby — this is consistent with a marginal "
                "contact between a finger and the *target* itself (e.g., the "
                "OBB-derived grasp pose is offset from the cube's true center "
                "by a few millimeters, so a finger tip clips the cube on the "
                "way down)."
            )
    elif leg == "grasp_verify":
        lines.append(
            "**Why it failed:** all motion phases succeeded — the robot "
            "descended cleanly to the grasp pose and closed the gripper. "
            "After closing, `agent.is_grasping(target)` returned False. "
            "ManiSkill's `is_grasping` requires both fingers to be pressing "
            "the *specific* target actor with sufficient bilateral contact "
            "force. A few mechanisms produce this leg:"
        )
        lines.append(
            "- The cube slipped out laterally during finger closure "
            "(contact forces on a corner can eject a small cube)."
        )
        lines.append(
            "- The grasp pose was offset from the cube's true center, so on "
            "closing one finger met the cube and the other passed by it."
        )
        if wall > 5:
            lines.append(
                f"- Wall time of {wall:.1f} s is unusually long for a "
                "grasp_verify leg, suggesting the planner re-tried multiple "
                "yaw candidates or reach attempts before settling on one "
                "that descended cleanly — the cube may have been disturbed "
                "by sub-threshold contacts during those earlier attempts."
            )
    elif leg == "lift":
        lines.append(
            "**Why it failed:** the gripper closed and `is_grasping` "
            "returned True (the cube *is* held). The lift step then tried "
            "to move the grasp pose +12 cm in world `+Z`. With contact "
            "monitoring off, the only way `move_to_pose` returns failure "
            "is `plan_screw` rejecting the trajectory — i.e., a kinematic "
            "issue:"
        )
        lines.append(
            "- The yaw committed during grasp search left the wrist near a "
            "joint-limit boundary; lifting another 12 cm pushes joint 6 or 7 "
            "past its limit."
        )
        lines.append(
            "- The straight-up screw motion is singular for this configuration "
            "(Jacobian rank-deficient), so mplib cannot synthesize a parameterizable "
            "joint trajectory."
        )
        if nbrs and nbrs[0][3] < 0.05:
            lines.append(
                f"- Note: the nearest neighbor is only {nbrs[0][3]*100:.1f} cm "
                "away, but contact monitoring is off during lift, so this is "
                "*not* the primary cause — the cube was successfully grasped "
                "before lift was attempted."
            )
    elif leg == "timeout":
        lines.append(
            f"**Why it failed:** the SIGALRM-based 60 s budget around `ctx.pick` "
            f"fired (wall time = {wall:.1f} s). The pick was still attempting "
            "skill phases when the runner aborted it. Likely cause:"
        )
        if source == "templated":
            lines.append(
                "- This templated scene presents many obstacle geometries; the "
                "planner repeatedly tries-and-fails yaw candidates and reach paths, "
                "each `plan_screw` call costing seconds, until the budget exhausts."
            )
        elif source == "canonical":
            lines.append(
                "- This canonical-program snapshot has multiple blocks already "
                "placed at grid targets, plus the remaining unplaced blocks. The "
                "dense scene means many planning retries before any phase "
                "succeeds. Cumulative planning time exceeds the 60 s budget "
                "before pick can complete."
            )
        else:
            lines.append(
                "- A randomly-sampled configuration where the geometry happens to "
                "force many planner retries (lots of contact aborts on reach + "
                "grasp_approach phases that re-trigger the yaw search). The "
                "60 s wall budget is consumed by accumulated planning + retries."
            )

    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Glob for parquet shards")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--grid-rows", type=int, default=3)
    p.add_argument("--grid-cols", type=int, default=3)
    args = p.parse_args()

    # Read manifest
    manifest_rows = []
    with open(args.manifest, "r") as f:
        for r in csv.DictReader(f):
            manifest_rows.append(r)

    failure_rows = [r for r in manifest_rows if r["category"].startswith("fail_")]
    print(f"manifest: {len(manifest_rows)} rows ({len(failure_rows)} failures)")

    # Build scene_id -> manifest row map
    by_scene = {r["scene_id"]: r for r in failure_rows}

    # Load dataset and select matching rows
    files = sorted(glob.glob(args.dataset))
    print(f"reading {len(files)} shards ...")
    table = pa.concat_tables([pq.read_table(f) for f in files])
    print(f"loaded {table.num_rows} rows; matching ...")

    # Filter to just our failure scene_ids (faster than full to_pylist)
    scene_ids = pa.array(list(by_scene.keys()))
    mask = pa.compute.is_in(table.column("scene_id"), value_set=scene_ids)
    sub = table.filter(mask)
    print(f"matched {sub.num_rows} rows")

    full_rows = {r["scene_id"]: r for r in sub.to_pylist()}

    n = args.grid_rows * args.grid_cols
    sections: list[str] = []
    sections.append("# Per-video failure explanations\n")
    sections.append(
        "Each subsection corresponds to one video under "
        "`visualizations/<category>/video_<scene_id>.mp4` "
        "(plus its matching `snapshot_*.png`). The text is generated "
        "from the dataset row's exact scene geometry, target index, "
        "and timing.\n"
    )

    by_cat: dict[str, list] = {}
    for r in failure_rows:
        by_cat.setdefault(r["category"], []).append(r)

    cat_order = [
        "fail_reach",
        "fail_grasp_approach",
        "fail_grasp_verify",
        "fail_lift",
        "fail_timeout",
    ]
    for cat in cat_order:
        if cat not in by_cat:
            continue
        sections.append(f"\n## {cat}\n")
        for mr in by_cat[cat]:
            scene_id = mr["scene_id"]
            full = full_rows.get(scene_id)
            if full is None:
                sections.append(f"### {scene_id}\n\n*(no matching dataset row found)*\n")
                continue
            sections.append(f"### {scene_id}")
            sections.append("")
            sections.append(_explain_failure(full, n))
            sections.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(sections))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
