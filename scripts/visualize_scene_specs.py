"""Render top-down visualizations of every scene-spec category and archetype.

Usage:
    uv run python scripts/visualize_scene_specs.py [--out DIR]

Writes:
    <out>/random_scenes.png
    <out>/canonical_snapshots.png
    <out>/templated_adjacent_obstacle.png
    <out>/templated_edge_target.png
    <out>/templated_ringed_target.png
    <out>/all_categories_overview.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mp
import matplotlib.pyplot as plt
import matplotlib.transforms as mt
import numpy as np

from taskbench.data.scene_specs import (
    BLOCK_HALF,
    WORKSPACE_X,
    WORKSPACE_Y,
    canonical_program_snapshot_specs,
    random_scene_spec,
    templated_scene_specs,
)


GRID_ORIGIN_X = 0.05
GRID_SPACING = 0.07
PANDA_BASE_XY = (-0.615, 0.0)  # approximate; just for context arrow


def _grid_target_xys(rows: int, cols: int):
    origin_y = -((cols - 1) * GRID_SPACING) / 2.0
    for i in range(rows):
        for j in range(cols):
            yield GRID_ORIGIN_X + i * GRID_SPACING, origin_y + j * GRID_SPACING


def _yaw_from_quat(qw: float, qz: float) -> float:
    """Recover yaw assuming pure-yaw quaternion (qx=qy=0)."""
    return 2.0 * np.arctan2(qz, qw)


def draw_scene(ax, spec, *, title: str = "", show_grid_targets: bool = True,
               show_workspace: bool = True, target_label: str = "T"):
    """Top-down render of a SceneSpec."""
    if show_workspace:
        ax.add_patch(mp.Rectangle(
            (WORKSPACE_X[0], WORKSPACE_Y[0]),
            WORKSPACE_X[1] - WORKSPACE_X[0],
            WORKSPACE_Y[1] - WORKSPACE_Y[0],
            fill=False, edgecolor="gray", linestyle="--", linewidth=0.6,
        ))
    if show_grid_targets:
        for tx, ty in _grid_target_xys(spec.grid_rows, spec.grid_cols):
            ax.add_patch(mp.Circle((tx, ty), 0.012,
                                   facecolor="#cfe9ff", edgecolor="#5a9fd9",
                                   linewidth=0.6, alpha=0.7, zorder=1))

    side = 2 * BLOCK_HALF
    for i in range(spec.block_poses.shape[0]):
        if not bool(spec.block_mask[i]):
            continue
        x, y = spec.block_poses[i, 0], spec.block_poses[i, 1]
        qw, qz = spec.block_poses[i, 3], spec.block_poses[i, 6]
        yaw = _yaw_from_quat(qw, qz)

        is_target = (spec.target_idx == i)
        face = "#d62728" if is_target else "#ff9f4a"   # red vs orange
        edge = "#7a0d0d" if is_target else "#5a3a14"

        rect = mp.Rectangle((-side / 2, -side / 2), side, side,
                            facecolor=face, edgecolor=edge,
                            linewidth=1.0, alpha=0.92, zorder=3)
        rect.set_transform(mt.Affine2D().rotate(yaw).translate(x, y) + ax.transData)
        ax.add_patch(rect)
        if is_target:
            ax.text(x, y, target_label, color="white", ha="center", va="center",
                    fontsize=8, fontweight="bold", zorder=4)

    # Mark the robot base for context (just an annotation, not the actual robot footprint).
    ax.scatter([0.0], [0.0], marker="P", s=60, c="#222", zorder=2)
    ax.annotate("robot\nbase", (0.0, 0.0), xytext=(0.005, -0.015),
                fontsize=6, color="#444")

    ax.set_xlim(-0.32, 0.32)
    ax.set_ylim(-0.30, 0.30)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.15, linewidth=0.4)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def figure_random(out_path: Path, seeds=(0, 17, 42, 123)):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, s in zip(axes.flat, seeds):
        spec = random_scene_spec(seed=s, grid_rows=3, grid_cols=3)
        draw_scene(ax, spec, title=f"random  seed={s}  target=block_{spec.target_idx}")
    fig.suptitle("Category: random — 9 blocks, random non-overlapping xy + yaw",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def figure_canonical(out_path: Path, seed=11, steps=(0, 2, 4, 8)):
    specs = canonical_program_snapshot_specs(seed=seed, grid_rows=3, grid_cols=3)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, k in zip(axes.flat, steps):
        spec = specs[k]
        draw_scene(ax, spec,
                   title=f"canonical  step={k}  target=block_{spec.target_idx}\n"
                         f"({k} placed, {9 - k} still on table)")
    fig.suptitle("Category: canonical — snapshots of the row-major Build2D program",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def figure_templated_archetype(out_path: Path, archetype: str, seed=0,
                               select_steps=(0, 3, 6, 11), title_suffix=""):
    specs = [s for s in templated_scene_specs(seed=seed, grid_rows=3, grid_cols=3)
             if s.metadata["archetype"] == archetype]
    select_steps = [k for k in select_steps if k < len(specs)]
    fig, axes = plt.subplots(1, len(select_steps), figsize=(4 * len(select_steps), 4.5))
    if len(select_steps) == 1:
        axes = [axes]
    for ax, k in zip(axes, select_steps):
        spec = specs[k]
        sweep_step = spec.metadata["sweep_step"]
        draw_scene(ax, spec, show_grid_targets=False,
                   title=f"sweep_step={sweep_step}  n_blocks={spec.metadata['n_real']}")
    fig.suptitle(f"Category: templated — archetype: {archetype}{title_suffix}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def figure_overview(out_path: Path, seed=0):
    """One-panel-per-category quick reference."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    draw_scene(axes[0, 0], random_scene_spec(seed=42, grid_rows=3, grid_cols=3),
               title="random")
    canon = canonical_program_snapshot_specs(seed=11, grid_rows=3, grid_cols=3)
    draw_scene(axes[0, 1], canon[4], title="canonical (step 4 of 9)")
    draw_scene(axes[0, 2], canon[8], title="canonical (step 8 of 9)")
    tpl = templated_scene_specs(seed=0, grid_rows=3, grid_cols=3)
    by_arch = {}
    for s in tpl:
        by_arch.setdefault(s.metadata["archetype"], []).append(s)
    draw_scene(axes[1, 0], by_arch["adjacent_obstacle"][3],
               title=f"templated · adjacent_obstacle (d={0.045 + 3*(0.12-0.045)/11:.3f} m)",
               show_grid_targets=False)
    draw_scene(axes[1, 1], by_arch["edge_target"][6],
               title=f"templated · edge_target (x={0.18 + 6*(0.255-0.18)/9:.3f} m)",
               show_grid_targets=False)
    draw_scene(axes[1, 2], by_arch["ringed_target"][2],
               title=f"templated · ringed_target (r={0.045 + 2*(0.10-0.045)/9:.3f} m)",
               show_grid_targets=False)
    fig.suptitle("Scene-spec categories — one representative each",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("outputs/visualizations"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    figure_random(args.out / "random_scenes.png")
    figure_canonical(args.out / "canonical_snapshots.png")
    figure_templated_archetype(
        args.out / "templated_adjacent_obstacle.png",
        archetype="adjacent_obstacle",
        select_steps=(0, 3, 7, 11),
        title_suffix="  (obstacle distance d sweeps 0.045 → 0.120 m)",
    )
    figure_templated_archetype(
        args.out / "templated_edge_target.png",
        archetype="edge_target",
        select_steps=(0, 3, 6, 9),
        title_suffix="  (target x sweeps 0.180 → 0.255 m toward workspace edge)",
    )
    figure_templated_archetype(
        args.out / "templated_ringed_target.png",
        archetype="ringed_target",
        select_steps=(0, 3, 6, 9),
        title_suffix="  (ring radius r sweeps 0.045 → 0.100 m)",
    )
    figure_overview(args.out / "all_categories_overview.png")

    print(f"wrote 6 figures under {args.out}/")


if __name__ == "__main__":
    main()
