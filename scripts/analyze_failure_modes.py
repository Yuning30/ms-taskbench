"""Failure-mode characterization for the c9+c18 production controller.

Combines the two 500-scene evals (in-sample seed=12345, OOS seed=98765) for
1000 total scenes. For each failure mode:

  - Classify and count
  - Compute geometric statistics (target reach, neighbor count, ...)
  - Render representative top-down scenes
  - Identify sub-mechanisms

Outputs go to outputs/controller_eval/analysis/.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

OUT_DIR = Path("/common/home/st1122/Projects/ms-taskbench/outputs/controller_eval/analysis_c21")
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ROBOT_BASE_X = -0.615  # world x of Panda base
ROBOT_REACH = 0.85     # workspace-gate threshold (robot frame)
BLOCK_HALF = 0.02      # cube half-size

# Colors per outcome
OUTCOME_COLOR = {
    "success": "#2ca02c",            # green
    "out_of_workspace": "#7f7f7f",   # grey
    "grasp_plan_failed": "#ff7f0e",  # orange
    "grasp_verification_failed": "#d62728",  # red
    "post_lift_slip": "#9467bd",     # purple
}
OUTCOME_LABEL = {
    "success": "success",
    "out_of_workspace": "out-of-workspace (c18)",
    "grasp_plan_failed": "plan-failed",
    "grasp_verification_failed": "close-time slip",
    "post_lift_slip": "post-lift slip",
}


def load_scenes(paths):
    """Load + merge multiple eval parquets, attach geometric features."""
    rows = []
    for src_path in paths:
        t = pq.read_table(src_path)
        n = len(t)
        for i in range(n):
            tgt = t.column("target_idx")[i].as_py()
            ok = t.column("success")[i].as_py()
            fr = t.column("failure_reason")[i].as_py()
            if ok:
                outcome = "success"
            else:
                outcome = fr or "unknown"

            tx = t.column(f"block_{tgt}_x")[i].as_py()
            ty = t.column(f"block_{tgt}_y")[i].as_py()
            tz = t.column(f"block_{tgt}_z")[i].as_py()
            tqw = t.column(f"block_{tgt}_qw")[i].as_py()
            tqz = t.column(f"block_{tgt}_qz")[i].as_py()

            # Target yaw in world frame (rotation around z)
            target_yaw = float(np.arctan2(2 * tqw * tqz, 1 - 2 * tqz * tqz))

            # Robot-frame target x
            tx_robot = tx - ROBOT_BASE_X

            # Neighbors at various radii (excluding target, excluding parked)
            nbr_dists = []
            for j in range(9):
                if j == tgt:
                    continue
                if not t.column(f"block_{j}_present")[i].as_py():
                    continue
                ox = t.column(f"block_{j}_x")[i].as_py()
                oy = t.column(f"block_{j}_y")[i].as_py()
                oz = t.column(f"block_{j}_z")[i].as_py()
                if oz < 0.0:
                    continue
                d = float(((ox - tx) ** 2 + (oy - ty) ** 2) ** 0.5)
                nbr_dists.append(d)
            nbrs_3 = sum(1 for d in nbr_dists if d < 0.03)
            nbrs_5 = sum(1 for d in nbr_dists if d < 0.05)
            nbrs_7 = sum(1 for d in nbr_dists if d < 0.07)
            min_d = float(min(nbr_dists)) if nbr_dists else float("inf")

            # Pull all block poses for rendering
            block_poses = []
            for j in range(9):
                bp = dict(
                    present=t.column(f"block_{j}_present")[i].as_py(),
                    x=t.column(f"block_{j}_x")[i].as_py(),
                    y=t.column(f"block_{j}_y")[i].as_py(),
                    z=t.column(f"block_{j}_z")[i].as_py(),
                    qw=t.column(f"block_{j}_qw")[i].as_py(),
                    qz=t.column(f"block_{j}_qz")[i].as_py(),
                    idx=j,
                )
                block_poses.append(bp)

            rows.append(dict(
                scene_id=t.column("scene_id")[i].as_py(),
                source=t.column("source")[i].as_py(),
                target_idx=tgt,
                success=ok,
                outcome=outcome,
                tx=tx, ty=ty, tz=tz, target_yaw=target_yaw,
                tx_robot=tx_robot,
                nbrs_3=nbrs_3, nbrs_5=nbrs_5, nbrs_7=nbrs_7,
                min_nbr_d=min_d,
                wall_time_s=t.column("wall_time_s")[i].as_py(),
                blocks=block_poses,
                src_file=Path(src_path).stem,
            ))
    return rows


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def plot_outcome_scatter(rows, save_path: Path):
    """Top-down scatter of target positions, colored by outcome."""
    fig, ax = plt.subplots(figsize=(7, 7))
    # workspace bounds approx (env spec: WORKSPACE_X/Y = (-0.26, 0.26) in world)
    ax.add_patch(mpatches.Rectangle((-0.26, -0.26), 0.52, 0.52,
                                    fill=False, edgecolor="black", linestyle=":",
                                    linewidth=1.0, label="grid workspace"))
    # Reach circle from the robot base in robot frame (transformed to world)
    theta = np.linspace(0, 2 * np.pi, 200)
    rx = ROBOT_BASE_X + ROBOT_REACH * np.cos(theta)
    ry = ROBOT_REACH * np.sin(theta)
    ax.plot(rx, ry, color="grey", linestyle="--", linewidth=0.8,
            label=f"workspace gate (robot-x={ROBOT_REACH}m)")
    # Robot base
    ax.plot([ROBOT_BASE_X], [0.0], "bs", markersize=10, label="robot base")
    # Targets
    for outcome in ["success", "grasp_plan_failed",
                    "grasp_verification_failed", "post_lift_slip",
                    "out_of_workspace"]:
        xs = [r["tx"] for r in rows if r["outcome"] == outcome]
        ys = [r["ty"] for r in rows if r["outcome"] == outcome]
        ax.scatter(xs, ys, c=OUTCOME_COLOR[outcome], s=14, alpha=0.6,
                   label=f"{OUTCOME_LABEL[outcome]} (n={len(xs)})")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_title("Target positions colored by outcome (1000 scenes)")
    ax.set_aspect("equal")
    ax.set_xlim(-0.7, 0.35)
    ax.set_ylim(-0.35, 0.35)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)


def plot_reach_histograms(rows, save_path: Path):
    """Stacked histogram of robot-frame target x by outcome."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(0.34, 0.92, 0.02)
    bottoms = np.zeros(len(bins) - 1)
    for outcome in ["success", "grasp_verification_failed", "post_lift_slip",
                    "grasp_plan_failed", "out_of_workspace"]:
        xs = [r["tx_robot"] for r in rows if r["outcome"] == outcome]
        hist, _ = np.histogram(xs, bins=bins)
        ax.bar(bins[:-1], hist, width=0.02, bottom=bottoms,
               color=OUTCOME_COLOR[outcome], align="edge", alpha=0.85,
               label=f"{OUTCOME_LABEL[outcome]} ({len(xs)})", edgecolor="white",
               linewidth=0.4)
        bottoms += hist
    ax.axvline(ROBOT_REACH, color="black", linestyle="--", linewidth=0.8)
    ax.text(ROBOT_REACH + 0.005, ax.get_ylim()[1] * 0.95,
            f"c18 gate\nx={ROBOT_REACH}m", fontsize=8, va="top")
    ax.set_xlabel("target distance from robot base in x (m, robot frame)")
    ax.set_ylabel("scenes")
    ax.set_title("Outcome composition vs reach distance (1000 scenes)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)


def plot_reach_failure_rates(rows, save_path: Path):
    """Failure rate (per type) as a function of reach distance bin."""
    bins = np.arange(0.35, 0.92, 0.05)
    fig, ax = plt.subplots(figsize=(8, 5))
    centers = (bins[:-1] + bins[1:]) / 2
    total_per_bin = np.zeros(len(bins) - 1)
    rates = {k: np.zeros(len(bins) - 1) for k in OUTCOME_COLOR if k != "success"}
    for r in rows:
        b = np.searchsorted(bins, r["tx_robot"], side="right") - 1
        if 0 <= b < len(bins) - 1:
            total_per_bin[b] += 1
            if r["outcome"] != "success":
                rates[r["outcome"]][b] += 1
    for outcome, arr in rates.items():
        # convert to percentage of bin total
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = np.where(total_per_bin > 0, 100 * arr / total_per_bin, 0)
        ax.plot(centers, pct, "o-", color=OUTCOME_COLOR[outcome], linewidth=2,
                markersize=6, label=OUTCOME_LABEL[outcome])
    ax.set_xlabel("target reach (m, robot frame)")
    ax.set_ylabel("failure rate (% of scenes in bin)")
    ax.set_title("Per-failure-mode rate vs reach (1000 scenes)")
    ax.legend(fontsize=9)
    ax.set_xlim(0.35, 0.92)
    ax.axvline(ROBOT_REACH, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
    # Annotate scene counts at the top
    ax2 = ax.twinx()
    ax2.bar(centers, total_per_bin, width=0.04, color="lightblue", alpha=0.2)
    ax2.set_ylabel("scenes in bin", color="grey", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="grey", labelsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)


def plot_neighbor_histograms(rows, save_path: Path):
    """Histogram of nearest-neighbor distance, grouped by outcome."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for ax, outcome in zip(axes.flatten(), ["success", "grasp_verification_failed",
                                            "post_lift_slip",
                                            "grasp_plan_failed", "out_of_workspace"]):
        ds = [r["min_nbr_d"] for r in rows
              if r["outcome"] == outcome and np.isfinite(r["min_nbr_d"])]
        n_iso = sum(1 for r in rows if r["outcome"] == outcome
                    and not np.isfinite(r["min_nbr_d"]))
        ax.hist(ds, bins=np.arange(0.0, 0.20, 0.005),
                color=OUTCOME_COLOR[outcome], edgecolor="white", linewidth=0.4)
        ax.axvline(0.041, color="red", linestyle=":", linewidth=0.8)  # cube touching = 4.1cm
        ax.text(0.045, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 1,
                "cube-touching\n(4.1cm)", fontsize=7, va="top")
        ax.set_title(f"{OUTCOME_LABEL[outcome]}  (n={len(ds) + n_iso}; "
                     f"{n_iso} isolated)")
        ax.set_xlabel("nearest-neighbor distance (m)")
        ax.set_ylabel("scenes")
    fig.suptitle("Distance to nearest other block, by outcome (1000 scenes)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------
# Per-scene rendering
# ----------------------------------------------------------------------

def _rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def _cube_corners(cx, cy, yaw, half=BLOCK_HALF):
    """4 world-frame corners of a top-down rotated square."""
    pts = np.array([[-half, -half], [+half, -half], [+half, +half], [-half, +half]])
    rot = _rot_z(yaw)
    pts = (rot @ pts.T).T + np.array([cx, cy])
    return pts


def render_scene(ax, row, *, title=None):
    """Top-down render of one scene on an axis."""
    # Workspace box
    ax.add_patch(mpatches.Rectangle((-0.26, -0.26), 0.52, 0.52,
                                    fill=False, edgecolor="black",
                                    linestyle=":", linewidth=0.6))
    # Reach arc (only the segment in the visible region)
    theta = np.linspace(-np.pi / 2.5, np.pi / 2.5, 100)
    rx = ROBOT_BASE_X + ROBOT_REACH * np.cos(theta)
    ry = ROBOT_REACH * np.sin(theta)
    ax.plot(rx, ry, color="grey", linestyle="--", linewidth=0.6)
    # Robot base
    ax.plot([ROBOT_BASE_X], [0.0], "bs", markersize=8)
    # Blocks
    target_idx = row["target_idx"]
    for bp in row["blocks"]:
        if not bp["present"]:
            continue
        if bp["z"] < 0.0:
            continue
        # extract yaw from quaternion
        yaw = float(np.arctan2(2 * bp["qw"] * bp["qz"],
                               1 - 2 * bp["qz"] * bp["qz"]))
        pts = _cube_corners(bp["x"], bp["y"], yaw)
        is_target = bp["idx"] == target_idx
        if is_target:
            color = OUTCOME_COLOR[row["outcome"]]
            edge = "black"; lw = 1.5
        else:
            color = "#9bc3e6"; edge = "grey"; lw = 0.5
        poly = mpatches.Polygon(pts, closed=True, facecolor=color,
                                edgecolor=edge, linewidth=lw, alpha=0.85)
        ax.add_patch(poly)
        ax.text(bp["x"], bp["y"], str(bp["idx"]), ha="center", va="center",
                fontsize=6, color="black" if is_target else "grey")
    ax.set_aspect("equal")
    ax.set_xlim(-0.70, 0.32)
    ax.set_ylim(-0.32, 0.32)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def render_failure_mode_grid(rows, outcome: str, scenes: list, save_path: Path,
                             title: str, suptitle_extra: str = ""):
    """Render N scenes on a 2-column grid for a single failure mode."""
    n = len(scenes)
    cols = 2
    rows_ = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_, cols, figsize=(11, 4.5 * rows_))
    if rows_ == 1:
        axes = np.array([axes])
    for k, scene in enumerate(scenes):
        ax = axes.flatten()[k]
        ssub = (f"x_robot={scene['tx_robot']:.3f}m, "
                f"min_nbr={scene['min_nbr_d']:.3f}m" if np.isfinite(scene['min_nbr_d'])
                else f"x_robot={scene['tx_robot']:.3f}m, isolated")
        render_scene(ax, scene,
                     title=f"{scene['scene_id']}  ({scene['source']})\n{ssub}")
    for k in range(n, rows_ * cols):
        axes.flatten()[k].axis("off")
    fig.suptitle(f"{title}{suptitle_extra}", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    paths = [
        "/common/home/st1122/Projects/ms-taskbench/outputs/controller_eval/c21_correct_eval.parquet",
        "/common/home/st1122/Projects/ms-taskbench/outputs/controller_eval/c21_correct_eval_oos.parquet",
    ]
    rows = load_scenes(paths)
    print(f"loaded {len(rows)} scenes from {len(paths)} evals")

    # Print failure breakdown
    by_outcome = Counter(r["outcome"] for r in rows)
    print("\n=== Outcome breakdown ===")
    for k, v in sorted(by_outcome.items(), key=lambda x: -x[1]):
        print(f"  {k:32s} {v:>4d}  ({100 * v / len(rows):.1f}%)")

    # Per-source × outcome
    src_outcome = Counter((r["source"], r["outcome"]) for r in rows)
    print("\n=== Per-source × outcome ===")
    for src in ("random", "canonical", "templated"):
        total = sum(v for (s, o), v in src_outcome.items() if s == src)
        if total == 0:
            continue
        print(f"  {src} (n={total}):")
        for outcome in OUTCOME_COLOR:
            cnt = src_outcome.get((src, outcome), 0)
            print(f"    {outcome:28s} {cnt:>3d}  ({100 * cnt / total:.1f}%)")

    # ----- Figures -----
    print("\n=== Generating figures ===")
    plot_outcome_scatter(rows, FIG_DIR / "01_outcome_scatter.png")
    print("  01_outcome_scatter.png")
    plot_reach_histograms(rows, FIG_DIR / "02_reach_histogram.png")
    print("  02_reach_histogram.png")
    plot_reach_failure_rates(rows, FIG_DIR / "03_failure_rate_vs_reach.png")
    print("  03_failure_rate_vs_reach.png")
    plot_neighbor_histograms(rows, FIG_DIR / "04_neighbor_distance.png")
    print("  04_neighbor_distance.png")

    # ----- Per-mode scene grids -----
    rng = np.random.default_rng(42)

    # Out-of-workspace: sample 4 across reach range
    oow = sorted([r for r in rows if r["outcome"] == "out_of_workspace"],
                 key=lambda r: r["tx_robot"])
    if oow:
        # Take min, mid-1, mid-2, max
        picks = [oow[0], oow[len(oow) // 3], oow[2 * len(oow) // 3], oow[-1]]
        render_failure_mode_grid(
            rows, "out_of_workspace", picks,
            FIG_DIR / "05_oow_examples.png",
            "Out-of-workspace (c18) examples: target at robot-x > 0.85m",
            f"\n{len(oow)} total scenes in this failure mode",
        )
        print("  05_oow_examples.png")

    # Plan-failed: split by sub-type
    plan_fail = [r for r in rows if r["outcome"] == "grasp_plan_failed"]
    # Templated plan-fails (mostly ringed)
    pf_tpl = [r for r in plan_fail if r["source"] == "templated"]
    pf_canon = [r for r in plan_fail if r["source"] == "canonical"]
    pf_rand = [r for r in plan_fail if r["source"] == "random"]
    picks = []
    if pf_tpl: picks.extend(pf_tpl[:2])
    if pf_canon: picks.extend(pf_canon[:2])
    if pf_rand: picks.extend(pf_rand[:2])
    if picks:
        render_failure_mode_grid(
            rows, "grasp_plan_failed", picks,
            FIG_DIR / "06_plan_fail_examples.png",
            "Plan-failed in-workspace: cuRobo found no feasible trajectory",
            f"\n{len(plan_fail)} total ({len(pf_tpl)} templated / "
            f"{len(pf_canon)} canonical / {len(pf_rand)} random)",
        )
        print("  06_plan_fail_examples.png")

    # Slip: split by reach-stress vs alt-grasp
    slips = [r for r in rows if r["outcome"] == "grasp_verification_failed"]
    slip_reach = [r for r in slips if r["tx_robot"] > 0.60 and r["nbrs_5"] == 0]
    slip_crowd = [r for r in slips if r["nbrs_5"] >= 1]
    slip_other = [r for r in slips if r not in slip_reach and r not in slip_crowd]

    if slip_reach:
        picks = sorted(slip_reach, key=lambda r: r["tx_robot"])[-6:]
        render_failure_mode_grid(
            rows, "grasp_verification_failed", picks,
            FIG_DIR / "07_slip_reach_examples.png",
            "Slip: reach-stress mechanism (isolated target at extended reach)",
            f"\n{len(slip_reach)} of {len(slips)} slips have this profile "
            f"(no neighbors, x > 0.60m)",
        )
        print("  07_slip_reach_examples.png")

    if slip_crowd:
        picks = sorted(slip_crowd, key=lambda r: r["min_nbr_d"])[:6]
        render_failure_mode_grid(
            rows, "grasp_verification_failed", picks,
            FIG_DIR / "08_slip_crowded_examples.png",
            "Slip: crowded-scene mechanism (neighbor within 5cm)",
            f"\n{len(slip_crowd)} of {len(slips)} slips have this profile",
        )
        print("  08_slip_crowded_examples.png")

    # Post-lift slip: contact at close, slips out during the lift trajectory.
    # Visible only after c21's post-lift verification was added.
    post_slip = [r for r in rows if r["outcome"] == "post_lift_slip"]
    if post_slip:
        post_reach = [r for r in post_slip if r["tx_robot"] > 0.60 and r["nbrs_5"] == 0]
        post_crowd = [r for r in post_slip if r["nbrs_5"] >= 1]
        if post_reach:
            picks = sorted(post_reach, key=lambda r: r["tx_robot"])[-6:]
            render_failure_mode_grid(
                rows, "post_lift_slip", picks,
                FIG_DIR / "08b_post_lift_slip_reach.png",
                "Post-lift slip: cube held briefly then dropped during lift "
                "(reach-stress)",
                f"\n{len(post_reach)} of {len(post_slip)} post-lift slips have this profile",
            )
            print("  08b_post_lift_slip_reach.png")
        if post_crowd:
            picks = sorted(post_crowd, key=lambda r: r["min_nbr_d"])[:6]
            render_failure_mode_grid(
                rows, "post_lift_slip", picks,
                FIG_DIR / "08c_post_lift_slip_crowded.png",
                "Post-lift slip: cube dropped during lift in crowded scene",
                f"\n{len(post_crowd)} of {len(post_slip)} post-lift slips have this profile",
            )
            print("  08c_post_lift_slip_crowded.png")

    # ----- Edge case: scenes near the workspace boundary -----
    boundary = [r for r in rows
                if 0.82 <= r["tx_robot"] <= 0.88]
    boundary.sort(key=lambda r: r["tx_robot"])
    if boundary:
        # Pick 6 spanning the boundary
        idxs = np.linspace(0, len(boundary) - 1, 6).astype(int)
        picks = [boundary[i] for i in idxs]
        render_failure_mode_grid(
            rows, "boundary", picks,
            FIG_DIR / "09_boundary_examples.png",
            "Workspace-gate boundary: scenes at robot-x in [0.82, 0.88]",
            f"\n{len(boundary)} total scenes near the gate threshold (0.85m)",
        )
        print("  09_boundary_examples.png")
        # Tabulate outcomes by boundary subrange
        print("\n=== Boundary scenes (x in [0.82, 0.88]) ===")
        for lo, hi in [(0.82, 0.84), (0.84, 0.85), (0.85, 0.86), (0.86, 0.88)]:
            sub = [r for r in boundary if lo <= r["tx_robot"] < hi]
            oc = Counter(r["outcome"] for r in sub)
            print(f"  x in [{lo:.2f}, {hi:.2f}) — n={len(sub):>3d}  {dict(oc)}")

    # ----- Edge case: templated archetypes -----
    print("\n=== Templated archetype breakdown ===")
    tpl_outcomes = Counter()
    for r in rows:
        if r["source"] != "templated":
            continue
        # Classify by min neighbor count
        if r["nbrs_5"] == 0:
            arch = "edge_target (isolated)"
        elif r["nbrs_5"] >= 4:
            arch = "ringed_target (4 neighbors)"
        else:
            arch = "adjacent_obstacle (1+ neighbors)"
        tpl_outcomes[(arch, r["outcome"])] += 1
    for (arch, outcome), cnt in sorted(tpl_outcomes.items()):
        print(f"  {arch:36s}  {outcome:32s}  {cnt}")

    # Save the rows as a CSV for reproducibility
    import csv as _csv
    out_csv = OUT_DIR / "scene_records.csv"
    with out_csv.open("w") as f:
        w = _csv.writer(f)
        w.writerow(["scene_id", "src_file", "source", "outcome",
                    "tx", "ty", "tz", "tx_robot", "target_yaw",
                    "nbrs_3", "nbrs_5", "nbrs_7", "min_nbr_d",
                    "wall_time_s"])
        for r in rows:
            w.writerow([
                r["scene_id"], r["src_file"], r["source"], r["outcome"],
                r["tx"], r["ty"], r["tz"], r["tx_robot"], r["target_yaw"],
                r["nbrs_3"], r["nbrs_5"], r["nbrs_7"],
                "" if not np.isfinite(r["min_nbr_d"]) else r["min_nbr_d"],
                r["wall_time_s"],
            ])
    print(f"\nWrote {out_csv}")
    print(f"Wrote {FIG_DIR}/")


if __name__ == "__main__":
    main()
