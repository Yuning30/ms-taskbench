"""
Build r*/d*/l0 matrices from a Z3 model (goals / GoalSort) and save a diagram PNG.

Used when high-level VC checks fail and Z3 returns sat so a counterexample model exists.
"""

from __future__ import annotations

import math
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from z3 import ModelRef, is_true

from taskbench.roboverify.verification_lib import highlevel_verification_lib as hlv


def _cover_edges(n: int, rel: Callable[[int, int], bool]) -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(n):
            if i == j or not rel(i, j) or rel(j, i):
                continue
            between = False
            for k in range(n):
                if k in (i, j):
                    continue
                if rel(i, k) and rel(k, j):
                    between = True
                    break
            if not between:
                edges.append((i, j))
    return edges


def _arrow(
    ax,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    *,
    color: str,
    lw: float = 2.2,
    mutation_scale: float = 20,
    zorder: int = 1,
    linestyle: str = "-",
) -> None:
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        return
    ux, uy = dx / norm, dy / norm
    off = 0.26
    a = (p0[0] + ux * off, p0[1] + uy * off)
    b = (p1[0] - ux * off, p1[1] - uy * off)
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=lw,
            color=color,
            zorder=zorder,
            linestyle=linestyle,
            shrinkA=0,
            shrinkB=0,
        )
    )


def _mid_label(
    ax,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    text: str,
    *,
    color: str,
    dy_off: float = 0.12,
) -> None:
    mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
    ax.text(
        mx,
        my - dy_off,
        text,
        fontsize=8,
        color=color,
        ha="center",
        va="bottom",
        fontweight="bold",
        zorder=4,
        bbox=dict(
            boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.9
        ),
    )


def _total_order_rank(n: int, rel: Callable[[int, int], bool]) -> List[int]:
    def less(a: int, b: int) -> bool:
        return a != b and rel(a, b) and not rel(b, a)

    order: List[int] = []
    rem = set(range(n))
    while rem:
        mins = [a for a in rem if not any(less(b, a) for b in rem)]
        pick = min(mins)
        order.append(pick)
        rem.remove(pick)
    rank = [0] * n
    for i, v in enumerate(order):
        rank[v] = i
    return rank


def _val_index(model: ModelRef, universe: Sequence, v) -> int:
    for i, u in enumerate(universe):
        if is_true(model.eval(u == v, model_completion=True)):
            return i
    raise ValueError(f"value not in Goal universe: {v}")


def build_goal_relation_matrices(
    context: hlv.HighLevelContext,
    model: ModelRef,
) -> Tuple[List, List[List[bool]], List[List[bool]], List[int], Dict[str, int]]:
    """
    Returns (universe, r_mat, d_mat, l0_of_index, named_indices)
    where named_indices maps const name -> index for known Goal constants in the model.
    """
    if context.GoalSort is None:
        raise ValueError("HighLevelContext must use verification_mode='goals'.")

    model_universe = model.get_universe(context.GoalSort)
    if model_universe is None:
        # EnumSort domains are finite and known a priori, but Z3 may report
        # `None` for get_universe on non-uninterpreted sorts.
        enum_goals = getattr(context, "enum_goals", None)
        if enum_goals:
            universe = list(enum_goals)
        else:
            raise ValueError(
                "Unable to obtain Goal universe from model. "
                "For enum-goal mode, ensure context.enum_goals is populated."
            )
    else:
        universe = list(model_universe)
    n = len(universe)
    r_mat: List[List[bool]] = []
    d_mat: List[List[bool]] = []
    for i in range(n):
        ri: List[bool] = []
        di: List[bool] = []
        for j in range(n):
            a, b = universe[i], universe[j]
            ri.append(is_true(model.eval(context.r_star(a, b), model_completion=True)))
            di.append(is_true(model.eval(context.d_star(a, b), model_completion=True)))
        r_mat.append(ri)
        d_mat.append(di)

    l0_of: List[int] = []
    for i in range(n):
        lv = model.eval(context.l0(universe[i]), model_completion=True)
        l0_of.append(_val_index(model, universe, lv))

    named: Dict[str, int] = {}
    for decl in model.decls():
        if decl.arity() != 0 or decl.range() != context.GoalSort:
            continue
        name = decl.name()
        if name not in ("h", "i", "j", "null"):
            continue
        v = model[decl]
        named[name] = _val_index(model, universe, v)

    return universe, r_mat, d_mat, l0_of, named


def draw_goal_model_panel(
    ax,
    *,
    r_mat: Sequence[Sequence[bool]],
    d_mat: Sequence[Sequence[bool]],
    l0_map: Sequence[int],
    named_indices: Dict[str, int],
    title: str,
) -> None:
    n = len(l0_map)
    r_rank = _total_order_rank(n, lambda a, b: r_mat[a][b])
    d_rank = _total_order_rank(n, lambda a, b: d_mat[a][b])
    pos = {i: (float(r_rank[i]), float(d_rank[i])) for i in range(n)}

    h_idx = named_indices.get("h")
    null_idx = named_indices.get("null")

    ax.set_title(title)
    ax.set_xlabel("layout x = r*-rank (blue: tail → head = increasing r*)")
    ax.set_ylabel("layout y = d*-rank (orange: tail → head = increasing d*)")
    ax.set_aspect("equal")

    for i in range(n):
        px, py = pos[i]
        if h_idx is not None and i == h_idx:
            fc, ew, ec = "lightyellow", 2.8, "#B8860B"
            txt = r"$h$" + "\n(head)"
        elif null_idx is not None and i == null_idx:
            fc, ew, ec = "lavender", 2.2, "#5E35B1"
            txt = "null"
        else:
            fc, ew, ec = "white", 1.1, "black"
            short = None
            for nm, ix in named_indices.items():
                if ix == i and nm not in ("h", "null"):
                    short = nm
                    break
            txt = short if short is not None else f"v{i}"
        ax.scatter([px], [py], s=950, c=fc, edgecolors=ec, linewidths=ew, zorder=2)
        ax.text(px, py, txt, ha="center", va="center", fontsize=9, zorder=3)

    r_covers = _cover_edges(n, lambda a, b: r_mat[a][b])
    for ei, (a, b) in enumerate(r_covers):
        _arrow(ax, pos[a], pos[b], color="#1565C0", lw=2.4, mutation_scale=22, zorder=1)
        if ei == 0:
            _mid_label(ax, pos[a], pos[b], r"$r^{\ast}$", color="#1565C0", dy_off=0.18)

    d_covers = _cover_edges(n, lambda a, b: d_mat[a][b])
    for ei, (a, b) in enumerate(d_covers):
        _arrow(ax, pos[a], pos[b], color="#E65100", lw=2.4, mutation_scale=22, zorder=1)
        if ei == 0:
            _mid_label(ax, pos[a], pos[b], r"$d^{\ast}$", color="#E65100", dy_off=-0.14)

    for x in range(n):
        lx = l0_map[x]
        if lx == x:
            continue
        pa, pb = pos[x], pos[lx]
        rad = 0.35 if (pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2 < 0.01 else 0.22
        ax.annotate(
            "",
            xy=pb,
            xytext=pa,
            arrowprops=dict(
                arrowstyle="-|>",
                color="#2E7D32",
                lw=2.2,
                mutation_scale=18,
                connectionstyle=f"arc3,rad={rad}",
            ),
            zorder=0,
        )
    if any(l0_map[x] != x for x in range(n)):
        xa = next(x for x in range(n) if l0_map[x] != x)
        xb = l0_map[xa]
        _mid_label(ax, pos[xa], pos[xb], r"$l_0$", color="#2E7D32", dy_off=0.2)

    hint_lines = [
        r"$r^{\ast}(i,j)$: blue $i \to j$",
        r"$d^{\ast}(i,j)$: orange $i \to j$",
        r"$l_0(x)$: green $x \to l_0(x)$",
    ]
    if h_idx is not None:
        hint_lines.insert(0, r"Yellow = $h$ (head) if present in VC.")
    if null_idx is not None:
        hint_lines.insert(1, "Lavender = null.")
    ax.text(
        0.98,
        0.02,
        "\n".join(hint_lines),
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.92),
    )
    ax.invert_yaxis()
    ax.grid(True, alpha=0.25)
    leg = [
        mpatches.Patch(color="#1565C0", label=r"$r^{\ast}$ cover"),
        mpatches.Patch(color="#E65100", label=r"$d^{\ast}$ cover"),
        mpatches.Patch(color="#2E7D32", label=r"$l_0$"),
    ]
    if h_idx is not None:
        leg.insert(
            0,
            mpatches.Patch(
                facecolor="lightyellow",
                edgecolor="#B8860B",
                linewidth=2,
                label=r"$h$",
            ),
        )
    if null_idx is not None:
        leg.insert(
            1 if h_idx is not None else 0,
            mpatches.Patch(
                facecolor="lavender",
                edgecolor="#5E35B1",
                linewidth=2,
                label="null",
            ),
        )
    ax.legend(handles=leg, loc="lower left", fontsize=8)


def draw_canonical_2d_grid(ax) -> None:
    """Reference layout: r to the right, d downward (+y)."""
    pos = {
        "x1": (0.0, 0.0),
        "x2": (1.0, 0.0),
        "x3": (2.0, 0.0),
        "x4": (0.0, 1.0),
        "x5": (1.0, 1.0),
        "x6": (2.0, 1.0),
    }
    ax.set_title("Reference: r →, d ↓ (inference 2D convention)")
    ax.set_aspect("equal")
    ax.set_xlabel("x  →  (r)")
    ax.set_ylabel("y  →  (d)")
    for name, (px, py) in pos.items():
        ax.scatter(
            [px], [py], s=900, c="white", edgecolors="black", linewidths=1.2, zorder=2
        )
        ax.text(px, py, name, ha="center", va="center", fontsize=11, zorder=3)
    off = 0.12
    for a, b in [("x1", "x2"), ("x2", "x3"), ("x4", "x5"), ("x5", "x6")]:
        pa, pb = pos[a], pos[b]
        _arrow(
            ax,
            (pa[0] + off, pa[1]),
            (pb[0] - off, pb[1]),
            color="C0",
            lw=2.0,
            mutation_scale=18,
            zorder=1,
        )
    _mid_label(
        ax,
        (pos["x1"][0] + off, pos["x1"][1]),
        (pos["x2"][0] - off, pos["x2"][1]),
        "r",
        color="C0",
    )
    for a, b in [("x1", "x4"), ("x2", "x5"), ("x3", "x6")]:
        pa, pb = pos[a], pos[b]
        _arrow(
            ax,
            (pa[0], pa[1] + off),
            (pb[0], pb[1] - off),
            color="C1",
            lw=2.0,
            mutation_scale=18,
            zorder=1,
        )
    _mid_label(
        ax,
        (pos["x1"][0], pos["x1"][1] + off),
        (pos["x4"][0], pos["x4"][1] - off),
        "d",
        color="C1",
        dy_off=-0.08,
    )
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(-0.4, 1.7)
    ax.invert_yaxis()
    ax.grid(False)
    ax.legend(
        handles=[
            mpatches.Patch(color="C0", label="r  (→)"),
            mpatches.Patch(color="C1", label="d  (↓)"),
        ],
        loc="upper right",
        fontsize=9,
    )


def save_goal_relation_png(
    output_path: str,
    r_mat: Sequence[Sequence[bool]],
    d_mat: Sequence[Sequence[bool]],
    l0_map: Sequence[int],
    named_indices: Dict[str, int],
    *,
    diagram_title: str,
) -> str:
    """Write goal-relation counterexample diagram from explicit matrices."""
    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    fig, ax1 = plt.subplots(1, 1, figsize=(8.4, 6.2))
    draw_goal_model_panel(
        ax1,
        r_mat=r_mat,
        d_mat=d_mat,
        l0_map=l0_map,
        named_indices=named_indices,
        title=diagram_title,
    )
    fig.suptitle(
        "Counterexample model: r* (blue), d* (orange), l0 (green); layout = (r-rank, d-rank)",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(output_path)


def save_goal_counterexample_figure(
    context: hlv.HighLevelContext,
    model: ModelRef,
    output_path: str,
    *,
    diagram_title: str,
) -> str:
    """
    Write a PNG of the Goal model from a Z3 counterexample.
    Returns the absolute path written.
    """
    _, r_mat, d_mat, l0_map, named = build_goal_relation_matrices(context, model)
    return save_goal_relation_png(
        output_path,
        r_mat,
        d_mat,
        l0_map,
        named,
        diagram_title=diagram_title,
    )
