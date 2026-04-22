"""Dry-run Build2D DSL without ManiSkill dependencies.

Useful on machines where simulation cannot run.
"""

from __future__ import annotations

import argparse

from taskbench.programs.build2d_dsl import Node, TraceRuntime, canonical_build2d_program


def make_linked_grid(rows: int, cols: int) -> Node:
    nodes = [[Node(float(i), float(j), 0.02) for j in range(cols)] for i in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if j + 1 < cols:
                nodes[i][j].r = nodes[i][j + 1]
            if i + 1 < rows:
                nodes[i][j].d = nodes[i + 1][j]
    return nodes[0][0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=3)
    args = parser.parse_args()

    h = make_linked_grid(args.rows, args.cols)
    runtime = TraceRuntime()
    canonical_build2d_program().eval(h, runtime)

    expected = args.rows * args.cols
    actual = len(runtime.placements)
    print(f"Expected placements: {expected}")
    print(f"Actual placements:   {actual}")
    print("Program run successful." if actual == expected else "Program run failed.")


if __name__ == "__main__":
    main()
