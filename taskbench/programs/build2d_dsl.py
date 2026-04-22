"""DSL and evaluator for Build2D linked-list programs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class Node:
    """2D linked-list node with world coordinates."""

    x: float
    y: float
    z: float
    r: "Node | None" = None
    d: "Node | None" = None


class Build2DRuntime(Protocol):
    """Runtime interface consumed by the Build2D program evaluator."""

    def put_block(self, node: Node | Any) -> None:
        """Place one block at node.(x, y, z)."""


class Stmt:
    """Base class for DSL statements."""

    def eval(self, state: dict[str, Any], runtime: Build2DRuntime) -> None:
        raise NotImplementedError


@dataclass
class Assign(Stmt):
    target: str
    source: str

    def eval(self, state: dict[str, Any], runtime: Build2DRuntime) -> None:
        state[self.target] = state.get(self.source)


@dataclass
class Advance(Stmt):
    var: str
    relation: str

    def eval(self, state: dict[str, Any], runtime: Build2DRuntime) -> None:
        cur = state.get(self.var)
        if cur is None:
            state[self.var] = None
            return
        if self.relation not in ("r", "d"):
            raise ValueError(f"Unknown relation: {self.relation!r}")
        state[self.var] = getattr(cur, self.relation)


@dataclass
class PutBlock(Stmt):
    node_var: str

    def eval(self, state: dict[str, Any], runtime: Build2DRuntime) -> None:
        node = state.get(self.node_var)
        if node is None:
            raise ValueError(f"Cannot put block on null variable {self.node_var!r}")
        runtime.put_block(node)


@dataclass
class WhileNotNull(Stmt):
    var: str
    body: list[Stmt]

    def eval(self, state: dict[str, Any], runtime: Build2DRuntime) -> None:
        while state.get(self.var) is not None:
            for stmt in self.body:
                stmt.eval(state, runtime)


@dataclass
class Build2DProgram:
    """Executable Build2D DSL program."""

    body: list[Stmt]

    def eval(self, h: Node | Any, runtime: Build2DRuntime) -> None:
        state: dict[str, Any] = {"h": h}
        for stmt in self.body:
            stmt.eval(state, runtime)


def canonical_build2d_program() -> Build2DProgram:
    """Create the exact nested-loop program requested by the user."""
    return Build2DProgram(
        body=[
            Assign("i", "h"),
            WhileNotNull(
                "i",
                body=[
                    Assign("j", "i"),
                    WhileNotNull(
                        "j",
                        body=[
                            PutBlock("j"),
                            Advance("j", "r"),
                        ],
                    ),
                    Advance("i", "d"),
                ],
            ),
        ]
    )


class TraceRuntime:
    """Runtime for dry-runs on machines without ManiSkill execution support."""

    def __init__(self):
        self.placements: list[tuple[float, float, float]] = []

    def put_block(self, node: Node | Any) -> None:
        self.placements.append((float(node.x), float(node.y), float(node.z)))

