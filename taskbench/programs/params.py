from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Union

import numpy as np

from taskbench.programs.ir import Program


PathElem = Union[str, int]


@dataclass(frozen=True)
class FloatParamRef:
    """Reference to a single float-valued parameter inside a Program.

    The parameter is identified by:
    - which instruction it belongs to (`instr_idx`)
    - the path inside `instruction.args` (`path`)
    """

    instr_idx: int
    path: tuple[PathElem, ...]


def _is_float(x: Any) -> bool:
    # Exclude bool explicitly (bool is a subclass of int in Python).
    if isinstance(x, bool):
        return False
    return isinstance(x, (float, np.floating))


def _walk(obj: Any, *, instr_idx: int, path: tuple[PathElem, ...]) -> Iterable[tuple[FloatParamRef, float]]:
    if _is_float(obj):
        yield FloatParamRef(instr_idx=instr_idx, path=path), float(obj)
        return

    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, instr_idx=instr_idx, path=path + (k,))
        return

    # We only support trainable vector parameters stored as lists.
    # If you store tuples, they will be treated as non-trainable in this helper.
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, instr_idx=instr_idx, path=path + (i,))
        return


def extract_float_parameters(program: Program) -> tuple[np.ndarray, list[FloatParamRef]]:
    """Flatten float-valued numeric fields from a Program into a vector.

    Returns:
        (x0, refs)
        - x0: shape (dim,) float vector
        - refs: list of parameter references; applying x in the same order
          will reconstruct the program's float fields.
    """
    refs: list[FloatParamRef] = []
    values: list[float] = []

    for instr_idx, instr in enumerate(program.instructions):
        for ref, val in _walk(instr.args, instr_idx=instr_idx, path=()):
            refs.append(ref)
            values.append(val)

    return np.asarray(values, dtype=float), refs


def _get_by_path(root: Any, path: tuple[PathElem, ...]) -> Any:
    cur = root
    for p in path:
        if isinstance(cur, dict):
            cur = cur[p]
        else:
            # list indexing
            cur = cur[p]  # type: ignore[index]
    return cur


def _set_by_path(root: Any, path: tuple[PathElem, ...], value: float) -> None:
    if not path:
        raise ValueError("empty path is not supported for set_by_path")

    cur = root
    for p in path[:-1]:
        if isinstance(cur, dict):
            cur = cur[p]
        else:
            cur = cur[p]  # type: ignore[index]

    last = path[-1]
    if isinstance(cur, dict):
        cur[last] = value
    else:
        cur[last] = value  # type: ignore[index]


def apply_float_parameters(program: Program, refs: list[FloatParamRef], x: np.ndarray) -> None:
    """In-place update of a Program's float fields from a parameter vector."""
    if len(refs) != int(x.shape[0]):
        raise ValueError(f"Parameter vector dim mismatch: refs={len(refs)} x={x.shape[0]}")

    for ref, val in zip(refs, x):
        instr = program.instructions[ref.instr_idx]
        _set_by_path(instr.args, ref.path, float(val))

