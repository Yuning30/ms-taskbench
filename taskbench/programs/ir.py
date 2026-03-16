from dataclasses import dataclass, field
from typing import Any


@dataclass
class Instruction:
    """Single high-level program instruction.

    Each instruction corresponds to invoking a skill on ``SkillContext``,
    e.g. ``pick``, ``move``, or ``place``. Arguments are stored in a generic
    dictionary so that synthesis algorithms can easily mutate both structure
    and parameters.
    """

    op: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class Program:
    """A sequence of high-level skill instructions."""

    instructions: list[Instruction] = field(default_factory=list)

