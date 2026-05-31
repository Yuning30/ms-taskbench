"""Generic MCMC utilities for program synthesis.

This module mirrors the structure of an external MCMC implementation, but is
adapted to the lightweight ``Program`` / ``Instruction`` IR used in this
codebase. It focuses on the MCMC loop itself and leaves the details of
mutation and cost computation to caller-provided callbacks.
"""

from __future__ import annotations

import math
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

from taskbench.programs.ir import Program


def sample_proportional(weights: list[float]) -> int:
    """Sample an index with probability proportional to the given weights."""
    if not weights:
        raise ValueError("weights must be non-empty")
    total = sum(weights)
    if total <= 0:
        raise ValueError("weights must be positive")
    return random.choices(range(len(weights)), weights=weights, k=1)[0]


def swap_two_elements_maybe_same(seq: list) -> None:
    """In-place swap of two (possibly identical) randomly chosen elements."""
    if not seq:
        raise ValueError("sequence must be non-empty")
    i = random.randint(0, len(seq) - 1)
    j = random.randint(0, len(seq) - 1)
    seq[i], seq[j] = seq[j], seq[i]


@dataclass
class MCMCSample:
    program: Program
    cost: float


def mcmc_optimize_program(
    initial_program: Program,
    *,
    cost_fn: Callable[[Program], float],
    mutate_fn: Callable[[Program], tuple[Program, bool]],
    iters: int,
) -> tuple[list[MCMCSample], Program]:
    """Run a simple Metropolis-Hastings MCMC over programs.

    Args:
        initial_program: Starting program.
        cost_fn: Maps a program to a scalar cost (higher is better).
        mutate_fn: Proposes a new program from the current one. Returns
            (new_program, changed) where changed indicates whether a genuine
            change was made.
        iters: Number of MCMC iterations.

    Returns:
        (samples, best_program)
        - samples: list of accepted samples (including the initial one)
        - best_program: program with the highest observed cost.
    """
    current_program = deepcopy(initial_program)
    current_cost = cost_fn(current_program)

    samples: list[MCMCSample] = [MCMCSample(deepcopy(current_program), current_cost)]
    best_program = deepcopy(current_program)
    best_cost = current_cost

    for _ in range(iters):
        proposed_program, changed = mutate_fn(current_program)
        if not changed:
            # No effective change; keep current sample.
            samples.append(MCMCSample(deepcopy(current_program), current_cost))
            continue

        proposed_cost = cost_fn(proposed_program)

        # Metropolis-Hastings acceptance ratio; higher cost is better.
        acceptance_ratio = math.exp(min(0.0, proposed_cost - current_cost))
        if random.random() < acceptance_ratio:
            current_program = proposed_program
            current_cost = proposed_cost

        # Track best program seen so far.
        if proposed_cost > best_cost:
            best_cost = proposed_cost
            best_program = deepcopy(proposed_program)

        samples.append(MCMCSample(deepcopy(current_program), current_cost))

    return samples, best_program

