"""Bounded Panda Stack layouts with physical clearance between block centers.

The default 14 cm spacing was validated on 300 distinct layouts each for three
and four cubes. The symbolic Scattered predicate keeps its separate 2L bound.
"""

import numpy as np

BLOCK_LENGTH = 0.04
DEFAULT_SEPARATION = 0.14
XY_LOW = np.array([-0.10, -0.20])
XY_HIGH = np.array([0.10, 0.20])


class LayoutSamplingError(ValueError):
    """The bounded workspace could not accommodate a complete layout."""


def sample_layout(num_blocks, separation, seed, *, max_layouts=100, max_trials=200):
    """Require abs(dx) >= separation OR abs(dy) >= separation for each pair.

    Retries continue within this seed's RNG stream, without widening the
    workspace or reducing clearance. Return a full layout and sampling counts.
    """
    if not 2 <= num_blocks <= 6:
        raise ValueError("Require 2 through 6 blocks")
    if not np.isfinite(separation) or separation < 2 * BLOCK_LENGTH:
        raise ValueError("Separation must be finite and at least 0.08 m")
    if max_layouts < 1 or max_trials < 1:
        raise ValueError("Sampling budgets must be positive")
    rng = np.random.default_rng(seed)
    proposals = 0
    for restart in range(max_layouts):
        positions = []
        for _ in range(num_blocks):
            for _ in range(max_trials):
                candidate = rng.uniform(XY_LOW, XY_HIGH)
                proposals += 1
                if all(
                    np.max(np.abs(candidate - other)) >= separation
                    for other in positions
                ):
                    positions.append(candidate)
                    break
            else:
                break
        else:
            return np.asarray(positions), dict(
                layout_attempts=restart + 1, proposals=proposals
            )
    raise LayoutSamplingError(
        f"No {num_blocks}-block layout at {separation:.3f} m separation "
        f"after {max_layouts} layouts / {proposals} proposals"
    )
