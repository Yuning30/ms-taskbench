from __future__ import annotations

import numpy as np


def cem_optimize(
    f,
    dim: int,
    *,
    iterations: int = 50,
    N: int = 100,
    K: int = 10,
    init_mu=None,
    init_std: float = 0.1,
    rng=None,
) -> tuple[float, np.ndarray]:
    """Maximize f over a continuous vector using Cross-Entropy Method (CEM).

    This is a lightweight, robust implementation intended for objectives
    that are expensive to evaluate (we default to sequential evaluation).

    Args:
        f: Objective function, called as ``f(x: np.ndarray) -> float``.
        dim: Search dimension.
        iterations: Number of CEM/CEM iterations.
        N: Number of sampled candidates per iteration.
        K: Number of elites used for updates.
        init_mu: Initial mean vector (shape (dim,)).
        init_std: Initial standard deviation (scalar; expanded to dim).
        rng: Optional numpy RNG.

    Returns:
        (best_score, best_mu)
    """
    if dim <= 0:
        raise ValueError("dim must be > 0")
    if not (0 < K <= N):
        raise ValueError("Must satisfy 0 < K <= N")

    rng = rng or np.random.default_rng()

    mu = np.zeros(dim, dtype=float) if init_mu is None else np.asarray(init_mu, dtype=float).reshape(dim)
    sigma = np.ones(dim, dtype=float) * float(init_std)

    # Track best over the entire run.
    best_score = float(f(mu))
    best_mu = mu.copy()

    for _ in range(iterations):
        samples = rng.standard_normal(size=(N, dim)) * sigma + mu

        # Sequential evaluation (keeps f as an arbitrary Python callable).
        scores = np.asarray([float(f(s)) for s in samples], dtype=float)

        # Pick top-K elites (maximize).
        elite_idx = np.argsort(scores)[-K:]
        elites = samples[elite_idx]

        mu = elites.mean(axis=0)
        sigma = elites.std(axis=0)

        # Optional numerical stability: avoid sigma collapsing to all zeros.
        sigma = np.maximum(sigma, 1e-6)

        mu_score = float(f(mu))
        if mu_score > best_score:
            best_score = mu_score
            best_mu = mu.copy()

    return best_score, best_mu

