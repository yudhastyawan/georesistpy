"""
Regularisation helpers.

Utilities for L-curve analysis, lambda estimation, and roughness
computation used by the inversion engines.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


def l_curve_corner(
    lambdas: Sequence[float],
    residual_norms: Sequence[float],
    model_norms: Sequence[float],
) -> Tuple[float, int]:
    """Find the L-curve corner (optimal λ) via maximum curvature.

    Parameters
    ----------
    lambdas : sequence of float
    residual_norms : sequence of float
        Data misfit for each λ.
    model_norms : sequence of float
        Model roughness (or norm) for each λ.

    Returns
    -------
    best_lambda : float
    best_index : int
    """
    log_res = np.log(np.asarray(residual_norms, dtype=float))
    log_mod = np.log(np.asarray(model_norms, dtype=float) + 1e-30)

    # Discrete curvature via finite differences
    n = len(log_res)
    if n < 3:
        return float(lambdas[0]), 0

    kappa = np.zeros(n)
    for i in range(1, n - 1):
        dx1 = log_res[i] - log_res[i - 1]
        dx2 = log_res[i + 1] - log_res[i]
        dy1 = log_mod[i] - log_mod[i - 1]
        dy2 = log_mod[i + 1] - log_mod[i]
        ddx = dx2 - dx1
        ddy = dy2 - dy1
        denom = (dx1 ** 2 + dy1 ** 2) ** 1.5
        if abs(denom) > 1e-30:
            kappa[i] = abs(dx1 * ddy - dy1 * ddx) / denom

    best_idx = int(np.argmax(kappa))
    return float(lambdas[best_idx]), best_idx


def estimate_lambda(
    data_size: int,
    model_size: int,
    noise_level: float = 0.03,
) -> float:
    """Heuristic initial λ estimate.

    Parameters
    ----------
    data_size : int
        Number of data points.
    model_size : int
        Number of model parameters.
    noise_level : float
        Expected relative data error (default 3 %).

    Returns
    -------
    float
        Suggested starting λ.
    """
    ratio = data_size / max(model_size, 1)
    return max(1.0, ratio / noise_level)


def roughness(model: np.ndarray) -> float:
    """Compute first-order roughness (sum of absolute differences).

    Parameters
    ----------
    model : np.ndarray
        1-D model vector.

    Returns
    -------
    float
    """
    return float(np.sum(np.abs(np.diff(model))))
