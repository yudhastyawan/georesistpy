"""
Depth-of-Investigation (DOI) estimation.

Implements sensitivity-based DOI index mapping and optional reliability
classification.  Sensitivities come from the Jacobian matrix; otherwise
a simple cumulative-sensitivity proxy is used.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def compute_doi(
    sensitivity: np.ndarray,
    mesh_cell_centers: np.ndarray,
    threshold: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the Depth-of-Investigation index for every mesh cell.

    Parameters
    ----------
    sensitivity : np.ndarray
        Per-cell cumulative absolute sensitivity (1-D, length = n_cells).
    mesh_cell_centers : np.ndarray
        (n_cells, 2) array with columns ``(x, z)``.
    threshold : float, optional
        Normalised sensitivity below which a cell is considered unreliable
        (default 0.1 = 10 % of max).

    Returns
    -------
    doi_index : np.ndarray
        Normalised DOI index in [0, 1] for each cell (1 = reliable).
    reliable_mask : np.ndarray
        Boolean mask — *True* where DOI index ≥ *threshold*.
    """
    sens = np.abs(sensitivity).astype(float)
    max_sens = sens.max()
    if max_sens > 0:
        doi_index = sens / max_sens
    else:
        doi_index = np.zeros_like(sens)

    reliable_mask = doi_index >= threshold

    return doi_index, reliable_mask


def classify_doi(
    doi_index: np.ndarray,
    thresholds: Tuple[float, float, float] = (0.05, 0.10, 0.30),
) -> np.ndarray:
    """Classify DOI reliability into categories.

    Parameters
    ----------
    doi_index : np.ndarray
        Normalised DOI values in [0, 1].
    thresholds : tuple of float
        Boundaries for *unreliable*, *marginal*, *acceptable*, *good*.

    Returns
    -------
    np.ndarray of int
        0 = unreliable, 1 = marginal, 2 = acceptable, 3 = good.
    """
    t1, t2, t3 = thresholds
    classes = np.zeros(len(doi_index), dtype=int)
    classes[doi_index >= t1] = 1
    classes[doi_index >= t2] = 2
    classes[doi_index >= t3] = 3
    return classes


def sensitivity_from_jacobian(jacobian: np.ndarray) -> np.ndarray:
    """Derive cumulative absolute sensitivity from a Jacobian matrix.

    Parameters
    ----------
    jacobian : np.ndarray
        (n_data, n_cells) Jacobian matrix from ERT forward or inversion.

    Returns
    -------
    np.ndarray
        (n_cells,) cumulative absolute column-sum of the Jacobian.
    """
    return np.abs(jacobian).sum(axis=0)
