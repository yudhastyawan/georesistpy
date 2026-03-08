"""
Topography correction utilities.

Provides helpers for embedding elevation data into electrode positions
and adjusting mesh coordinates for surface topography.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def apply_topography(
    electrode_positions: np.ndarray,
    elevation: Sequence[float],
) -> np.ndarray:
    """Attach elevation to electrode positions.

    Parameters
    ----------
    electrode_positions : np.ndarray
        (N,) or (N, 2) array.  If 1-D, treated as x-coordinates.
    elevation : sequence of float
        Elevation at each electrode (length N).

    Returns
    -------
    np.ndarray
        (N, 2) array with columns ``(x, z)``.
    """
    pos = np.asarray(electrode_positions, dtype=float)
    elev = np.asarray(elevation, dtype=float)

    if pos.ndim == 1:
        return np.column_stack([pos, elev])

    if pos.shape[1] >= 2:
        out = pos.copy()
        out[:, 1] = elev
        return out

    return np.column_stack([pos[:, 0], elev])


def interpolate_topography(
    x_known: np.ndarray,
    z_known: np.ndarray,
    x_query: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    """Interpolate topography at arbitrary x-positions.

    Parameters
    ----------
    x_known, z_known : np.ndarray
        Known (x, z) control points.
    x_query : np.ndarray
        X-positions at which to evaluate elevation.
    method : str
        ``'linear'`` (default) or ``'cubic'``.

    Returns
    -------
    np.ndarray
        Interpolated elevation at *x_query*.
    """
    from scipy.interpolate import interp1d

    f = interp1d(x_known, z_known, kind=method, fill_value="extrapolate")
    return f(x_query)


def correct_dataframe_topo(
    df: pd.DataFrame,
    elevation_col: str = "elevation",
) -> pd.DataFrame:
    """Add an ``elevation`` column to a survey DataFrame when absent.

    If the column already exists the DataFrame is returned unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        Survey data.
    elevation_col : str
        Name of the elevation column.

    Returns
    -------
    pd.DataFrame
    """
    if elevation_col not in df.columns:
        df = df.copy()
        df[elevation_col] = 0.0
    return df
