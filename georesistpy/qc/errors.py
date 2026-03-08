"""
Error estimation and reciprocal error analysis.

Provides routines for estimating data errors from repeat / reciprocal
measurements, which feed into inversion weighting.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def reciprocal_error(
    df_forward: pd.DataFrame,
    df_reciprocal: pd.DataFrame,
    merge_on: Tuple[str, ...] = ("a", "b", "m", "n"),
    value_col: str = "rhoa",
) -> pd.DataFrame:
    """Compute reciprocal errors between forward and reciprocal data.

    For each measurement the reciprocal error is defined as:

        e = |ρ_fwd − ρ_rec| / ((ρ_fwd + ρ_rec) / 2) × 100  [%]

    Parameters
    ----------
    df_forward, df_reciprocal : pd.DataFrame
        Forward and reciprocal datasets.
    merge_on : tuple of str
        Electrode columns used to match measurements.
    value_col : str
        Column containing apparent resistivity.

    Returns
    -------
    pd.DataFrame
        Merged table with columns ``rhoa_fwd``, ``rhoa_rec``,
        ``reciprocal_error_pct``.
    """
    # In reciprocal, current and potential electrodes are swapped
    rec = df_reciprocal.rename(
        columns={merge_on[0]: merge_on[2], merge_on[1]: merge_on[3],
                 merge_on[2]: merge_on[0], merge_on[3]: merge_on[1]}
    )

    merged = pd.merge(
        df_forward[list(merge_on) + [value_col]],
        rec[list(merge_on) + [value_col]],
        on=list(merge_on),
        suffixes=("_fwd", "_rec"),
    )

    fwd_col = f"{value_col}_fwd"
    rec_col = f"{value_col}_rec"
    mean_val = (merged[fwd_col] + merged[rec_col]) / 2.0
    merged["reciprocal_error_pct"] = (
        np.abs(merged[fwd_col] - merged[rec_col]) / mean_val * 100.0
    )

    return merged


def estimate_error_model(
    df: pd.DataFrame,
    value_col: str = "rhoa",
    relative: float = 0.03,
    absolute: float = 0.001,
) -> pd.DataFrame:
    """Assign an error estimate to each datum.

    Error model:  ``err = relative × |ρ_a| + absolute``

    Parameters
    ----------
    df : pd.DataFrame
        Survey data.
    value_col : str
        Column with apparent resistivity.
    relative : float
        Relative error fraction (default 3 %).
    absolute : float
        Absolute floor error in Ω·m (default 0.001).

    Returns
    -------
    pd.DataFrame
        Copy with an ``error`` column added.
    """
    df = df.copy()
    if value_col in df.columns:
        df["error"] = relative * np.abs(df[value_col]) + absolute
    else:
        df["error"] = absolute
    return df


def error_statistics(df: pd.DataFrame, error_col: str = "error") -> dict:
    """Return summary statistics for the error column.

    Parameters
    ----------
    df : pd.DataFrame
    error_col : str

    Returns
    -------
    dict
        Keys: ``mean``, ``median``, ``std``, ``min``, ``max``, ``count``.
    """
    if error_col not in df.columns:
        return {}
    s = df[error_col].dropna()
    return {
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std()),
        "min": float(s.min()),
        "max": float(s.max()),
        "count": int(len(s)),
    }
