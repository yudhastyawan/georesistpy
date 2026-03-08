"""
Quality-control filters for geoelectrical data.

Automatic routines for removing invalid data points, filtering outliers,
and flagging suspect measurements.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def remove_negative_resistivity(
    df: pd.DataFrame,
    col: str = "rhoa",
) -> pd.DataFrame:
    """Remove rows with negative apparent resistivity.

    Parameters
    ----------
    df : pd.DataFrame
        Survey data containing column *col*.
    col : str
        Column name for apparent resistivity (default ``'rhoa'``).

    Returns
    -------
    pd.DataFrame
        Filtered copy without negative values.
    """
    if col not in df.columns:
        return df.copy()
    mask = df[col] > 0
    return df.loc[mask].reset_index(drop=True)


def filter_outliers_mad(
    df: pd.DataFrame,
    col: str = "rhoa",
    threshold: float = 3.5,
) -> pd.DataFrame:
    """Remove outliers using Median Absolute Deviation (MAD).

    Parameters
    ----------
    df : pd.DataFrame
        Survey data.
    col : str
        Column to analyse.
    threshold : float
        Modified Z-score cutoff (default 3.5).

    Returns
    -------
    pd.DataFrame
        Filtered copy.
    """
    if col not in df.columns or len(df) == 0:
        return df.copy()

    values = df[col].values.astype(float)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad < 1e-30:
        return df.copy()

    modified_z = 0.6745 * (values - median) / mad
    mask = np.abs(modified_z) < threshold
    return df.loc[mask].reset_index(drop=True)


def filter_outliers_iqr(
    df: pd.DataFrame,
    col: str = "rhoa",
    factor: float = 1.5,
) -> pd.DataFrame:
    """Remove outliers using the Inter-Quartile Range (IQR) method.

    Parameters
    ----------
    df : pd.DataFrame
        Survey data.
    col : str
        Column to analyse.
    factor : float
        IQR multiplier (default 1.5).

    Returns
    -------
    pd.DataFrame
        Filtered copy.
    """
    if col not in df.columns or len(df) == 0:
        return df.copy()

    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    mask = (df[col] >= lower) & (df[col] <= upper)
    return df.loc[mask].reset_index(drop=True)


def flag_outliers(
    df: pd.DataFrame,
    col: str = "rhoa",
    method: str = "mad",
    threshold: float = 3.5,
) -> pd.DataFrame:
    """Return a copy of *df* with an ``outlier`` boolean column.

    Parameters
    ----------
    df : pd.DataFrame
    col : str
    method : str
        ``'mad'`` or ``'iqr'``.
    threshold : float
        Cutoff value for the chosen method.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    if col not in df.columns or len(df) == 0:
        df["outlier"] = False
        return df

    if method == "iqr":
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        df["outlier"] = ~((df[col] >= lower) & (df[col] <= upper))
    else:
        values = df[col].values.astype(float)
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        if mad < 1e-30:
            df["outlier"] = False
        else:
            mz = 0.6745 * (values - median) / mad
            df["outlier"] = np.abs(mz) >= threshold

    return df
