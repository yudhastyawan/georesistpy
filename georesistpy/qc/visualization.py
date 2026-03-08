"""
QC visualization helpers.

Quick-view functions for data quality assessment: histograms, scatter
plots, flagged pseudosections.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_histogram(
    df: pd.DataFrame,
    col: str = "rhoa",
    bins: int = 40,
    log_scale: bool = True,
    title: str = "Apparent Resistivity Distribution",
    figsize: Tuple[int, int] = (8, 4),
) -> Figure:
    """Histogram of a data column.

    Parameters
    ----------
    df : pd.DataFrame
    col : str
    bins : int
    log_scale : bool
    title : str
    figsize : tuple

    Returns
    -------
    Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    values = df[col].dropna().values
    if log_scale and (values > 0).all():
        values = np.log10(values)
        ax.set_xlabel(f"log₁₀({col})")
    else:
        ax.set_xlabel(col)
    ax.hist(values, bins=bins, edgecolor="k", alpha=0.75)
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_scatter_qc(
    df: pd.DataFrame,
    x_col: str = "a",
    y_col: str = "rhoa",
    flag_col: str = "outlier",
    title: str = "QC Scatter",
    figsize: Tuple[int, int] = (10, 5),
) -> Figure:
    """Scatter plot with flagged outliers highlighted.

    Parameters
    ----------
    df : pd.DataFrame
    x_col, y_col : str
    flag_col : str
        Boolean column; *True* = outlier.
    title : str
    figsize : tuple

    Returns
    -------
    Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    good = df[~df.get(flag_col, False)]  # type: ignore[arg-type]
    bad = df[df.get(flag_col, False)]    # type: ignore[arg-type]

    ax.scatter(good[x_col], good[y_col], s=12, label="Good", alpha=0.7)
    if len(bad) > 0:
        ax.scatter(bad[x_col], bad[y_col], s=20, c="red", marker="x",
                   label="Outlier")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_error_distribution(
    df: pd.DataFrame,
    error_col: str = "error",
    bins: int = 30,
    title: str = "Data Error Distribution",
    figsize: Tuple[int, int] = (8, 4),
) -> Figure:
    """Histogram of estimated data errors.

    Parameters
    ----------
    df : pd.DataFrame
    error_col : str
    bins : int
    title : str
    figsize : tuple

    Returns
    -------
    Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    values = df[error_col].dropna().values
    ax.hist(values, bins=bins, edgecolor="k", alpha=0.75, color="coral")
    ax.set_xlabel(error_col)
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_pseudosection_qc(
    df: pd.DataFrame,
    value_col: str = "rhoa",
    flag_col: str = "outlier",
    cmap: str = "jet",
    title: str = "QC Pseudosection",
    figsize: Tuple[int, int] = (10, 5),
) -> Figure:
    """Plot an apparent-resistivity pseudosection with outliers flagged.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns 'a', 'b', 'm', 'n'.
    value_col : str
    flag_col : str
        Boolean column; *True* = outlier.
    cmap : str
    title : str
    figsize : tuple
    
    Returns
    -------
    Figure
    """
    import matplotlib.colors as mcolors
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate midpoint and pseudo-depth
    df_plot = df.copy()
    df_plot["midpoint"] = (df_plot["a"] + df_plot["b"] + df_plot["m"] + df_plot["n"]) / 4.0
    df_plot["pseudo_depth"] = (np.abs(df_plot["a"] - df_plot["b"]) + np.abs(df_plot["m"] - df_plot["n"])) / 4.0
    
    good = df_plot[~df_plot.get(flag_col, False)]
    bad = df_plot[df_plot.get(flag_col, False)]
    
    all_vals = df_plot[value_col].dropna().values.astype(float)
    if len(all_vals) > 0 and (all_vals > 0).all():
        norm = mcolors.LogNorm(vmin=np.nanmin(all_vals), vmax=np.nanmax(all_vals))
    else:
        norm = None

    if len(good) > 0:
        sc = ax.scatter(
            good["midpoint"], -good["pseudo_depth"],
            c=good[value_col].astype(float), cmap=cmap, norm=norm,
            s=30, edgecolors="k", linewidths=0.3, label="Kept"
        )
        plt.colorbar(sc, ax=ax, label=f"{value_col} (Ω·m)")
        
    if len(bad) > 0:
        ax.scatter(
            bad["midpoint"], -bad["pseudo_depth"],
            marker="x", color="red", s=50, label="Removed (Outlier)", zorder=10
        )
        
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Pseudo-depth (m)")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig

