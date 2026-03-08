"""
Interactive visualizations using Plotly and Holoviews.

These produce embeddable, zoomable plots suitable for the Panel web UI.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


# -------------------------------------------------------------------------
# Plotly-based plots
# -------------------------------------------------------------------------

def plotly_pseudosection(
    data: pd.DataFrame,
    value_col: str = "rhoa",
    log_scale: bool = True,
    title: str = "Apparent Resistivity Pseudosection",
    width: int = 900,
    height: int = 450,
):
    """Create a Plotly pseudosection scatter.

    Parameters
    ----------
    data : pd.DataFrame
        Columns ``a, b, m, n`` and *value_col*.
    value_col : str
    log_scale : bool
    title : str
    width, height : int

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.express as px

    df = data.copy()
    df["midpoint"] = (df["a"] + df["b"] + df["m"] + df["n"]) / 4.0
    df["pseudo_depth"] = (
        np.abs(df["a"] - df["b"]) + np.abs(df["m"] - df["n"])
    ) / 4.0
    df["neg_depth"] = -df["pseudo_depth"]

    color_col = value_col
    if log_scale:
        df["log_rhoa"] = np.log10(df[value_col].clip(lower=1e-6))
        color_col = "log_rhoa"

    fig = px.scatter(
        df,
        x="midpoint",
        y="neg_depth",
        color=color_col,
        color_continuous_scale="Jet",
        title=title,
        labels={
            "midpoint": "Distance (m)",
            "neg_depth": "Pseudo-depth (m)",
            color_col: "log₁₀(ρₐ)" if log_scale else "ρₐ (Ω·m)",
        },
        width=width,
        height=height,
    )
    fig.update_traces(marker=dict(size=6))
    max_d = df["pseudo_depth"].max()
    # Explicitly set bounds without autorange reversal to keep negative depths natural Cartesian 
    fig.update_layout(yaxis=dict(range=[-max_d * 1.05, max_d * 0.05]))
    return fig


def plotly_inverted_section(
    cell_centers: np.ndarray,
    resistivity: np.ndarray,
    log_scale: bool = True,
    title: str = "Inverted Resistivity Section",
    width: int = 900,
    height: int = 450,
):
    """Create a Plotly scatter of inverted resistivity.

    Parameters
    ----------
    cell_centers : np.ndarray
        (N, 2) ``(x, z)`` cell centres.
    resistivity : np.ndarray
    log_scale : bool
    title : str
    width, height : int

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.express as px

    df = pd.DataFrame({
        "x": cell_centers[:, 0],
        "z": cell_centers[:, 1],
        "resistivity": resistivity,
    })

    color_col = "resistivity"
    if log_scale:
        df["log_rho"] = np.log10(df["resistivity"].clip(lower=1e-6))
        color_col = "log_rho"

    fig = px.scatter(
        df, x="x", y="z", color=color_col,
        color_continuous_scale="Jet",
        title=title,
        labels={"x": "Distance (m)", "z": "Depth (m)",
                color_col: "log₁₀(ρ)" if log_scale else "ρ (Ω·m)"},
        width=width, height=height,
    )
    fig.update_traces(marker=dict(size=5))
    
    # Y=0 visualization padding
    max_d = np.abs(cell_centers[:, 1]).max()
    fig.update_layout(
        yaxis_scaleanchor="x",
        yaxis=dict(range=[-max_d * 1.05, max_d * 0.05])
    )
    return fig


def plotly_residuals(
    observed: np.ndarray,
    predicted: np.ndarray,
    title: str = "Data Misfit",
    width: int = 800,
    height: int = 400,
):
    """Plotly observed-vs-predicted scatter.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2,
                         subplot_titles=("Obs. vs Pred.", "Residual (%)"))

    fig.add_trace(
        go.Scatter(x=observed, y=predicted, mode="markers",
                   marker=dict(size=4), name="Data"),
        row=1, col=1,
    )
    lo = min(observed.min(), predicted.min())
    hi = max(observed.max(), predicted.max())
    fig.add_trace(
        go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                   line=dict(color="red", dash="dash"), name="1:1"),
        row=1, col=1,
    )

    residual = (predicted - observed) / observed * 100
    fig.add_trace(
        go.Histogram(x=residual, nbinsx=30, name="Residual"),
        row=1, col=2,
    )

    fig.update_layout(title=title, width=width, height=height, showlegend=False)
    return fig


# -------------------------------------------------------------------------
# Holoviews-based plots
# -------------------------------------------------------------------------

def hv_pseudosection(
    data: pd.DataFrame,
    value_col: str = "rhoa",
    log_scale: bool = True,
    title: str = "Pseudosection",
):
    """Create a Holoviews Points pseudosection.

    Returns
    -------
    holoviews.Element
    """
    import holoviews as hv
    hv.extension("bokeh")

    df = data.copy()
    df["midpoint"] = (df["a"] + df["b"] + df["m"] + df["n"]) / 4.0
    df["pseudo_depth"] = -(
        np.abs(df["a"] - df["b"]) + np.abs(df["m"] - df["n"])
    ) / 4.0

    if log_scale:
        df["log_rhoa"] = np.log10(df[value_col].clip(lower=1e-6))
        vdim = "log_rhoa"
    else:
        vdim = value_col

    pts = hv.Points(
        df, kdims=["midpoint", "pseudo_depth"], vdims=[vdim],
    ).opts(
        color=vdim, cmap="jet", size=5, colorbar=True,
        width=800, height=400, title=title,
        xlabel="Distance (m)", ylabel="Pseudo-depth (m)",
        invert_yaxis=True, tools=["hover"],
    )
    return pts
