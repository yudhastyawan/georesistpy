"""
Static (matplotlib) plotting functions for ERT data and inversion results.

Functions
---------
- pseudosection
- inverted resistivity section
- mesh overlay
- residual (misfit) plot
- sensitivity map
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server use
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.tri import Triangulation
import matplotlib.colors as mcolors


# -------------------------------------------------------------------------
# Pseudosection
# -------------------------------------------------------------------------

def plot_pseudosection(
    data: pd.DataFrame,
    value_col: str = "rhoa",
    ax: Optional[plt.Axes] = None,
    cmap: str = "jet",
    log_scale: bool = True,
    title: str = "Apparent Resistivity Pseudosection",
    figsize: Tuple[int, int] = (12, 5),
) -> Figure:
    """Plot an apparent-resistivity pseudosection.

    The midpoint of each quadrupole is on the x-axis; the pseudo-depth
    is the median spacing.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain columns ``a, b, m, n`` and *value_col*.
    value_col : str
        Column to colour-map (default ``'rhoa'``).
    ax : matplotlib Axes, optional
    cmap : str
    log_scale : bool
    title : str
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = data.copy()
    df["midpoint"] = (df["a"] + df["b"] + df["m"] + df["n"]) / 4.0
    df["pseudo_depth"] = (
        np.abs(df["a"] - df["b"]) + np.abs(df["m"] - df["n"])
    ) / 4.0

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    values = df[value_col].values.astype(float)
    norm = mcolors.LogNorm(vmin=np.nanmin(values[values > 0]), vmax=np.nanmax(values)) if log_scale else None

    # Trapesium contour plotting
    try:
        x_pts = df["midpoint"].values
        y_pts = df["pseudo_depth"].values
        tri = Triangulation(x_pts, y_pts)
        
        # Plot continuous filled contours
        levels = np.logspace(np.log10(np.nanmin(values[values > 0])), np.log10(np.nanmax(values)), 20) if log_scale else 20
        cax = ax.tricontourf(tri, values, levels=levels, cmap=cmap, norm=norm, extend="both")
        
        # Optional: draw the discrete points lightly over the contour
        sc = ax.scatter(x_pts, y_pts, c="k", s=5, alpha=0.3)
        plt.colorbar(cax, ax=ax, label=f"{value_col} (Ω·m)")
    except Exception:
        # Fallback to scatter if triangulation fails
        sc = ax.scatter(
            df["midpoint"],
            df["pseudo_depth"],
            c=values,
            cmap=cmap,
            norm=norm,
            s=30,
            edgecolors="k",
            linewidths=0.3,
        )
        plt.colorbar(sc, ax=ax, label=f"{value_col} (Ω·m)")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Pseudo-depth (m)")
    ax.set_title(title)
    
    # Enforce Y=0 line is visible globally, with 5% padding
    z_max = df["pseudo_depth"].max()
    z_pad = z_max * 0.05
    ax.set_ylim(z_max + z_pad, -z_pad)  # Inverted via reversed limits
    
    fig.tight_layout()
    return fig


# -------------------------------------------------------------------------
# Inverted section
# -------------------------------------------------------------------------

def plot_inverted_section(
    cell_centers: np.ndarray,
    resistivity: np.ndarray,
    ax: Optional[plt.Axes] = None,
    cmap: str = "jet",
    log_scale: bool = True,
    title: str = "Inverted Resistivity Section",
    figsize: Tuple[int, int] = (12, 5),
    doi_mask: Optional[np.ndarray] = None,
) -> Figure:
    """Plot the inverted resistivity cross-section.

    Parameters
    ----------
    cell_centers : np.ndarray
        (N, 2) cell centre coordinates ``(x, z)``.
    resistivity : np.ndarray
        (N,) inverted resistivity values.
    ax : matplotlib Axes, optional
    cmap : str
    log_scale : bool
    title : str
    figsize : tuple
    doi_mask : np.ndarray, optional
        Boolean mask; cells with *False* are greyed out.

    Returns
    -------
    Figure
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x = cell_centers[:, 0]
    z = cell_centers[:, 1]
    values = resistivity.copy().astype(float)

    if doi_mask is not None:
        values[~doi_mask] = np.nan

    nan_mask = ~np.isnan(values)
    if nan_mask.any():
        norm = mcolors.LogNorm(vmin=np.nanmin(values[nan_mask]),
                               vmax=np.nanmax(values[nan_mask])) if log_scale else None
    else:
        norm = None

    try:
        tri = Triangulation(x, z)
        if doi_mask is not None:
            tri.set_mask(~doi_mask[tri.triangles].any(axis=1))
        tc = ax.tripcolor(tri, values, cmap=cmap, norm=norm, shading="flat")
    except Exception:
        if doi_mask is not None:
            tc = ax.scatter(x[doi_mask], z[doi_mask], c=values[doi_mask], cmap=cmap, norm=norm, s=8)
        else:
            tc = ax.scatter(x, z, c=values, cmap=cmap, norm=norm, s=8)

    plt.colorbar(tc, ax=ax, label="Resistivity (Ω·m)")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def plot_inversion_progress(
    data: pd.DataFrame,
    dpred: np.ndarray,
    cell_centers: np.ndarray,
    resistivity: np.ndarray,
    iteration: int,
    rms: float,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "jet",
) -> Figure:
    """Create a 3-panel real-time progress figure similar to RES2DINV.
    
    1. Observed Apparent Resistivity Pseudosection
    2. Calculated Apparent Resistivity Pseudosection
    3. Inverted Resistivity Model (Section)
    """
    df = data.copy()
    df["midpoint"] = (df["a"] + df["b"] + df["m"] + df["n"]) / 4.0
    df["pseudo_depth"] = (np.abs(df["a"] - df["b"]) + np.abs(df["m"] - df["n"])) / 4.0
    
    x_pseud = df["midpoint"]
    y_pseud = -df["pseudo_depth"]
    
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    
    # Common log scale for pseudosections
    v_obs = df["rhoa"].values.astype(float)
    # Ensure vmin is positive for LogNorm
    vmin_ps = max(1e-2, min(np.nanmin(v_obs), np.nanmin(dpred)))
    vmax_ps = max(vmin_ps * 1.1, max(np.nanmax(v_obs), np.nanmax(dpred)))
    norm_ps = mcolors.LogNorm(vmin=vmin_ps, vmax=vmax_ps)
    
    # Pseudosections (Observed & Calculated)
    try:
        tri_ps = Triangulation(x_pseud.values, y_pseud.values)
        levels_ps = np.logspace(np.log10(vmin_ps), np.log10(vmax_ps), 20)
        
        # 1. Observed
        cax1 = axes[0].tricontourf(tri_ps, v_obs, levels=levels_ps, cmap=cmap, norm=norm_ps, extend="both")
        axes[0].scatter(x_pseud, y_pseud, c="k", s=5, alpha=0.3)
        plt.colorbar(cax1, ax=axes[0], label="Observed ρₐ (Ω·m)")
        axes[0].set_title(f"Iteration {iteration} — Observed Data")
        axes[0].set_ylabel("Pseudo-depth (m)")
        
        # 2. Calculated
        cax2 = axes[1].tricontourf(tri_ps, dpred, levels=levels_ps, cmap=cmap, norm=norm_ps, extend="both")
        axes[1].scatter(x_pseud, y_pseud, c="k", s=5, alpha=0.3)
        plt.colorbar(cax2, ax=axes[1], label="Calculated ρₐ (Ω·m)")
        axes[1].set_title(f"Iteration {iteration} — Calculated Data (RMS: {rms:.2f}%)")
        axes[1].set_ylabel("Pseudo-depth (m)")
    except Exception:
        # Fallback to scatter
        sc1 = axes[0].scatter(x_pseud, y_pseud, c=v_obs, cmap=cmap, norm=norm_ps, s=40, edgecolors="k", linewidths=0.3)
        plt.colorbar(sc1, ax=axes[0], label="Observed ρₐ (Ω·m)")
        axes[0].set_title(f"Iteration {iteration} — Observed Data")
        axes[0].set_ylabel("Pseudo-depth (m)")
        
        sc2 = axes[1].scatter(x_pseud, y_pseud, c=dpred, cmap=cmap, norm=norm_ps, s=40, edgecolors="k", linewidths=0.3)
        plt.colorbar(sc2, ax=axes[1], label="Calculated ρₐ (Ω·m)")
        axes[1].set_title(f"Iteration {iteration} — Calculated Data (RMS: {rms:.2f}%)")
        axes[1].set_ylabel("Pseudo-depth (m)")
    
    # 3. Model
    x_mod = cell_centers[:, 0]
    z_mod = cell_centers[:, 1]
    res_vals = resistivity.copy().astype(float)
    
    # Clamp to positive values for LogNorm
    res_vals = np.clip(res_vals, 1e-2, None)
    
    # Render full tri mesh
    r_vis = res_vals
    
    if len(r_vis) > 0:
        vmin_m = max(1e-2, np.nanmin(r_vis))
        vmax_m = max(vmin_m * 1.1, np.nanmax(r_vis))
    else:
        vmin_m, vmax_m = 1.0, 100.0
    
    norm_mod = mcolors.LogNorm(vmin=vmin_m, vmax=vmax_m)
    levels_mod = np.logspace(np.log10(vmin_m), np.log10(vmax_m), 20)
    
    try:
        tri_mod = Triangulation(x_mod, z_mod)
        tc = axes[2].tricontourf(tri_mod, r_vis, levels=levels_mod, cmap=cmap, norm=norm_mod, extend="both")
    except Exception:
        try:
            tc = axes[2].scatter(x_mod, z_mod, c=r_vis, cmap=cmap, norm=norm_mod, s=15)
        except Exception:
            tc = None
            
    # Set shared global bounds
    x_min_bound = x_mod.min()
    x_max_bound = x_mod.max()
    z_min_bound = z_mod.min()
    z_max_bound = z_mod.max()
    
    # Sync all axes horizontally and apply 5% padding above Y=0
    x_pad = (x_max_bound - x_min_bound) * 0.02
    for ax in axes:
        ax.set_xlim(x_min_bound - x_pad, x_max_bound + x_pad)
        
    z_pad = (z_max_bound - z_min_bound) * 0.05
    # Since z is negative, 0 is typically the max bound. Padding it upwards (positive)
    axes[2].set_ylim(z_min_bound - z_pad, z_max_bound + z_pad)
    
    p_min = y_pseud.min()
    p_max = y_pseud.max() # usually closely negative
    p_pad = np.abs(p_min) * 0.05
    axes[0].set_ylim(p_min - p_pad, p_pad)
    axes[1].set_ylim(p_min - p_pad, p_pad)
    
    # Draw a thin grey line at Y=0 to demark the surface
    for ax in axes:
        ax.axhline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
        
    if tc is not None:
        plt.colorbar(tc, ax=axes[2], label="Resistivity (Ω·m)")
    axes[2].set_title(f"Iteration {iteration} — Inverted Model")
    axes[2].set_xlabel("Distance (m)")
    axes[2].set_ylabel("Depth (m)")
    
    fig.tight_layout()
    return fig


# -------------------------------------------------------------------------
# Mesh visualization
# -------------------------------------------------------------------------

def plot_mesh(
    mesh,
    ax: Optional[plt.Axes] = None,
    title: str = "Inversion Mesh",
    figsize: Tuple[int, int] = (12, 5),
) -> Figure:
    """Plot a discretize mesh.

    Parameters
    ----------
    mesh : discretize.TensorMesh
    ax : matplotlib Axes, optional
    title : str
    figsize : tuple

    Returns
    -------
    Figure
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    try:
        mesh.plot_grid(ax=ax)
    except Exception:
        ax.text(0.5, 0.5, "Could not render mesh grid",
                ha="center", va="center", transform=ax.transAxes)

    ax.set_title(title)
    fig.tight_layout()
    return fig


# -------------------------------------------------------------------------
# Residual plot
# -------------------------------------------------------------------------

def plot_residuals(
    observed: np.ndarray,
    predicted: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Data Misfit",
    figsize: Tuple[int, int] = (8, 6),
) -> Figure:
    """Plot observed vs predicted and the residual histogram.

    Parameters
    ----------
    observed, predicted : np.ndarray
    ax : matplotlib Axes, optional
    title : str
    figsize : tuple

    Returns
    -------
    Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    # Scatter
    axes[0].scatter(observed, predicted, s=10, alpha=0.7)
    lo = min(observed.min(), predicted.min())
    hi = max(observed.max(), predicted.max())
    axes[0].plot([lo, hi], [lo, hi], "r--", lw=1)
    axes[0].set_xlabel("Observed ρₐ")
    axes[0].set_ylabel("Predicted ρₐ")
    axes[0].set_title("Obs. vs Pred.")

    # Histogram of relative residuals
    residual = (predicted - observed) / observed * 100
    axes[1].hist(residual, bins=30, edgecolor="k", alpha=0.7)
    axes[1].set_xlabel("Relative Residual (%)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")

    fig.suptitle(title)
    fig.tight_layout()
    return fig


# -------------------------------------------------------------------------
# Sensitivity map
# -------------------------------------------------------------------------

def plot_sensitivity(
    cell_centers: np.ndarray,
    sensitivity: np.ndarray,
    ax: Optional[plt.Axes] = None,
    cmap: str = "magma",
    title: str = "Sensitivity Map",
    figsize: Tuple[int, int] = (12, 5),
) -> Figure:
    """Plot the cumulative sensitivity map.

    Parameters
    ----------
    cell_centers : np.ndarray
        (N, 2) coordinates.
    sensitivity : np.ndarray
        (N,) sensitivity values.
    ax : matplotlib Axes, optional
    cmap : str
    title : str
    figsize : tuple

    Returns
    -------
    Figure
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x = cell_centers[:, 0]
    z = cell_centers[:, 1]
    values = np.log10(np.abs(sensitivity) + 1e-30)

    try:
        tri = Triangulation(x, z)
        tc = ax.tripcolor(tri, values, cmap=cmap, shading="flat")
    except Exception:
        tc = ax.scatter(x, z, c=values, cmap=cmap, s=8)

    plt.colorbar(tc, ax=ax, label="log₁₀(Sensitivity)")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig
