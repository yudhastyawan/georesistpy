"""
Mesh generation for ERT modelling and inversion.

Uses ``discretize`` to create tensor meshes for 2-D ERT problems
and simple numpy arrays for 1-D VES layer models.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def create_mesh_2d(
    electrode_positions: np.ndarray,
    depth_max: Optional[float] = None,
    n_cells_x: int = 0,
    n_cells_z: int = 0,
    padding_factor: float = 1.5,
    expansion_z: float = 1.0,
    max_cells_x: int = 60,
    max_cells_z: int = 30,
):
    """Create a 2-D tensor mesh suitable for ERT.

    Uses ``discretize.TensorMesh`` to build a rectilinear mesh with
    core cells covering the electrode spread and padding cells on
    the sides and bottom.

    Parameters
    ----------
    electrode_positions : np.ndarray
        (N, 2) array of ``(x, z)`` electrode positions.
    depth_max : float, optional
        Maximum model depth. Defaults to half the profile length.
    n_cells_x : int
        Number of core cells in x. If 0, auto-computed.
    n_cells_z : int
        Number of core cells in z. If 0, auto-computed.
    padding_factor : float
        Expansion factor for padding cells (default 1.5).
    expansion_z : float
        Geometric expansion factor for core cells in Z (default 1.0, no expansion).
    max_cells_x : int
        Maximum number of core cells in X (default 60).
    max_cells_z : int
        Maximum number of core cells in Z (default 30).

    Returns
    -------
    discretize.TensorMesh
        The generated mesh object.
    """
    from discretize import TensorMesh

    pos = np.atleast_2d(electrode_positions)
    if pos.shape[1] < 2:
        pos = np.column_stack([pos.ravel(), np.zeros(len(pos))])

    x_min, x_max = pos[:, 0].min(), pos[:, 0].max()
    spread = x_max - x_min

    if depth_max is None:
        depth_max = spread * 0.5

    # Use minimum electrode spacing
    x_sorted = np.sort(np.unique(pos[:, 0]))
    if len(x_sorted) > 1:
        dx_elec = np.min(np.diff(x_sorted))
    else:
        dx_elec = spread if spread > 0 else 1.0

    # Core cell size (dx is constant)
    core_dx = max(dx_elec, spread / float(max_cells_x)) if dx_elec > 0 else max(spread / float(max_cells_x), 1.0)
    
    if n_cells_x <= 0:
        n_cells_x = min(max(int(np.ceil(spread / core_dx)), 10), max_cells_x)
    
    # Recalculate dx to fit exactly
    core_dx = spread / float(n_cells_x)

    # Core cell widths in Z
    core_dz_init = core_dx # Start with square cells
    if n_cells_z <= 0:
        if expansion_z > 1.0:
            # Solve for n such that sum(dz_init * exp^i) >= depth_max
            # depth = dz_init * (exp^n - 1) / (exp - 1)
            # exp^n = depth * (exp - 1) / dz_init + 1
            # n = log(depth * (exp - 1) / dz_init + 1) / log(exp)
            n_float = np.log(depth_max * (expansion_z - 1.0) / core_dz_init + 1.0) / np.log(expansion_z)
            n_cells_z = min(max(int(np.ceil(n_float)), 10), max_cells_z)
        else:
            n_cells_z = min(max(int(np.ceil(depth_max / core_dz_init)), 10), max_cells_z)
    
    if expansion_z > 1.0:
        hz_core = [core_dz_init * expansion_z**i for i in range(n_cells_z)]
        hz_core = hz_core[::-1]  # Large to small, since SimPEG builds from bottom to top
    else:
        # Uniform dz to fit depth_max exactly
        core_dz = depth_max / float(n_cells_z)
        hz_core = [core_dz] * n_cells_z

    # Padding cells (bottom only for Z, geometric expansion downwards)
    n_pad = 5
    dz_bottom = hz_core[0]  # The bottom-most core cell (largest)
    
    # Keep downward padding geometric, even if core expansion is 1.0
    pad_z = [(dz_bottom * padding_factor ** i) for i in range(1, n_pad + 1)]
    pad_z = pad_z[::-1]  # Deepest cell is largest, so place it at the beginning

    # No geometric expansion in X as requested
    pad_x = [core_dx] * n_pad

    # Build cell widths
    hx = pad_x + [core_dx] * n_cells_x + pad_x
    hz = pad_z + hz_core

    mesh = TensorMesh([hx, hz], origin="CN")

    # Shift mesh so top is at z=0 and centered on electrode spread
    x_center = (x_min + x_max) / 2.0
    mesh.origin = np.array([
        x_center - mesh.h[0].sum() / 2.0,
        -mesh.h[1].sum(),
    ])

    return mesh


def create_mesh_1d(
    n_layers: int = 5,
    thickness: Optional[Sequence[float]] = None,
    max_depth: float = 50.0,
) -> np.ndarray:
    """Create a 1-D layer-depth model.

    Parameters
    ----------
    n_layers : int
        Number of layers (default 5).
    thickness : sequence of float, optional
        Layer thicknesses from top to bottom.  Length should be
        ``n_layers - 1`` (last layer extends to infinity).
        If *None*, thicknesses increase logarithmically.
    max_depth : float
        Maximum depth used when generating automatic thicknesses.

    Returns
    -------
    np.ndarray
        (n_layers - 1,) array of layer thicknesses in metres.
    """
    if thickness is not None:
        return np.asarray(thickness, dtype=float)

    # Log-spaced thicknesses
    depths = np.logspace(
        np.log10(0.5),
        np.log10(max_depth),
        n_layers,
    )
    t = np.diff(depths)
    return t
