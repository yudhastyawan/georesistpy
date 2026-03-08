"""
2-D forward modelling for Electrical Resistivity Tomography.

Uses SimPEG's DC resistivity module to simulate apparent resistivity
data on a tensor mesh with a given resistivity distribution.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def forward_2d(
    electrode_positions: np.ndarray,
    scheme: pd.DataFrame,
    resistivity_model: np.ndarray,
    mesh=None,
    noise_level: float = 0.0,
    noise_abs: float = 0.0,
):
    """Run a 2-D ERT forward simulation.

    Parameters
    ----------
    electrode_positions : np.ndarray
        (N, 2) electrode coordinates ``(x, z)``.
    scheme : pd.DataFrame
        Measurement schedule with columns ``a, b, m, n`` (electrode
        *indices* 0-based).
    resistivity_model : np.ndarray
        Resistivity per mesh cell (Ω·m).
    mesh : discretize.TensorMesh, optional
        Pre-built mesh.  If *None*, one is generated automatically.
    noise_level : float, optional
        Relative Gaussian noise to add (e.g. 0.03 = 3 %).
    noise_abs : float, optional
        Absolute Gaussian noise floor (Ω·m).

    Returns
    -------
    dict
        ``'rhoa'`` — simulated apparent resistivity (np.ndarray),
        ``'mesh'`` — the mesh used,
        ``'data'`` — None.
    """
    from simpeg.electromagnetics.static import resistivity as dc
    from simpeg import maps

    # --- Build sensor list ------------------------------------------------
    pos = np.atleast_2d(electrode_positions)
    if pos.shape[1] < 2:
        pos = np.column_stack([pos.ravel(), np.zeros(len(pos))])

    # --- Mesh ------------------------------------------------------------
    if mesh is None:
        from georesistpy.mesh.mesh_builder import create_mesh_2d
        mesh = create_mesh_2d(pos)

    # --- Build survey from scheme -----------------------------------------
    source_list = []
    for _, row in scheme.iterrows():
        a_idx, b_idx = int(row["a"]), int(row["b"])
        m_idx, n_idx = int(row["m"]), int(row["n"])

        # Electrode locations
        a_loc = pos[a_idx]
        b_loc = pos[b_idx]
        m_loc = pos[m_idx]
        n_loc = pos[n_idx]

        rx = dc.receivers.Dipole(
            locations_m=m_loc.reshape(1, -1),
            locations_n=n_loc.reshape(1, -1),
            data_type="apparent_resistivity",
        )
        src = dc.sources.Dipole(
            receiver_list=[rx],
            location_a=a_loc,
            location_b=b_loc,
        )
        source_list.append(src)

    survey = dc.survey.Survey(source_list)
    survey.set_geometric_factor()

    # --- Simulation ------------------------------------------------------
    sigma_model = 1.0 / resistivity_model
    sigma_map = maps.IdentityMap(mesh)

    sim = dc.Simulation2DNodal(
        mesh,
        survey=survey,
        sigmaMap=sigma_map,
    )

    dpred = sim.dpred(sigma_model)
    rhoa = np.abs(dpred)

    # Add noise
    if noise_level > 0 or noise_abs > 0:
        noise = np.random.default_rng().normal(
            0, noise_level * rhoa + noise_abs
        )
        rhoa = rhoa + noise

    return {
        "rhoa": rhoa,
        "mesh": mesh,
        "data": None,
    }
