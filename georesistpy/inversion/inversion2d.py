"""
2-D ERT inversion.

Smooth-constrained and robust (L1) inversion of 2-D apparent-resistivity
data using SimPEG's DC resistivity module.  Includes L-curve regularisation
data using SimPEG's DC resistivity module. Includes L-curve regularisation
search and automatic lambda estimation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from georesistpy.mesh.mesh_builder import create_mesh_2d


@dataclass
class InversionResult2D:
    """Store 2-D ERT inversion results."""
    resistivity: np.ndarray
    mesh: Any  # discretize.TensorMesh
    rms: float
    chi2: float
    n_iterations: int
    response: np.ndarray
    sensitivity: Optional[np.ndarray] = None
    coverage: Optional[np.ndarray] = None


from simpeg.directives import InversionDirective

class IterationProgressDirective(InversionDirective):
    """SimPEG directive to fire a callback at the end of each iteration."""

    def __init__(self, callback: Callable):
        super().__init__()
        self.callback = callback

    def initialize(self):
        pass

    def endIter(self):
        if self.callback is None:
            return

        try:
            m_current = self.opt.xc
            dpred_current = self.invProb.dpred
        except Exception:
            return

        # Navigate dmisfit — may be wrapped in ComboObjectiveFunction
        dmis_obj = self.invProb.dmisfit
        # If wrapped in ComboObjectiveFunction, get first child
        if hasattr(dmis_obj, 'objfcts'):
            for obj in dmis_obj.objfcts:
                if hasattr(obj, 'data') and hasattr(obj, 'simulation'):
                    dmis_obj = obj
                    break

        # Calculate RMS
        rms = 0.0
        try:
            dobs = dmis_obj.data.dobs
            residuals = (np.abs(dpred_current) - dobs) / dobs * 100.0
            rms = float(np.sqrt(np.mean(residuals ** 2)))
        except Exception:
            pass

        # Convert model to resistivity
        sigma_current = np.exp(m_current)
        rho_current = 1.0 / sigma_current
        # Extract cell centers
        cell_centers = None
        try:
            cell_centers = dmis_obj.simulation.mesh.cell_centers
        except Exception:
            pass
            
        self.callback(self.opt.iter, rho_current, np.abs(dpred_current), rms, cell_centers)


def invert_2d(
    data: pd.DataFrame,
    m0: Optional[np.ndarray] = None,
    lam: float = 10.0,
    max_iter: int = 20,
    robust: bool = False,
    error_level: float = 0.03,
    verbose: bool = True,
    iteration_callback: Optional[Callable] = None,
    expansion_z: float = 1.0,
    max_cells_x: int = 60,
    max_cells_z: int = 30,
    depth_max: Optional[float] = None,
) -> InversionResult2D:
    """Run 2-D ERT inversion using SimPEG."""
    from simpeg import (
        maps,
        data as simpeg_data,
        data_misfit,
        regularization,
        optimization,
        inverse_problem,
        directives,
        inversion as simpeg_inversion,
    )
    from simpeg.electromagnetics.static import resistivity as dc
    from georesistpy.mesh.mesh_builder import create_mesh_2d

    # 1. Prepare electrode positions and mesh
    elec_x = np.unique(np.concatenate([data['a'], data['b'], data['m'], data['n']]))
    elec_pos = np.column_stack([elec_x, np.zeros_like(elec_x)])

    mesh = create_mesh_2d(
        elec_pos,
        depth_max=depth_max,
        expansion_z=expansion_z,
        max_cells_x=max_cells_x,
        max_cells_z=max_cells_z
    )

    # 2. Map electrode positions to indices
    def _idx(val):
        return int(np.argmin(np.abs(elec_pos[:, 0] - val)))

    # 3. Build SimPEG survey
    source_list = []
    for _, row in data.iterrows():
        a_i, b_i = _idx(row["a"]), _idx(row["b"])
        m_i, n_i = _idx(row["m"]), _idx(row["n"])

        rx = dc.receivers.Dipole(
            locations_m=elec_pos[m_i].reshape(1, -1),
            locations_n=elec_pos[n_i].reshape(1, -1),
            data_type="apparent_resistivity",
        )
        src = dc.sources.Dipole(
            receiver_list=[rx],
            location_a=elec_pos[a_i],
            location_b=elec_pos[b_i],
        )
        source_list.append(src)

    survey = dc.survey.Survey(source_list)
    survey.set_geometric_factor()

    # 4. Observed data & errors
    dobs = data["rhoa"].values.astype(float)
    std = np.abs(dobs) * error_level
    data_obj = simpeg_data.Data(survey, dobs=dobs, standard_deviation=std)

    # 5. Mapping & initial model
    sigma_map = maps.ExpMap(mesh)
    if m0 is None:
        m0 = np.log(1.0 / np.median(dobs)) * np.ones(mesh.nC)

    # 6. Simulation
    sim = dc.Simulation2DNodal(
        mesh,
        survey=survey,
        sigmaMap=sigma_map,
    )

    # 7. Inversion components
    dmis = data_misfit.L2DataMisfit(data=data_obj, simulation=sim)
    if robust:
        reg = regularization.Sparse(mesh, mapping=maps.IdentityMap(mesh))
        reg.norms = [1, 1, 1, 1]
    else:
        reg = regularization.WeightedLeastSquares(mesh)

    reg.alpha_s = 1e-4
    reg.alpha_x = 1.0
    reg.alpha_y = 1.0

    opt = optimization.InexactGaussNewton(maxIter=max_iter, maxIterCG=20)
    opt.remember("xc")

    inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

    # Directives
    beta_est = directives.BetaEstimate_ByEig(beta0_ratio=lam)
    beta_cool = directives.BetaSchedule(coolingFactor=2, coolingRate=1)
    target = directives.TargetMisfit()
    
    dir_list = [beta_est, beta_cool, target]
    if iteration_callback is not None:
        dir_list.append(IterationProgressDirective(iteration_callback))

    inv = simpeg_inversion.BaseInversion(inv_prob, directiveList=dir_list)

    # Fire callback for Initial Model (Iter 0) if requested
    if iteration_callback is not None:
        try:
            dpred_0 = sim.dpred(m0)
            rho_0 = 1.0 / np.exp(m0)
            try:
                cell_centers_0 = sim.mesh.cell_centers
            except Exception:
                cell_centers_0 = None
                
            # Compute initial RMS
            dobs_0 = data["rhoa"].values.astype(float)
            res_0 = (np.abs(dpred_0) - dobs_0) / dobs_0 * 100.0
            rms_0 = float(np.sqrt(np.mean(res_0 ** 2)))
            
            iteration_callback(0, rho_0, np.abs(dpred_0), rms_0, cell_centers_0)
        except Exception:
            pass

    # 8. Run inversion
    m_recovered = inv.run(m0)
    rho_recovered = 1.0 / np.exp(m_recovered)
    dpred = sim.dpred(m_recovered)
    response = np.abs(dpred)
    dobs = data["rhoa"].values.astype(float)
    rms = float(np.sqrt(np.mean(((response - dobs) / dobs) ** 2))) * 100.0

    return InversionResult2D(
        resistivity=rho_recovered,
        mesh=mesh,
        rms=rms,
        chi2=0.0,
        n_iterations=opt.iter,
        response=response,
        sensitivity=None,
        coverage=None,
    )


def SimPEG_data(survey, dobs, standard_deviation):
    """Create a SimPEG Data object (handles API differences)."""
    from simpeg import data as simpeg_data
    return simpeg_data.Data(survey, dobs=dobs, standard_deviation=standard_deviation)


def l_curve_search(
    data: pd.DataFrame,
    lambdas: Optional[Sequence[float]] = None,
    **invert_kwargs,
) -> Tuple[List[float], List[float], List[float]]:
    """Perform an L-curve search over regularisation parameters.

    Parameters
    ----------
    data : pd.DataFrame
        Survey data (same format as :func:`invert_2d`).
    lambdas : sequence of float, optional
        λ values to test (default logspace from 1 to 1000).
    **invert_kwargs
        Additional keyword arguments forwarded to :func:`invert_2d`.

    Returns
    -------
    lambdas : list of float
    rms_values : list of float
    roughness_values : list of float
    """
    if lambdas is None:
        lambdas_arr = np.logspace(0, 3, 10)
    else:
        lambdas_arr = np.asarray(lambdas, dtype=float)

    rms_list: List[float] = []
    rough_list: List[float] = []
    lam_list: List[float] = []

    for lam_val in lambdas_arr:
        try:
            result = invert_2d(data, lam=float(lam_val), **invert_kwargs)
            rms_list.append(result.rms)
            # Roughness proxy = std of model
            rough_list.append(float(np.std(result.resistivity)))
            lam_list.append(float(lam_val))
        except Exception:
            continue

    return lam_list, rms_list, rough_list
