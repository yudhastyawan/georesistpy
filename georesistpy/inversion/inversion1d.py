"""
1-D VES inversion.

Invert apparent-resistivity sounding curves to recover a layered-earth
resistivity model using scipy least-squares optimisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class InversionResult1D:
    """Container for 1-D inversion results.

    Attributes
    ----------
    resistivity : np.ndarray
        Inverted layer resistivities (Ω·m).
    thickness : np.ndarray
        Inverted layer thicknesses (m).
    rms : float
        Final RMS misfit (%).
    chi2 : float
        Chi-squared misfit.
    n_iterations : int
        Number of iterations performed.
    response : np.ndarray
        Model response (fitted apparent resistivity).
    rms_history : list of float
        RMS at each iteration.
    """

    resistivity: np.ndarray
    thickness: np.ndarray
    rms: float = 0.0
    chi2: float = 0.0
    n_iterations: int = 0
    response: np.ndarray = field(default_factory=lambda: np.array([]))
    rms_history: List[float] = field(default_factory=list)


def invert_1d(
    ab2: Sequence[float],
    rhoa_obs: Sequence[float],
    n_layers: int = 5,
    mn2: Optional[Sequence[float]] = None,
    start_model: Optional[Sequence[float]] = None,
    lam: float = 20.0,
    max_iter: int = 30,
    error: Optional[Sequence[float]] = None,
    verbose: bool = False,
) -> InversionResult1D:
    """Run 1-D VES inversion.

    Parameters
    ----------
    ab2 : sequence of float
        Half current-electrode spacings (m).
    rhoa_obs : sequence of float
        Observed apparent resistivity (Ω·m).
    n_layers : int
        Number of layers in the model (default 5).
    mn2 : sequence of float, optional
        Half potential-electrode spacings; defaults to ``ab2 / 10``.
    start_model : sequence of float, optional
        Starting model ``[thk_1 … thk_{n-1}, rho_1 … rho_n]``.
    lam : float
        Regularisation parameter λ (default 20).
    max_iter : int
        Maximum iterations (default 30).
    error : sequence of float, optional
        Data error (fraction); defaults to 3 %.
    verbose : bool
        Print iteration info.

    Returns
    -------
    InversionResult1D
    """
    ab2_arr = np.asarray(ab2, dtype=float)
    rhoa_arr = np.asarray(rhoa_obs, dtype=float)
    if mn2 is None:
        mn2_arr = ab2_arr * 0.1
    else:
        mn2_arr = np.asarray(mn2, dtype=float)

    return _invert_1d_scipy(
        ab2_arr, mn2_arr, rhoa_arr, n_layers, start_model, max_iter, error,
    )


# ------------------------------------------------------------------
# scipy backend
# ------------------------------------------------------------------


def _invert_1d_scipy(
    ab2, mn2, rhoa_obs, n_layers, start_model, max_iter, error,
) -> InversionResult1D:
    from scipy.optimize import least_squares
    from georesistpy.forward.forward1d import forward_1d

    n_thk = n_layers - 1

    if start_model is not None:
        x0 = np.log(np.asarray(start_model, dtype=float))
    else:
        # Simple starting model
        rho0 = np.full(n_layers, np.median(rhoa_obs))
        thk0 = np.logspace(np.log10(1.0), np.log10(ab2.max() / 2), n_thk)
        x0 = np.log(np.concatenate([thk0, rho0]))

    err = np.full(len(rhoa_obs), 0.03) if error is None else np.asarray(error)
    weights = 1.0 / (err * rhoa_obs)

    rms_hist: List[float] = []

    def residuals(x):
        params = np.exp(x)
        thk = params[:n_thk]
        rho = params[n_thk:]
        pred = forward_1d(rho, thk, ab2, mn2)
        r = (pred - rhoa_obs) * weights
        rms_val = float(np.sqrt(np.mean(((pred - rhoa_obs) / rhoa_obs) ** 2)) * 100)
        rms_hist.append(rms_val)
        return r

    result = least_squares(residuals, x0, max_nfev=max_iter * len(x0))
    params = np.exp(result.x)
    thk = params[:n_thk]
    rho = params[n_thk:]

    resp = forward_1d(rho, thk, ab2, mn2)
    rms = float(np.sqrt(np.mean(((resp - rhoa_obs) / rhoa_obs) ** 2)) * 100)

    return InversionResult1D(
        resistivity=rho,
        thickness=thk,
        rms=rms,
        chi2=0.0,
        n_iterations=result.nfev,
        response=resp,
        rms_history=rms_hist,
    )
