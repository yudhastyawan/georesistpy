"""
1-D forward modelling for Vertical Electrical Sounding (VES).

Compute apparent resistivity for a horizontally layered earth model
using the linear filter method (Ghosh, 1971 / Anderson, 1979).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def forward_1d(
    resistivity: Sequence[float],
    thickness: Sequence[float],
    ab2: Sequence[float],
    mn2: Optional[Sequence[float]] = None,
    array_type: str = "schlumberger",
) -> np.ndarray:
    """Compute 1-D VES apparent resistivity.

    Parameters
    ----------
    resistivity : sequence of float
        Layer resistivities from top to bottom (Ω·m).
    thickness : sequence of float
        Layer thicknesses (m).  Length = ``len(resistivity) - 1``.
    ab2 : sequence of float
        Half current-electrode spacings AB/2 (m).
    mn2 : sequence of float, optional
        Half potential-electrode spacings MN/2 (m).  Required for
        Schlumberger; for Wenner ``mn2 = ab2 / 3``.
    array_type : str
        ``'schlumberger'`` or ``'wenner'`` (default Schlumberger).

    Returns
    -------
    np.ndarray
        Apparent resistivity at each AB/2 value.
    """
    rho = np.asarray(resistivity, dtype=float)
    thk = np.asarray(thickness, dtype=float)
    ab2_arr = np.asarray(ab2, dtype=float)

    if mn2 is None:
        if array_type.lower().startswith("w"):
            mn2_arr = ab2_arr / 3.0
        else:
            mn2_arr = np.minimum(ab2_arr * 0.1, np.ones_like(ab2_arr))
    else:
        mn2_arr = np.asarray(mn2, dtype=float)

    return _forward_1d_numpy(rho, thk, ab2_arr, mn2_arr)


# ------------------------------------------------------------------
# Pure-numpy implementation using Hankel transform
# ------------------------------------------------------------------

# Hankel filter coefficients (short Anderson filter)
_FC = np.array([
    2.8955942e-03, 1.3076872e-02, 3.1467350e-02, 4.6440350e-02,
    3.4458170e-02, -2.0427750e-02, -8.5782530e-02, -4.2472510e-02,
    1.2107610e-01, 1.8828220e-01, -1.2601250e-01, -5.2975850e-01,
    -2.4931480e-01, 1.4069260e+00, 4.0956670e+00, 4.0956670e+00,
    1.4069260e+00, -2.4931480e-01, -5.2975850e-01, -1.2601250e-01,
    1.8828220e-01, 1.2107610e-01, -4.2472510e-02, -8.5782530e-02,
    -2.0427750e-02, 3.4458170e-02, 4.6440350e-02, 3.1467350e-02,
    1.3076872e-02, 2.8955942e-03,
])

_ABSC_SHIFT = np.arange(len(_FC)) - len(_FC) // 2


def _kernel_1d(rho: np.ndarray, thk: np.ndarray, lam: float) -> float:
    """Compute the 1-D resistivity kernel T(λ) via recurrence."""
    n = len(rho)
    T = rho[-1]
    for i in range(n - 2, -1, -1):
        tanh_val = np.tanh(lam * thk[i])
        num = T + rho[i] * tanh_val
        den = 1.0 + T * tanh_val / rho[i]
        T = rho[i] * num / den
    return T


def _forward_1d_numpy(
    rho: np.ndarray,
    thk: np.ndarray,
    ab2: np.ndarray,
    mn2: np.ndarray,
) -> np.ndarray:
    """Pure-numpy 1-D VES forward using Hankel transform."""
    rhoa = np.zeros(len(ab2))
    for i, (a, m) in enumerate(zip(ab2, mn2)):
        val = 0.0
        for j, fc in enumerate(_FC):
            # Lambda values for the filter
            lam = np.exp(_ABSC_SHIFT[j] * np.log(10) / 6.0) / a
            T = _kernel_1d(rho, thk, lam)
            val += fc * T * lam
        rhoa[i] = val * a
    return rhoa
