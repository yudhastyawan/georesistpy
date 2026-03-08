"""
Array configuration helpers for common electrode arrangements.

Supports Wenner, Schlumberger, Dipole-Dipole, Pole-Dipole, and Pole-Pole
arrays.  Each helper returns a DataFrame of (A, B, M, N) electrode indices
suitable for forward modelling or survey planning.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARRAY_TYPES: List[str] = [
    "wenner",
    "schlumberger",
    "dipole-dipole",
    "pole-dipole",
    "pole-pole",
]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def generate_array(
    array_type: str,
    n_electrodes: int,
    spacing: float = 1.0,
    max_n: int = 6,
) -> pd.DataFrame:
    """Generate an electrode configuration table.

    Parameters
    ----------
    array_type : str
        One of :data:`ARRAY_TYPES`.
    n_electrodes : int
        Total number of electrodes on the profile line.
    spacing : float, optional
        Unit electrode spacing in metres (default 1.0).
    max_n : int, optional
        Maximum dipole separation factor *n* (default 6).

    Returns
    -------
    pd.DataFrame
        Columns ``a``, ``b``, ``m``, ``n`` (electrode positions in metres)
        plus ``k`` (geometric factor).
    """
    array_type = array_type.lower().strip()
    if array_type not in ARRAY_TYPES:
        raise ValueError(
            f"Unknown array type '{array_type}'. Choose from {ARRAY_TYPES}"
        )

    positions = np.arange(n_electrodes, dtype=float) * spacing

    func = {
        "wenner": _wenner,
        "schlumberger": _schlumberger,
        "dipole-dipole": _dipole_dipole,
        "pole-dipole": _pole_dipole,
        "pole-pole": _pole_pole,
    }[array_type]

    return func(positions, max_n=max_n)


def electrode_positions(
    n_electrodes: int,
    spacing: float = 1.0,
    topo: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Return an (N, 2) or (N, 3) array of electrode coordinates.

    Parameters
    ----------
    n_electrodes : int
        Number of electrodes.
    spacing : float
        Uniform spacing along the profile.
    topo : sequence of float, optional
        Elevation at each electrode.  If *None*, z = 0 for all.

    Returns
    -------
    np.ndarray
        Shape ``(N, 2)`` with columns ``(x, z)`` or ``(N, 3)`` with
        ``(x, y, z)`` when topography is supplied.
    """
    x = np.arange(n_electrodes, dtype=float) * spacing
    if topo is not None:
        z = np.asarray(topo, dtype=float)
        return np.column_stack([x, z])
    return np.column_stack([x, np.zeros(n_electrodes)])


def geometric_factor(
    a: float, b: float, m: float, n: float
) -> float:
    """Compute the geometric factor *k* for a four-electrode configuration.

    Parameters
    ----------
    a, b, m, n : float
        Positions of current (A, B) and potential (M, N) electrodes.

    Returns
    -------
    float
        Geometric factor *k*.
    """
    am = abs(a - m)
    an = abs(a - n)
    bm = abs(b - m)
    bn = abs(b - n)

    # Guard against zero distances (pole configs)
    def _inv(d: float) -> float:
        return 1.0 / d if d > 1e-12 else 0.0

    k_inv = _inv(am) - _inv(an) - _inv(bm) + _inv(bn)
    if abs(k_inv) < 1e-30:
        return np.inf
    return 2.0 * np.pi / k_inv


# ---------------------------------------------------------------------------
# Private array generators
# ---------------------------------------------------------------------------


def _wenner(positions: np.ndarray, **_kw) -> pd.DataFrame:
    """Wenner-alpha array: AM = MN = NB = a."""
    rows = []
    ne = len(positions)
    a_vals = range(1, ne)  # spacing multiplier
    for a in a_vals:
        for i in range(ne - 3 * a):
            ai = positions[i]
            mi = positions[i + a]
            ni = positions[i + 2 * a]
            bi = positions[i + 3 * a]
            k = geometric_factor(ai, bi, mi, ni)
            rows.append(dict(a=ai, b=bi, m=mi, n=ni, k=k))
    return pd.DataFrame(rows)


def _schlumberger(positions: np.ndarray, max_n: int = 6, **_kw) -> pd.DataFrame:
    """Schlumberger array: AB >> MN, symmetric about centre."""
    rows = []
    ne = len(positions)
    spacing = positions[1] - positions[0] if ne > 1 else 1.0
    for mn_mult in range(1, max(2, ne // 4)):
        mn = mn_mult * spacing
        for ab_mult in range(mn_mult + 1, ne // 2 + 1):
            ab_half = ab_mult * spacing
            centre_indices = range(ab_mult, ne - ab_mult)
            for ci in centre_indices:
                cx = positions[ci]
                ai = cx - ab_half
                bi = cx + ab_half
                mi = cx - mn / 2
                ni = cx + mn / 2
                if ai < positions[0] - 1e-6 or bi > positions[-1] + 1e-6:
                    continue
                k = geometric_factor(ai, bi, mi, ni)
                rows.append(dict(a=ai, b=bi, m=mi, n=ni, k=k))
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["a", "b", "m", "n", "k"])


def _dipole_dipole(positions: np.ndarray, max_n: int = 6, **_kw) -> pd.DataFrame:
    """Dipole-dipole array: AB dipole separated from MN dipole by n×a."""
    rows = []
    ne = len(positions)
    for a_mult in range(1, ne):
        for i in range(ne - a_mult):
            for n_sep in range(1, min(max_n, ne) + 1):
                j = i + a_mult + (n_sep - 1) * a_mult
                if j + a_mult >= ne:
                    break
                ai = positions[i]
                bi = positions[i + a_mult]
                mi = positions[j]
                ni = positions[j + a_mult]
                k = geometric_factor(ai, bi, mi, ni)
                rows.append(dict(a=ai, b=bi, m=mi, n=ni, k=k))
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["a", "b", "m", "n", "k"])


def _pole_dipole(positions: np.ndarray, max_n: int = 6, **_kw) -> pd.DataFrame:
    """Pole-dipole array: single current electrode A, remote B at infinity."""
    rows = []
    ne = len(positions)
    b_inf = positions[-1] + 1000.0 * (positions[-1] - positions[0])
    for a_mult in range(1, ne):
        for i in range(ne - a_mult):
            for n_sep in range(1, min(max_n, ne) + 1):
                j = i + 1 + (n_sep - 1)
                if j + a_mult >= ne:
                    break
                ai = positions[i]
                mi = positions[j]
                ni = positions[j + a_mult]
                k = geometric_factor(ai, b_inf, mi, ni)
                rows.append(dict(a=ai, b=b_inf, m=mi, n=ni, k=k))
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["a", "b", "m", "n", "k"])


def _pole_pole(positions: np.ndarray, max_n: int = 6, **_kw) -> pd.DataFrame:
    """Pole-pole array: A injects, B and N at infinity."""
    rows = []
    ne = len(positions)
    far = positions[-1] + 1000.0 * (positions[-1] - positions[0])
    for sep in range(1, min(max_n, ne)):
        for i in range(ne - sep):
            ai = positions[i]
            mi = positions[i + sep]
            k = geometric_factor(ai, far, mi, far * 1.01)
            rows.append(dict(a=ai, b=far, m=mi, n=far * 1.01, k=k))
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["a", "b", "m", "n", "k"])
