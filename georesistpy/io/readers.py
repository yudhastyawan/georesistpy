"""
Data readers for common geoelectrical survey formats.

Supported formats
-----------------
- CSV / TXT — generic tabular data
- ABEM — Terrameter LS / SAS export
- Syscal — Iris Instruments Syscal Pro/R8 export
- RES2DINV — Loke's RES2DINV data format (e.g., .dat, .txt)
- Generic electrode-spacing tables
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column-name aliases used for auto-detection
# ---------------------------------------------------------------------------

_COL_ALIASES: Dict[str, Sequence[str]] = {
    "a": ["a", "c1", "c+", "a_pos", "electrode_a", "current_a"],
    "b": ["b", "c2", "c-", "b_pos", "electrode_b", "current_b"],
    "m": ["m", "p1", "p+", "m_pos", "electrode_m", "potential_m"],
    "n": ["n", "p2", "p-", "n_pos", "electrode_n", "potential_n"],
    "rhoa": [
        "rhoa", "rho_a", "app_res", "apparent_resistivity",
        "resistivity", "res", "ohm_m",
    ],
    "i": ["i", "current", "i_ma"],
    "v": ["v", "voltage", "potential", "v_mv"],
    "k": ["k", "geom_factor", "geometric_factor"],
}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case column names and map known aliases to canonical names."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    rename_map = {}
    for canonical, aliases in _COL_ALIASES.items():
        for alias in aliases:
            if alias in df.columns and canonical not in df.columns:
                rename_map[alias] = canonical
                break
    df.rename(columns=rename_map, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Public readers
# ---------------------------------------------------------------------------


def read_csv(
    filepath: Union[str, Path],
    sep: str = ",",
    **kwargs,
) -> pd.DataFrame:
    """Read a CSV file containing apparent-resistivity survey data.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.
    sep : str, optional
        Column separator (default ``','``).
    **kwargs
        Forwarded to :func:`pandas.read_csv`.

    Returns
    -------
    pd.DataFrame
        Normalised survey data.
    """
    df = pd.read_csv(filepath, sep=sep, **kwargs)
    return _normalise_columns(df)


def read_txt(
    filepath: Union[str, Path],
    sep: Optional[str] = None,
    comment: str = "#",
    **kwargs,
) -> pd.DataFrame:
    """Read a whitespace- or tab-separated text file.

    Parameters
    ----------
    filepath : str or Path
        Path to the text file.
    sep : str or None
        Separator — ``None`` lets pandas infer whitespace.
    comment : str
        Comment character.
    **kwargs
        Forwarded to :func:`pandas.read_csv`.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(
        filepath,
        sep=sep,
        comment=comment,
        engine="python",
        **kwargs,
    )
    return _normalise_columns(df)


def read_abem(filepath: Union[str, Path]) -> pd.DataFrame:
    """Read an ABEM Terrameter export file (.dat / .txt).

    The ABEM format typically has a header block followed by tabular data
    with columns like ``Spa.1  Spa.2  Spa.3  Spa.4  Rho(a)``.

    Parameters
    ----------
    filepath : str or Path
        Path to the ABEM file.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(filepath)
    lines = path.read_text(errors="replace").splitlines()

    # Skip header lines
    data_start = 0
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "" or stripped.startswith(("#", "/", "!")):
            data_start = idx + 1
            continue
        # First line with mostly numeric tokens → data start
        tokens = stripped.split()
        try:
            [float(t.replace(",", ".")) for t in tokens[:4]]
            data_start = idx
            break
        except ValueError:
            # Might be column header row
            if any(k in stripped.lower() for k in ("spa", "rho", "elec")):
                data_start = idx
                break
            data_start = idx + 1

    df = pd.read_csv(
        io.StringIO("\n".join(lines[data_start:])),
        sep=r"\s+",
        engine="python",
        header=0 if not lines[data_start].strip()[0].isdigit() else None,
    )

    # If no column names, assign standard ones
    if all(isinstance(c, int) for c in df.columns):
        ncols = len(df.columns)
        if ncols >= 5:
            names = ["a", "b", "m", "n", "rhoa"] + [
                f"col{i}" for i in range(5, ncols)
            ]
            df.columns = names[:ncols]

    return _normalise_columns(df)


def read_syscal(filepath: Union[str, Path]) -> pd.DataFrame:
    """Read an Iris Instruments Syscal Pro export file.

    Parameters
    ----------
    filepath : str or Path
        Path to the Syscal file.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(filepath)
    text = path.read_text(errors="replace")

    # Syscal files often start with metadata lines ending at a "---" or blank
    lines = text.splitlines()
    data_start = 0
    for idx, line in enumerate(lines):
        if re.match(r"^\s*-{3,}", line) or line.strip() == "":
            data_start = idx + 1
            continue
        tokens = line.strip().split()
        if len(tokens) >= 5:
            try:
                float(tokens[0])
                data_start = idx
                break
            except ValueError:
                if idx > 0:
                    data_start = idx
                    break

    df = pd.read_csv(
        io.StringIO("\n".join(lines[data_start:])),
        sep=r"\s+",
        engine="python",
        header=0 if not lines[data_start].strip()[0].isdigit() else None,
    )

    if all(isinstance(c, int) for c in df.columns):
        ncols = len(df.columns)
        base_names = ["a", "b", "m", "n", "rhoa", "i", "v"]
        names = base_names[:ncols] + [f"col{i}" for i in range(len(base_names), ncols)]
        df.columns = names[:ncols]

    return _normalise_columns(df)


def read_generic(
    filepath: Union[str, Path],
    columns: Optional[Sequence[str]] = None,
    sep: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """Read a generic electrode-spacing table.

    Parameters
    ----------
    filepath : str or Path
        Path to the data file.
    columns : list of str, optional
        Explicit column names.  If *None*, first row is used as header.
    sep : str or None
        Separator; ``None`` lets pandas auto-detect.
    **kwargs
        Forwarded to :func:`pandas.read_csv`.

    Returns
    -------
    pd.DataFrame
    """
    if columns is not None:
        kwargs["header"] = None
        kwargs["names"] = columns
    df = pd.read_csv(filepath, sep=sep, engine="python", **kwargs)
    return _normalise_columns(df)


def read_res2dinv(filepath: Union[str, Path]) -> pd.DataFrame:
    """Read a RES2DINV format data file.
    
    The RES2DINV format starts with a header of metadata (Name, Spacing, 
    Array type number, Total data points) followed by rows of data.
    The most common general format is: x, a, n, rho_a.
    Where x is center, a is spacing, n is factor. We convert this to
    electrode positions a, b, m, n depending on the array type.
    
    Parameters
    ----------
    filepath : str, Path, or file-like object
        Path to the RES2DINV file or a file-like object yielding text.
        
    Returns
    -------
    pd.DataFrame
        Data block converted to canonical A, B, M, N, Rhoa columns.
    """
    if hasattr(filepath, "read"):
        text = filepath.read()
    else:
        text = Path(filepath).read_text(errors="replace")
    
    lines = text.splitlines()
    
    if len(lines) < 9:
        raise ValueError("File too short to be RES2DINV format.")
        
    line_idx = 0
    # Skip potential empty lines at start
    while line_idx < len(lines) and not lines[line_idx].strip():
        line_idx += 1
        
    name = lines[line_idx].strip()
    
    # Sometimes spacing has a trailing '.0' like '3.5.0', so we clean it
    spacing_str = lines[line_idx + 1].strip()
    if spacing_str.count('.') > 1:
        parts = spacing_str.split('.')
        spacing_str = f"{parts[0]}.{parts[1]}"
    a_spacing = float(spacing_str)
    
    array_type = int(lines[line_idx + 2].split()[0].strip())
    n_data = int(lines[line_idx + 3].split()[0].strip())
    x_loc_type = int(lines[line_idx + 4].split()[0].strip())
    
    # Header usually takes ~9-10 lines depending on flags (IP, topography, etc)
    # The first line with exactly 4 numeric tokens represents the first data point
    # Form: x, a, n, rhoa (for general array type)
    
    data_start = -1
    for i in range(line_idx + 4, min(line_idx + 20, len(lines))):
        tokens = lines[i].strip().split()
        if len(tokens) >= 3:
            try:
                # Require at least the first 3 tokens to be valid floats
                [float(t.replace(",", ".")) for t in tokens[:3]]
                data_start = i
                break
            except ValueError:
                pass
                
    if data_start == -1:
        raise ValueError("Could not find start of data block in RES2DINV file.")
        
    # Read the data block
    data_lines = lines[data_start:data_start + n_data]
    
    # 3 = dipole-dipole, 1 = wenner, 7 = schlumberger, etc
    # To keep it generic and simple, RES2DINV often uses a general format:
    # x (center), a (spacing), n (multiple), rhoa
    # Or for general array (type 11): n_electrodes, then electrode x,z coords
    # We will handle the standard x, a, n, rhoa format here based on array_type
    
    records = []
    for line in data_lines:
        tokens = line.strip().split()
        if len(tokens) < 3:
            continue
            
        x = float(tokens[0])
        a = float(tokens[1])
        
        # Determine n and rhoa based on column count and array type
        if len(tokens) >= 4:
            n_val = float(tokens[2])
            rhoa = float(tokens[3])
        else:
            # For 3-column formats like Wenner
            n_val = 1.0
            rhoa = float(tokens[2])
        
        # Convert (x, a, n) to (A, B, M, N) center coordinates based on array
        # x_loc_type 0 means x is the first electrode. 1 means x is the mid-point.
        if array_type == 3:  # Dipole-Dipole
            # length of array is (n+2)*a
            if x_loc_type == 0:
                center = x + ((n_val + 2) * a) / 2
            else:
                center = x
            
            C_center = center - (n_val * a + a) / 2
            P_center = center + (n_val * a + a) / 2
            A = C_center - a / 2
            B = C_center + a / 2
            M = P_center - a / 2
            N = P_center + a / 2
            
        elif array_type == 1:  # Wenner
            # length is 3a
            if x_loc_type == 0:
                center = x + 1.5 * a
            else:
                center = x
                
            A = center - 1.5 * a
            M = center - 0.5 * a
            N = center + 0.5 * a
            B = center + 1.5 * a
            
        elif array_type == 7:  # Schlumberger
            # total length is 2*n_val
            if x_loc_type == 0:
                center = x + n_val
            else:
                center = x
                
            M = center - a
            N = center + a
            A = center - n_val
            B = center + n_val
            
        elif array_type == 2:  # Pole-Pole
            # length is a
            if x_loc_type == 0:
                center = x + a / 2
            else:
                center = x
                
            A = center - a / 2
            M = center + a / 2
            B = np.nan
            N = np.nan
            
        else:
            # Fallback
            if x_loc_type == 0:
                center = x + 1.5 * a
            else:
                center = x
                
            A = center - a * 1.5
            B = center + a * 1.5
            M = center - a * 0.5
            N = center + a * 0.5
            
        records.append({"a": A, "b": B, "m": M, "n": N, "rhoa": rhoa})
        
    df = pd.DataFrame(records)
    
    # Attach metadata so the Geometry Tab can auto-populate
    df.attrs["res2dinv_spacing"] = a_spacing
    df.attrs["res2dinv_array_type"] = array_type
    
    return df


def auto_read(filepath: Union[str, Path]) -> pd.DataFrame:
    """Auto-detect format and read a geoelectrical data file.

    Tries ABEM, Syscal, CSV, and TXT readers in order; returns the
    first successful parse.

    Parameters
    ----------
    filepath : str or Path

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
        If none of the readers can parse the file.
    """
    path = Path(filepath)
    suffix = path.suffix.lower()
    errors = []

    readers = [
        ("res2dinv", read_res2dinv),
        ("csv", read_csv),
        ("txt", read_txt),
        ("abem", read_abem),
        ("syscal", read_syscal),
    ]

    # Prioritise by file extension
    if suffix in (".dat",):
        readers = [("abem", read_abem), ("syscal", read_syscal)] + readers

    for name, reader in readers:
        try:
            df = reader(filepath)
            if len(df) > 0 and len(df.columns) >= 2:
                return df
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    raise ValueError(
        f"Could not parse '{filepath}' with any reader.\n"
        + "\n".join(errors)
    )
