"""
Data export / writer utilities.

Export inversion results and survey data to CSV, PNG, GeoTIFF, and NetCDF.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd


def export_csv(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    **kwargs,
) -> Path:
    """Export a DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Data to export.
    filepath : str or Path
        Destination file path.
    **kwargs
        Forwarded to :meth:`DataFrame.to_csv`.

    Returns
    -------
    Path
        Absolute path to the written file.
    """
    path = Path(filepath)
    df.to_csv(path, index=False, **kwargs)
    return path.resolve()


def export_png(
    fig: Any,
    filepath: Union[str, Path],
    dpi: int = 150,
) -> Path:
    """Save a matplotlib figure to PNG.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    filepath : str or Path
        Destination path.
    dpi : int, optional
        Resolution (default 150).

    Returns
    -------
    Path
    """
    path = Path(filepath)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    return path.resolve()


def export_geotiff(
    data: np.ndarray,
    filepath: Union[str, Path],
    x_min: float = 0.0,
    x_max: float = 1.0,
    z_min: float = 0.0,
    z_max: float = 1.0,
    crs: str = "EPSG:32650",
) -> Path:
    """Export a 2-D resistivity grid as a single-band GeoTIFF.

    Requires the ``rasterio`` package.

    Parameters
    ----------
    data : np.ndarray
        2-D array (rows = depth, cols = lateral).
    filepath : str or Path
        Destination file path.
    x_min, x_max, z_min, z_max : float
        Spatial extent.
    crs : str
        Coordinate reference system (default UTM zone 50N).

    Returns
    -------
    Path

    Raises
    ------
    ImportError
        If ``rasterio`` is not installed.
    """
    try:
        import rasterio
        from rasterio.transform import from_bounds
    except ImportError as exc:
        raise ImportError(
            "GeoTIFF export requires 'rasterio'. "
            "Install it with: pip install rasterio"
        ) from exc

    path = Path(filepath)
    nrows, ncols = data.shape
    transform = from_bounds(x_min, z_min, x_max, z_max, ncols, nrows)

    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=nrows,
        width=ncols,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    return path.resolve()


def export_netcdf(
    data: np.ndarray,
    filepath: Union[str, Path],
    x_coords: Optional[np.ndarray] = None,
    z_coords: Optional[np.ndarray] = None,
    variable_name: str = "resistivity",
    attrs: Optional[Dict[str, Any]] = None,
) -> Path:
    """Export a 2-D resistivity grid to NetCDF.

    Requires the ``xarray`` package.

    Parameters
    ----------
    data : np.ndarray
        2-D array.
    filepath : str or Path
        Destination path.
    x_coords, z_coords : np.ndarray, optional
        1-D coordinate arrays.
    variable_name : str
        NetCDF variable name (default ``'resistivity'``).
    attrs : dict, optional
        Global attributes.

    Returns
    -------
    Path

    Raises
    ------
    ImportError
        If ``xarray`` is not installed.
    """
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError(
            "NetCDF export requires 'xarray'. "
            "Install it with: pip install xarray netCDF4"
        ) from exc

    path = Path(filepath)
    nrows, ncols = data.shape

    if x_coords is None:
        x_coords = np.arange(ncols, dtype=float)
    if z_coords is None:
        z_coords = np.arange(nrows, dtype=float)

    ds = xr.Dataset(
        {variable_name: (["depth", "distance"], data)},
        coords={"depth": z_coords, "distance": x_coords},
        attrs=attrs or {"description": "GeoResistPy export"},
    )
    ds.to_netcdf(str(path))
    return path.resolve()
