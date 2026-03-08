"""
GeoResistPy — Electrical Resistivity Tomography processing library.

Provides 1D VES and 2D ERT workflows: data import, quality control,
mesh generation, forward modelling, inversion, and interactive
visualization — all accessible through a Panel web UI.
"""

__version__ = "0.1.0"

from georesistpy.io import readers, writers          # noqa: F401
from georesistpy.qc import filters, errors           # noqa: F401
from georesistpy.utils import arrays, doi            # noqa: F401
