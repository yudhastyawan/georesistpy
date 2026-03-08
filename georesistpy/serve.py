"""
Servable entry point for ``panel serve``.

Usage
-----
    panel serve georesistpy/serve.py --show

This file is designed to be used with ``panel serve`` which
expects a module-level ``.servable()`` call.
"""

import panel as pn

pn.extension(
    "tabulator",
    "plotly",
    notifications=True,
    sizing_mode="stretch_width",
)

from georesistpy.ui.app import build_app

template = build_app()
template.servable(title="GeoResistPy — ERT Processing Suite")
