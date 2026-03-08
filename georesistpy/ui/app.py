"""
Main Panel application assembly.

Brings together the sidebar, all workflow tabs, and the shared state
into a single servable Panel template.
"""

from __future__ import annotations

from typing import Any, Dict

import panel as pn

pn.extension(
    "tabulator",
    "plotly",
    notifications=True,
    sizing_mode="stretch_width",
)


def build_app() -> pn.Template:
    """Construct and return the GeoResistPy Panel application.

    Returns
    -------
    pn.Template
        A configured ``MaterialTemplate`` ready to be served.
    """
    # Shared mutable state passed between tabs
    shared_state: Dict[str, Any] = {}

    # --- Import tabs lazily to avoid heavy imports at startup -------------
    from georesistpy.ui.tabs import ImportTab
    from georesistpy.ui.tabs.qc_tab import QCTab
    from georesistpy.ui.tabs.geometry_tab import GeometryTab
    from georesistpy.ui.tabs.forward_tab import ForwardTab
    from georesistpy.ui.tabs.inversion_tab import InversionTab
    from georesistpy.ui.tabs.viz_tab import VizTab
    from georesistpy.ui.tabs.export_tab import ExportTab
    from georesistpy.ui.sidebar import create_sidebar

    # Instantiate tabs
    import_tab = ImportTab(shared_state)
    qc_tab = QCTab(shared_state)
    geom_tab = GeometryTab(shared_state)
    fwd_tab = ForwardTab(shared_state)
    inv_tab = InversionTab(shared_state)
    viz_tab = VizTab(shared_state)
    export_tab = ExportTab(shared_state)

    tab_components = [
        import_tab, qc_tab, geom_tab, fwd_tab, inv_tab, viz_tab, export_tab
    ]

    # Assemble tabs (main visual area)
    tabs = pn.Tabs(
        ("📂 Import", import_tab.main_panel()),
        ("🔍 QC", qc_tab.main_panel()),
        ("📐 Geometry", geom_tab.main_panel()),
        ("⚡ Forward", fwd_tab.main_panel()),
        ("🔄 Inversion", inv_tab.main_panel()),
        ("📊 Visualize", viz_tab.main_panel()),
        ("💾 Export", export_tab.main_panel()),
        dynamic=True,
    )

    # Setup the dynamic sidebar
    sidebar_base = create_sidebar()
    
    @pn.depends(tabs.param.active)
    def get_sidebar_controls(active_index):
        # Ensure index is within range
        if 0 <= active_index < len(tab_components):
            return tab_components[active_index].controls()
        return pn.Column()

    sidebar_container = pn.Column(
        sidebar_base,
        get_sidebar_controls,
        sizing_mode="stretch_width"
    )

    # Template
    template = pn.template.MaterialTemplate(
        title="GeoResistPy — ERT Processing Suite",
        sidebar=[sidebar_container],
        main=[tabs],
        header_background="#1a237e",
    )

    return template
