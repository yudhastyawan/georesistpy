"""
Sidebar navigation component for the GeoResistPy web UI.
"""

from __future__ import annotations

import panel as pn


def create_sidebar() -> pn.Column:
    """Build the workflow sidebar.

    Returns a Panel Column that displays the application title,
    a brief description, and version info.  The actual tab switching
    is handled by the ``pn.Tabs`` component in the main app.
    """
    logo_md = pn.pane.Markdown(
        """
# ⚡ GeoResistPy

**Electrical Resistivity Tomography**
*Processing & Inversion Suite*

---

### Workflow Steps

1. 📂 Import Data
2. 🔍 Quality Control
3. 📐 Survey Geometry
4. ⚡ Forward Model
5. 🔄 Inversion
6. 📊 Visualization
7. 💾 Export Results

---

*v0.1.0*
""",
        width=260,
    )
    return pn.Column(logo_md, width=280)
