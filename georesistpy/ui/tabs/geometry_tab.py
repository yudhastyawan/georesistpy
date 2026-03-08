"""
Survey Geometry tab — array configuration picker and electrode plot.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import param
import panel as pn

pn.extension()


class GeometryTab(param.Parameterized):
    """Panel component for Survey Geometry configuration."""

    def __init__(self, shared_state: Dict[str, Any], **params):
        super().__init__(**params)
        self.shared = shared_state

        self._array_type = pn.widgets.Select(
            name="Array Type",
            options=["wenner", "schlumberger", "dipole-dipole", "pole-dipole", "pole-pole"],
            value="wenner",
        )
        self._n_electrodes = pn.widgets.IntSlider(
            name="Number of Electrodes", start=4, end=128, value=24, step=1,
        )
        self._spacing = pn.widgets.FloatInput(
            name="Spacing (m)", value=1.0, start=0.1, step=0.5,
        )
        self._max_n = pn.widgets.IntSlider(
            name="Max n-factor", start=1, end=12, value=6,
        )
        self._gen_btn = pn.widgets.Button(
            name="Generate Array", button_type="primary",
        )
        self._gen_btn.on_click(self._on_generate)

        self._status = pn.pane.Alert("Configure array geometry.", alert_type="info")
        self._plot_pane = pn.pane.Matplotlib(None, tight=True, sizing_mode="stretch_width")
        self._table = pn.widgets.Tabulator(
            pd.DataFrame(), page_size=15, sizing_mode="stretch_width",
        )
        
        # Subscribe to shared data updates
        pn.state.add_periodic_callback(self._check_shared_data, period=1000)
        self._last_data_id = None

    def _check_shared_data(self):
        """Watch the shared dictionary to see if new data was imported."""
        df = self.shared.get("data")
        if df is not None and id(df) != self._last_data_id:
            self._last_data_id = id(df)
            
            # Map RES2DINV array types to our strings
            res2dinv_types = {
                1: "wenner",
                2: "pole-pole",
                3: "dipole-dipole",
                6: "pole-dipole",
                7: "schlumberger"
            }
            
            # Try to populate from attributes (if read from RES2DINV)
            array_type_int = df.attrs.get("res2dinv_array_type")
            if array_type_int in res2dinv_types:
                self._array_type.value = res2dinv_types[array_type_int]
                
            spacing = df.attrs.get("res2dinv_spacing")
            if spacing is not None:
                self._spacing.value = spacing
                
            if array_type_int is not None or spacing is not None:
                pn.state.notifications.info(
                    "Geometry parameters auto-populated from imported file."
                )

    def _on_generate(self, event):
        from georesistpy.utils.arrays import generate_array, electrode_positions

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        try:
            scheme = generate_array(
                array_type=self._array_type.value,
                n_electrodes=self._n_electrodes.value,
                spacing=self._spacing.value,
                max_n=self._max_n.value,
            )
            pos = electrode_positions(
                self._n_electrodes.value, self._spacing.value,
            )

            self.shared["scheme"] = scheme
            self.shared["electrode_positions"] = pos
            self._table.value = scheme.head(200)

            # Electrode plot
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.scatter(pos[:, 0], pos[:, 1], marker="v", s=60, c="navy")
            for i, (x, z) in enumerate(pos):
                ax.annotate(str(i), (x, z), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=7)
            ax.set_xlabel("Distance (m)")
            ax.set_ylabel("Elevation (m)")
            ax.set_title(f"{self._array_type.value.title()} — {len(pos)} electrodes")
            
            # Fix aspect ratio issue when topography is flat
            if np.ptp(pos[:, 1]) == 0:
                ax.set_ylim(-5, 5)
            else:
                ax.set_aspect("equal")
                
            fig.tight_layout()
            self._plot_pane.object = fig

            self._status.object = (
                f"✅ Generated **{len(scheme)}** measurements with "
                f"**{len(pos)}** electrodes."
            )
            self._status.alert_type = "success"
        except Exception as exc:
            self._status.object = f"❌ {exc}"
            self._status.alert_type = "danger"

    def controls(self) -> pn.Column:
        """Return the input controls layout."""
        return pn.Column(
            self._array_type,
            self._n_electrodes,
            self._spacing,
            self._max_n,
            self._gen_btn,
            sizing_mode="stretch_width",
        )

    def main_panel(self) -> pn.Column:
        """Return the main visualization layout."""
        return pn.Column(
            pn.pane.Markdown("## 📐 Survey Geometry"),
            self._status,
            self._plot_pane,
            pn.pane.Markdown("### Measurement Schedule"),
            self._table,
            sizing_mode="stretch_width",
        )
