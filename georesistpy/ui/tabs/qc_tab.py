"""
Quality Control tab — filtering, error estimation, QC plots.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import param
import panel as pn
import numpy as np

pn.extension("tabulator")


class QCTab(param.Parameterized):
    """Panel component for the Quality Control workflow step."""

    def __init__(self, shared_state: Dict[str, Any], **params):
        super().__init__(**params)
        self.shared = shared_state

        self._method = pn.widgets.Select(
            name="Outlier Method", options=["mad", "iqr"], value="mad"
        )
        self._threshold = pn.widgets.FloatSlider(
            name="Threshold", start=1.0, end=10.0, value=3.5, step=0.5
        )
        self._remove_neg = pn.widgets.Checkbox(
            name="Remove negative ρₐ", value=True
        )
        self._run_btn = pn.widgets.Button(
            name="Run QC", button_type="primary"
        )
        self._run_btn.on_click(self._on_run)

        self._status = pn.pane.Alert(
            "Configure QC parameters and click Run.", alert_type="info",
        )
        self._table = pn.widgets.Tabulator(
            pd.DataFrame(), pagination="remote", page_size=15, height=400,
            sizing_mode="stretch_width"
        )
        self._plot_pane = pn.pane.Matplotlib(
            None, tight=True, sizing_mode="stretch_width", min_height=400
        )
        self._scatter_pane = pn.pane.Matplotlib(
            None, tight=True, sizing_mode="stretch_width", min_height=400
        )

    def _on_run(self, event):
        from georesistpy.qc.filters import (
            flag_outliers,
            remove_negative_resistivity,
        )
        from georesistpy.qc.errors import estimate_error_model
        from georesistpy.qc.visualization import plot_scatter_qc, plot_pseudosection_qc

        df: pd.DataFrame = self.shared.get("data")
        if df is None or len(df) == 0:
            self._status.object = "⚠️ No data loaded."
            self._status.alert_type = "warning"
            return

        original_len = len(df)

        if self._remove_neg.value and "rhoa" in df.columns:
            df = remove_negative_resistivity(df)

        df = flag_outliers(
            df, method=self._method.value, threshold=self._threshold.value,
        )
        n_outliers = int(df["outlier"].sum()) if "outlier" in df.columns else 0

        df = estimate_error_model(df)

        self.shared["data_qc"] = df
        self._table.value = df.head(200)

        # Plot
        try:
            if "outlier" in df.columns and "rhoa" in df.columns:
                # Pseudosection plot
                required_2d = {"a", "b", "m", "n"}
                if required_2d.issubset(df.columns):
                    self._plot_pane.object = plot_pseudosection_qc(df, flag_col="outlier")
                else:
                    self._plot_pane.object = None # Clear pseudosection if not 2D data
                
                # Scatter plot
                self._scatter_pane.object = plot_scatter_qc(
                    df, x_col=df.columns[0], y_col="rhoa", flag_col="outlier",
                )
            else:
                self._plot_pane.object = None
                self._scatter_pane.object = None
        except Exception as e:
            self._status.object = f"⚠️ Error plotting: {e}"
            self._status.alert_type = "error"
            self._plot_pane.object = None
            self._scatter_pane.object = None


        self._status.object = (
            f"✅ QC complete: {original_len} → {len(df)} points, "
            f"**{n_outliers}** flagged outliers."
        )
        self._status.alert_type = "success"

    def controls(self) -> pn.Column:
        """Return the input controls layout."""
        return pn.Column(
            self._method,
            self._threshold,
            self._remove_neg,
            self._run_btn,
            sizing_mode="stretch_width",
        )

    def main_panel(self) -> pn.Column:
        """Return the main visual layout."""
        return pn.Column(
            self._status,
            self._plot_pane,
            self._scatter_pane,
            pn.pane.Markdown("### Processed Data"),
            self._table,
            sizing_mode="stretch_width",
        )
