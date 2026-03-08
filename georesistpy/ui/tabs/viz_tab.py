"""
Visualization tab — interactive pseudosection, resistivity section, residuals.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import panel as pn
import param

pn.extension("plotly")


class VizTab(param.Parameterized):
    """Panel component for interactive visualizations."""

    def __init__(self, shared_state: Dict[str, Any], **params):
        super().__init__(**params)
        self.shared = shared_state

        self._plot_type = pn.widgets.Select(
            name="Plot",
            options=[
                "Pseudosection",
                "Inverted Section (1D)",
                "Inverted Section (2D)",
                "Residuals",
                "Sensitivity / DOI",
            ],
            value="Pseudosection",
        )
        self._zip_upload = pn.widgets.FileInput(
            name="Load History ZIP", accept=".zip"
        )
        self._iteration_slider = pn.widgets.IntSlider(
            name="Iteration", start=0, end=0, value=0, visible=False
        )
        self._backend = pn.widgets.RadioButtonGroup(
            name="Backend",
            options=["Plotly", "Matplotlib"],
            value="Plotly",
        )

        self._draw_btn = pn.widgets.Button(
            name="Draw", button_type="primary",
        )
        self._draw_btn.on_click(self._on_draw)
        self._plot_type.param.watch(self._on_plot_type_change, "value")
        self._zip_upload.param.watch(self._on_load_zip, "value")

        self._status = pn.pane.Alert("Select a plot type.", alert_type="info")
        self._plot_area = pn.Column(sizing_mode="stretch_width")
        
        # In-memory custom history store for uploaded ZIPs
        self._loaded_history = []
        self._loaded_centers = None
        self._loaded_qc_data = None
        
    def _on_plot_type_change(self, event):
        ptype = event.new
        needs_iter = ptype in ["Inverted Section (2D)", "Calculated Pseudosection (2D)"]
        self._iteration_slider.visible = needs_iter
        if needs_iter:
            self._update_slider_bounds()
            
    def _update_slider_bounds(self):
        # Prefer uploaded history if it exists, otherwise use shared memory
        hist = self._loaded_history if len(self._loaded_history) > 0 else self.shared.get("inversion_history", [])
        if hist and len(hist) > 0:
            max_it = max(h["iteration"] for h in hist)
            self._iteration_slider.end = max_it
            if self._iteration_slider.value > max_it:
                self._iteration_slider.value = max_it
        else:
            self._iteration_slider.end = 0
            self._iteration_slider.value = 0
            
    def _on_load_zip(self, event):
        val = event.new
        if not val:
            return
            
        import zipfile
        import io
        import pandas as pd
        
        try:
            with zipfile.ZipFile(io.BytesIO(val)) as zf:
                files = zf.namelist()
                hist = []
                centers = None
                qc_data = None
                
                # We need to discover max iteration
                iters = set()
                for fn in files:
                    if fn.startswith("iteration_"):
                        parts = fn.split("_")
                        if len(parts) >= 2 and parts[1].isdigit():
                            iters.add(int(parts[1]))
                
                for it in sorted(list(iters)):
                    mod_fn = f"iteration_{it}_model.csv"
                    calc_fn = f"iteration_{it}_calc_data.csv"
                    
                    hist_entry = {"iteration": it}
                    
                    if mod_fn in files:
                        df_m = pd.read_csv(zf.open(mod_fn))
                        hist_entry["rho_model"] = df_m["resistivity"].values
                        if centers is None:
                            centers = np.column_stack([df_m["x"].values, df_m["z"].values])
                            
                    if calc_fn in files:
                        df_d = pd.read_csv(zf.open(calc_fn))
                        hist_entry["dpred"] = df_d["rhoa_calc"].values
                        if qc_data is None:
                            qc_data = df_d.copy()
                            
                    hist.append(hist_entry)
                    
            if hist:
                self._loaded_history = hist
                self._loaded_centers = centers
                self._loaded_qc_data = qc_data
                self._update_slider_bounds()
                self._status.object = f"✅ Loaded {len(hist)} iterations from ZIP."
                self._status.alert_type = "success"
                
        except Exception as e:
            self._status.object = f"❌ Failed to parse ZIP: {e}"
            self._status.alert_type = "danger"

    def _on_draw(self, event):
        try:
            ptype = self._plot_type.value
            backend = self._backend.value
            it = self._iteration_slider.value
            
            if ptype == "Pseudosection":
                self._draw_pseudosection(backend)
            elif ptype == "Calculated Pseudosection (2D)":
                self._draw_calc_pseudosection(backend, it)
            elif ptype == "Inverted Section (1D)":
                self._draw_1d_section(backend)
            elif ptype == "Inverted Section (2D)":
                self._draw_2d_section(backend, it)
            elif ptype == "Residuals":
                self._draw_residuals(backend)
            elif ptype == "Sensitivity / DOI":
                self._draw_sensitivity()
            self._status.alert_type = "success"
        except Exception as exc:
            self._status.object = f"❌ {exc}"
            self._status.alert_type = "danger"

    def _draw_pseudosection(self, backend):
        data = self.shared.get("data_qc") or self.shared.get("data")
        if data is None:
            raise ValueError("No data loaded.")

        if backend == "Plotly":
            from georesistpy.visualization.interactive import plotly_pseudosection
            fig = plotly_pseudosection(data)
            self._plot_area.objects = [pn.pane.Plotly(fig, sizing_mode="stretch_width")]
        else:
            from georesistpy.visualization.plots import plot_pseudosection
            fig = plot_pseudosection(data)
            self._plot_area.objects = [pn.pane.Matplotlib(fig, tight=True, sizing_mode="stretch_width")]

        self._status.object = "✅ Pseudosection plotted."

    def _draw_1d_section(self, backend):
        result = self.shared.get("result_1d")
        if result is None:
            raise ValueError("Run 1-D inversion first.")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 8))
        depths = np.concatenate([[0], np.cumsum(result.thickness)])
        for i, r in enumerate(result.resistivity):
            d_top = depths[i]
            d_bot = depths[i + 1] if i + 1 < len(depths) else d_top + result.thickness[-1]
            ax.barh(
                (d_top + d_bot) / 2, r, height=d_bot - d_top,
                align="center", alpha=0.7, edgecolor="k",
            )
        ax.set_xlabel("Resistivity (Ω·m)")
        ax.set_ylabel("Depth (m)")
        ax.set_title("Inverted 1-D Model")
        ax.invert_yaxis()
        ax.set_xscale("log")
        fig.tight_layout()
        self._plot_area.objects = [pn.pane.Matplotlib(fig, tight=True, sizing_mode="stretch_width")]
        self._status.object = "✅ 1-D inverted section shown."

    def _draw_calc_pseudosection(self, backend, iteration):
        hist = self._loaded_history if len(self._loaded_history) > 0 else self.shared.get("inversion_history", [])
        if not hist:
            raise ValueError("No inversion history available. Run inversion or load a ZIP.")
            
        data = self._loaded_qc_data if self._loaded_qc_data is not None else (self.shared.get("data_qc") or self.shared.get("data"))
        if data is None:
            raise ValueError("No survey data layout available.")
            
        # Find iteration data
        entry = next((h for h in hist if h["iteration"] == iteration), None)
        if entry is None or "dpred" not in entry:
            raise ValueError(f"No calculated data for iteration {iteration}.")
            
        df = data.copy()
        
        # Ensure rhoa values match geometry indices correctly
        # Here entry["dpred"] usually matches the number of rows in the dataframe directly
        calc_arr = entry["dpred"]
        if len(calc_arr) != len(df):
            raise ValueError(f"Length mismatch: {len(calc_arr)} calculated data vs {len(df)} geometry rows.")
        
        df["rhoa"] = calc_arr
        
        if backend == "Plotly":
            from georesistpy.visualization.interactive import plotly_pseudosection
            fig = plotly_pseudosection(df, title=f"Calculated Data (Iteration {iteration})")
            self._plot_area.objects = [pn.pane.Plotly(fig, sizing_mode="stretch_width")]
        else:
            from georesistpy.visualization.plots import plot_pseudosection
            fig = plot_pseudosection(df, title=f"Calculated Data (Iteration {iteration})")
            self._plot_area.objects = [pn.pane.Matplotlib(fig, tight=True, sizing_mode="stretch_width")]
            
        self._status.object = f"✅ Calculated pseudosection for Iteration {iteration} shown."

    def _draw_2d_section(self, backend, iteration):
        hist = self._loaded_history if len(self._loaded_history) > 0 else self.shared.get("inversion_history", [])
        
        if not hist:
            raise ValueError("No inversion history available. Run inversion or load a ZIP.")
            
        entry = next((h for h in hist if h["iteration"] == iteration), None)
        if entry is None or "rho_model" not in entry:
            raise ValueError(f"No resistivity model for iteration {iteration}.")
            
        cell_centers = self._loaded_centers if self._loaded_centers is not None else self.shared.get("mesh_centers")
        if cell_centers is None:
            raise ValueError("No mesh cell centers available.")
            
        data = self._loaded_qc_data if self._loaded_qc_data is not None else (self.shared.get("data_qc") or self.shared.get("data"))
        if data is None:
            raise ValueError("No survey data layout available.")

        from georesistpy.visualization.plots import plot_inverted_section
        
        cx = cell_centers[:, 0]
        cz = cell_centers[:, 1]
        
        # Calculate geometric trapezoidal mask (DOI) from survey data
        a, b, m, n = data["a"], data["b"], data["m"], data["n"]
        midpoints = (a + b + m + n) / 4.0
        pseudo_depths = (np.abs(a - b) + np.abs(m - n)) / 4.0
        
        max_depth = pseudo_depths.max()
        min_x_surf = midpoints[pseudo_depths == pseudo_depths.min()].min()
        max_x_surf = midpoints[pseudo_depths == pseudo_depths.min()].max()
        
        mask = np.ones(len(cx), dtype=bool)
        mask[cz < -max_depth * 1.5] = False
        mask[cx < min_x_surf + np.abs(cz)] = False
        mask[cx > max_x_surf - np.abs(cz)] = False
        
        fig = plot_inverted_section(
            cell_centers=cell_centers,
            resistivity=entry["rho_model"],
            title=f"Inverted Section (Iteration {iteration})",
            doi_mask=mask
        )
        self._plot_area.objects = [pn.pane.Matplotlib(fig, tight=True, sizing_mode="stretch_width")]
        self._status.object = f"✅ 2-D inverted section shown for Iteration {iteration}."

    def _draw_residuals(self, backend):
        result = self.shared.get("result_2d") or self.shared.get("result_1d")
        fwd = self.shared.get("data_qc") or self.shared.get("data")
        
        if result is None or fwd is None:
            raise ValueError("Run inversion first or provide forward data.")

        obs = fwd["rhoa"].values.astype(float)
        
        # Find appropriate predicted data depends on the model dimensionality
        if hasattr(result, "response"):
            pred = result.response
        elif hasattr(result, "dpred"):
            pred = result.dpred
        else:
            raise ValueError("Inversion result missing predicted data attributes.")

        if backend == "Plotly":
            from georesistpy.visualization.interactive import plotly_residuals
            fig = plotly_residuals(obs, pred)
            self._plot_area.objects = [pn.pane.Plotly(fig, sizing_mode="stretch_width")]
        else:
            from georesistpy.visualization.plots import plot_residuals
            fig = plot_residuals(obs, pred)
            self._plot_area.objects = [pn.pane.Matplotlib(fig, tight=True, sizing_mode="stretch_width")]

        self._status.object = "✅ Residuals plotted."

    def _draw_sensitivity(self):
        self._status.object = (
            "ℹ️ Sensitivity / DOI requires a 2-D inversion result with coverage data."
        )
        self._status.alert_type = "info"

    def controls(self) -> pn.Column:
        try:
            return pn.Column(
                self._plot_type,
                self._zip_upload,
                self._iteration_slider,
                self._backend,
                self._draw_btn,
                sizing_mode="stretch_width",
            )
        except Exception as e:
            return pn.Column(pn.pane.Markdown(f"**Sidebar Error**: {e}"))

    def main_panel(self) -> pn.Column:
        return pn.Column(
            pn.pane.Markdown("## 📊 Visualization"),
            self._status,
            self._plot_area,
            sizing_mode="stretch_width",
        )
