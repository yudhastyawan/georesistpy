"""
Forward Model tab — 1-D / 2-D forward simulation interface.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import param
import panel as pn

pn.extension()


class ForwardTab(param.Parameterized):
    """Panel component for Forward Modelling."""

    def __init__(self, shared_state: Dict[str, Any], **params):
        super().__init__(**params)
        self.shared = shared_state

        self._mode = pn.widgets.RadioButtonGroup(
            name="Mode", options=["1D VES", "2D ERT"], value="1D VES",
        )

        # 1-D controls
        self._n_layers = pn.widgets.IntSlider(
            name="Layers", start=2, end=10, value=4,
        )
        self._rho_input = pn.widgets.TextInput(
            name="Resistivities (Ω·m, comma-sep)",
            value="100, 50, 200, 500",
        )
        self._thk_input = pn.widgets.TextInput(
            name="Thicknesses (m, comma-sep)",
            value="2, 5, 10",
        )
        self._ab2_input = pn.widgets.TextInput(
            name="AB/2 values (m)",
            value="1, 2, 3, 5, 7, 10, 15, 20, 30, 50",
        )

        self._run_btn = pn.widgets.Button(
            name="Run Forward Model", button_type="primary",
        )
        self._run_btn.on_click(self._on_run)

        self._status = pn.pane.Alert(
            "Set model parameters and run.", alert_type="info"
        )
        self._plot_pane = pn.pane.Matplotlib(
            None, tight=True, sizing_mode="stretch_width"
        )

    def _on_run(self, event):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        try:
            if self._mode.value == "1D VES":
                self._run_1d(plt)
            else:
                self._run_2d(plt)
        except Exception as exc:
            self._status.object = f"❌ {exc}"
            self._status.alert_type = "danger"

    def _run_1d(self, plt):
        from georesistpy.forward.forward1d import forward_1d

        rho = np.array([float(x) for x in self._rho_input.value.split(",")])
        thk = np.array([float(x) for x in self._thk_input.value.split(",")])
        ab2 = np.array([float(x) for x in self._ab2_input.value.split(",")])

        rhoa = forward_1d(rho, thk, ab2)

        self.shared["forward_1d"] = {"ab2": ab2, "rhoa": rhoa, "rho": rho, "thk": thk}

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # Sounding curve
        axes[0].loglog(ab2, rhoa, "bo-")
        axes[0].set_xlabel("AB/2 (m)")
        axes[0].set_ylabel("Apparent ρₐ (Ω·m)")
        axes[0].set_title("VES Sounding Curve")
        axes[0].grid(True, which="both", ls="--", alpha=0.4)

        # Layer model
        depths = np.concatenate([[0], np.cumsum(thk)])
        for i, r in enumerate(rho):
            d_top = depths[i] if i < len(depths) else depths[-1]
            d_bot = depths[i + 1] if i + 1 < len(depths) else d_top + thk[-1]
            axes[1].fill_betweenx(
                [d_top, d_bot], 0, r, alpha=0.6, step="post",
            )
            axes[1].plot([r, r], [d_top, d_bot], "k-", lw=1.5)
        axes[1].set_xlabel("Resistivity (Ω·m)")
        axes[1].set_ylabel("Depth (m)")
        axes[1].set_title("Layer Model")
        axes[1].invert_yaxis()
        axes[1].set_xscale("log")
        axes[1].grid(True, which="both", ls="--", alpha=0.4)

        fig.tight_layout()
        self._plot_pane.object = fig
        self._status.object = f"✅ 1-D forward complete — {len(ab2)} data points."
        self._status.alert_type = "success"

    def _run_2d(self, plt):
        self._status.object = (
            "ℹ️ 2-D forward requires electrode geometry. "
            "Generate a survey in the Geometry tab first."
        )
        self._status.alert_type = "info"

    def _dynamic_controls(self, mode_value):
        """Return UI controls depending on the selected mode."""
        if mode_value == "1D VES":
            return pn.Column(
                self._n_layers, 
                self._rho_input, 
                self._thk_input,
                self._ab2_input,
                self._run_btn,
                sizing_mode="stretch_width"
            )
        else:
            return pn.Column(
                pn.pane.Markdown("*2-D Forward Modelling requires a generated array geometry.*"),
                self._run_btn,
                sizing_mode="stretch_width"
            )

    def controls(self) -> pn.Column:
        bound_controls = pn.bind(self._dynamic_controls, self._mode)
        return pn.Column(
            self._mode,
            bound_controls,
            sizing_mode="stretch_width",
        )

    def main_panel(self) -> pn.Column:
        return pn.Column(
            pn.pane.Markdown("## ⚡ Forward Model"),
            self._status,
            self._plot_pane,
            sizing_mode="stretch_width",
        )
