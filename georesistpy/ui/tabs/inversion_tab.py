"""
Inversion tab — 1-D VES and 2-D ERT inversion with parameter controls.
"""

from __future__ import annotations

from typing import Any, Dict

import sys
import threading
from io import StringIO
import numpy as np
import pandas as pd
import param
import panel as pn

pn.extension('terminal')


class InversionTab(param.Parameterized):
    """Panel component for Inversion workflow."""

    def __init__(self, shared_state: Dict[str, Any], **params):
        super().__init__(**params)
        self.shared = shared_state

        self._mode = pn.widgets.RadioButtonGroup(
            name="Mode", options=["1D VES", "2D ERT"], value="1D VES",
        )
        self._n_layers = pn.widgets.IntSlider(
            name="Layers (1-D)", start=2, end=10, value=5,
        )
        self._lam = pn.widgets.FloatInput(
            name="Lambda (λ)", value=20.0, start=0.1, step=5.0,
        )
        self._max_iter = pn.widgets.IntInput(
            name="Max Iterations", start=1, value=20,
        )
        self._robust = pn.widgets.Checkbox(
            name="Robust (L1)", value=False,
        )
        self._error_level = pn.widgets.FloatInput(
            name="Error Level (%)", value=3.0, start=0.1, step=0.5,
        )

        # 2D Mesh Parameters
        self._expansion_z = pn.widgets.FloatInput(
            name="Vertical Expansion", value=1.0, start=1.0, end=1.5, step=0.05,
        )
        self._max_cells_x = pn.widgets.IntInput(
            name="Max Cells (Horiz)", value=60, start=10, end=200,
        )
        self._max_cells_z = pn.widgets.IntInput(
            name="Max Cells (Vert)", value=30, start=10, end=150,
        )

        self._preview_mesh_btn = pn.widgets.Button(
            name="🔍 Preview Mesh", button_type="primary",
            width=200,
        )
        self._preview_mesh_btn.on_click(self._on_preview_mesh)

        self._run_btn = pn.widgets.Button(
            name="▶ Run Inversion", button_type="success",
            width=200, height=45,
        )
        self._run_btn.on_click(self._on_run)

        self._status = pn.pane.Alert(
            "Configure parameters and click Run.", alert_type="info",
        )
        self._rms_pane = pn.pane.Markdown("")
        
        # Terminal for logs
        self._terminal = pn.widgets.Terminal(
            sizing_mode="stretch_width", min_height=150, max_height=200,
            options={"cursorBlink": True}
        )
        self._log_buffer = ["Ready.\n"]
        self._terminal.write("Ready.\n")
        
        self._plot_pane = pn.pane.Matplotlib(
            None, tight=True, sizing_mode="stretch_width", min_height=600
        )
        self._mesh_pane = pn.pane.Matplotlib(
            None, tight=True, sizing_mode="stretch_width", min_height=300
        )

    def _on_run(self, event):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self._status.object = "⏳ Running inversion…"
        self._status.alert_type = "info"

        try:
            if self._mode.value == "1D VES":
                self._run_btn.disabled = True
                self._run_1d(plt)
                self._run_btn.disabled = False
            else:
                self._run_btn.disabled = True
                self._terminal.clear()
                self._log_buffer = ["Starting 2-D Inversion...\n"]
                self._terminal.write("Starting 2-D Inversion...\n")
                # Switch tabs to the Plot pane if possible, and just clear the plot pane
                self._plot_pane.object = None
                # Run in background to keep UI responsive
                thread = threading.Thread(target=self._run_2d_thread, daemon=True)
                thread.start()
        except Exception as exc:
            self._status.object = f"❌ {exc}"
            self._status.alert_type = "danger"
            self._run_btn.disabled = False

    def _on_preview_mesh(self, event):
        import matplotlib.pyplot as plt
        from georesistpy.mesh.mesh_builder import create_mesh_2d

        data = self.shared.get("data_qc")
        if data is None or (isinstance(data, pd.DataFrame) and data.empty):
            data = self.shared.get("data")
        
        if data is None or (isinstance(data, pd.DataFrame) and data.empty):
            self._status.object = "❌ No data to build mesh. Import data first."
            self._status.alert_type = "danger"
            return

        try:
            elec_x = np.unique(np.concatenate([data['a'], data['b'], data['m'], data['n']]))
            elec_pos = np.column_stack([elec_x, np.zeros_like(elec_x)])
            
            # Calculate actual max empirical depth from array geometries
            pseudo_depth = (np.abs(data["a"] - data["b"]) + np.abs(data["m"] - data["n"])) / 4.0
            actual_max_depth = float(pseudo_depth.max())
            
            # Apply safety factor of 1.2 to standard depth to ensure boundary conditions work
            depth_limit = actual_max_depth * 1.2

            mesh = create_mesh_2d(
                elec_pos,
                depth_max=depth_limit,
                expansion_z=self._expansion_z.value,
                max_cells_x=self._max_cells_x.value,
                max_cells_z=self._max_cells_z.value
            )

            fig, ax = plt.subplots(figsize=(10, 4))
            mesh.plot_grid(ax=ax)
            ax.set_title(f"Mesh Preview: {mesh.nC} cells")
            ax.set_aspect("equal")
            plt.tight_layout()
            
            self._mesh_pane.object = fig
            self._status.object = f"✅ Mesh generated: {mesh.nC} cells ({mesh.shape_cells[0]}x{mesh.shape_cells[1]})"
            self._status.alert_type = "success"
        except Exception as exc:
            self._status.object = f"❌ Mesh error: {exc}"
            self._status.alert_type = "danger"

    def _run_1d(self, plt):
        from georesistpy.inversion.inversion1d import invert_1d

        # Look for data in shared state
        fwd = self.shared.get("forward_1d")
        
        data = self.shared.get("data_qc")
        if data is None or (isinstance(data, pd.DataFrame) and data.empty):
            data = self.shared.get("data")

        if fwd:
            ab2 = fwd["ab2"]
            rhoa = fwd["rhoa"]
        elif data is not None and "rhoa" in data.columns:
            ab2 = data.get("ab2", data.get("a", pd.Series())).values
            rhoa = data["rhoa"].values
            if len(ab2) == 0:
                raise ValueError("Cannot derive AB/2 from data columns.")
        else:
            raise ValueError(
                "No sounding data available. Run Forward Model or import data first."
            )

        result = invert_1d(
            ab2=ab2,
            rhoa_obs=rhoa,
            n_layers=self._n_layers.value,
            lam=self._lam.value,
            max_iter=self._max_iter.value,
            error=np.full(len(rhoa), self._error_level.value / 100.0),
        )
        self.shared["result_1d"] = result

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Observed vs fitted
        axes[0].loglog(ab2, rhoa, "ko", label="Observed")
        axes[0].loglog(ab2, result.response, "r-", lw=2, label="Fitted")
        axes[0].set_xlabel("AB/2 (m)")
        axes[0].set_ylabel("ρₐ (Ω·m)")
        axes[0].set_title("Sounding Curve Fit")
        axes[0].legend()
        axes[0].grid(True, which="both", ls="--", alpha=0.4)

        # Layer model
        depths = np.concatenate([[0], np.cumsum(result.thickness)])
        for i, r in enumerate(result.resistivity):
            d_top = depths[i]
            d_bot = depths[i + 1] if i + 1 < len(depths) else d_top + result.thickness[-1]
            axes[1].barh(
                (d_top + d_bot) / 2, r, height=d_bot - d_top,
                align="center", alpha=0.7, edgecolor="k",
            )
        axes[1].set_xlabel("Resistivity (Ω·m)")
        axes[1].set_ylabel("Depth (m)")
        axes[1].set_title("Inverted Model")
        axes[1].invert_yaxis()
        axes[1].set_xscale("log")

        # RMS history
        if result.rms_history:
            axes[2].plot(result.rms_history, "b.-")
            axes[2].set_xlabel("Iteration")
            axes[2].set_ylabel("RMS (%)")
            axes[2].set_title("Convergence")
            axes[2].grid(True)

        fig.tight_layout()
        self._plot_pane.object = fig

        self._rms_pane.object = (
            f"**RMS**: {result.rms:.2f}% &emsp; "
            f"**χ²**: {result.chi2:.3f} &emsp; "
            f"**Iterations**: {result.n_iterations}"
        )
        self._status.object = f"✅ 1-D inversion complete — RMS = {result.rms:.2f}%"
        self._status.alert_type = "success"

    def _run_2d_thread(self):
        try:
            self._run_2d()
        except Exception as exc:
            def update_err():
                self._status.object = f"❌ {exc}"
                self._status.alert_type = "danger"
                self._run_btn.disabled = False
            pn.state.execute(update_err)

    def _run_2d(self):
        import matplotlib
        matplotlib.use("Agg")
        from georesistpy.inversion.inversion2d import invert_2d
        from georesistpy.visualization.plots import plot_inversion_progress

        data = self.shared.get("data_qc")
        if data is None or (isinstance(data, pd.DataFrame) and data.empty):
            data = self.shared.get("data")
            
        if data is None or (isinstance(data, pd.DataFrame) and data.empty):
            raise ValueError("No data loaded. Import data first.")

        # Prepare shared state lists
        self.shared["inversion_history"] = []
        
        required = {"a", "b", "m", "n", "rhoa"}
        if not required.issubset(data.columns):
            raise ValueError(f"Data must contain columns: {required}")

        import matplotlib.pyplot as plt

        # Capture stdout for Terminal — avoids lambda closure leak
        # Capture stdout for Terminal — avoids lambda closure leak
        class TerminalEcho:
            def __init__(self, terminal, buffer):
                self.terminal = terminal
                self.original_stdout = sys.stdout
                self.buffer = buffer
            def write(self, s):
                self.original_stdout.write(s)
                text = str(s)
                self.buffer.append(text)
                try:
                    pn.state.execute(lambda _t=text: self.terminal.write(_t))
                except Exception:
                    pass
            def flush(self):
                self.original_stdout.flush()

        def iteration_update(iteration, rho_model, dpred, rms, cell_centers):
            if cell_centers is None:
                return
                
            # Store in shared state for export
            self.shared["mesh_centers"] = cell_centers
            if iteration == 0 and "model_0" not in self.shared:
                self.shared["model_0"] = rho_model
                
            self.shared["inversion_history"].append({
                "iteration": int(iteration),
                "rho_model": rho_model.copy(),
                "dpred": dpred.copy(),
                "rms": float(rms)
            })
            
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                
                fig = plot_inversion_progress(
                    data=data,
                    dpred=dpred,
                    cell_centers=cell_centers,
                    resistivity=rho_model,
                    iteration=iteration,
                    rms=rms
                )
                it_num = int(iteration)
                rms_val = float(rms)
                def update_ui(_fig=fig, _it=it_num, _rms=rms_val):
                    # Close old figure to prevent memory leak
                    old_fig = self._plot_pane.object
                    if old_fig is not None:
                        plt.close(old_fig)
                    self._plot_pane.object = _fig
                    self._rms_pane.object = f"**Iteration {_it} RMS**: {_rms:.2f}%"
                pn.state.execute(update_ui)
            except Exception as e:
                # Log to terminal so we can see why it fails
                print(f"Plotting error on iter {iteration}: {e}")

        original_stdout = sys.stdout
        sys.stdout = TerminalEcho(self._terminal, self._log_buffer)
        
        result = None
        fig = None
        try:
            import time
            
            # Wait a tiny bit and draw final
            time.sleep(0.5)
            
            # Calculate depth_max manually to match preview precisely
            pseudo_depth = (np.abs(data["a"] - data["b"]) + np.abs(data["m"] - data["n"])) / 4.0
            depth_limit = float(pseudo_depth.max()) * 1.2
            
            result = invert_2d(
                data,
                lam=self._lam.value,
                max_iter=self._max_iter.value,
                robust=self._robust.value,
                error_level=self._error_level.value / 100.0,
                expansion_z=self._expansion_z.value,
                max_cells_x=self._max_cells_x.value,
                max_cells_z=self._max_cells_z.value,
                depth_max=depth_limit,
                iteration_callback=iteration_update,
            )
            
            # Since cell centers require the mesh which is built inside invert_2d,
            # we will just plot the final result using plot_inversion_progress here.
            fig = plot_inversion_progress(
                data=data,
                dpred=result.response,
                cell_centers=result.mesh.cell_centers,
                resistivity=result.resistivity,
                iteration=result.n_iterations,
                rms=result.rms
            )
        except Exception as e:
            self._status.object = f"❌ Inversion failed: {e}"
            self._status.alert_type = "danger"
        finally:
            sys.stdout = original_stdout

        def update_success():
            if result is not None and fig is not None:
                self.shared["result_2d"] = result
                self._plot_pane.object = fig
                self._rms_pane.object = (
                    f"**RMS**: {result.rms:.2f}% &emsp; "
                    f"**χ²**: {result.chi2:.3f} &emsp; "
                    f"**Iterations**: {result.n_iterations}"
                )
                self._status.object = f"✅ 2-D Inversion complete! (RMS: {result.rms:.2f}%)"
                self._status.alert_type = "success"
                self._run_btn.disabled = False
            else:
                self._run_btn.disabled = False
        pn.state.execute(update_success)

    def _dynamic_controls(self, mode_value):
        if mode_value == "1D VES":
            return pn.Column(
                self._n_layers, 
                self._lam, 
                self._max_iter,
                self._error_level,
                self._run_btn,
                sizing_mode="stretch_width"
            )
        else:
            return pn.Column(
                self._lam, 
                self._max_iter,
                self._robust, 
                self._error_level,
                pn.layout.Divider(),
                pn.pane.Markdown("### Mesh Settings"),
                self._expansion_z, 
                self._max_cells_x,
                self._max_cells_z,
                self._preview_mesh_btn,
                pn.layout.Divider(),
                self._run_btn,
                sizing_mode="stretch_width"
            )

    def controls(self) -> pn.Column:
        bound_controls = pn.bind(self._dynamic_controls, self._mode)
        
        # Restore terminal buffer if tab was switched
        try:
            full_log = "".join(self._log_buffer)
            self._terminal.clear()
            self._terminal.write(full_log)
        except Exception:
            pass

        return pn.Column(
            self._mode,
            bound_controls,
            sizing_mode="stretch_width",
        )

    def main_panel(self) -> pn.Column:
        """Return the main visual layout."""
        
        # Group visualizations to prevent endless vertical scrolling
        viz_tabs = pn.Tabs(
            ("Inversion Progress", self._plot_pane),
            ("Mesh Preview", self._mesh_pane),
            sizing_mode="stretch_width"
        )
        
        return pn.Column(
            self._status,
            self._rms_pane,
            viz_tabs,
            pn.pane.Markdown("### Inversion Log"),
            self._terminal,
            sizing_mode="stretch_width"
        )
