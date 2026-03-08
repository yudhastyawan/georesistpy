"""
Export Results tab — download data and figures.
"""

from __future__ import annotations

import io
from typing import Any, Dict

import numpy as np
import pandas as pd
import param
import panel as pn

pn.extension()


class ExportTab(param.Parameterized):
    """Panel component for exporting results."""

    def __init__(self, shared_state: Dict[str, Any], **params):
        super().__init__(**params)
        self.shared = shared_state

        self._format = pn.widgets.Select(
            name="Export Format",
            options=["CSV", "PNG", "NetCDF"],
            value="CSV",
        )
        self._export_what = pn.widgets.Select(
            name="Export What",
            options=[
                "Imported Data",
                "QC-filtered Data",
                "2-D Initial Model (CSV)",
                "2-D Inversion History (ZIP)",
                "1-D Inversion Result",
            ],
            value="Imported Data",
        )
        self._export_btn = pn.widgets.Button(
            name="Export", button_type="primary",
        )
        self._export_btn.on_click(self._on_export)

        self._status = pn.pane.Alert("Choose format and data to export.", alert_type="info")
        self._download = pn.widgets.FileDownload(
            label="Download File",
            button_type="success",
            auto=False,
            visible=False,
        )

    def _on_export(self, event):
        import zipfile
        try:
            what = self._export_what.value
            fmt = self._format.value

            df = None
            zip_buffer = None
            filename = "georesistpy_export.csv"
            
            if what == "Imported Data":
                df = self.shared.get("data")
            elif what == "QC-filtered Data":
                df = self.shared.get("data_qc")
            elif what == "1-D Inversion Result":
                result = self.shared.get("result_1d")
                if result:
                    depths = np.concatenate([[0], np.cumsum(result.thickness)])
                    rows = []
                    for i, r in enumerate(result.resistivity):
                        d_top = depths[i]
                        d_bot = depths[i + 1] if i + 1 < len(depths) else None
                        rows.append({"layer": i + 1, "depth_top": d_top,
                                     "depth_bottom": d_bot, "resistivity": r})
                    df = pd.DataFrame(rows)
            elif what == "2-D Initial Model (CSV)":
                init_model = self.shared.get("model_0")
                centers = self.shared.get("mesh_centers")
                if init_model is not None and centers is not None:
                    df = pd.DataFrame({
                        "x": centers[:, 0],
                        "z": centers[:, 1],
                        "resistivity": init_model
                    })
                    filename = "initial_model.csv"
            elif what == "2-D Inversion History (ZIP)":
                history = self.shared.get("inversion_history")
                centers = self.shared.get("mesh_centers")
                if history and centers is not None:
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                        for it_data in history:
                            it = it_data["iteration"]
                            # Write Model
                            m_df = pd.DataFrame({
                                "x": centers[:, 0],
                                "z": centers[:, 1],
                                "resistivity": it_data["rho_model"]
                            })
                            m_csv = m_df.to_csv(index=False)
                            zf.writestr(f"iteration_{it}_model.csv", m_csv)
                            
                            # Write Calculated Data
                            # Assuming original data is in data_qc
                            data = self.shared.get("data_qc")
                            if data is not None:
                                d_df = data.copy()
                                d_df["rhoa_calc"] = it_data["dpred"]
                                d_csv = d_df.to_csv(index=False)
                                zf.writestr(f"iteration_{it}_calc_data.csv", d_csv)
                            
                    filename = "inversion_history.zip"

            if (df is None or len(df) == 0) and zip_buffer is None:
                self._status.object = "⚠️ No data available for this selection."
                self._status.alert_type = "warning"
                return

            if zip_buffer is not None:
                # Deliver ZIP
                self._download.file = io.BytesIO(zip_buffer.getvalue())
                self._download.filename = filename
                self._download.visible = True
            elif fmt == "CSV":
                buf = io.StringIO()
                df.to_csv(buf, index=False)
                self._download.file = io.BytesIO(buf.getvalue().encode())
                self._download.filename = filename # or generic
                self._download.visible = True
            elif fmt == "PNG":
                self._status.object = (
                    "ℹ️ PNG export: use the Visualization tab to render a plot, "
                    "then right-click → Save Image."
                )
                self._status.alert_type = "info"
                return
            elif fmt == "NetCDF":
                self._status.object = (
                    "ℹ️ NetCDF export requires xarray and a 2-D grid result."
                )
                self._status.alert_type = "info"
                return

            self._status.object = "✅ File ready for download."
            self._status.alert_type = "success"
        except Exception as exc:
            self._status.object = f"❌ {exc}"
            self._status.alert_type = "danger"

    def controls(self) -> pn.Column:
        return pn.Column(
            self._export_what,
            self._format,
            self._export_btn,
            sizing_mode="stretch_width",
        )

    def main_panel(self) -> pn.Column:
        return pn.Column(
            pn.pane.Markdown("## 💾 Export Results"),
            self._status,
            self._download,
            sizing_mode="stretch_width",
        )
