"""
Import Data tab — file upload, format detection, and data preview.
"""

from __future__ import annotations

import io
from typing import Any, Dict

import pandas as pd
import param
import panel as pn

pn.extension("tabulator", notifications=True)


class ImportTab(param.Parameterized):
    """Panel component for the 'Import Data' workflow step."""

    file_input = param.Parameter(doc="Panel FileInput widget")
    format_selector = param.Selector(
        default="auto",
        objects=["auto", "csv", "txt", "abem", "syscal", "res2dinv", "generic"],
        doc="Data format",
    )
    separator = param.String(default=",", doc="Column separator (CSV/TXT)")
    data = param.DataFrame(doc="Loaded survey data")

    def __init__(self, shared_state: Dict[str, Any], **params):
        super().__init__(**params)
        self.shared = shared_state
        self._file_widget = pn.widgets.FileInput(
            accept=".csv,.txt,.dat,.res",
            multiple=False,
            name="Upload Survey File",
        )
        self._format_widget = pn.widgets.Select(
            name="Format",
            options=["auto", "csv", "txt", "abem", "syscal", "res2dinv", "generic"],
            value="auto",
        )
        self._sep_widget = pn.widgets.TextInput(
            name="Separator", value=",", width=80
        )
        self._load_btn = pn.widgets.Button(
            name="Load Data", button_type="primary"
        )
        self._load_btn.on_click(self._on_load)

        self._preview = pn.widgets.Tabulator(
            pd.DataFrame(), page_size=15, sizing_mode="stretch_width",
        )
        self._plot_pane = pn.pane.Matplotlib(
            None, tight=True, sizing_mode="stretch_width"
        )
        self._status = pn.pane.Alert(
            "Upload a file to begin.", alert_type="info"
        )

    # ---- callbacks -------------------------------------------------------

    def _on_load(self, event):
        from georesistpy.io import readers

        if self._file_widget.value is None:
            self._status.object = "⚠️ No file selected."
            self._status.alert_type = "warning"
            return

        try:
            raw = io.BytesIO(self._file_widget.value)
            text = raw.read().decode("utf-8", errors="replace")
            fmt = self._format_widget.value

            reader_map = {
                "csv": lambda: readers.read_csv(io.StringIO(text), sep=self._sep_widget.value),
                "txt": lambda: readers.read_txt(io.StringIO(text)),
                "abem": lambda: readers.read_abem(io.StringIO(text)),
                "syscal": lambda: readers.read_syscal(io.StringIO(text)),
                "res2dinv": lambda: readers.read_res2dinv(io.StringIO(text)),
                "generic": lambda: readers.read_generic(io.StringIO(text)),
                "auto": lambda: _auto_read_text(text),
            }

            df = reader_map[fmt]()
            self.data = df
            self.shared["data"] = df
            self._preview.value = df.head(200)
            self._status.object = (
                f"✅ Loaded **{len(df)}** measurements, "
                f"**{len(df.columns)}** columns."
            )
            self._status.alert_type = "success"
            
            # Plot data if possible
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                
                required_2d = {"a", "b", "m", "n", "rhoa"}
                
                if required_2d.issubset(df.columns):
                    from georesistpy.visualization.plots import plot_pseudosection
                    fig = plot_pseudosection(df, title="Imported Apparent Resistivity Pseudosection")
                    self._plot_pane.object = fig
                elif "rhoa" in df.columns and ("ab2" in df.columns or "a" in df.columns):
                    x_col = "ab2" if "ab2" in df.columns else "a"
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.loglog(df[x_col], df["rhoa"], "bo-")
                    ax.set_xlabel(f"{x_col} (m)")
                    ax.set_ylabel("ρₐ (Ω·m)")
                    ax.set_title("Imported VES Sounding")
                    ax.grid(True, which="both", ls="--", alpha=0.5)
                    fig.tight_layout()
                    self._plot_pane.object = fig
                else:
                    self._plot_pane.object = None
            except Exception:
                pass # Don't fail the whole import if plotting fails
                
            pn.state.notifications.success(f"Loaded {len(df)} measurements")
        except Exception as exc:
            self._status.object = f"❌ Error: {exc}"
            self._status.alert_type = "danger"

    # ---- layout ----------------------------------------------------------

    def controls(self) -> pn.Column:
        """Return the input controls layout."""
        return pn.Column(
            self._file_widget,
            self._format_widget,
            self._sep_widget,
            self._load_btn,
            sizing_mode="stretch_width",
        )

    def main_panel(self) -> pn.Column:
        """Return the main visualization layout."""
        return pn.Column(
            pn.pane.Markdown("## 📂 Import Data"),
            self._status,
            self._plot_pane,
            pn.pane.Markdown("### Data Preview"),
            self._preview,
            sizing_mode="stretch_width",
        )


def _auto_read_text(text: str) -> pd.DataFrame:
    """Try multiple readers on raw text."""
    from georesistpy.io import readers

    for reader in [readers.read_res2dinv, readers.read_csv, readers.read_txt]:
        try:
            df = reader(io.StringIO(text))
            if len(df) > 0 and len(df.columns) >= 2:
                return df
        except Exception:
            continue
    raise ValueError("Could not auto-detect format.")
