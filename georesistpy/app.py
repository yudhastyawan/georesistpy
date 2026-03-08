"""
GeoResistPy application entry point.

Usage
-----
    python -m georesistpy.app          # launch the web UI
    georesistpy                        # same, via console_scripts entry point

The Panel server starts on http://localhost:5006 by default.
"""

from __future__ import annotations

import sys


def main(port: int = 5006, show: bool = True) -> None:
    """Launch the GeoResistPy Panel web application.

    Parameters
    ----------
    port : int
        TCP port to bind (default 5006).
    show : bool
        Open a browser window automatically (default True).
    """
    import panel as pn

    from georesistpy.ui.app import build_app

    template = build_app()
    template.servable()

    pn.serve(
        {"/": template},
        port=port,
        address="127.0.0.1",
        show=show,
        title="GeoResistPy",
        websocket_origin="*",
    )


if __name__ == "__main__":
    # Allow optional --port and --no-show flags
    port = 5006
    show = True
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--port" and i < len(sys.argv):
            port = int(sys.argv[i + 1])
        if arg in ("--no-show", "--noshow"):
            show = False
    main(port=port, show=show)
