"""
GeoResistPy application entry point.

Usage
-----
    georesistpy                        # CLI shortcut via console_scripts
    georesistpy --port 8080            # custom port
    georesistpy --no-show              # don't open browser
    python -m georesistpy              # same, via __main__.py

The Panel server starts on http://localhost:5006 by default.
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    """Launch the GeoResistPy Panel web application."""
    parser = argparse.ArgumentParser(
        prog="georesistpy",
        description="GeoResistPy — ERT Processing Suite",
    )
    parser.add_argument(
        "--port", type=int, default=5006,
        help="TCP port to bind (default: 5006)",
    )
    parser.add_argument(
        "--no-show", action="store_true", default=False,
        help="Do not open a browser window automatically",
    )
    args = parser.parse_args()

    import panel as pn
    from georesistpy.ui.app import build_app

    template = build_app()

    pn.serve(
        {"/": template},
        port=args.port,
        address="127.0.0.1",
        show=not args.no_show,
        title="GeoResistPy — ERT Processing Suite",
        websocket_origin="*",
    )


if __name__ == "__main__":
    main()
