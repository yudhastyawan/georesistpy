# GeoResistPy

**Python library for Electrical Resistivity Tomography (ERT) data processing**

GeoResistPy provides a complete, workflow-based toolset for 1D Vertical
Electrical Sounding (VES) and 2D resistivity imaging — similar to professional
software such as RES2DINV or BERT, but built entirely in Python with a modern
web UI.

---

## Features

| Area | Capabilities |
|------|-------------|
| **Data Import** | CSV, TXT, ABEM, Syscal, generic spacing tables |
| **Quality Control** | Negative-resistivity removal, outlier filtering, reciprocal error analysis |
| **Survey Config** | Wenner, Schlumberger, Dipole-Dipole, Pole-Dipole, Pole-Pole |
| **Forward Modelling** | 1D layered earth, 2D resistivity (pygimli / SimPEG) |
| **Mesh Generation** | Adaptive triangular mesh with electrode & depth refinement |
| **Inversion** | 1D VES, 2D ERT — smooth / robust (L1), L-curve, auto-λ |
| **DOI** | Sensitivity-based depth-of-investigation index |
| **Visualization** | Pseudosection, inverted section, residuals, sensitivity (Plotly / Holoviews) |
| **Web UI** | Panel dashboard with 7 workflow tabs |
| **Export** | CSV, GeoTIFF, PNG, NetCDF |

## Quick Start

### 1. Installation

The easiest way to install GeoResistPy with the Web UI is via PyPI:
```bash
pip install "georesistpy[ui]"
```

*Note: For complete 2D inversion capabilities, ensure you have the required dependencies (such as SimPEG and discretize).*

### 2. Launching the Web UI

After installation, you can launch the interactive dashboard directly from your terminal:
```bash
georesistpy
```

Optional arguments:
```bash
georesistpy --port 8080      # run on a custom port instead of default 5006
georesistpy --no-show        # start the server without automatically opening a browser tab
```

## Package Layout

```
georesistpy/
    __init__.py
    app.py / __main__.py
    io/          # data readers & writers
    qc/          # quality control & filtering
    mesh/        # mesh generation
    forward/     # forward modelling (1D / 2D)
    inversion/   # inversion engines
    visualization/  # plotting (matplotlib, plotly, holoviews)
    utils/       # array configs, DOI, topography
    ui/          # Panel web application
examples/
    sample_ves.csv
    sample_ert2d.csv
    demo.ipynb
```

## License

MIT
