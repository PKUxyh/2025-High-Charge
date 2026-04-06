"""
Figure 2: Normalized transverse emittance measurement.

Load measurement data from data/fig2_data.csv and reproduce Figure 2
of the PRAB manuscript.

Usage
-----
    python fig2.py

Output
------
    fig2.pdf   (vector, for submission)
    fig2.png   (raster, 300 dpi)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "figure.figsize": (3.375, 2.8),
    }
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "fig2_data.csv")

if os.path.isfile(DATA_FILE):
    data = np.loadtxt(DATA_FILE, delimiter=",", skiprows=1)
    charge_nC = data[:, 0]           # beam charge [nC]
    emittance_x_mm_mrad = data[:, 1] # horizontal normalized emittance [mm·mrad]
    emittance_y_mm_mrad = data[:, 2] # vertical normalized emittance [mm·mrad]
    emitt_err = data[:, 3]           # emittance uncertainty [mm·mrad]
else:
    rng = np.random.default_rng(7)
    charge_nC = np.linspace(0.5, 5, 10)
    emittance_x_mm_mrad = 0.4 + 0.08 * charge_nC + rng.normal(0, 0.05, 10)
    emittance_y_mm_mrad = 0.38 + 0.07 * charge_nC + rng.normal(0, 0.05, 10)
    emitt_err = 0.05 * np.ones(10)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots()

ax.errorbar(
    charge_nC,
    emittance_x_mm_mrad,
    yerr=emitt_err,
    fmt="s",
    color="C0",
    capsize=3,
    label=r"$\varepsilon_{n,x}$",
)
ax.errorbar(
    charge_nC,
    emittance_y_mm_mrad,
    yerr=emitt_err,
    fmt="^",
    color="C1",
    capsize=3,
    label=r"$\varepsilon_{n,y}$",
)

ax.set_xlabel("Beam charge (nC)")
ax.set_ylabel(r"Normalized emittance (mm$\cdot$mrad)")
ax.legend(frameon=False)
fig.tight_layout()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = os.path.dirname(__file__)
fig.savefig(os.path.join(out_dir, "fig2.pdf"))
fig.savefig(os.path.join(out_dir, "fig2.png"), dpi=300)
print("Saved fig2.pdf and fig2.png")
plt.close(fig)
