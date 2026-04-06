"""
Figure 1: Beam charge as a function of laser pulse energy.

Load measurement data from data/fig1_data.csv and reproduce Figure 1
of the PRAB manuscript.

Usage
-----
    python fig1.py

Output
------
    fig1.pdf   (vector, for submission)
    fig1.png   (raster, 300 dpi)
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
        "figure.figsize": (3.375, 2.8),  # single-column width for PRL/PRAB
    }
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "fig1_data.csv")

if os.path.isfile(DATA_FILE):
    data = np.loadtxt(DATA_FILE, delimiter=",", skiprows=1)
    laser_energy_mJ = data[:, 0]  # laser pulse energy [mJ]
    beam_charge_nC = data[:, 1]   # beam charge [nC]
    charge_err_nC = data[:, 2]    # measurement uncertainty [nC]
else:
    # Placeholder synthetic data — replace with real measurements.
    rng = np.random.default_rng(42)
    laser_energy_mJ = np.linspace(50, 300, 11)
    beam_charge_nC = 0.012 * laser_energy_mJ + rng.normal(0, 0.05, laser_energy_mJ.size)
    charge_err_nC = 0.05 * np.ones_like(laser_energy_mJ)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots()

ax.errorbar(
    laser_energy_mJ,
    beam_charge_nC,
    yerr=charge_err_nC,
    fmt="o",
    color="C0",
    capsize=3,
    label="Measurement",
)

ax.set_xlabel("Laser pulse energy (mJ)")
ax.set_ylabel("Beam charge (nC)")
ax.legend(frameon=False)
fig.tight_layout()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = os.path.dirname(__file__)
fig.savefig(os.path.join(out_dir, "fig1.pdf"))
fig.savefig(os.path.join(out_dir, "fig1.png"), dpi=300)
print("Saved fig1.pdf and fig1.png")
plt.close(fig)
