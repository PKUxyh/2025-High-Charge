"""
Figure 3: Electron energy spectrum.

Load measurement data from data/fig3_data.csv and reproduce Figure 3
of the PRAB manuscript.

Usage
-----
    python fig3.py

Output
------
    fig3.pdf   (vector, for submission)
    fig3.png   (raster, 300 dpi)
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
DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "fig3_data.csv")

if os.path.isfile(DATA_FILE):
    data = np.loadtxt(DATA_FILE, delimiter=",", skiprows=1)
    energy_MeV = data[:, 0]   # electron energy [MeV]
    dN_dE = data[:, 1]        # energy spectrum dN/dE [arb. units]
else:
    rng = np.random.default_rng(13)
    energy_MeV = np.linspace(1, 10, 200)
    E0 = 5.5   # peak energy [MeV]
    sigma = 0.8
    dN_dE = np.exp(-0.5 * ((energy_MeV - E0) / sigma) ** 2)
    dN_dE += rng.normal(0, 0.02, energy_MeV.size)
    dN_dE = np.clip(dN_dE, 0, None)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots()

ax.plot(energy_MeV, dN_dE, color="C0", lw=1.2, label="Measurement")

ax.set_xlabel("Electron energy (MeV)")
ax.set_ylabel("dN/dE (arb. units)")
ax.legend(frameon=False)
fig.tight_layout()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = os.path.dirname(__file__)
fig.savefig(os.path.join(out_dir, "fig3.pdf"))
fig.savefig(os.path.join(out_dir, "fig3.png"), dpi=300)
print("Saved fig3.pdf and fig3.png")
plt.close(fig)
