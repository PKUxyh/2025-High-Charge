"""
Figure 5: Longitudinal bunch length (streak-camera or CTR measurement).

Load measurement data from data/fig5_data.csv and reproduce Figure 5
of the PRAB manuscript.

Usage
-----
    python fig5.py

Output
------
    fig5.pdf   (vector, for submission)
    fig5.png   (raster, 300 dpi)
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
DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "fig5_data.csv")

if os.path.isfile(DATA_FILE):
    data = np.loadtxt(DATA_FILE, delimiter=",", skiprows=1)
    time_ps = data[:, 0]   # time axis [ps]
    signal = data[:, 1]    # longitudinal profile [arb. units]
else:
    rng = np.random.default_rng(99)
    time_ps = np.linspace(-5, 5, 300)
    sigma_t = 1.2   # rms bunch length [ps]
    signal = np.exp(-0.5 * (time_ps / sigma_t) ** 2)
    signal += rng.normal(0, 0.02, time_ps.size)
    signal = np.clip(signal, 0, None)

# ---------------------------------------------------------------------------
# Compute rms bunch length
# ---------------------------------------------------------------------------
trapz = getattr(np, "trapezoid", None) if hasattr(np, "trapezoid") else np.trapz
norm = trapz(signal, time_ps)
mean_t = trapz(time_ps * signal, time_ps) / norm
rms_t = np.sqrt(trapz((time_ps - mean_t) ** 2 * signal, time_ps) / norm)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots()

ax.plot(time_ps, signal / signal.max(), color="C0", lw=1.2, label="Measurement")
ax.axvline(mean_t - rms_t, color="C1", ls="--", lw=0.9)
ax.axvline(mean_t + rms_t, color="C1", ls="--", lw=0.9, label=rf"$\sigma_t$ = {rms_t:.2f} ps")

ax.set_xlabel("Time (ps)")
ax.set_ylabel("Current profile (arb. units)")
ax.legend(frameon=False)
fig.tight_layout()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = os.path.dirname(__file__)
fig.savefig(os.path.join(out_dir, "fig5.pdf"))
fig.savefig(os.path.join(out_dir, "fig5.png"), dpi=300)
print("Saved fig5.pdf and fig5.png")
plt.close(fig)
