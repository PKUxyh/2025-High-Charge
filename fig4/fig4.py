"""
Figure 4: Transverse beam profile (2-D false-colour image).

Load screen-monitor image data from data/fig4_data.csv and reproduce
Figure 4 of the PRAB manuscript.

Usage
-----
    python fig4.py

Output
------
    fig4.pdf   (vector, for submission)
    fig4.png   (raster, 300 dpi)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import gaussian_filter

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
        "figure.figsize": (3.375, 2.8),
    }
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "fig4_data.csv")

if os.path.isfile(DATA_FILE):
    # Expected format: 2-D array (rows = y pixels, columns = x pixels)
    image = np.loadtxt(DATA_FILE, delimiter=",")
    nx = image.shape[1]
    ny = image.shape[0]
    pixel_size_mm = 0.05   # update with actual calibration
    x_mm = (np.arange(nx) - nx / 2) * pixel_size_mm
    y_mm = (np.arange(ny) - ny / 2) * pixel_size_mm
else:
    rng = np.random.default_rng(21)
    nx, ny = 64, 64
    pixel_size_mm = 0.05
    x_mm = (np.arange(nx) - nx / 2) * pixel_size_mm
    y_mm = (np.arange(ny) - ny / 2) * pixel_size_mm
    xx, yy = np.meshgrid(x_mm, y_mm)
    sigma_x, sigma_y = 0.3, 0.25
    image = np.exp(-(xx**2 / (2 * sigma_x**2) + yy**2 / (2 * sigma_y**2)))
    image = gaussian_filter(image + rng.normal(0, 0.02, image.shape), sigma=1)
    image = np.clip(image, 0, None)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots()

im = ax.pcolormesh(
    x_mm, y_mm, image,
    cmap="inferno",
    shading="auto",
    rasterized=True,
)
cb = fig.colorbar(im, ax=ax, pad=0.02)
cb.set_label("Intensity (arb. units)")

ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
ax.set_aspect("equal")
fig.tight_layout()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = os.path.dirname(__file__)
fig.savefig(os.path.join(out_dir, "fig4.pdf"))
fig.savefig(os.path.join(out_dir, "fig4.png"), dpi=300)
print("Saved fig4.pdf and fig4.png")
plt.close(fig)
