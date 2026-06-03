from pathlib import Path
import sys

import contextily as ctx
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]

from simulation import (
    CANDIDATE_LOCATIONS_PATH,
    EV_DEMAND_HEATMAP_PATH,
    HEATMAP_DENSITY_PATH,
    load_blended_heatmap_weights,
    load_candidate_locations,
)

SAVE_TO = ROOT / "other data" / "blended_heatmap.png"

RD_TO_3857 = Transformer.from_crs("EPSG:28992", "EPSG:3857", always_xy=True)

locations = load_candidate_locations(CANDIDATE_LOCATIONS_PATH)
weights = load_blended_heatmap_weights(
    locations, HEATMAP_DENSITY_PATH, EV_DEMAND_HEATMAP_PATH
)

xs3857, ys3857 = zip(*(RD_TO_3857.transform(loc.x, loc.y) for loc in locations))
xs3857 = np.array(xs3857)
ys3857 = np.array(ys3857)
w = np.array(weights, dtype=float)

pad = 300  # metres around the point cloud
xmin, xmax = xs3857.min() - pad, xs3857.max() + pad
ymin, ymax = ys3857.min() - pad, ys3857.max() + pad

fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_aspect("equal")

# Normalise weights to [0, 1] for colouring
w_norm = (w - w.min()) / (w.max() - w.min()) if w.max() > w.min() else np.ones_like(w)

cmap = plt.get_cmap("hot_r")
norm = mcolors.Normalize(vmin=w.min(), vmax=w.max())

sc = ax.scatter(
    xs3857,
    ys3857,
    c=w,
    cmap=cmap,
    norm=norm,
    s=60 + 200 * w_norm,  # size also encodes weight
    edgecolors="black",
    linewidths=0.4,
    alpha=0.85,
    zorder=4,
)

cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01)
cbar.set_label("Blended weight (20 % GPX + 80 % EV demand)", fontsize=9)

ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom="auto")

ax.set_title("Candidate charger locations — blended demand heatmap", fontsize=14)
ax.set_axis_off()

fig.text(
    0.01,
    0.01,
    f"{len(locations)} candidate locations · weight range {w.min():.3f}–{w.max():.3f}",
    fontsize=8,
    color="gray",
)

plt.tight_layout()
SAVE_TO.parent.mkdir(exist_ok=True)
fig.savefig(SAVE_TO, dpi=200, bbox_inches="tight")
print(f"Saved: {SAVE_TO}")
plt.close(fig)
