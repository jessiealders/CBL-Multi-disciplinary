from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import KernelDensity

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "processed data" / "heatmap4.csv"
OUT_PATH = ROOT / "processed data" / "heatmap4_density.npz"

BINS = 1024
BANDWIDTH_M = 100

df = pd.read_csv(CSV_PATH)  # X=lon, Y=lat (EPSG:4326)

to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
xs, ys = to_3857.transform(df["X"].values, df["Y"].values)
weights = df["traffic_weight"].values.astype(float)

xmin, xmax = xs.min(), xs.max()
ymin, ymax = ys.min(), ys.max()

margin = 200.0
xmin -= margin
xmax += margin
ymin -= margin
ymax += margin

coords = np.column_stack([xs, ys])
kde = KernelDensity(bandwidth=BANDWIDTH_M, kernel="gaussian", metric="euclidean")
kde.fit(coords, sample_weight=weights)

x_grid = np.linspace(xmin, xmax, BINS)
y_grid = np.linspace(ymin, ymax, BINS)
xx, yy = np.meshgrid(x_grid, y_grid)
grid_pts = np.column_stack([xx.ravel(), yy.ravel()])

log_density = kde.score_samples(grid_pts)
density = np.exp(log_density).reshape(BINS, BINS)

counts = gaussian_filter(density, sigma=1)

np.savez(
    OUT_PATH,
    counts=counts,
    xmin=np.float64(xmin),
    xmax=np.float64(xmax),
    ymin=np.float64(ymin),
    ymax=np.float64(ymax),
)
print(f"Saved density grid {counts.shape} → {OUT_PATH}")
print(f"  x: {xmin:.1f} – {xmax:.1f}  ({xmax - xmin:.0f} m)")
print(f"  y: {ymin:.1f} – {ymax:.1f}  ({ymax - ymin:.0f} m)")
print(f"  density range: {counts.min():.3e} – {counts.max():.3e}")
