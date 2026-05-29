import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import contextily as ctx
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter
from shapely.geometry import LineString

ROOT = Path(__file__).resolve().parents[1]
GPX_FILE = ROOT / "other data" / "bbox_traces.gpx"

NS = {"gpx": "http://www.topografix.com/GPX/1/0"}


def load_page(path: Path) -> list[dict]:
    tree = ET.parse(path)
    root = tree.getroot()
    result = []
    for trk in root.findall("gpx:trk", NS):
        segments = []
        for trkseg in trk.findall("gpx:trkseg", NS):
            pts = [
                (float(pt.attrib["lon"]), float(pt.attrib["lat"]))
                for pt in trkseg.findall("gpx:trkpt", NS)
            ]
            if pts:
                segments.append(pts)
        if not segments:
            continue
        first_pt = trk.find(".//gpx:trkpt", NS)
        has_ts = first_pt is not None and first_pt.find("gpx:time", NS) is not None
        result.append({"segments": segments, "has_timestamp": has_ts})
    return result


def load_data() -> gpd.GeoDataFrame:
    rows = []
    for trk in load_page(GPX_FILE):
        for seg in trk["segments"]:
            if len(seg) >= 2:
                rows.append(
                    {
                        "geometry": LineString(seg),
                        "has_timestamp": trk["has_timestamp"],
                        "n_pts": len(seg),
                    }
                )

    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    gdf = gdf.to_crs(epsg=3857)

    ts = gdf["has_timestamp"].sum()
    total_pts = gdf["n_pts"].sum()
    ts_pts = gdf.loc[gdf["has_timestamp"], "n_pts"].sum()
    print(
        f"Timestamped: {ts} segments · {ts_pts:,} points\n"
        f"All segments: {len(gdf)} · {total_pts:,} points"
    )
    return gdf


def plot_heatmap(gdf, ax) -> None:
    lines = [
        np.array(row.geometry.coords) for row in gdf[gdf["has_timestamp"]].itertuples()
    ]
    lc = LineCollection(lines, colors="#0057b8", linewidths=0.8, alpha=0.25)
    ax.add_collection(lc)


density_overlay = True
save_to = ROOT / "other data" / "gpx_density.png"
density_save_to = ROOT / "other data" / "gpx_heatmap_density.npz"

gdf = load_data()
xmin, ymin, xmax, ymax = gdf.total_bounds

fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_aspect("equal")

plot_heatmap(gdf, ax)
if density_overlay:
    all_pts = np.concatenate([np.array(geom.coords) for geom in gdf.geometry])
    xs, ys = all_pts[:, 0], all_pts[:, 1]

    counts, _, _ = np.histogram2d(xs, ys, bins=512, range=[[xmin, xmax], [ymin, ymax]])
    counts = gaussian_filter(counts.T.astype(float), sigma=3)

    np.savez(
        density_save_to,
        counts=counts,
        xmin=np.float64(xmin),
        xmax=np.float64(xmax),
        ymin=np.float64(ymin),
        ymax=np.float64(ymax),
    )
    print(f"Saved density grid: {density_save_to}")

    masked = np.ma.masked_where(counts == 0, counts)
    norm = mcolors.LogNorm(vmin=max(counts[counts > 0].min(), 1), vmax=counts.max())

    hot = plt.get_cmap("hot")
    colors = hot(np.linspace(0, 1, 256))
    colors[:, 3] = np.clip(np.linspace(0, 1, 256) ** 0.3, 0, 1)
    cmap_transparent = mcolors.ListedColormap(colors)

    ax.imshow(
        masked,
        extent=(xmin, xmax, ymin, ymax),
        origin="lower",
        cmap=cmap_transparent,
        norm=norm,
        alpha=0.85,
        aspect="auto",
        zorder=3,
    )

ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom="auto")

ax.set_title("GPX data density", fontsize=14)
ax.set_axis_off()

ts_pts = gdf.loc[gdf["has_timestamp"], "n_pts"].sum()
all_pts = gdf["n_pts"].sum()
label = f"{gdf['has_timestamp'].sum()} segments · {ts_pts:,} pts (timestamped)"
if density_overlay:
    label += f" · {all_pts:,} pts total (density)"
fig.text(0.01, 0.01, label, fontsize=8, color="gray")

plt.tight_layout()

fig.savefig(save_to, dpi=200, bbox_inches="tight")
print(f"Saved heatmap: {save_to}")
plt.close(fig)
