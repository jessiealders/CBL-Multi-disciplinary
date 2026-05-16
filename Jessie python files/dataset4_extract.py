from shapely import wkb
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import contextily as ctx


df = pd.read_csv(
    'other data/enexis_elektra_csv/nbnl_e_ls_verbinding.csv',
    sep=';'
)

gdf_buurten = gpd.read_file('other data/buurten.geojson')

# Convert WKB hex to geometry
df['geometry'] = df['geografischeligging'].apply(
    lambda x: wkb.loads(bytes.fromhex(x))
)

# Create GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:28992')

# Match CRS
gdf = gdf.to_crs(gdf_buurten.crs)

# Eindhoven boundary
eindhoven = gdf_buurten.dissolve()

# Spatial filter
gdf_ehv = gdf[gdf.intersects(eindhoven.geometry.iloc[0])]

# Convert to web mercator for basemap
gdf_ehv = gdf_ehv.to_crs(epsg=3857)

# Plot
fig, ax = plt.subplots(figsize=(10,10))

gdf_ehv.plot(ax=ax)

ctx.add_basemap(ax)

ax.set_axis_off()

plt.show()