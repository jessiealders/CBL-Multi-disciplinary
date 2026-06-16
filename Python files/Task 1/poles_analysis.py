from pathlib import Path

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]


def p(rel_windows_path: str) -> Path:
    """This is to make sure the backslashes work on Windows and Linux OS as well."""
    return ROOT.joinpath(*rel_windows_path.split("\\"))


# Load charging station locations and neighborhood boundaries
gdf_chargers = gpd.read_file(
    ROOT / "Data_Set" / "Dataset 3 – Existing EV Charging Points" / "oplaadpalen.geojson"
)

gdf_buurten_path = ROOT / "other data" / "spatial" / "buurten.geojson"
if not gdf_buurten_path.exists():
    gdf_buurten_path = ROOT / "processed data" / "spatial" / "buurten.geojson"

gdf_buurten = gpd.read_file(gdf_buurten_path)

# Load number of households per neighborhood
density_data = pd.read_csv(
    ROOT / "processed data" / "lili_populationdesnity_districts.csv"
)[['buurtnaam', 'aantalHuishoudens']]

density_data.columns = ['buurtnaam', 'density']


# Assign each charging station to a neighborhood
joined = gpd.sjoin(
    gdf_chargers,
    gdf_buurten,
    predicate="within"
)

# Count the number of chargers in each neighborhood
counts = (joined.groupby("buurtcode").size().reset_index(name="aantal_laadpalen"))

# Add charger counts to the neighborhood GeoDataFrame
gdf_buurten = gdf_buurten.merge(
    counts,
    on="buurtcode",
    how="left"
)

# Create a table with charger counts per neighborhood
chargers_per_nbh = gdf_buurten[['buurtnaam', 'aantal_laadpalen']]

# Combine charger counts with household data
chargers_density = density_data.merge(chargers_per_nbh, on='buurtnaam')

# Remove neighborhoods with missing values or zero households
chargers_density_no_neg = (chargers_density[chargers_density['density'] > 0].dropna())

# Define explanatory and response variables
x = chargers_density_no_neg[['density']]
y = chargers_density_no_neg['aantal_laadpalen']

# Extract Strijp-S for highlighting in the plot
strijp_x = chargers_density_no_neg[chargers_density_no_neg['buurtnaam'] == 'Strijp S']['density']

strijp_y = chargers_density_no_neg[chargers_density_no_neg['buurtnaam'] == 'Strijp S']['aantal_laadpalen']

# Create scatter plot
fig, ax = plt.subplots()

data_plot = ax.scatter(x=x, y=y, alpha=0.5)

# Highlight Strijp-S
strijp_plot = ax.scatter(strijp_x, strijp_y, color='red')

ax.set_xlabel('Nr. of households')
ax.set_ylabel('Nr. of chargers')
ax.set_title('Neighborhood nr of households against nr of EV chargers')

# Fit a linear regression model
lr = LinearRegression().fit(x, y)

# Generate regression line endpoints
x_train = [[0], [max(chargers_density_no_neg['density'])]]
y_test = lr.predict(x_train)

# Plot regression line
line_plot = ax.plot(x_train, y_test, color='black')

# Add legend
ax.legend([data_plot, strijp_plot, line_plot[0]],['Data', 'Strijp S', 'Linear regression'])

plt.show()