from pathlib import Path
import pandas as pd
from poles_analysis import chargers_per_nbh

ROOT = Path(__file__).resolve().parents[2]

# Load population density data and keep only Eindhoven neighborhoods
population_data = pd.read_csv(
    ROOT / "processed data" / "lili_populationdesnity_districts.csv"
)[['buurtnaam', 'gemeentenaam', 'omgevingsadressendichtheid']]

ehv_population_data = population_data[population_data['gemeentenaam'] == 'Eindhoven']

# Load OD matrix
od_matrix = pd.read_csv(ROOT / "Data_Set" / "Dataset 1 – Mobility Demand (Origin–Destination)" / "eindhoven_od_matrix.csv")


# Select the 30 most address-dense Eindhoven neighborhoods
most_address_dense = (
    ehv_population_data.sort_values('omgevingsadressendichtheid', ascending=False).head(30)[['buurtnaam', 'omgevingsadressendichtheid']])

dense_neighborhoods = list(most_address_dense['buurtnaam'])

# Calculate total OD volume (origins + destinations) for each anonymous zone
zone_totals = pd.DataFrame({
    'od_sum':od_matrix.iloc[:, 1:].sum(axis=1).to_numpy() + od_matrix.iloc[:, 1:].sum().to_numpy()
}, index=od_matrix.columns[1:]).sort_values('od_sum', ascending=False)

# Match anonymous OD zones to Eindhoven neighborhoods
nbh_dict = dict(zip(zone_totals.index, dense_neighborhoods))

# Rename OD matrix rows and columns using neighborhood names
od_matrix_named = od_matrix.rename(columns=nbh_dict)
od_matrix_named['origin_district'] = (od_matrix_named['origin_district'].map(nbh_dict))

# Calculate total arrivals (destinations) per neighborhood
destination_sums = pd.DataFrame({
    'buurtnaam': od_matrix_named['origin_district'],
    'destinations': od_matrix_named.iloc[:, 1:].sum(axis=1)
})

# Combine destination demand with number of chargers
destination_chargers = destination_sums.merge(
    chargers_per_nbh,
    on='buurtnaam',
    how='left'
)

# Destination-to-charger index
destination_chargers['dc_index'] = (destination_chargers['destinations'] / destination_chargers['aantal_laadpalen'])

print(destination_chargers[['buurtnaam', 'dc_index']].sort_values('dc_index', ascending=False).reset_index(drop=True))