import pandas as pd
import matplotlib.pyplot as plt
import regex as re
from pathlib import Path


# Make Windows-style relative paths (with backslashes) work cross-platform.
ROOT = Path(__file__).resolve().parents[1]


def p(rel_windows_path: str) -> Path:
	"""This is to make sure the backlashes work on windows and linux OS as well"""
	return ROOT.joinpath(*rel_windows_path.split("\\"))

# Load dataset and print summary statistics
zonal_load_data = pd.read_csv(p(r"Data_Set\Dataset 6 – Electricity Load (Demand)\eindhoven_zonal_load.csv"))
# Split timestamps into date and time
zonal_load_data['date'] = re.findall(r'\d\/\d\/\d{4}', str(list(zonal_load_data['timestamp'])))
zonal_load_data['time'] = re.findall(r'\d+:\d{2}', str(list(zonal_load_data['timestamp'])))
# print(zonal_load_data[['date','time']])
# print(zonal_load_data.describe())

# Group by and sum over zone_id
grouped_summed_zl_data = zonal_load_data.groupby('zone_id').sum()
# print(grouped_zl_data['demand_MW'])

# Timestamp analysis
zonal_load_times = zonal_load_data['timestamp'].drop_duplicates()
# print(zonal_load_times)

# Choose a zone and display (crowded) graph of demand
chosen_zone = 'Z2'
filtered_zone_data = zonal_load_data[zonal_load_data['zone_id'] == chosen_zone]
plt.figure()
plt.bar(filtered_zone_data['timestamp'], filtered_zone_data['demand_MW'])
# plt.show()

# Choose day to display (from zone chosen above)
chosen_day = '1/1/2025'
filtered_zone_day = filtered_zone_data[filtered_zone_data['date'] == chosen_day]
plt.figure()
plt.bar(filtered_zone_day['time'], filtered_zone_day['demand_MW'])
plt.show()