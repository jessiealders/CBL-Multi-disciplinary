from pathlib import Path

import pandas as pd
from poles_analysis import chargers_per_nbh
import matplotlib.pyplot as plt

# Found column aantalPersonenautosMetOverigeBrandstof -> num of cars with other fuel (probably electric)

ROOT = Path(__file__).resolve().parents[1]


def p(rel_windows_path: str) -> Path:
    """This is to make sure the backlashes works on windows and linux OS as well"""
    return ROOT.joinpath(*rel_windows_path.split("\\"))

population_data = pd.read_csv(
    ROOT / "other data" / "lili_populationdesnity_districts.csv"
)[['buurtnaam', 'gemeentenaam', 'omgevingsadressendichtheid']]
ehv_population_data = population_data[population_data['gemeentenaam'] == 'Eindhoven']
od_matrix = pd.read_csv(
    ROOT / "Data_Set" / "Dataset 1 – Mobility Demand (Origin–Destination)" / "eindhoven_od_matrix.csv"
)

# population_data.columns =

most_address_dense = ehv_population_data.sort_values(
    'omgevingsadressendichtheid',ascending=False).reset_index()[0:30][
        ['buurtnaam', 'omgevingsadressendichtheid']]
dense_neighborhoods = list(most_address_dense['buurtnaam'])

# od_matrix.columns = ['origin_district'] + dense_neighborhoods
# od_matrix['origin_district'] = dense_neighborhoods
od_sum_columns = od_matrix[od_matrix.columns[1:]].sum(axis=1)
od_sum_rows = od_matrix[od_matrix.columns[1:]].sum()

sum_df = pd.DataFrame([od_sum_columns])
sum_df.columns = od_matrix.columns[1:]
# Sum of origin and destination values
sum_df += od_sum_rows
sum_df = sum_df.transpose().sort_values(0,ascending=False)
# print(list(sum_df.index))

nbh_dict = dict(zip(sum_df.index, dense_neighborhoods))
# print(nbh_dict)

od_matrix_named = od_matrix.rename(columns=nbh_dict)
od_matrix_named['origin_district'] = od_matrix_named['origin_district'].map(nbh_dict)

origin_sums = od_matrix_named[od_matrix_named.columns[1:]].sum().reset_index()
origin_sums.columns = ['buurtnaam', 'origins']

destination_sums = pd.DataFrame({'buurtnaam': origin_sums['buurtnaam'],
                                 'destinations': od_matrix_named[od_matrix_named.columns[1:]].sum(axis=1)})

total_sums = sum_df.rename(index=nbh_dict).reset_index()
total_sums.columns = ['buurtnaam', 'od_sum']

origin_chargers = origin_sums.merge(
    chargers_per_nbh,
    on="buurtnaam",
    how="left"
)

destination_chargers = destination_sums.merge(
    chargers_per_nbh,
    on="buurtnaam",
    how="left"
)

summed_chargers = total_sums.merge(
    chargers_per_nbh,
    on="buurtnaam",
    how="left"
)

# Amount of trips from... vs chargers
# Amount ot trips to... vs chargers
# Total trips... vs chargers

# fig, ax = plt.subplots(1,3, figsize=[15,4])
# ax[0].scatter(x=origin_chargers['origins'],y=origin_chargers['aantal_laadpalen'])
# ax[0].set_title('Origins')
# ax[0].set_xlim(0,max(origin_chargers['origins']) + 1000)
# ax[0].set_ylabel('Nr of EV chargers')
# ax[0].set_xlabel('Nr of trips originated')
# ax[1].scatter(x=destination_chargers['destinations'],y=destination_chargers['aantal_laadpalen'])
# ax[1].set_title('Destinations')
# ax[1].set_xlim(0,max(destination_chargers['destinations']) + 1000)
# ax[1].set_xlabel('Nr of trip destinations')
# ax[2].scatter(x=summed_chargers['od_sum'],y=summed_chargers['aantal_laadpalen'])
# ax[2].set_title('Summed')
# ax[2].set_xlim(0,max(summed_chargers['od_sum']) + 1000)
# ax[2].set_xlabel('Trip origins + destinations')

# fig.suptitle('Number of EV chargers vs number of trips per neighborhood')

# fig, ax = plt.subplots()
# ax.scatter(x=destination_chargers['destinations'],y=destination_chargers['aantal_laadpalen'])
# ax.set_xlim(0,max(destination_chargers['destinations']) + 1000)
# ax.set_xlabel('Nr of trip destinations')
# ax.set_ylabel('Nr of EV chargers')

# ax.set_title('Number of EV chargers vs number of trip destinations per neighborhood')

destination_chargers['dc_index'] = destination_chargers['destinations'] / destination_chargers['aantal_laadpalen']
print(destination_chargers[['buurtnaam','dc_index']].sort_values('dc_index', ascending=False).reset_index(drop=True))


# fig, ax = plt.subplots()
# ax.bar(x='buurtnaam',height='dc_index',data=destination_chargers)
# # ax.set_xticklabels(labels=destination_chargers['buurtnaam'], rotation=70, fontsize=5)
# interesting_nbh = ['Strijp S', 'Fellenoord', 'Woensel-West']


# for label in ax.get_xticklabels():
#     if label.get_text() in interesting_nbh:
#         label.set_color("black")
#         label.set_fontweight("bold")
#         label.set_rotation(45)
#         label.set_fontsize(7)
#     else:
#         label.set_visible(False)
# plt.show()