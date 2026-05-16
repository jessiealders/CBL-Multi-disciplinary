import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

gdf_chargers = gpd.read_file("Data_Set\Dataset 3 – Existing EV Charging Points\oplaadpalen.geojson")
gdf_buurten = gpd.read_file('other data/buurten.geojson')
density_data = pd.read_csv('other data\lili_populationdesnity_districts.csv')[['buurtnaam', 'aantalHuishoudens']]
density_data.columns = ['buurtnaam','density']
# spatial join
joined = gpd.sjoin(
    gdf_chargers,
    gdf_buurten,
    predicate="within"
)


counts = (
    joined.groupby("buurtcode")
    .size()
    .reset_index(name="aantal_laadpalen")
)

gdf_buurten = gdf_buurten.merge(
    counts,
    on="buurtcode",
    how="left"
)
# gdf_buurten.plot(
#     column="aantal_laadpalen",
#     legend=True,
#     figsize=(10,10)
# )
#plt.figure()
#gdf_buurten.plot()
# ax = gdf_buurten.plot(figsize=(10,10))
# gdf_chargers.plot(ax=ax, markersize=5, color='red')
# plt.show()

chargers_per_nbh = gdf_buurten[['buurtnaam','aantal_laadpalen']]
#print(density_data)

chargers_density = density_data.merge(chargers_per_nbh,on='buurtnaam')

chargers_density_no_neg = chargers_density[chargers_density['density'] > 0].dropna()
x = chargers_density_no_neg[['density']]
y = chargers_density_no_neg['aantal_laadpalen']
strijp_x = chargers_density_no_neg[chargers_density_no_neg['buurtnaam'] == 'Strijp S']['density']
strijp_y = chargers_density_no_neg[chargers_density_no_neg['buurtnaam'] == 'Strijp S']['aantal_laadpalen']
# print(max(chargers_density_no_neg['density']))

# fig, ax = plt.subplots()
# data_plot = ax.scatter(x=x, y=y, alpha=0.5)
# strijp_plot = ax.scatter(strijp_x, strijp_y, color='red')
# ax.set_xlabel('Nr. of households')
# ax.set_ylabel('Nr. of chargers')
# ax.set_title('Neighborhood nr of households against nr of EV chargers')

# lr = LinearRegression().fit(x,y)
# x_train = [[0],[max(chargers_density_no_neg['density'])]]
# y_test = lr.predict(x_train)
# line_plot = ax.plot(x_train, y_test, color='black')
# ax.legend([data_plot,strijp_plot,line_plot[0]],['Data','Strijp S','Linear regression'])

# plt.show()