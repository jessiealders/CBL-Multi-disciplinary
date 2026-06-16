import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


random_results_fit = pd.read_csv('output\scenario_kpi_fit_summary.csv')
optimization_results = pd.read_csv('output\optimization\optimization_results_avg.csv')
random_results = pd.read_csv('output\scenarios\scenario_results_avg.csv')

# print(random_results_fit[['demand_case','minimum_total_chargers']])
# print(optimization_results[['year','average_minimum_chargers']].iloc[[0,2,4,9]])

# print(random_results_fit[['served_rate','gave_up','pct_gave_up','avg_waiting_time','avg_walk_m','util_mean']])
# print(optimization_results[['average_served_rate','average_unmet_rate','average_served_demand','average_unmet_demand',
#                             'average_walking_distance','average_utilization']])


num_arrivals_base = 159
num_arrivals_2028 = 280

# print(random_results[(random_results['total_arrivals'] == 449) & (random_results['avg_walk_m'] < 300)])


baseline_results = random_results_fit[random_results_fit['demand_case'] == 'baseline']
random_fit_now = random_results[(random_results['total_arrivals'] == num_arrivals_base) & (random_results['pct_gave_up'] < 10)]
random_focus_walkdist_now = random_results[(random_results['total_arrivals'] == num_arrivals_base) & 
                                                      (random_results['avg_walk_m'] < 300)].head(1)
optim_now = optimization_results[optimization_results['year'] == 2026]

index_names = ['baseline', 'random_fit_kpi', 'random_focus_walkdist', 'optim']
num_chargers_now = [baseline_results['minimum_total_chargers'].iloc[0], random_fit_now['num_chargers'].iloc[0], 
                    random_focus_walkdist_now['num_chargers'].iloc[0], optim_now['average_minimum_chargers'].iloc[0]]
walkdist_now = [baseline_results['avg_walk_m'].iloc[0], random_fit_now['avg_walk_m'].iloc[0], 
                    random_focus_walkdist_now['avg_walk_m'].iloc[0], optim_now['average_walking_distance'].iloc[0]]
gaveup_now = [baseline_results['pct_gave_up'].iloc[0], random_fit_now['pct_gave_up'].iloc[0], 
                    random_focus_walkdist_now['pct_gave_up'].iloc[0], optim_now['average_unmet_rate'].iloc[0]]
util_now = [baseline_results['util_mean'].iloc[0], random_fit_now['util_mean'].iloc[0], 
                    random_focus_walkdist_now['util_mean'].iloc[0], optim_now['average_utilization'].iloc[0]]


now_results = pd.DataFrame({'Min_chargers': num_chargers_now, 'Avg_walkdist': walkdist_now, 'Unmet_rate': gaveup_now, 'Avg_util': util_now}, 
                           index=index_names)
now_results['Min_chargers'] = now_results['Min_chargers'].astype(float).astype(int)
now_results['Avg_walkdist'] = now_results['Avg_walkdist'].round(3)
now_results['Unmet_rate'] = now_results['Unmet_rate'].round(3)
now_results['Avg_util'] = now_results['Avg_util'].round(3)
print(now_results)


no_change_results = random_results[random_results['scenario'] == 'ev_adoption_2028_current_chargers']
random_fit_2028 = random_results_fit[random_results_fit['demand_case'] == 'ev_adoption_2028']
random_focus_walkdist_2028 = random_results[(random_results['total_arrivals'] == num_arrivals_2028) & 
                                                      (random_results['avg_walk_m'] < 300)]
optim_2028 = optimization_results[optimization_results['year'] == 2028]

num_chargers_2028 = [no_change_results['num_chargers'].iloc[0], random_fit_2028['num_chargers'].iloc[0], 
                    random_focus_walkdist_2028['num_chargers'].iloc[0], optim_2028['average_minimum_chargers'].iloc[0]]
walkdist_2028 = [no_change_results['avg_walk_m'].iloc[0], random_fit_2028['avg_walk_m'].iloc[0], 
                    random_focus_walkdist_2028['avg_walk_m'].iloc[0], optim_2028['average_walking_distance'].iloc[0]]
gaveup_2028 = [no_change_results['pct_gave_up'].iloc[0], random_fit_2028['pct_gave_up'].iloc[0], 
                    random_focus_walkdist_2028['pct_gave_up'].iloc[0], optim_2028['average_unmet_rate'].iloc[0]]
util_2028 = [no_change_results['util_mean'].iloc[0], random_fit_2028['util_mean'].iloc[0], 
                    random_focus_walkdist_2028['util_mean'].iloc[0], optim_2028['average_utilization'].iloc[0]]

results_2028 = pd.DataFrame({'Min_chargers': num_chargers_2028, 'Avg_walkdist': walkdist_2028, 'Unmet_rate': gaveup_2028, 'Avg_util': util_2028}, 
                           index=index_names)
results_2028['Min_chargers'] = results_2028['Min_chargers'].astype(float).astype(int)
results_2028['Avg_walkdist'] = results_2028['Avg_walkdist'].round(3)
results_2028['Unmet_rate'] = results_2028['Unmet_rate'].round(3)
results_2028['Avg_util'] = results_2028['Avg_util'].round(3)
print(results_2028)

def plot_kpis(df, year, demand):
    bar_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    titles = ['Minimum number of chargers', 'Average walking distance (m)', 'Percentage of unmet demand', 'Average utilization rate']
    xticks = ['Baseline', 'Randomized', 'Walkdist<300', 'Optimized']

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,11))
    ax = ax.flatten()


    for i in range(4):
        bars = ax[i].bar(index_names, df[now_results.columns[i]],color=bar_colors)
        ax[i].set_xticklabels(xticks, fontsize=13.5)
        ax[i].set_title(titles[i], fontsize=16, fontweight='bold')
        ax[i].bar_label(
        bars,
        fmt='%.1f',
        fontsize=10,
        fontweight='bold'
    
    )
        
    legend_handles = [
        Patch(color=color, label=name)
        for color, name in zip(bar_colors, index_names)
    ]

    fig.suptitle('KPI\'s for simulation and optimization scenario\'s',fontweight='bold', fontsize=22)
    plt.tight_layout(rect=[0, 0, 1, 0.94], h_pad=4)    
    fig.text(    0.5, 0.92,
        f"Year={year}, demand={demand} cars",
        ha='center',
        fontsize=18)
    plt.show()
    
plot_kpis(results_2028, 2028, num_arrivals_2028)