from pathlib import Path

import pandas as pd
import regex as re


ROOT = Path(__file__).resolve().parents[1]


def p(rel_windows_path: str) -> Path:
    """This is to make sure the backlashes works on windows and linux OS as well"""
    return ROOT.joinpath(*rel_windows_path.split("\\"))

# Reading the data from the csv file and converting to a Pandas dataframe
# There were some errors after row 347174, but none of these rows contained data about
# Eindhoven, so these 'bad lines' are skipped
congestion_data = pd.read_csv(p(r"other data\congestie_pc6.csv"), on_bad_lines='skip', sep=';')
# Using a regular expression (regex) to find all Eindhoven postcodes and save them in a list
# (used https://postcodebijadres.nl/eindhoven to find the postcodes)
ehv_postcodes_list = re.findall(r'56(?:0[0-6]|1[1-7]|2[1-9]|3[1-3]|4[1-7]|5[1-8])[A-Z]{2}',
                                str(list(congestion_data['postcode'])))
# Use the list as a filter to get only Eindhoven postcodes
ehv_congestion_data = congestion_data[congestion_data['postcode'].isin(ehv_postcodes_list)]
# Print summary statistics
# print(ehv_congestion_data.describe())

# Find only the rows that show congestion in Eindhoven
ehv_congested_rows = ehv_congestion_data[(ehv_congestion_data['afname'] > 0) |
                                         (ehv_congestion_data['opwek'] > 0)]
# Resetting the indexes for a more logical dataset
ehv_congested_rows = ehv_congested_rows.reset_index()
# Renaming the old columns and translating them to English, the old index is kept in case
# we want to look these rows up in the old dataset
ehv_congested_rows.columns = ['old_index','postcode','consumption','generation',
                              'supply_area_id','supply_area_name','tennet_id','RNB_postcode']
# Exporting the filtered dataset to a csv file
ehv_congested_rows.to_csv('ehv_congestion.csv')