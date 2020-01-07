# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 13:30:47 2018

@author: caoa
"""
import pandas as pd
import time
import argparse
from hsip_utils import parse_excel_file, standardize_columns, filter_invalid_rows
from hsip_utils import prep_data_for_matching, create_linkage_dataframe
from hsip_utils import record_linkage, assign_ids, override_rollupid
from hsip_utils import aggregate_data, prepare_output_excel
from hsip_utils import identify_possible_FN, save_excel_file

pd.options.display.max_rows = 16
pd.options.display.max_columns = 25
pd.options.mode.chained_assignment = None # suppress SettingWithCopyWarning

desc = 'HSIP person record linkage algorithm'
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('-f','--filename', type=str, help='name of Excel file')
args = parser.parse_args()
if args.filename:
    filename = args.filename
else:
    filename = 'Master_Data_File-December_python_input.xlsx'

t0 = time.time()

#%% Record Linkage Algorithm
df_raw, kathy, invalid_records = parse_excel_file(filename)
df_raw = standardize_columns(df_raw)
df, invalid_records, keep_rows, column_dict = filter_invalid_rows(df_raw, invalid_records)
df = prep_data_for_matching(df)
df_linkage = create_linkage_dataframe(df, column_dict)
print(f'{time.time()-t0:.0f} seconds have elapsed already')
matches = record_linkage(df_linkage)
master = assign_ids(df, matches)
master = override_rollupid(master, kathy, keep_rows)
master = aggregate_data(master)
xlsx = prepare_output_excel(master, kathy)
xlsx = identify_possible_FN(xlsx, df_linkage, column_dict)
save_excel_file(xlsx, invalid_records, filename, t0)
