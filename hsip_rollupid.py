# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 13:30:47 2018

@author: caoa
"""
import pandas as pd
import networkx as nx
from collections import defaultdict
import time
import argparse
from hsip_utils import suffix_dict, standardize_ssn, standardize_name, standardize_address
from hsip_utils import standardize_email, parse_name, get_local_part, get_rules
from hsip_utils import get_index_pairs, score_calculation, get_total_and_counts

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
    filename = 'Master_Data_File-December_mod.xlsx'

t0 = time.time()

#%%
print(f'\nReading excel file {filename}')
sheet_dict = pd.read_excel(filename, sheet_name=[0,1])
print(f'This section took {time.time()-t0:.0f} seconds')

sheet_names = list(sheet_dict.keys())
df_input = sheet_dict[0]
#if 'rollupid' in filename:
df_raw = df_input.loc[:,'hsip':'AP Control']
kathy = df_input[['AP Control','new_rollupid','TIN MATCH','NOTES']]
invalid_records = sheet_dict[1]
#else:
#    df_raw = df_input.copy()

print('Standardizing ssn')
df_raw = standardize_ssn(df_raw)
print('Standardizing name')
df_raw = standardize_name(df_raw)
print('Standardizing address')
df_raw = standardize_address(df_raw)
print('Standardizing email')
df_raw = standardize_email(df_raw)
df_raw['date'] = pd.to_datetime(df_raw['date'])
df_raw['entered'] = pd.to_datetime(df_raw['entered'])

print('Reading rules.txt file')
Rules = pd.read_csv('rules.txt', sep='|')

#%% Filter dataset based on rules.txt
print('Filtering out invalid rows')
ssn_list = list(Rules.loc[Rules['column'] == 'ssn','value'])
ssn_dict = {ssn:None for ssn in ssn_list}

name_list = list(Rules.loc[Rules['column'] == 'name','value'])
name_dict = {name:None for name in name_list}

address_list = list(Rules.loc[Rules['column'] == 'address_1','value'])
address_dict = {addr.replace(' ','').lower():None for addr in address_list}

email_list = list(Rules.loc[Rules['column'] == 'email','value'])
email_dict = {email.lower():None for email in email_list}

rules_dict = defaultdict(list)
for col, rule in Rules.groupby('column'):
    rules_dict[col] = rule['value'].tolist()

list_k = []   
for col, invalid_entries in rules_dict.items():
    if col == 'email':
        rule1 = df_raw[col].isin(invalid_entries) | df_raw[col].isnull()
        rule2 = df_raw[col].str.contains('@') & df_raw[col].notnull()
        valid = (~rule1) & rule2
    else:
        rule1 = df_raw[col].isin(invalid_entries) | df_raw[col].isnull()
        valid = ~rule1
    list_k.append(valid)
columns = pd.concat(list_k, axis=1)
columns = columns.astype(int)

score = columns[['name','email','ssn','address_1']]
score['total'] = score.sum(axis=1)
score['AP Control'] = df_raw['AP Control']

keep_rows = score['total'] >= 2
df = df_raw[keep_rows]
#if 'rollupid' in filename:
invalid_records = invalid_records.append(df_raw[~keep_rows], sort=False)
#else:
#    invalid_records = df_raw[~keep_rows]

#%% data wrangling for matching purposes
print('Preparing address, email, name columns for matching purposes')
# Address Section
df['address_'] = df['address_1'].replace(suffix_dict, regex=True)
df['address_'] = df['address_'].str.replace(' ','').str.replace(',','').str.lower()
# Email Section
tf = df['email'].notnull()
df.loc[tf,'email_'] = df.loc[tf,'email'].apply(get_local_part)
# Name Section
df['name_'] = df['name'].str.replace(' - ','-').str.replace('-',' ')
assert df['name_'].notnull().all()
names = parse_name(df)
names['initials'] = names['first'].str[0] + names['last'].str[0]
df = df.merge(names, left_index=True, right_index=True)

#%% create dataframe for linking
df_linkage = df[['first','last','initials','email_','ssn','address_']]
df_linkage.replace({'ssn':ssn_dict,
                    'address_':address_dict,
                    'email_':email_dict,
                    }, inplace=True)
print(f'{time.time()-t0:.0f} seconds have elapsed already')

#%% blocking and linking
blocks = ['ssn','email_','address_',['last','initials'],['first','initials']]
pair_score = []
print('Starting Matching Process\n')
for block in blocks:
    rules = get_rules(block)
    idx_pairs = get_index_pairs(df_linkage, block)
    print(f'{block} pairs = {len(idx_pairs):,}')
    t1 = time.time()
    features = rules.compute(idx_pairs, df_linkage)
    t2 = time.time()
    print(f'Matching took {t2-t1:.1f} seconds')
    print(f'Pairs per second: {len(idx_pairs)/(t2-t1):.0f}\n')
    pair_score.append(score_calculation(features))

scores = pd.concat(pair_score, ignore_index=True, sort=True)
matches = scores[scores['total'] >= 2]
matches.reset_index(drop=True, inplace=True)

#%% assigns id to each person based on a graph connected components
print('Assigning rollupid to rows')
G = nx.Graph()
edgelist = list(zip(matches['rec1'],matches['rec2']))
G.add_edges_from(edgelist)
cc = nx.connected_components(G)
labels = {}
labels_rev = defaultdict(list)
for rollupid, recids in enumerate(cc, start=1):
    for recid in recids:
        labels[recid] = rollupid
        labels_rev[rollupid].append(recid)
personid = pd.DataFrame.from_dict(labels, orient='index')
personid.columns = ['rollupid']
maxid = rollupid + 1

#%% assigns id to singletons and merge with groups
dakota = df.merge(personid, how='left', left_index=True, right_index=True)
alex = dakota[dakota['rollupid'].notnull()]
alex['rollupid'] = alex['rollupid'].astype(int)
singletons = dakota[dakota['rollupid'].isnull()]
singletons.dropna(axis=1, how='all', inplace=True)
singletons['rollupid'] = range(maxid, maxid+singletons.shape[0])
master = pd.concat([alex, singletons], ignore_index=False)
assert master['name_'].notnull().all()
assert master['ssn'].notnull().all()

#%% manually override rollupid with new rollupid
#if 'rollupid' in filename:
df_override = kathy[(kathy['new_rollupid'] > 1e6) & keep_rows]
master.loc[df_override.index,'rollupid'] = df_override['new_rollupid']
print(f'Replaced {df_override.shape[0]} rows with manual rollupid')        

#%% Aggregating data
print('Calculating _ct columns and total_rollup')
total_cts = get_total_and_counts(master)
master = master.merge(total_cts, how='left', left_on='rollupid', right_index=True)
master = master.sort_values(['rollupid','date'], ascending=[True,False])
master['rollup_rank'] = master.groupby('rollupid').cumcount() + 1
master.sort_values(['total_rollup','rollupid','rollup_rank'], ascending=[False,True,True], inplace=True)
# formatting output
df_date = master.select_dtypes(include='datetime')
if 'date' in df_date.columns:
    master['date'] = master['date'].dt.strftime('%m-%d-%Y')
if 'entered' in df_date.columns:
    master['entered'] = master['entered'].dt.strftime('%m-%d-%Y')

#%%
xlsx = master.drop(['first','last','initials'], axis=1)
if not kathy['AP Control'].isnull().all():
    xlsx = xlsx.merge(kathy, how='left', on='AP Control')

#%% Identify possible false negatives
print('\nIdentifying possible false negatives')
key_columns = ['name_','email_','address_','ssn','rollupid']
xlsx['valid_ssn'] = xlsx['ssn'] != '000000000'
common_addresses = df_linkage['address_'].value_counts()
special_addresses = common_addresses[common_addresses <= 5].index
xlsx['less_common_address'] = xlsx['address_'].isin(special_addresses)

ssn_rollupid = xlsx.groupby('ssn')['rollupid'].nunique()
ssn_suspects = ssn_rollupid[ssn_rollupid > 1]
ssn_set = set(ssn_suspects.index) - set(ssn_dict.keys())
ssn_flag = xlsx['ssn'].isin(ssn_set)
xlsx.loc[ssn_flag,'same_ssn_diff_rollupid'] = 1

name_rollupid = xlsx.groupby('name_')['rollupid'].nunique()
name_ssn = xlsx.groupby('name_')['valid_ssn'].any()
name_suspects = name_ssn[(name_rollupid > 1) & (name_ssn)]
name_set = set(name_suspects.index) - set(name_dict.keys())
name_flag = xlsx['name_'].isin(name_set)
dupes = xlsx.loc[name_flag].duplicated(key_columns, keep='first')
tf = dupes[~dupes]
xlsx.loc[tf.index,'same_name_diff_rollupid'] = 1

address_rollupid = xlsx.groupby('address_')['rollupid'].nunique()
address_ssn = xlsx.groupby('address_')['valid_ssn'].any()
address_lesscommon = xlsx.groupby('address_')['less_common_address'].any()
address_suspects = address_ssn[(address_rollupid > 1) & (address_ssn) & (address_lesscommon)]
address_set = set(address_suspects.index) - set(address_dict.keys())
address_flag = xlsx['address_'].isin(address_set)
dupes = xlsx.loc[address_flag].duplicated(key_columns, keep='first')
tf = dupes[~dupes]
xlsx.loc[tf.index,'same_address1_diff_rollupid'] = 1

email_rollupid = xlsx.groupby('email_')['rollupid'].nunique()
email_ssn = xlsx.groupby('email_')['valid_ssn'].any()
email_suspects = email_rollupid[(email_rollupid > 1) & (email_ssn)]
email_set = set(email_suspects.index) - set(email_dict.keys())
email_flag = xlsx['email_'].isin(email_set)
dupes = xlsx.loc[email_flag].duplicated(key_columns, keep='first')
tf = dupes[~dupes]
xlsx.loc[tf.index,'same_email_diff_rollupid'] = 1

xlsx.drop(['name_','address_','email_','valid_ssn'], axis=1, inplace=True)

print(f'{len(ssn_set)} same ssn have different rollupids')
print(f'{len(name_set)} same names have different rollupids') 
print(f'{len(address_set)} same addresses have different rollupids') 
print(f'{len(email_set)} same emails have different rollupids') 

#%% Save results
outputfile = filename.replace('.xlsx','_processed.xlsx')
print(f'Creating output excel file {outputfile}')
print(f'{time.time()-t0:.0f} seconds have elapsed already')
t5 = time.time()
writer = pd.ExcelWriter(outputfile)
xlsx.to_excel(writer, 'Working', index=False, float_format='%.2f')
invalid_records.to_excel(writer, 'invalid_rows', index=False)
writer.save()
print(f'{outputfile} created in {time.time()-t5:.0f} seconds')
print(f'This whole process took too long: {time.time()-t0:.0f} seconds')
