# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 13:30:47 2018

@author: caoa
"""
import pandas as pd
import recordlinkage as rl
from collections import defaultdict
import argparse
import time
import sys

pd.options.display.max_rows = 16
pd.options.display.max_columns = 25
pd.options.mode.chained_assignment = None # suppress SettingWithCopyWarning

desc = 'HSIP person record linkage algorithm'
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('-f','--filename', type=str, help='name of Excel file')
parser.add_argument('-t','--threshold', type=float, default=0.85, help='threshold to use for string match using Jaro-Winkler distance')
args = parser.parse_args()
args.filename = '1 HSIP_Data_File-December_Copy.xlsx'

if not args.filename:
    print("You need to supply a --filename argument")
    sys.exit()
    
filename = args.filename
df_raw = pd.read_excel(filename, sheet=0)
df_raw.columns = ['hsip', 'sid', 'name', 'email', 'ssn', 
              'address_1', 'address_2', 'city', 'country', 'state',
              'postal', 'method', 'date', 'amt', 'entered',
              'updated','status']
rules = pd.read_csv('rules.txt', sep='|')
   
#%% Filter dataset based on rules.txt
rules_dict = defaultdict(list)
for col, rule in rules.groupby('column'):
    rules_dict[col] = rule['value'].tolist()

list_k = []   
for col, invalid_entries in rules_dict.items():
    rule1 = df_raw[col].isin(invalid_entries) | df_raw[col].isnull()
    if col == 'ssn':
        rule2 = df_raw['ssn'].str.startswith('11111')
        valid = ~(rule1 | rule2)
    else:
        valid = ~rule1
    list_k.append(valid)
columns = pd.concat(list_k, axis=1)
columns = columns.astype(int)

# special address handling
df_address = columns[['address_1','city','postal']]
df_address['address_score'] = 0.6*df_address['address_1'] + 0.4*df_address['city'] + 0.4*df_address['postal']

score = columns[['name','ssn']]
score['address'] = df_address['address_score']
score['total'] = score['name'] + score['ssn'] + score['address']

keep_rows = score['total'] >= 2
df = df_raw[keep_rows].reset_index(drop=True)

# data cleanup
df = df[df['amt'] > 0]
df['address_1'] = df['address_1'].str.replace(' ','').str.lower()
df['name'] = df['name'].str.lower()
df['n'] = df['name'].apply(lambda x: len(x.split()) )
names = df['name'].str.split(' ', expand=True, n=2)
names.columns = ['first','middle','last']
df = df.merge(names, left_index=True, right_index=True)

tf = (df['n'] == 2)
df1 = df[~tf]
df2 = df[tf]
df2.rename(columns={'last':'middle','middle':'last'}, inplace=True)
df = pd.concat([df1, df2], sort=False)

#%% blocking
indexer = rl.BlockIndex(on='last')
pairs = indexer.index(df)
print('Pairs to compare = {:,}'.format(len(pairs)) )

#%% Set rules for matching
threshold = args.threshold
rules = rl.Compare()
rules.exact('last', 'last', label='last_name')
rules.string('first', 'first', label='first_name', method='jarowinkler', threshold=threshold)
rules.string('ssn', 'ssn', label='ssn', method='jarowinkler', threshold=threshold)
rules.string('address_1', 'address_1', label='address', threshold=threshold)

t1 = time.time()
computation = rules.compute(pairs, df)
t2 = time.time()
print('Matching took {:.1f} mins'.format((t2-t1)/60) )
print('Pairs per second: {:.0f}'.format(len(pairs)/(t2-t1)))
features = computation.copy()

#%%
features['name'] = 0.5*features['first_name'] + 0.5*features['last_name']
features.drop(['first_name','last_name'], axis=1, inplace=True)
features['score'] = features.sum(axis=1)
features.reset_index(inplace=True)
features = features.astype(int)
if 'level_0' in features.columns:
    features.rename(columns={'level_0':'rec1','level_1':'rec2'}, inplace=True)
features.sort_values(['score','rec1','rec2'], ascending=[False,True,True], inplace=True)
matches = features[features['score'] >= 2].reset_index(drop=True)

#%% assigns id to each person based on match
labels = {}
sid = 1
for i in range(matches.shape[0]):
    if matches.at[i,'rec1'] in labels:
        labels[matches.at[i,'rec2']] = labels[matches.at[i,'rec1']]
    else:
        labels[matches.at[i,'rec1']] = sid
        labels[matches.at[i,'rec2']] = sid
        sid += 1
pid = pd.DataFrame.from_dict(labels, orient='index')
pid.columns = ['alexid']

#%%
dakota = df.merge(pid, how='left', left_index=True, right_index=True)
maxid = int(dakota['alexid'].max())
dupes = dakota[dakota['alexid'].notnull()]
singletons = dakota[dakota['alexid'].isnull()]
singletons.dropna(axis=1, how='all', inplace=True)

#%% Re-block on first name
indexer2 = rl.BlockIndex(on='first')
pairs2 = indexer.index(singletons)
print('Pairs to compare = {:,}'.format(len(pairs2)) )

#%%
rules2 = rl.Compare()
rules2.string('last', 'last', label='last_name', method='jarowinkler', threshold=threshold)
rules2.exact('first', 'first', label='first_name')
rules2.string('ssn', 'ssn', label='ssn', method='jarowinkler', threshold=threshold)
rules2.string('address_1', 'address_1', label='address', threshold=threshold)

t1 = time.time()
computation2 = rules2.compute(pairs2, singletons)
t2 = time.time()
print('Matching took {:.1f} mins'.format((t2-t1)/60) )
print('Pairs per second: {:.0f}'.format(len(pairs2)/(t2-t1)))
features2 = computation2.copy()

#%% Make sure there are no matches; otherwise more code is needed
features2['name'] = 0.5*features2['first_name'] + 0.5*features2['last_name']
features2.drop(['first_name','last_name'], axis=1, inplace=True)
features2['score'] = features2.sum(axis=1)
assert features2['score'].max() < 2

#%%
singletons['alexid'] = range(singletons.shape[0])
singletons['alexid'] += maxid + 1

lily = pd.concat([dupes, singletons])
lily['alexid'] = lily['alexid'].astype(int)
lily.sort_values('alexid', inplace=True)

#%%
ppl = lily.groupby('alexid')['amt'].sum().to_frame()
ppl.columns = ['total']
master = lily.merge(ppl, left_on='alexid', right_index=True)
master.sort_values(['total','alexid'], ascending=[False,True], inplace=True)
nodupes = master.drop_duplicates(['alexid','total'])
f1099 = nodupes[nodupes['total'] > 600]

#%%
outputfile = filename.replace('.xlsx','_alexid.xlsx')
writer = pd.ExcelWriter(outputfile)
xlsx = master.drop(['n','first','middle','last'], axis=1)
xlsx.to_excel(writer, 'alexid', index=False)
df_raw[~keep_rows].to_excel(writer, 'invalid_rows', index=False)
writer.save()
print('{} created'.format(outputfile))



