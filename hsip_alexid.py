# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 13:30:47 2018

@author: caoa
"""
import pandas as pd
import recordlinkage as rl
import networkx as nx
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
parser.add_argument('-t','--threshold', type=float, default=0.82, help='threshold to use for string match using Jaro-Winkler distance')
args = parser.parse_args()
args.filename = '1 HSIP_Data_File-December_Copy.xlsx'
threshold = args.threshold

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
ssn_list = list(rules.loc[rules['column'] == 'ssn','value'])
ssn_dict = {}
for ssn in ssn_list:
    ssn_dict[ssn] = None

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

#%% data wrangling for matching purposes
# swap c/o address_1 with address_2
careof = df['address_1'].apply(lambda x: 'C/O' in x)
df.loc[careof,['address_1','address_2']] = df.loc[careof,['address_2','address_1']].values
# add address_1 and address_2 for numbers only address_1
numbersonly = df['address_1'].str.isnumeric()
df.loc[numbersonly,'address_1'] = df.loc[numbersonly,'address_1'] + ' ' + df.loc[numbersonly,'address_2']

# remove common trailing address descriptions to increase # of matched pairs
fetype = ['ST','DR','RD','AVE','CT','LN','BLVD','CIR','WAY','TRL',
          'PL','SE','HWY','PKWY','NE','SW','E','W','N','S',
          'NW','AVENUE','HTS','CTR','LANE','DRIVE','CIRCLE','STREET','CRES','SOUTH',
          'ROAD','PT','TR','TRAIL','WEST','HEIGHTS','COURT','CR','EAST',
          'NORTH','SQ','APT',
]
# standardize suffixes to increase # of matched pairs
suffix = [('SOUTH','S'),
('NORTH','N'),
('EAST','E'),
('WEST','W'),
('NORTHWEST','NW'),
('SOUTHWEST','SW'),
('NORTHEAST','NE'),
('SOUTHEAST','SE'),
('LANE','LN'),
('DRIVE','DR'),
('CIRCLE','CIR'),
('AVENUE','AVE'),
('STREET','ST'),
('COURT','CT'),
('ROAD','RD'),
('TRAIL','TRL'),
('TR','TRL'),
('CRESCENT','CRES'),
('HEIGHTS','HTS'),
('HIGHWAY','HWY'),
('BOULEVARD','BLVD'),
('LAKE','LK'),
('PARKWAY','PKWY'),
('PLACE','PL'),
('CENTER','CTR'),
('POINT','PT'),
('SQUARE','SQ'),
]
suffix_dict = {rf'\b{x[0]}\b':x[1] for x in suffix}

df['address_'] = df['address_1'].replace(suffix_dict, regex=True)
df['address_'] = df['address_'].str.replace(' ','').str.lower()
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

#%% create dataframe for linking
df['initials'] = df['first'].str[0] + df['last'].str[0]
df_linkage = df[['first','last','initials','ssn','address_']]
df_linkage.replace({'ssn':ssn_dict}, inplace=True)

#%%
def get_index_pairs_rules(df, columns):
    if isinstance(columns, str):
        columns = [columns]
    rules = rl.Compare()
    rules.string('first', 'first', label='first', method='jarowinkler', threshold=threshold)
    rules.string('last', 'last', label='last', method='jarowinkler', threshold=threshold)
    rules.string('ssn', 'ssn', label='ssn', method='damerau_levenshtein', threshold=0.77)
    rules.string('address_', 'address_', label='address_', method='jarowinkler', threshold=threshold)
    for col in columns:
        rules.exact(col, col, label=col)
    indexer = rl.BlockIndex(on=columns)
    idx_pairs = indexer.index(df)
    return idx_pairs, rules

def score_calculation(df):    
    df['name'] = 0.5*df['first'] + 0.5*df['last']
    df = df[['name','ssn','address_']]       
    df['total'] = df.sum(axis=1)
    df.reset_index(inplace=True)
    df.rename(columns={'level_0':'rec1','level_1':'rec2'}, inplace=True)
    df.sort_values(['total','rec1','rec2'], ascending=[False,True,True], inplace=True)
    return df

#%% blocking and linking
blocks = ['ssn','address_',['last','initials'],['first','initials']]
pair_score = []
for block in blocks:
    idx_pairs, rules = get_index_pairs_rules(df_linkage, block)
    t1 = time.time()
    features = rules.compute(idx_pairs, df_linkage)
    t2 = time.time()
    print(f'{block} Block Matching took {t2-t1:.1f} sec' )
    print('Pairs per second: {:.0f}'.format(len(idx_pairs)/(t2-t1)))
    pair_score.append(score_calculation(features))

scores = pd.concat(pair_score, ignore_index=True, sort=False)
matches = scores[scores['total'] >= 2]
matches.reset_index(drop=True, inplace=True)

#%% assigns id to each person based on a graph connected components
G = nx.Graph()
edgelist = list(zip(matches['rec1'],matches['rec2']))
G.add_edges_from(edgelist)
cc = nx.connected_components(G)
labels = {}
for sid, pids in enumerate(cc, start=1):
    for pid in pids:
        labels[pid] = sid
personid = pd.DataFrame.from_dict(labels, orient='index')
personid.columns = ['alexid']
maxid = sid + 1

#%%
dakota = df.merge(personid, how='left', left_index=True, right_index=True)
alex = dakota[dakota['alexid'].notnull()]
alex['alexid'] = alex['alexid'].astype(int)
singletons = dakota[dakota['alexid'].isnull()]
singletons.dropna(axis=1, how='all', inplace=True)
singletons['alexid'] = range(maxid, maxid+singletons.shape[0])

#%%
def get_count(data):
    name_ct = len(set(data['name']))
    ssn_ct = len(set(data['ssn']))
    address_ct = len(set(data['address_']))
    return (name_ct, ssn_ct, address_ct)

t3 = time.time()
master = pd.concat([alex, singletons], ignore_index=True)
total = master.groupby('alexid')['amt'].sum().to_frame()
total.columns = ['total']
cts = master.groupby('alexid').apply(get_count)
counts = pd.DataFrame(cts.tolist(), index=cts.index, columns=['name_ct','ssn_ct','address_ct'])
counts['ct_sum'] = counts.sum(axis=1)
total_cts = pd.concat([total, counts], axis=1)
master = master.merge(total_cts, how='left', left_on='alexid', right_index=True)
master = master.sort_values(['alexid','date'], ascending=[True,False]).reset_index()
master['record'] = master.groupby('alexid').cumcount() + 1
t4 = time.time()
print(f'Adding MetaData {t4-t3:.1f}s')

#%%
master.sort_values(['total','alexid','record'], ascending=[False,True,True], inplace=True)
#nodupes = master.drop_duplicates(['alexid'])
#f1099 = nodupes[nodupes['total'] > 600]

#%%
ssn_alexid = master.groupby('ssn')['alexid'].nunique()
ssn_suspects = ssn_alexid[ssn_alexid > 1]
print(ssn_suspects.shape) #33

name_alexid = master.groupby('name')['alexid'].nunique()
name_suspects = name_alexid[name_alexid > 1]
print(name_suspects.shape) #1781

address_alexid = master.groupby('address_')['alexid'].nunique()
address_suspects = address_alexid[address_alexid > 1]
print(address_suspects.shape) #4239

#%%
outputfile = filename.replace('.xlsx','_alexid.xlsx')
writer = pd.ExcelWriter(outputfile)
xlsx = master.drop(['address_','n','first','middle','last','initials'], axis=1)
xlsx['name'] = xlsx['name'].str.upper()
xlsx.to_excel(writer, 'alexid', index=False)
df_raw[~keep_rows].to_excel(writer, 'invalid_rows', index=False)
writer.save()
print('{} created'.format(outputfile))

#%%
#kathy = pd.read_excel('2nd Run Differences.xlsx', sheet_name='Dec Rollup')
#kathy['key'] = kathy['HSIP Control No'].astype(str) + '_' + kathy['Subject#'].astype(str)
#kathy.sort_values('key', inplace=True)
#kathy.reset_index(drop=True, inplace=True)
#knodupes = kathy.drop_duplicates('Formatted SSN')
#kset = set(knodupes['key'])
#
#kmaster = master.copy()
#kmaster['key'] = kmaster['hsip'].astype(str) + '_' + kmaster['sid'].astype(str)
#kmaster.sort_values('key', inplace=True)
#kmaster.reset_index(drop=True, inplace=True)
#nodupes_ = kmaster.drop_duplicates(['alexid'])
#aset = set(nodupes_['key'])
#
#diffset1 = aset.difference(kset)
#diffset2 = kset.difference(aset)
#print(len(diffset1),len(diffset2))
#
##%%
#cfile = 'differences.xlsx'
#writer = pd.ExcelWriter(cfile)
#xlsx = nodupes_.drop(['address_','n','first','middle','last','initials'], axis=1)
#xlsx = xlsx[xlsx['key'].isin(diffset1)]
#xlsx.sort_values('ssn', ascending=False, inplace=True)
#xlsx.to_excel(writer, 'alex diff kathy', index=False)
#xlsx2 = knodupes[knodupes['key'].isin(diffset2)]
#xlsx2 = xlsx2.merge(kmaster[['key','alexid']], how='left', on='key')
#xlsx2.sort_values('alexid', inplace=True)
#xlsx2.to_excel(writer, 'kathy diff alex', index=False)
#writer.save()
#print('{} created'.format(cfile))
