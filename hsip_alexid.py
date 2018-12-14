# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 13:30:47 2018

@author: caoa
"""
import pandas as pd
import recordlinkage as rl
import networkx as nx
from collections import defaultdict
import time
import string
import re
import os

pd.options.display.max_rows = 16
pd.options.display.max_columns = 25
pd.options.mode.chained_assignment = None # suppress SettingWithCopyWarning

#%%
def standardize_ssn(df):
    if df['ssn'].dtype == 'float':
        tf = df['ssn'].notnull()
        df.loc[tf,'ssn'] = df.loc[tf,'ssn'].apply(lambda x: f'{x:.0f}')
    return df

def standardize_name(df):
    tf = df['name'].notnull() # check for blank names
    df = df[tf]
    tf = df['name'].str.contains('@')
    df.loc[tf,'email'] = df.loc[tf,'name'] # assign email to correct column
    df.loc[tf,'name'] = ''
    name_invalid_punctuation = re.sub(r'[-&@]','',string.punctuation)
    regex1 = re.compile(rf'[{name_invalid_punctuation}]')
    regex2 = re.compile(r'\b(MD|PHD|FCCP|DDS|MBA|MS|MHS|)\b')
    tmp = df['name'].apply(lambda x: re.sub(regex1, '', x))
    df['name'] = tmp.apply(lambda x: re.sub(regex2, '', x).strip(' '))
    df['name'].replace({'': None}, inplace=True)
    return df

def standardize_address(df):
    tf = df['address_1'].str.contains('@')
    df.loc[tf,'email'] = df.loc[tf,'address_1'] # assign email to correct column
    df.loc[tf,'address_1'] = ''
    address_invalid_punctuation = re.sub(r'[-&#/@]','',string.punctuation)
    regex = re.compile(rf'[{address_invalid_punctuation}]')
    df['address_1'] = df['address_1'].apply(lambda x: re.sub(regex, '', x).strip(' ') )
    df['address_1'].replace({'': None}, inplace=True)
    return df

wdir = r'X:\HSIP'
filename = 'HSIP_2018_Dec_to_CSCAR.xlsx'
df_raw = pd.read_excel(os.path.join(wdir,filename), sheet_name=0)
#cols = ['HSIP Control No', 'Subject#', 'Name', 'Email', 'SSN', 'Address 1',
#       'Address 2', 'City', 'Country', 'State', 'Postal', 'Payment Type',
#       'Date', 'Payment Amount', 'Entered', 'Last Updt', 'Form Status']
#newcols = ['hsip', 'sid', 'name', 'email', 'ssn', 
#              'address_1', 'address_2', 'city', 'country', 'state',
#              'postal', 'method', 'date', 'amt', 'entered',
#              'updated','status']
#colnames = dict(zip(cols,newcols))
#df_raw.rename(columns=colnames, inplace=True)
df_raw = standardize_ssn(df_raw)
df_raw = standardize_name(df_raw)
df_raw = standardize_address(df_raw)
df_raw['uid'] = df_raw.index + 1
df_raw['uid'] = df_raw['uid'].apply(lambda x: f'HSIPDEC{x:0>6}')

Rules = pd.read_csv('rules.txt', sep='|')

#%% additional files
#df_extra = pd.read_excel('JAN_MAR_HSIP_AWARD_MSTR_COPY.xlsx', sheet_name=None)
#list_df = []
#sheets = ['HSIPJAN','HSIPMAR','AWARDJAN','AWARDMAR']
#for df, sheet in zip(df_extra.values(), sheets):
#    df['uid'] = df.index + 2
#    df['uid'] = df['uid'].apply(lambda x: f'{sheet}{x:0>5}')
#    list_df.append(df)
#df_ext = pd.concat(list_df, ignore_index=True, sort=False)
#df_ext.dropna(axis=1, how='all', inplace=True)
#df_ext = standardize_ssn(df_ext)
#    
#df_rawext = pd.concat([df_raw, df_ext], ignore_index=True)
df_rawext = df_raw.copy()

#%% Filter dataset based on rules.txt
ssn_list = list(Rules.loc[Rules['column'] == 'ssn','value'])
ssn_dict = {ssn:None for ssn in ssn_list}

address_list = list(Rules.loc[Rules['column'] == 'address_1','value'])
address_dict = {addr.replace(' ','').lower():None for addr in address_list}

rules_dict = defaultdict(list)
for col, rule in Rules.groupby('column'):
    rules_dict[col] = rule['value'].tolist()
rules_dict['email'] = []

list_k = []   
for col, invalid_entries in rules_dict.items():
    if col == 'email':
        valid = df_rawext['email'].str.contains('@') & df_rawext['email'].notnull()
    else:
        rule1 = df_rawext[col].isin(invalid_entries) | df_rawext[col].isnull()
        valid = ~rule1
    list_k.append(valid)
columns = pd.concat(list_k, axis=1)
columns = columns.astype(int)

# special address handling
score = columns[['name','email','ssn','address_1']]
score['total'] = score.sum(axis=1)
score['uid'] = df_rawext['uid']

keep_rows = score['total'] >= 2
df = df_rawext[keep_rows].reset_index(drop=True)

#%% data wrangling for matching purposes
# swap c/o address_1 with address_2
careof = df['address_1'].apply(lambda x: 'C/O' in x)
df.loc[careof,['address_1','address_2']] = df.loc[careof,['address_2','address_1']].values
# add address_1 and address_2 for numbers only address_1
numbersonly = df['address_1'].str.isnumeric()
notblank = df['address_2'].notnull()
tf = numbersonly & notblank
df.loc[tf,'address_1'] = df.loc[tf,'address_1'].astype(str) + ' ' + df.loc[tf,'address_2'].astype(str)

# standardize suffixes to increase # of matched pairs
suffix = [('SOUTH','S'),('NORTH','N'),('EAST','E'),('WEST','W'),
('NORTHWEST','NW'),('SOUTHWEST','SW'),('NORTHEAST','NE'),('SOUTHEAST','SE'),
('AVENUE','AVE'),('BOULEVARD','BLVD'),('CENTER','CTR'),('CIRCLE','CIR'),
('COURT','CT'),('CRESCENT','CRES'),('DRIVE','DR'),('HEIGHTS','HTS'),
('HIGHWAY','HWY'),('LANE','LN'),('LAKE','LK'),('PARKWAY','PKWY'),
('PLACE','PL'),('POINT','PT'),('ROAD','RD'),('SQUARE','SQ'),
('STREET','ST'),('TR','TRL'),('TRAIL','TRL'),
]
suffix_dict = {rf'\b{x[0]}\b':x[1] for x in suffix}

df['address_'] = df['address_1'].replace(suffix_dict, regex=True)
df['address_'] = df['address_'].str.replace(' ','').str.replace(',','').str.lower()

regex_email = re.compile('^(.+)@')

def get_local_part(x):
    match = re.search(regex_email, x)
    local_part = match.group(1).lower()
    return re.sub(r'[.0-9_]', '', local_part)

tf = df['email'].str.contains('@') & df['email'].notnull()
df.loc[tf,'email_'] = df.loc[tf,'email'].apply(get_local_part)

df['name'] = df['name'].str.lower()
df['n'] = df['name'].apply(lambda x: len(x.split()) )
names = df['name'].str.split(' ', expand=True, n=2)
names.columns = ['first','middle','last']
df = df.merge(names, left_index=True, right_index=True)

# swap middle and last names for ppl with only two names
twonamesonly = (df['n'] == 2)
df.loc[twonamesonly,['last','middle']] = df.loc[twonamesonly,['middle','last']].values

#%% create dataframe for linking
df['initials'] = df['first'].str[0] + df['last'].str[0]
df_linkage = df[['first','last','initials','email_','ssn','address_']]
df_linkage.replace({'ssn':ssn_dict,'address_':address_dict}, inplace=True)

#%%
def get_index_pairs_rules(df, columns):
    if isinstance(columns, str):
        columns = [columns]
    threshold = 0.82
    rules = rl.Compare()
    if 'first' not in columns:
        rules.string('first', 'first', label='first', method='jarowinkler', threshold=threshold)
    if 'last' not in columns:
        rules.string('last', 'last', label='last', method='jarowinkler', threshold=threshold)
    if 'ssn' not in columns:
        rules.string('ssn', 'ssn', label='ssn', method='damerau_levenshtein', threshold=0.77)
    if 'address_' not in columns:
        rules.string('address_', 'address_', label='address_', method='jarowinkler', threshold=threshold)
    if 'email_' not in columns:
        rules.string('email_', 'email_', label='email_', method='jarowinkler', threshold=threshold)
    for col in columns:
        rules.exact(col, col, label=col)
    indexer = rl.index.Block(left_on=columns, right_on=None)
    idx_pairs = indexer.index(df)
    return idx_pairs, rules

def score_calculation(df):    
    df['name'] = 0.5*df['first'] + 0.5*df['last']
    df = df[['name','email_','ssn','address_']]       
    df['total'] = df.sum(axis=1)
    df.reset_index(inplace=True)
    df.rename(columns={'level_0':'rec1','level_1':'rec2'}, inplace=True)
    df.sort_values(['total','rec1','rec2'], ascending=[False,True,True], inplace=True)
    return df

#%% blocking and linking
blocks = ['ssn','email_','address_',['last','initials'],['first','initials']]
pair_score = []
for block in blocks:
    idx_pairs, rules = get_index_pairs_rules(df_linkage, block)
    print(f'{block} pairs = {len(idx_pairs):,}')
    t1 = time.time()
    features = rules.compute(idx_pairs, df_linkage)
    t2 = time.time()
    print(f'Matching took {t2-t1:.1f} sec' )
    print('Pairs per second: {:.0f}\n'.format(len(idx_pairs)/(t2-t1)))
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
    name_ct = len(set(data['name'].astype(str)))
    ssn_ct = len(set(data['ssn'].astype(str)))
    address_ct = len(set(data['address_'].astype(str)))
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
master = master.sort_values(['alexid','date'], ascending=[True,False])
master['record'] = master.groupby('alexid').cumcount() + 1
t4 = time.time()
print(f'Adding MetaData {t4-t3:.1f}s')

#%%
master.sort_values(['total','alexid','record'], ascending=[False,True,True], inplace=True)
# formatting output
master['date'] = master['date'].dt.strftime('%m-%d-%Y')
master['entered'] = master['entered'].dt.strftime('%m-%d-%Y')
master['name'] = master['name'].str.upper()
master['ssn'].fillna('000000000', inplace=True)
master['ssn'] = master['ssn'].map(lambda x: f'{x:0>9}')

#%%
outputfile = filename.replace('.xlsx','_alexid.xlsx')
writer = pd.ExcelWriter(os.path.join(wdir,outputfile))
xlsx = master.drop(['n','first','middle','last','initials'], axis=1)

#%% Identify possible false negatives
address_alexid = xlsx.groupby('address_')['alexid'].nunique()
address_suspects = address_alexid[address_alexid > 1]
address_list = list(Rules.loc[Rules['column'] == 'name','value'])
address_set = set(address_suspects.index) - set(address_list)
wb3 = xlsx[xlsx['address_'].isin(address_set)]
wb3.sort_values(['address_','alexid'], inplace=True)

wb3.drop('address_', axis=1, inplace=True)
xlsx.drop('address_', axis=1, inplace=True)

ssn_alexid = xlsx.groupby('ssn')['alexid'].nunique()
ssn_suspects = ssn_alexid[ssn_alexid > 1]
ssn_set = set(ssn_suspects.index) - set(ssn_dict.keys())
wb1 = xlsx[xlsx['ssn'].isin(ssn_set)]
wb1.sort_values(['ssn','alexid'], inplace=True)

name_alexid = xlsx.groupby('name')['alexid'].nunique()
name_suspects = name_alexid[name_alexid > 1]
name_list = list(Rules.loc[Rules['column'] == 'name','value'])
name_set = set(name_suspects.index) - set(name_list)
wb2 = xlsx[xlsx['name'].isin(name_set)]
wb2.sort_values(['name','alexid'], inplace=True)

print(ssn_suspects.shape) 
print(name_suspects.shape)
print(address_suspects.shape)

#%%
xlsx.to_excel(writer, 'alexid', index=False, float_format='%.2f')
df_rawext[~keep_rows].to_excel(writer, 'invalid_rows', index=False)
wb1.to_excel(writer, 'same_ssn_diff_alexid', index=False)
wb2.to_excel(writer, 'same_name_diff_alexid', index=False)
wb3.to_excel(writer, 'same_address1_diff_alexid', index=False)
writer.save()
print('{} created'.format(outputfile))
