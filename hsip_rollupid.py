# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 13:30:47 2018

@author: caoa
"""
import pandas as pd
import recordlinkage as rl
import networkx as nx
from collections import defaultdict
from nameparser import HumanName
import time
import string
import re
import os
from itertools import combinations, chain

pd.options.display.max_rows = 16
pd.options.display.max_columns = 25
pd.options.mode.chained_assignment = None # suppress SettingWithCopyWarning

t0 = time.time()

#%%
def standardize_ssn(df):
    df['ssn'].fillna('000000000', inplace=True)
    df['ssn'] = df['ssn'].astype(str).str.replace('-','')
    df['ssn'] = df['ssn'].str.replace('1{9}','000000000')
    df['ssn'] = df['ssn'].str.replace('.0','',regex=False)
    df['ssn'] = df['ssn'].str.zfill(9)
    return df

def standardize_name(df):
    tf = (df_input['address_1'].isnull()) & (df_input['address_2'].notnull())
    df.loc[tf,['address_1','address_2']] = df.loc[tf,['address_2','address_1']].values
    df['name'].fillna('', inplace=True) # check for blank names
    tf = df['name'].str.contains('@')
    df.loc[tf,'email'] = df.loc[tf,'name'] # assign email to correct column
    df.loc[tf,'name'] = ''
    name_invalid_punctuation = re.sub(r'[-&]','',string.punctuation)
    regex_punct = re.compile(rf'[{name_invalid_punctuation}]')
    regex_titles = re.compile(r'\b(MD|PHD|FCCP|DDS|MBA|MHS|)\b')
    tmp = df['name'].apply(lambda x: re.sub(regex_punct, '', x))
    df['name'] = tmp.apply(lambda x: re.sub(regex_titles, '', x).strip(' ').upper() )
    df['name'].replace({'': None}, inplace=True)
    return df

def standardize_address(df):
    tf = (df_input['address_1'].isnull()) & (df_input['address_2'].notnull())
    df.loc[tf,['address_1','address_2']] = df.loc[tf,['address_2','address_1']].values    
    columns = ['address_1','address_2','city','state']
    for column in columns:
        df[column].fillna('', inplace=True) # fill in blank entries temporarily
    tf = df['address_1'].str.contains('@')
    df.loc[tf,'email'] = df.loc[tf,'address_1'] # assign email to correct column
    df.loc[tf,'address_1'] = ''
    address_invalid_punctuation = re.sub(r'[-#/]','',string.punctuation)
    regex_punct = re.compile(rf'[{address_invalid_punctuation}]')
    df['address_1'] = df['address_1'].apply(lambda x: re.sub(regex_punct, '', x).strip() )
    df['address_2'] = df['address_2'].apply(lambda x: re.sub(regex_punct, '', x).strip() )
    df['city'] = df['city'].apply(lambda x: re.sub(regex_punct, '', x).strip() )
    for column in columns:
        df[column] = df[column].str.upper().replace({'': None})
    return df

def standardize_email(df):
    df['email'].fillna('', inplace=True) # check for blank emails
    df['email'] = df['email'].str.lower()
    df['email'] = df['email'].apply(lambda x: re.sub(r'[._]','',x).strip() )
    df['email'].replace({'': None}, inplace=True)
    return df

#%%    
wdir = r'X:\HSIP'
filename = '20190125_rev_rollupid.xlsx'
sheet_dict = pd.read_excel(os.path.join(wdir,filename), sheet_name=[0,1])
print(f'Read and standardization took {time.time()-t0:.1f} sec')

sheet_names = list(sheet_dict.keys())
df_input = sheet_dict[0]
if 'rollupid' in filename:
    df_raw = df_input.loc[:,'hsip':'AP Control']
    kathy = df_input[['AP Control','new_rollupid','TIN MATCH','NOTES']]
    invalid_records = sheet_dict[1]
else:
    df_raw = df_input.copy()

df_raw = standardize_ssn(df_raw)
df_raw = standardize_name(df_raw)
df_raw = standardize_address(df_raw)
df_raw = standardize_email(df_raw)

Rules = pd.read_csv('rules.txt', sep='|')

#%% Filter dataset based on rules.txt
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
if 'rollupid' in filename:
    invalid_records = invalid_records.append(df_raw[~keep_rows], sort=False)
else:
    invalid_records = df_raw[~keep_rows]

#%% data wrangling for matching purposes
# Address Section
# swap c/o address_1 with address_2
def look4careof(address):
    if address:
        if 'C/O' in address:
            return True
    return False
    
careof = df['address_1'].astype(str).apply(look4careof)
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

# Email Section
regex_email = re.compile('^([^@]+)@?', flags=re.IGNORECASE)

def get_local_part(x):
    match = re.search(regex_email, x)
    local_part = match.group(1).lower()
    return re.sub(r'[._]', '', local_part)

tf = df['email'].notnull()
df.loc[tf,'email_'] = df.loc[tf,'email'].apply(get_local_part)

# Name Section
def parse_name(df):
    names = []
    for row in df.itertuples():
        nom = HumanName(row.name_)
        names.append((nom.first, nom.last))
    return pd.DataFrame(names, index=df.index, columns=['first','last'])

df['name_'] = df['name'].str.replace(' - ','-').str.replace('-',' ')
names = parse_name(df)
names['initials'] = names['first'].str[0] + names['last'].str[0]
df = df.merge(names, left_index=True, right_index=True)

#%% create dataframe for linking
df_linkage = df[['first','last','initials','email_','ssn','address_']]
df_linkage.replace({'ssn':ssn_dict,
                    'address_':address_dict,
                    'email_':email_dict,
                    }, inplace=True)

#%%
def get_rules(columns):
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
        rules.exact('email_', 'email_', label='email_')
    for col in columns:
        rules.exact(col, col, label=col)
    return rules

def get_index_pairs(df, columns):
    if isinstance(columns, str):
        columns = [columns]
    indexer = rl.index.Block(left_on=columns, right_on=None)
    idx_pairs = indexer.index(df)
    return idx_pairs

def create_index_pairs(df, columns):
    if isinstance(columns, str):
        columns = [columns]
    columns_dict = df.groupby(columns).indices
    pairs = []
    for k,v in columns_dict.items():
        if len(v) > 1 and k != 'nan':
            pairs.append(combinations(v,2))
    pairs = list(chain.from_iterable(pairs))
    return pd.MultiIndex.from_tuples(pairs)

def score_calculation(df):    
    df['total'] = 0.5*df['first'] + 0.5*df['last'] + 1.5*df['email_'] + df['ssn'] + df['address_']
    df.reset_index(inplace=True)
    df.rename(columns={'level_0':'rec1','level_1':'rec2'}, inplace=True)
    df.sort_values(['total','rec1','rec2'], ascending=[False,True,True], inplace=True)
    return df

#%% blocking and linking
blocks = ['ssn','email_','address_',['last','initials'],['first','initials']]
pair_score = []
print(f'Start of Matching Process {time.time()-t0:.1f} sec')
for block in blocks:
    rules = get_rules(block)
    idx_pairs = get_index_pairs(df_linkage, block)
    print(f'{block} pairs = {len(idx_pairs):,}')
    t1 = time.time()
    features = rules.compute(idx_pairs, df_linkage)
    t2 = time.time()
    print(f'Matching took {t2-t1:.1f} sec' )
    print('Pairs per second: {:.0f}\n'.format(len(idx_pairs)/(t2-t1)))
    pair_score.append(score_calculation(features))

scores = pd.concat(pair_score, ignore_index=True, sort=True)
matches = scores[scores['total'] >= 2]
matches.reset_index(drop=True, inplace=True)

#%% assigns id to each person based on a graph connected components
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
if 'rollupid' in filename:
    df_override = kathy[kathy['new_rollupid'] > 1e6]
    master.loc[df_override.index,'rollupid'] = df_override['new_rollupid']
    print(f'Replaced {df_override.shape[0]} rows with new rollupid')        

#%%
def get_total_and_counts(master):
    name_ct = master.groupby('rollupid')['name_'].nunique(dropna=False)
    email_ct = master.groupby('rollupid')['email_'].nunique(dropna=False)
    ssn_ct = master.groupby('rollupid')['ssn'].nunique(dropna=False)
    address_ct = master.groupby('rollupid')['address_'].nunique(dropna=False)
    total = master.groupby('rollupid')['amt'].sum()
    df = pd.DataFrame({'name_ct':name_ct,
                        'email_ct':email_ct,
                        'ssn_ct':ssn_ct,
                        'address_ct':address_ct,
                        })
    df['ct_sum'] = df.sum(axis=1)
    df.insert(0,'total_rollup',total)
    return df
    
total_cts = get_total_and_counts(master)
master = master.merge(total_cts, how='left', left_on='rollupid', right_index=True)
master = master.sort_values(['rollupid','date'], ascending=[True,False])
master['rollup_rank'] = master.groupby('rollupid').cumcount() + 1

#%%
master.sort_values(['total_rollup','rollupid','rollup_rank'], ascending=[False,True,True], inplace=True)
# formatting output
df_date = master.select_dtypes(include='datetime')
if 'date' in df_date.columns:
    master['date'] = master['date'].dt.strftime('%m-%d-%Y')
if 'entered' in df_date.columns:
    master['entered'] = master['entered'].dt.strftime('%m-%d-%Y')

#%%
if 'rollupid' in filename:
    outputfile = filename.replace('rollupid','rollupidv2')
else:
    outputfile = filename.replace('.xlsx','_rollupid.xlsx')
writer = pd.ExcelWriter(os.path.join(wdir,outputfile))
#%%
xlsx = master.drop(['first','last','initials'], axis=1)
if 'rollupid' in filename:
    xlsx = xlsx.merge(kathy, how='left', on='AP Control')

#%% Identify possible false negatives
key_columns = ['name_','email_','address_','ssn','rollupid']
xlsx['valid_ssn'] = xlsx['ssn'].map(lambda x: x != '000000000')

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
address_suspects = address_ssn[(address_rollupid > 1) & (address_ssn)]
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

print('ssn', ssn_suspects.shape) 
print('name', name_suspects.shape)
print('address', address_suspects.shape)
print('email', email_suspects.shape)

#%% Save results
t5 = time.time()
xlsx.to_excel(writer, 'Working', index=False, float_format='%.2f')
invalid_records.to_excel(writer, 'invalid_rows', index=False)
writer.save()
print(f'{outputfile} created in {time.time()-t5:.0f} sec')
print(f'This whole process took too long: {time.time()-t0:.0f} sec')
