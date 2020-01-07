# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:19:56 2019

@author: caoa
"""
import pandas as pd
import re
import string
from nameparser import HumanName
import recordlinkage as rl

suffix = [('SOUTH','S'),('NORTH','N'),('EAST','E'),('WEST','W'),
    ('NORTHWEST','NW'),('SOUTHWEST','SW'),('NORTHEAST','NE'),('SOUTHEAST','SE'),
    ('AVENUE','AVE'),('BOULEVARD','BLVD'),('CENTER','CTR'),('CIRCLE','CIR'),
    ('COURT','CT'),('CRESCENT','CRES'),('DRIVE','DR'),('HEIGHTS','HTS'),
    ('HIGHWAY','HWY'),('LANE','LN'),('LAKE','LK'),('PARKWAY','PKWY'),
    ('PLACE','PL'),('POINT','PT'),('ROAD','RD'),('SQUARE','SQ'),
    ('STREET','ST'),('TR','TRL'),('TRAIL','TRL'),
]
suffix_dict = {rf'\b{x[0]}\b':x[1] for x in suffix}

#%% Functions
def standardize_ssn(df):
    df['ssn'].fillna('000000000', inplace=True)
    df['ssn'] = df['ssn'].astype(str).str.replace('-','')
    df['ssn'] = df['ssn'].str.replace('1{9}','000000000')
    df['ssn'] = df['ssn'].str.replace('.0','',regex=False)
    df['ssn'] = df['ssn'].str.zfill(9)
    return df

def standardize_name(df):
    df['name'].fillna('', inplace=True) # check for blank names
    tf = df['name'].str.contains('@') # check if name column contains an email
    df.loc[tf,'email'] = df.loc[tf,'name'] # assign email to correct column
    df.loc[tf,'name'] = ''
    name_invalid_punctuation = re.sub(r'[-&]','',string.punctuation)
    regex_punct = re.compile(rf'[{name_invalid_punctuation}]')
    regex_titles = re.compile(r'\b(MD|PHD|FCCP|DDS|MBA|MHS)\b')
    tmp = df['name'].apply(lambda x: re.sub(regex_punct, '', x))
    df['name'] = tmp.apply(lambda x: re.sub(regex_titles, '', x).strip(' ').upper() )
    df['name'].replace({'': 'UNKNOWN'}, inplace=True)
    return df

def look4careof(address):
    if address:
        if 'C/O' in address:
            return True
    return False

def standardize_address(df):
    tf = (df['address_1'].isnull()) & (df['address_2'].notnull())
    df.loc[tf,['address_1','address_2']] = df.loc[tf,['address_2','address_1']].values    
    # swap c/o address_1 with address_2      
    careof = df['address_1'].astype(str).apply(look4careof)
    df.loc[careof,['address_1','address_2']] = df.loc[careof,['address_2','address_1']].values
    # add address_1 and address_2 for numbers only address_1
    numbersonly = df['address_1'].str.isnumeric()
    notblank = df['address_2'].notnull()
    tf = numbersonly & notblank
    df.loc[tf,'address_1'] = df.loc[tf,'address_1'].astype(str) + ' ' + df.loc[tf,'address_2'].astype(str)
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

def parse_name(df):
    names = []
    for row in df.itertuples():
        nom = HumanName(row.name_)
        names.append((nom.first, nom.last))
    return pd.DataFrame(names, index=df.index, columns=['first','last'])

regex_email = re.compile('^([^@]+)@?', flags=re.IGNORECASE)
def get_local_part(x):
    match = re.search(regex_email, x)
    return match.group(1)

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

def score_calculation(df):    
    df['total'] = 0.5*df['first'] + 0.5*df['last'] + 1.5*df['email_'] + df['ssn'] + df['address_']
    df.reset_index(inplace=True)
    df.rename(columns={'level_0':'rec1','level_1':'rec2'}, inplace=True)
    df.sort_values(['total','rec1','rec2'], ascending=[False,True,True], inplace=True)
    return df

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