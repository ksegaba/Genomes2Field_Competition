#!/usr/bin/env python3
""" Generate pheno_{env}.csv files to run models on """
import pandas as pd
df = pd.read_csv('1_Training_Trait_Data_2014_2021_cleaned.csv')

## pheno for each year
years = df.Year.unique()
for yr in years:
    sub = df[df.Year==yr]
    sub['ID'] = df['Env'].str.cat(df['Hybrid'], sep='__')
    print(sub.head())
    sub[['ID', 'Yield_Mg_ha']].to_csv(f'pheno_{yr}.csv', index=False)

## pheno for low, mid, high yield

## pheno for each env
# envs = df.Env.unique()
# for env in envs:
#     sub = df[df.Env==env]
#     sub.rename({'Hybrid':'ID'}, inplace=True)
#     print(sub.head())
#     sub[['ID', 'Yield_Mg_ha']].to_csv(f'pheno_{env}.csv', index=False)