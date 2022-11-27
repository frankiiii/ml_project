# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 23:40:49 2022

@author: thoma
"""

### Import libraries
import pandas as pd

### Load UCDP data 
df = pd.read_csv("https://ucdp.uu.se/downloads/ged/ged221-csv.zip",parse_dates=['date_start','date_end'])

### creation blank dataframe (with zero)
df_tot = pd.DataFrame(columns=df.country.unique(),index=pd.date_range(df.date_start.min(),df.date_end.max()))
df_tot=df_tot.fillna(0)
### Filling of the dataset
for i in df.country.unique():
    df_sub=df[df.country==i]
    for j in range(len(df_sub)):
        if df_sub.date_start.iloc[j] == df_sub.date_end.iloc[j]:
            df_tot.loc[df_sub.date_start.iloc[j],i]=df_tot.loc[df_sub.date_start.iloc[j],i]+df_sub.best.iloc[j]
        else:
            df_tot.loc[df_sub.date_start.iloc[j]:df_sub.date_end.iloc[j],i]=df_tot.loc[df_sub.date_start.iloc[j]:df_sub.date_end.iloc[j],i]+df_sub.best.iloc[j]/(df_sub.date_end.iloc[j]-df_sub.date_start.iloc[j]).days       

### From day to month sampling
df_tot=df_tot.resample('M').sum()                     
### We keep only countries with at least one value
df_tot = df_tot.loc[:, (df_tot != 0).any(axis=0)]  

df_tot.to_csv('C:/Users/thoma/Desktop/PhD/machine learning/ml_project-main/ucdp_month.csv')
   
 