### Import libraries -----
import pandas as pd

### Load UCDP data -----
df = pd.read_csv("https://ucdp.uu.se/downloads/ged/ged221-csv.zip",
                 parse_dates=['date_start',
                              'date_end'],
                 low_memory=False)

### Create blank dataframe -----
df_tot = pd.DataFrame(columns=df.country.unique(),
                      index=pd.date_range(df.date_start.min(),
                                          df.date_end.max()))
# Fill with zeros
df_tot=df_tot.fillna(0)

### Fill in the data in blank dataframe ----

# For each country
for i in df.country.unique():
    
    # Make subset for country
    df_sub=df[df.country==i]
    
    # For each observation per country
    for j in range(len(df_sub)):
        
        if df_sub.date_start.iloc[j] == df_sub.date_end.iloc[j]:
            df_tot.loc[df_sub.date_start.iloc[j],i]=df_tot.loc[df_sub.date_start.iloc[j],i]+df_sub.best.iloc[j]
        
        else:
            df_tot.loc[df_sub.date_start.iloc[j]:
            df_sub.date_end.iloc[j],i]=df_tot.loc[df_sub.date_start.iloc[j]: \
                                                  df_sub.date_end.iloc[j],i]+ \
                                                  df_sub.best.iloc[j]/ \
                                                  (df_sub.date_end.iloc[j]- \
                                                  df_sub.date_start.iloc[j]).days       

### Aggregate to month level -----
df_tot=df_tot.resample('M').sum() 

# Reset index
df_tot.reset_index(inplace=True)

# Rename index column
# Source: https://www.statology.org/pandas-rename-columns/
df_tot.rename(columns = {'index': 'date'}, inplace = True)

# Change format of date variable
# Source: https://www.folkstalk.com/2022/10/pandas-convert-date-column-to-year-and-month-with-code-examples.html
df_tot['date'] = df_tot['date'].dt.to_period('M')

### Drop countries that always have zero fatalities ------
df_tot = df_tot.loc[:, (df_tot != 0).any(axis=0)]  

### Save df -----
df_tot.to_csv('ucdp_month.csv', index=False)
   
 