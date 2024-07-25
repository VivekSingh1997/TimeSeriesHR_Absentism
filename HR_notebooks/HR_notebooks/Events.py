#!/usr/bin/env python
# coding: utf-8

# ### Events API

# #### Source Directory

# In[1]:


get_ipython().run_line_magic('pwd', '')


# In[2]:


source_path = 'E:\HR-Analytics\source'
source_path


# In[3]:


import os
os.chdir(source_path)
get_ipython().run_line_magic('pwd', '')


# In[4]:


ls


# #### Imports

# In[5]:


import calendar
import datetime
import itertools
import jaydebeapi as jdb
import json
import logging
import math
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.io.sql as psql
import predicthq
import psycopg2
import random
import requests
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
import statsmodels as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import urllib.request



# In[6]:


# Internal Python Packages
import calendrical
from calendrical import get_nth_kday_of_month
from database import connect_greenplum
from database import create_frame_from_pg
from database import create_sqlalchemy_engine
from main import get_pipeline_config
from main import get_plant_config
from model import calendrical_features
from model import get_holidays
from model import holiday_mean


# In[7]:


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.debug("test")


# In[8]:


pd.set_option('display.max_rows', 1000)


# #### Read in Pipeline and Plant Specs

# In[9]:





# In[10]:


plant_id = 'jnap'
plant_id


# In[11]:


pipeline_specs['plant_id'] = plant_id
pipeline_specs['project_directory'] = source_path
pipeline_specs


# In[12]:


plant_specs = get_plant_config(pipeline_specs)
plant_specs


# #### Set Proxy

# In[13]:


https_proxy = 'https://t9939vs:Jeepnov@2019@iproxy.appl.chrysler.com:9090'
proxyDict = {"https" : https_proxy}
proxyDict


# #### Read Events Table

# In[14]:


https_proxy = 'https://t9939vs:Jeepnov@2019@iproxy.appl.chrysler.com:9090'
proxyDict = {"https" : https_proxy}
proxyDict


# #### TicketMaster API

# In[ ]:


# TicketMaster API
url_string = "https://app.ticketmaster.com/discovery/v2/events.json?apikey=YGFCvuvjiyjJmT7KbvluGhTASZsBRQxt&page=0"
marketid = 7
url_string += "&marketId={}".format(marketid)
nevents = 200
url_string += "&size={}".format(nevents)
url_string


# In[ ]:


# add to URL string for a given date range
start_date = "2019-10-01T00:00:00Z"
end_date = "2019-11-01T00:00:00Z"
start_date_string = "&startDateTime={}".format(start_date)
end_date_string = "&endDateTime={}".format(end_date)
url_string += start_date_string + end_date_string
url_string


# In[ ]:


response = requests.get(
    url=url_string,
    proxies=proxyDict
)


# In[ ]:


# TicketMaster API
response = requests.get(
    url="https://app.ticketmaster.com/discovery/v2/events.json?apikey=YGFCvuvjiyjJmT7KbvluGhTASZsBRQxt&marketId=7&size=200&page=0",
    proxies=proxyDict
)


# In[ ]:


# TicketMaster API
response = requests.get(
    url="https://app.ticketmaster.com/discovery/v2/events.json?apikey=YGFCvuvjiyjJmT7KbvluGhTASZsBRQxt&dmaId=266&page=0",
    proxies=proxyDict
)


# In[ ]:


response_json = response.json()


# In[ ]:


response


# In[ ]:


response.status_code


# In[ ]:


json_data = json.loads(response.text)


# In[ ]:


json_data


# In[ ]:


json_data['_embedded']['events']


# In[ ]:


len(json_data['_embedded']['events'])


# In[ ]:


for i, event in enumerate(json_data['_embedded']['events']):
    print("\nNext Event\n")
    print(i+1, event['name'], event['dates']['start']['localDate'])


# In[ ]:


event


# In[ ]:


event['name']


# In[ ]:


event['dates']['start']['localDate']


# In[ ]:


event['dates']['start']['localTime']


# In[ ]:


event['classifications'][0]['segment']['name']


# In[ ]:


event['classifications'][0]['genre']['name']


# In[ ]:


event['priceRanges'][0]['max']


# In[ ]:


event_list = []


# In[ ]:


for i, event in enumerate(json_data['_embedded']['events']):
    print("\nNext Event\n")
    print(i+1, event['name'], event['dates']['start']['localDate'])
    try:
        local_time = event['dates']['start']['localTime']
    except:
        local_time = 'Not Available'
    event_list.append((marketid,
                       event['name'],
                       event['dates']['start']['localDate'],
                       local_time,
                       event['classifications'][0]['genre']['name']))        


# In[ ]:


# Create event table
df_event = pd.DataFrame(event_list, columns=['market_id', 'event_name', 'event_date', 'event_time', 'genre'])


# In[ ]:


df_event.sort_values(by=['market_id', 'event_date', 'event_time'], inplace=True)


# In[ ]:


df_event.shape


# In[ ]:


df_event['genre'].value_counts()


# In[ ]:


def write_frame_to_pg(df, table_name, pipeline_specs, data_path):
    # extract specifications
    schema = pipeline_specs['datalake']['schema']
    # establish connection
    print("Establishing connection to Greenplum")
    conn_dl, curs_dl = connect_greenplum(pipeline_specs['datalake'])
    # establish SQL Alchemy connection
    engine_dl = create_sqlalchemy_engine(pipeline_specs['datalake'])
    # create table name
    table = str.lower('_'.join([table_name, 'tbl']))
    full_table_name = '.'.join([schema, table])
    print("Table Name: %s" % full_table_name)
    # drop table if already exists
    print("Dropping table")
    drop_table = 'drop table if exists ' + full_table_name
    curs_dl.execute(drop_table)
    # create empty table
    print("Creating table %s" % full_table_name)
    empty_table = pd.io.sql.get_schema(df, full_table_name, con=engine_dl)
    empty_table = empty_table.replace('"', '')
    print(empty_table)
    curs_dl.execute(empty_table)
    # save the CSV file
    file_name = table + '.csv'
    csv_file = '/'.join([data_path, file_name])
    print("Saving CSV file %s" % csv_file)
    df.fillna(0, inplace=True)
    df.to_csv(csv_file, index=False)
    # create sql for copying table
    SQL_STATEMENT = """
        COPY %s FROM STDIN WITH
            CSV
            HEADER
            DELIMITER AS ','
        """
    # copy file to the table
    print("Copying table from %s" % csv_file)
    f = open(csv_file)
    curs_dl.copy_expert(sql=SQL_STATEMENT % full_table_name, file=f)
    # execute grants
    print("Executing grants")
    grant = 'grant select on table ' + full_table_name + ' to datasci'
    curs_dl.execute(grant)
    grant = 'grant select on table ' + full_table_name + ' to hrba'
    curs_dl.execute(grant)
    # close connection
    print("Closing connection")
    conn_dl.commit()
    curs_dl.close()
    return


# In[ ]:


data_path = 'E:\HR-Analytics\data'
data_path


# In[15]:


table_name = '_'.join(['abs', 'event_calendar'])
table_name


# In[ ]:


write_frame_to_pg(df_event, table_name, pipeline_specs, data_path)


# In[16]:


# Connect to Greenplum
conn_dl, curs_dl = connect_greenplum(pipeline_specs['datalake'])
conn_dl, curs_dl


# In[17]:


schema = pipeline_specs['datalake']['schema']
query = "select * from \"" + schema + "\".\"" + table_name + "_tbl\""
query


# In[18]:


df_event_in = create_frame_from_pg(conn_dl, table_name, query)


# In[19]:


df_event_in.shape


# In[20]:


df_event_in.columns


# In[21]:


df_event_in


# In[22]:


df_event_in.iloc[246:250]


# In[23]:


df_event_in.index[246:250]


# In[60]:


df_event_in['genre'].value_counts(dropna=False)


# In[54]:


lf_join = lambda x : ';'.join(x)
lf_join_str = lambda x : ';'.join(map(str, x))
agg_dict = {'market_id'  : lf_join_str,
            'event_name' : lf_join,
            'event_time' : lf_join,
            'genre'      : lf_join}
agg_dict


# In[55]:


# Test of multiple events on same day
df_event_merge = df_event_in.groupby(['event_date']).agg(agg_dict).reset_index()
df_event_merge


# In[61]:


df_event_merge['genre'].value_counts()


# In[58]:


df_event_merge['genre'][0].split(';')


# In[ ]:


df_event_out = df_event_in.drop(df_event_in.index[246:250])


# In[ ]:


df_event_out


# In[ ]:


write_frame_to_pg(df_event_out, table_name, pipeline_specs, data_path)


# #### Update Events Table

# In[ ]:


def update_events(pipeline_specs):
    # set proxy
    https_proxy = 'https://t0272m1:AlphaPy2019$@iproxy.appl.chrysler.com:9090'
    proxyDict = {"https" : https_proxy}
    # read in event table
    conn_dl, curs_dl = connect_greenplum(pipeline_specs['datalake'])
    schema = pipeline_specs['datalake']['schema']
    table_name = 'abs_event_calendar_tbl'
    query = "select * from \"" + schema + "\".\"" + table_name + "\""
    df_event = create_frame_from_pg(conn_dl, table_name, query)
    # get latest event snapshot
    url_string = "https://app.ticketmaster.com/discovery/v2/events.json?apikey=YGFCvuvjiyjJmT7KbvluGhTASZsBRQxt&page=0"
    marketid = 7    # this will be stored in plant_specs
    url_string += "&marketId={}".format(marketid)
    nevents = 200   # this will be stored in plant_specs
    url_string += "&size={}".format(nevents)
    response = requests.get(url=url_string, proxies=proxyDict)
    if response.status_code == 200:
        json_data = json.loads(response.text)
        event_list = []
        for event in json_data['_embedded']['events']:
            try:
                local_time = event['dates']['start']['localTime']
            except:
                local_time = 'Not Available'
            event_list.append((marketid,
                               event['name'],
                               event['dates']['start']['localDate'],
                               local_time,
                               event['classifications'][0]['genre']['name']))
        event_cols = ['market_id', 'event_name', 'event_date', 'event_time', 'genre']
        df_event_new = pd.DataFrame(event_list, columns=event_cols)
        # concatenate new events with existing table
        df_event = pd.concat([df_event, df_event_new])
        # sort concatenated frame and drop duplicates
        df_event.sort_values(by=['market_id', 'event_date', 'event_time'], inplace=True)
        df_event.drop_duplicates(keep='last', inplace=True)
        # write updated table
        # write_frame_to_pg()
    else:
        print("Status Code from Events API: %d" % response.status_code)
    return df_event


# In[ ]:


df_event = update_events(pipeline_specs)


# In[ ]:


df_event.shape


# In[ ]:


df_event


# In[ ]:


df_event.drop_duplicates(subset=['event_date'], keep='first', inplace=True)


# In[ ]:


df_event.shape


# In[ ]:


df_event


# #### Calendrical Analysis

# In[ ]:


table_date = '20190908'
table_date


# In[ ]:


# Read in plant table
table_name = '_'.join(['abs', plant_id, 'plant', table_date, 'tbl'])
file_name = '.'.join([table_name, 'csv'])
file_path = '/'.join([data_path, file_name])
df_plant = pd.read_csv(file_path)


# In[ ]:


df_plant.columns


# In[ ]:


# drop columns to simulate recreation of calendrical features
drop_cols = ['year',
             'month',
             'quarter',
             'week',
             'day',
             'day_of_week',
             'day_of_year',
             'nth_kday',
             'au_group_mean',
             'au_dow_mean',
             'au_week_mean',
             'au_month_mean',
             'au_nth_kday_mean',
             'holiday',
             'holiday_offset',
             'holiday_dates',
             'au_holiday_mean',
             'au_dow_pct',
             'au_week_pct',
             'au_month_pct',
             'au_nth_kday_pct',
             'au_holiday_pct',
             'day_name',
             'month_name',
             'feature1',
             'feature2',
             'feature3',
             'feature4',
             'feature5']
df_plant.drop(columns=drop_cols, inplace=True)


# In[ ]:


df_plant.sample(20)


# In[ ]:


def create_calendrical_stats(dfm, group_cols):
    logger.info("Creating Calendrical Statistics")
    # Fill in date information for all rows vis a vis the prediction frame
    dfm['workdate_dt'] = pd.to_datetime(dfm['workdate'])
    dfm['year'] = dfm['workdate_dt'].dt.year
    dfm['month'] = dfm['workdate_dt'].dt.month
    dfm['quarter'] = dfm['workdate_dt'].dt.quarter
    dfm['week'] = dfm['workdate_dt'].dt.week
    dfm['day'] = dfm['workdate_dt'].dt.day
    dfm['day_of_week'] = dfm['workdate_dt'].dt.dayofweek
    dfm['day_of_year'] = dfm['workdate_dt'].dt.dayofyear
    dfm['nth_kday'] = dfm[['day', 'month', 'year']].apply(lambda x: get_nth_kday_of_month(*x), axis=1)
    # Calculate means and merge all frames together
    dfm_mean = dfm.groupby(group_cols)[['absences_unplanned']].mean().reset_index()
    dfm1 = pd.merge(dfm, dfm_mean, left_on=group_cols, right_on=group_cols, how='left')
    dfm1.rename(index=str, columns={'absences_unplanned_x': 'absences_unplanned', 'absences_unplanned_y': 'au_group_mean'}, inplace=True)
    join_cols = group_cols + ['day_of_week']
    dfm_dow = dfm.groupby(join_cols)[['absences_unplanned']].mean().reset_index()
    dfm2 = pd.merge(dfm1, dfm_dow, left_on=join_cols, right_on=join_cols, how='left')
    dfm2.rename(index=str, columns={'absences_unplanned_x': 'absences_unplanned', 'absences_unplanned_y': 'au_dow_mean'}, inplace=True)
    join_cols = group_cols + ['week']
    dfm_week = dfm.groupby(join_cols)[['absences_unplanned']].mean().reset_index()
    dfm3 = pd.merge(dfm2, dfm_week, left_on=join_cols, right_on=join_cols, how='left')
    dfm3.rename(index=str, columns={'absences_unplanned_x': 'absences_unplanned', 'absences_unplanned_y': 'au_week_mean'}, inplace=True)
    join_cols = group_cols + ['month']
    dfm_month = dfm.groupby(join_cols)[['absences_unplanned']].mean().reset_index()
    dfm4 = pd.merge(dfm3, dfm_month, left_on=join_cols, right_on=join_cols, how='left')
    dfm4.rename(index=str, columns={'absences_unplanned_x': 'absences_unplanned', 'absences_unplanned_y': 'au_month_mean'}, inplace=True)
    join_cols = group_cols + ['nth_kday', 'day_of_week']
    dfm_nth = dfm.groupby(join_cols)[['absences_unplanned']].mean().reset_index()
    dfm5 = pd.merge(dfm4, dfm_nth, left_on=join_cols, right_on=join_cols, how='left')
    dfm5.rename(index=str, columns={'absences_unplanned_x': 'absences_unplanned', 'absences_unplanned_y': 'au_nth_kday_mean'}, inplace=True)
    # Holiday Features
    holidays = get_holidays(dfm5)
    dfh = dfm5.apply(calendrical_features, holidays=holidays, axis=1)
    dfm5 = pd.concat([dfm5, dfh], axis=1)
    dfm5['au_holiday_mean'] = dfm5.apply(holiday_mean, df=dfm5, axis=1)
    # Mean Imputation for NA values
    dfm5['au_dow_mean'].replace([np.inf, -np.inf], np.nan, inplace=True)
    dfm5['au_dow_mean'].fillna(dfm5['au_group_mean'], inplace=True)
    dfm5['au_week_mean'].replace([np.inf, -np.inf], np.nan, inplace=True)
    dfm5['au_week_mean'].fillna(dfm5['au_group_mean'], inplace=True)
    dfm5['au_month_mean'].replace([np.inf, -np.inf], np.nan, inplace=True)
    dfm5['au_month_mean'].fillna(dfm5['au_group_mean'], inplace=True)
    dfm5['au_nth_kday_mean'].replace([np.inf, -np.inf], np.nan, inplace=True)
    dfm5['au_nth_kday_mean'].fillna(dfm5['au_group_mean'], inplace=True)
    dfm5['au_holiday_mean'].replace([np.inf, -np.inf], np.nan, inplace=True)
    dfm5['au_holiday_mean'].fillna(dfm5['au_group_mean'], inplace=True)
    # Historical Percentages
    dfm5['au_dow_pct'] = round(100 * (dfm5['au_dow_mean'] / dfm5['au_group_mean'] - 1), 0)
    dfm5['au_dow_pct'].fillna(0, inplace=True)
    dfm5['au_dow_pct'] = dfm5['au_dow_pct'].astype(int)
    dfm5['au_week_pct'] = round(100 * (dfm5['au_week_mean'] / dfm5['au_group_mean'] - 1), 0)
    dfm5['au_week_pct'].fillna(0, inplace=True)
    dfm5['au_week_pct'] = dfm5['au_week_pct'].astype(int)
    dfm5['au_month_pct'] = round(100 * (dfm5['au_month_mean'] / dfm5['au_group_mean'] - 1), 0)
    dfm5['au_month_pct'].fillna(0, inplace=True)
    dfm5['au_month_pct'] = dfm5['au_month_pct'].astype(int)
    dfm5['au_nth_kday_pct'] = round(100 * (dfm5['au_nth_kday_mean'] / dfm5['au_group_mean'] - 1), 0)
    dfm5['au_nth_kday_pct'].fillna(0, inplace=True)
    dfm5['au_nth_kday_pct'] = dfm5['au_nth_kday_pct'].astype(int)
    dfm5['au_holiday_pct'] = round(100 * (dfm5['au_holiday_mean'] / dfm5['au_group_mean'] - 1), 0)
    dfm5['au_holiday_pct'].fillna(0, inplace=True)
    dfm5['au_holiday_pct'] = dfm5['au_holiday_pct'].astype(int)
    # Set day and month names   
    dfm5['day_name'] = dfm5['workdate_dt'].dt.day_name()
    dfm5['month_name'] = dfm5['workdate_dt'].dt.month_name()
    # Assemble calendar features
    dfm5['feature1'] = dfm5['day_name'] + ': ' + dfm5['au_dow_pct'].astype(str) + '%'
    dfm5['feature2'] = 'Week ' + dfm5['week'].astype(str) + ': ' + dfm5['au_week_pct'].astype(str) + '%'
    dfm5['feature3'] = dfm5['month_name'] + ': ' + dfm5['au_month_pct'].astype(str) + '%'
    dfm5['feature4'] = dfm5['nth_kday'].astype(str) + ' ' + dfm5['day_name'] +  ': ' + dfm5['au_nth_kday_pct'].astype(str) + '%'
    dfm5['feature4'] = dfm5['feature4'].str.replace('^1', '1st', regex=True)
    dfm5['feature4'] = dfm5['feature4'].str.replace('^2', '2nd', regex=True)
    dfm5['feature4'] = dfm5['feature4'].str.replace('^3', '3rd', regex=True)
    dfm5['feature4'] = dfm5['feature4'].str.replace('^4', '4th', regex=True)
    dfm5['feature4'] = dfm5['feature4'].str.replace('^5', '5th', regex=True)
    dfm5['holiday_plus_minus'] = dfm5['holiday_offset'].apply(lambda x: 'after' if x > 0 else 'before')
    dfm5.loc[dfm5['holiday_offset'] == 0, 'holiday_plus_minus'] = 'it\'s'
    dfm5['feature5'] = abs(dfm5['holiday_offset']).astype(str) + ' days ' + dfm5['holiday_plus_minus'] + ' '                        + dfm5['holiday'] + ': ' + dfm5['au_holiday_pct'].astype(str) + '%'
    # Read in event table
    df_event = update_events(pipeline_specs)
    df_event.drop_duplicates(subset=['event_date'], keep='first', inplace=True)
    dfm5 = pd.merge(dfm5, df_event, left_on=['workdate'], right_on=['event_date'], how='left')
    # Read in event table
    drop_cols = ['workdate_dt',
                 'holiday_plus_minus',
                 'market_id',
                 'event_date',
                 'event_time',
                 'genre']
    dfm5.drop(columns=drop_cols, inplace=True)
    return dfm5


# In[ ]:


dfc = create_calendrical_stats(df_plant, plant_specs['model']['levels'])


# In[ ]:


dfc_event = pd.merge(dfc, df_event, left_on=['workdate'], right_on=['event_date'], how='left')


# In[ ]:


dfc_event.columns


# In[ ]:


dfc_event[['workdate', 'holiday', 'holiday_offset', 'feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'event_name']].tail(50)


# #### PredictHQ API

# In[ ]:


radius = 100
lat_long = "42.375292,-82.966222"
area_string = str(radius) + "mi@" + lat_long
area_string


# In[ ]:


maximum_events = 20
maximum_events


# In[ ]:


offset = 0
offset


# In[ ]:


# This is the event search.
response = requests.get(
    url="https://api.predicthq.com/v1/events/",
    headers={"Authorization": "Bearer CIve3xYUdfXFn0Wbznseqr8ngCzgfR"},
    params={"within"           : area_string,
            "limit"            : maximum_events,
            "offset"           : offset,
            "start.gte"        : "2018-01-01",
            "start.lte"        : "2019-01-01",
            "category"         : "festivals",
            "rank_level"       : "4,5"},
    proxies=proxyDict
)


# In[ ]:


# This is the event calendar, which is more of a summary.
response = requests.get(
    url="https://api.predicthq.com/v1/events/calendar/",
    headers={"Authorization": "Bearer CIve3xYUdfXFn0Wbznseqr8ngCzgfR"},
    params={"within"           : area_string,
            "limit"            : maximum_events,
            "offset"           : offset,
            "category"         : "concerts,festivals,sports",
            "rank_level"       : "4,5"},
    proxies=proxyDict
)


# In[ ]:


response_json = response.json()


# In[ ]:


response


# In[ ]:


json_data = json.loads(response.text)


# In[ ]:


json_data

