# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:40:53 2019

@author: t0272m1
"""


#
# Imports (External and Internal)
#
import subprocess
import datetime
import itertools
import json
import logging
import math
import numpy as np
import pandas as pd
import requests
import re
import os
import sys
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

import calendrical
from calendrical import expand_dates
from calendrical import get_nth_kday_of_month
from database import connect_greenplum
from database import create_frame_from_pg
from database import write_frame_to_pg
# @ T8828FA Added for enryption
from cryptography.fernet import Fernet

#@t9939vs - Added for Compression Algorithm
import subprocess


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function streaks
#

def streaks(x):
    sign = np.sign(x)
    s = sign.groupby((sign != sign.shift()).cumsum()).cumsum()
    return (s.where(s > 0, 0.0))

def streak_1(x):
    return np.sum(x == 1)

def streak_2(x):
    return np.sum(x == 2)

def streak_3(x):
    return np.sum(x == 3)

def streak_4_plus(x):
    return np.sum(x >= 4)


#
# Function nearest
#

def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))


#
# Function sequence_frame
#

def sequence_frame(df, target, forecast_periods=1, leaders=[], lag_period=1):
    r"""Sequence a frame with lags

    Parameters
    ----------
    df : pandas.DataFrame
        The original dataframe.
    target : str
        The target variable for prediction.
    forecast_periods : int
        The periods for forecasting the target of the analysis.
    leaders : list
        The features that are contemporaneous with the target.
    lag_period : int
        The number of lagged rows for prediction.

    Returns
    -------
    new_frame : pandas.DataFrame
        The transformed dataframe with variable sequences.
    """

    # Set Leading and Lagging Columns
    le_cols = sorted(leaders)
    le_len = len(le_cols)
    df_cols = sorted(list(set(df.columns) - set(le_cols) - set([target])))
    df_len = len(df_cols)

    # Lag Features
    new_cols, new_names = list(), list()
    for i in range(lag_period, 0, -1):
        new_cols.append(df[df_cols].shift(i))
        new_names += ['%s_lag_%d' % (df_cols[j], i) for j in range(df_len)]

    # Lag Leaders
    new_cols.append(df[le_cols])
    new_names += [le_cols[j] for j in range(le_len)]

    # Forecast Target(s)
    for i in range(forecast_periods):
        fp = i + 1
        new_cols.append(pd.DataFrame(df[target].shift(1-fp)))
        new_names.append('_'.join([target, 'forecast', str(fp)]))

    # Collect all columns into new frame
    new_frame = pd.concat(new_cols, axis=1)
    new_frame.columns = new_names
    return new_frame


#
# Function get_crew_dates
#
    
def get_crew_dates(start_date, npreds, crew_days, shift_days):
    nweeks = math.ceil(npreds / shift_days) + 1
    crew_days = sorted(crew_days.copy())
    pred_days = []
    for i in range(nweeks):
        offsets = len(crew_days)*[i*7]
        this_week = [sum(x) for x in zip(crew_days, offsets)]
        pred_days.extend(this_week)
    day_index = start_date.weekday()
    pred_days = [x - day_index for x in pred_days if x > day_index]
    pred_days = pred_days[:npreds]
    # calculate dates
    crew_dates = [(start_date + datetime.timedelta(x)).strftime('%Y-%m-%d') for x in pred_days]
    return crew_dates


#
# Function get_actuals
#

def get_actuals(npreds, df_ts, target, target_dates):
    actuals = npreds * [np.nan]
    # index into target
    try:
        actuals = df_ts.loc[df_ts['workdate'].isin(target_dates)][target].tolist()
    except KeyError:
        logger.info("Could not get actuals for %s" % target_dates)
    # return actuals
    return actuals


#
# This section is for generating the exogenous variables for
# forward predictions. For example, we use an ARIMA model
# to forecast real-valued time series.
#

def set_quarter(npreds, df, col, offset):
    future_dates = df['workdate'][offset:]
    df[col][offset:] = [pd.to_datetime(x).quarter for x in future_dates]
    return df

def set_month(npreds, df, col, offset):
    future_dates = df['workdate'][offset:]
    df[col][offset:] = [pd.to_datetime(x).month for x in future_dates]
    return df

def set_week(npreds, df, col, offset):
    future_dates = df['workdate'][offset:]
    df[col][offset:] = [pd.to_datetime(x).week for x in future_dates]
    return df

def set_day(npreds, df, col, offset):
    future_dates = df['workdate'][offset:]
    df[col][offset:] = [pd.to_datetime(x).day for x in future_dates]
    return df

def set_day_of_week(npreds, df, col, offset):
    future_dates = df['workdate'][offset:]
    df[col][offset:] = [pd.to_datetime(x).dayofweek for x in future_dates]
    return df

def set_day_of_year(npreds, df, col, offset):
    future_dates = df['workdate'][offset:]
    df[col][offset:] = [pd.to_datetime(x).dayofyear for x in future_dates]
    return df

def predict_arima(npreds, df, col, offset, p, d, q):
    model = ARIMA(df[col][:offset], order=(p, d, q))
    try:
        model_fit = model.fit(solver='powell')
        predictions = model_fit.predict(offset, offset+npreds-1)
        df[col][offset:] = predictions.tolist()
    except:
        df[col][offset:] = [df[col][-1:].values[0]] * npreds
    return df


#
# Exogenous Function Dictionary
#

exog_funcs = {
    'actual_hours'                          : predict_arima,
    'lost_hours'                            : predict_arima,
    'paa_hours'                             : predict_arima,
    'absences_late'                         : predict_arima,
    'absences_noshow'                       : predict_arima,
    'absences_any'                          : predict_arima,
    'absences_fmla'                         : predict_arima,
    'mean_absence_pct'                      : predict_arima,
    'mean_experience'                       : predict_arima,
    'absences_planned'                      : predict_arima,
    'absences_unplanned_rolling_sum_5'      : predict_arima,
    'absences_unplanned_rolling_median_5'   : predict_arima,
    'absences_unplanned_rolling_sum_12'     : predict_arima,
    'absences_unplanned_rolling_median_12'  : predict_arima,
    'absences_unplanned_rolling_sum_20'     : predict_arima,
    'absences_unplanned_rolling_median_20'  : predict_arima,
    'actual_hours_rolling_mean_20'          : predict_arima,
    'actual_hours_rolling_median_20'        : predict_arima,
    'lost_hours_rolling_mean_20'            : predict_arima,
    'lost_hours_rolling_median_20'          : predict_arima,
    'quarter'                               : set_quarter,
    'month'                                 : set_month,
    'week'                                  : set_week,
    'day'                                   : set_day,
    'day_of_week'                           : set_day_of_week,
    'day_of_year'                           : set_day_of_year
    }


#
# Function generate_future_frame
#

def generate_future_frame(npreds, df_ts, future_dates, exog_cols, p, d, q):
    # extend data frame by npreds rows
    nrows = df_ts.shape[0]
    ncols = df_ts.shape[1]
    for i in range(npreds):
        df_ts.loc[nrows + i] = [0 for n in range(ncols)]
    # apply future crew dates to extension
    df_ts['workdate'][nrows:] = future_dates
    # call a feature function represented by each exogenous column
    for e in exog_cols:
        try:
            if exog_funcs[e] == predict_arima:
                df_ts = exog_funcs[e](npreds, df_ts, e, nrows, p, d, q)
            else:
                df_ts = exog_funcs[e](npreds, df_ts, e, nrows)
        except:
            logger.info("\nCould not find feature %s in the master dictionary" % e)
    # return augmented frame
    return df_ts


#
# Function get_holidays
#

def get_holidays(df):
    years = sorted(df['year'].unique().tolist())
    all_holidays = []
    for y in years:
        holidays = calendrical.set_holidays(y, True)
        htuples = list(holidays.items())
        all_holidays += htuples
    return all_holidays


#
# Function calendrical_features
#

def calendrical_features(row, holidays):
    gyear = row['year']
    gmonth = row['month']
    gday = row['day']
    rdate = calendrical.gdate_to_rdate(gyear, gmonth, gday)
    holiday_rdates = [x[1] for x in holidays]
    rdate_nearest = min(holiday_rdates, key=lambda x: abs(x - rdate))
    rdate_offset = rdate - rdate_nearest
    nearest_holiday = [item for item in holidays if item[1] == rdate_nearest][0][0]
    holiday_dates = []
    for h in holidays:
        if h[0] == nearest_holiday:
            rd = calendrical.rdate_to_gdate(h[1] + rdate_offset)
            date_str = datetime.datetime(rd[0], rd[1], rd[2]).strftime("%Y-%m-%d")
            holiday_dates.append(date_str)
    return pd.Series([nearest_holiday, rdate_offset, holiday_dates], index=['holiday', 'holiday_offset', 'holiday_dates'])


#
# Function holiday_mean
#

def holiday_mean(row, df, levels):
    holiday_dates = row['holiday_dates']
    crew = row['crew']
    if 'production_line' in levels:
        pline = row['production_line']
        holiday_mean = df['absences_unplanned'][(df['workdate'].isin(holiday_dates)) & (df['crew'] == crew) & (df['production_line'] == pline)].mean()
    else:
        holiday_mean = df['absences_unplanned'][(df['workdate'].isin(holiday_dates)) & (df['crew'] == crew)].mean()        
    return holiday_mean


#
# Function event_percent
#

def event_percent(row, df, levels):
    # Calculate event mean and percentages by genre
    genre = row['genre']
    crew = row['crew']
    group_mean = row['au_group_mean']
    # See if we have splits
    try:
        # Split for multiple values
        genres = genre.split(';')
        splits = True
    except:
        splits = False
    # Calculate event percentages
    if splits:
        event_pcts = []
        for g in genres:
            if 'production_line' in levels:
                pline = row['production_line']
                event_mean = df['absences_unplanned'][(df['genre'] == g) & (df['crew'] == crew) & (df['production_line'] == pline)].mean()
            else:
                event_mean = df['absences_unplanned'][(df['genre'] == g) & (df['crew'] == crew)].mean()        
            try:
                event_pct = round(100 * (event_mean / group_mean - 1), 0)
            except:
                event_pct = 0.0
            event_pct = 0.0 if math.isnan(event_pct) else event_pct
            event_pcts.append(str(event_pct))
        event_string = ';'.join(event_pcts)
    else:
        event_string = ''
    # Join percentages for final event feature
    return event_string


#
# Function create_event_Feature
#

def create_event_feature(row):
    # See if we have splits
    try:
        # Split multiples
        events = row['event_name'].split(';')
        genres = row['genre'].split(';')
        event_pcts = row['au_event_pct'].split(';')
        splits = True
    except:
        splits = False
    # Construct features
    if splits:
        features = []
        for e, g, p in zip(events, genres, event_pcts):
            if e:
                event_feature = e + ' [' + g + ' ' + str(p) + '%]'
            else:
                event_feature = ''
            features.append(event_feature)
        feature_string = ';'.join(features)
    else:
        feature_string = ''
    # Join features
    return feature_string



#
# Function update_events
#

def update_events(pipeline_specs, plant_specs):

    logger.info("Updating Events Table")
    # TicketMaster maximum
    nevents = 200
    # Limit event name specified characters
    limit_event_name_characters = 30
    # set proxy
    # @T8828FA  START- Added below code to implement encryption and  credentials from pipeline configuration file
    tm_encrypt_key=pipeline_specs['externalAPI']['ticket_master']['encrypt_key']
    f = Fernet(tm_encrypt_key)
    tm_userID=pipeline_specs['externalAPI']['ticket_master']['userID']
    tm_password_encrypted=str.encode(pipeline_specs['externalAPI']['ticket_master']['password'])
    tm_password_decrypted=f.decrypt(tm_password_encrypted )
    tm_password=tm_password_decrypted.decode("utf-8")
    https_proxy = 'https://'+tm_userID+':'+tm_password+'@iproxy.appl.chrysler.com:9090'
    tm_URL=pipeline_specs['externalAPI']['ticket_master']['URL']
    tm_API_key=pipeline_specs['externalAPI']['ticket_master']['API_key']
    url_string=tm_URL+tm_API_key
    logger.info("URL-TicketmasterAPI %s" % url_string)
    # @T8828FA  END
    proxyDict = {"https" : https_proxy}
    # read in event table
    conn_dl, curs_dl = connect_greenplum(pipeline_specs['datalake'])
    schema = pipeline_specs['datalake']['schema']
    table_tag = 'event_calendar'
    table_name = '_'.join(['abs', table_tag, 'tbl'])
    query = "select * from \"" + schema + "\".\"" + table_name + "\""
    df_event = create_frame_from_pg(conn_dl, table_name, query)
    df_event['event_name'].fillna('', inplace=True)
    df_event['event_time'].fillna('', inplace=True)
    df_event['genre'].fillna('', inplace=True)
    # @t9939vs -Regular expression and limit to avoid special chracters in event name
    df_event['event_name'] = df_event[['event_name']].apply(lambda x : re.sub('[^A-Za-z0-9]+', ' ', x[0][:limit_event_name_characters]) , axis =1)
    # close connection
    logger.info("Closing connection")
    conn_dl.commit()
    curs_dl.close()
    # get latest event snapshot
    marketid = plant_specs['plant']['market_id']
    url_string += "&marketId={}".format(marketid)
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
        
        # @t9939vs -Regular expression and limit to avoid special chracters in event name
        df_event_new['event_name'] = df_event_new[['event_name']].apply(lambda x : re.sub('[^A-Za-z0-9]+', ' ', x[0][:limit_event_name_characters]) , axis =1)
        
        # concatenate new events with existing table
        df_event = pd.concat([df_event, df_event_new])
        # sort concatenated frame and drop duplicates
        df_event.sort_values(by=['market_id', 'event_date', 'event_time'], inplace=True)
        df_event.drop_duplicates(keep='last', inplace=True)
        # write updated table
        write_frame_to_pg(df_event, table_tag, pipeline_specs)
    else:
        print("Status Code from Events API: %d" % response.status_code)
    return df_event


#
# Function create_calendrical_stats
#

def create_calendrical_stats(dfm, pipeline_specs, plant_specs, group_cols):
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
    dfm5['au_holiday_mean'] = dfm5.apply(holiday_mean, df=dfm5, levels=group_cols, axis=1)
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
    dfm5['feature5'] = abs(dfm5['holiday_offset']).astype(str) + ' days ' + dfm5['holiday_plus_minus'] + ' ' \
                       + dfm5['holiday'] + ': ' + dfm5['au_holiday_pct'].astype(str) + '%'
    # Read in event table
    df_event = update_events(pipeline_specs, plant_specs)
    # Filter out events only for this market
    market_id = plant_specs['plant']['market_id']
    df_event2 = df_event[df_event['market_id'] == market_id]
    df_event2.drop(columns=['market_id'], inplace=True)
    # Consolidate events on the same day so we can join tables
    lf_join = lambda x : ';'.join(x)
    agg_dict = {'event_name' : lf_join,
                'event_time' : lf_join,
                'genre'      : lf_join}
    df_event3 = df_event2.groupby(['event_date']).agg(agg_dict).reset_index()
    # Merge with the event table with multiple events per day
    dfm5 = pd.merge(dfm5, df_event3, left_on=['workdate'], right_on=['event_date'], how='left')
    # Calculate the event percentages for each group
    dfm5['au_event_pct'] = dfm5.apply(event_percent, df=dfm5, levels=group_cols, axis=1)
    # Create the event feature
    dfm5['feature_event'] = dfm5.apply(create_event_feature, axis=1)
    # Drop extraneous columns
    drop_cols = ['workdate_dt',
                 'holiday_plus_minus']
    dfm5.drop(columns=drop_cols, inplace=True)
    return dfm5


#
# Function merge_peak_table
#

def merge_peak_table(pipeline_specs, df_model, peak_table, group_levels):
    logger.info("Merging Peak Values from %s" % peak_table)

    # Extract values from specifications
    schema = pipeline_specs['datalake']['schema']
    plant_id = pipeline_specs['plant_id']

    # Get Peak Values

    try:
        # Create dataframe from Peak Table
        conn_dl, curs_dl = connect_greenplum(pipeline_specs['datalake'])
        query = "select * from \"" + schema + "\".\"" + peak_table + "\""
        df_peak = create_frame_from_pg(conn_dl, peak_table, query)
        # Replace relevant predictions with peak values
        df_peak = df_peak[df_peak['plant'] == plant_id]
        join_cols = ['workdate'] + group_levels
        df_model = pd.merge(df_model, df_peak,
                            left_on=join_cols,
                            right_on=join_cols,
                            how='left')
        df_model['absences_unplanned'] = df_model['absences_unplanned_y'].fillna(df_model['absences_unplanned_x'])
        # Drop extraneous merged columns
        drop_cols = ['absences_unplanned_x', 'absences_unplanned_y', 'plant', 'extrema', 'imp_extrema']
        df_model.drop(columns=drop_cols, inplace=True)
    except:
        logger.info("Could not access Peak Table: %s" % peak_table)

    return df_model


#
# Function create_model_table
#

def create_model_table(df, pipeline_specs, plant_specs, levels=None):
    logger.info("Building Model Table")

    # Get variables from specifications
    unplanned_codes = plant_specs['plant']['absence_codes']
    plant_shift_hours = plant_specs['plant']['shift_hours']
    exclude_dates = plant_specs['plant']['exclude_dates']
    group_levels = plant_specs['model']['levels']
    target = plant_specs['model']['target']
    use_peaks = plant_specs['model']['use_peaks']
    peak_table = plant_specs['model']['peak_table']

    # Override specs to build a model table with different levels
    if levels:
        group_levels = levels

    # Model Frame Start
    df_model = df.copy(deep=True)

    # Use only work days
    logger.info("Retaining work days only")
    df_model = df_model[df_model['is_work_day'] == 'Y']
    df_model.drop(columns=['is_work_day'], inplace=True)

    # Filter out plant shutdown days
    expanded_dates = expand_dates(exclude_dates)
    df_model = df_model[df_model['workdate'].isin(expanded_dates) == False]

    # Define CID-Level Features
    logger.info("Defining CID-Level Features")

    # Lost Hours Percentage
    lost_hours_period = 20
    df_model['actual_hours_sum'] = df_model.groupby('cid')['ch_actl_hrs'].transform(lambda x: x.rolling(lost_hours_period).sum().fillna(0))
    df_model['lost_hours_sum'] = df_model.groupby('cid')['lost_hrs'].transform(lambda x: x.rolling(lost_hours_period).sum().fillna(0))
    df_model['lost_hours_pct'] = round(100 * df_model['lost_hours_sum'] / (df_model['actual_hours_sum'] + df_model['lost_hours_sum']))
    df_model['lost_hours_pct'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df_model['lost_hours_pct'].fillna(0.0, inplace=True)
    df_model['lost_hours_pct'].loc[df_model['lost_hours_pct'] < 0.0] = 0.0
    df_model['lost_hours_pct'].loc[df_model['lost_hours_pct'] > 100.0] = 100.0
    df_model.drop(columns=['actual_hours_sum', 'lost_hours_sum'], inplace=True)
    # Team Experience
    te_cols = ['cid'] + group_levels
    df_model['days_on_team'] = df_model.groupby(te_cols).cumcount()
    # Bin Age
    bins = [0, 21, 30, 40, 50, 60, 100]
    labels = ['0-21', '22-30', '31-40', '41-50', '51-60', '>60']
    df_model['age'] = pd.cut(df_model['age'], bins=bins, labels=labels)
    df_model['age'] = df_model['age'].astype('category')
    # Bin Tenure
    bins = [0, 2, 5, 10, 15, 20, 100]
    labels = ['0-2', '3-5', '6-10', '11-15', '16-20', '>20']
    df_model['tenure'] = pd.cut(df_model['tenure'], bins=bins, labels=labels)
    df_model['tenure'] = df_model['tenure'].astype('category')
    # Bin Hourly Pay
    bins = [0, 15, 20, 25, 30, 35, 100]
    labels = ['0-15', '16-20', '21-25', '26-30', '31-35', '>35']
    df_model['hrlypayrate'] = pd.cut(df_model['hrlypayrate'], bins=bins, labels=labels)
    df_model['hrlypayrate'] = df_model['hrlypayrate'].astype('category')
    # Unplanned Absences
    df_model['abs_code'].fillna('', inplace=True)
    df_model['abs_code'] = df_model['abs_code'].str.strip()
    unplanned_codes_str = '|'.join(unplanned_codes)
    df_model['absence_unplanned'] = df_model['abs_code'].isin(unplanned_codes).astype(int)
    df_model['absence_hours'] = 0.0
    df_model.loc[df_model['abs_code'].isin(unplanned_codes), 'absence_hours'] = df_model['lost_hrs']
    tardy_condition = (df_model['c_typ'] == 'T')
    df_model['absence_late'] = ((df_model['abs_code'].isin(unplanned_codes)) & (tardy_condition)).astype(int)
    df_model['absence_noshow'] = ((df_model['absence_unplanned'] > 0) & (df_model['lost_hrs'] >= plant_shift_hours)).astype(int)
    df_model['absence_any'] = (df_model['abs_code'].str.len() == 4).astype(int)
    # Streaks
    df_model['streak'] = df_model.groupby('cid')['absence_unplanned'].apply(streaks)
    df_model['cluster'] = df_model['streak'] >= 2
    df_model['cluster'] = df_model['cluster'].astype(int)

    # Lambda Functions

    # Total Working CIDs
    lf_cid = lambda g: g[df_model.loc[g.index]['ch_actl_hrs'] > 0.0].count()
    lf_cid.__name__ = 'cid_lfunc'

    # Lost Hours Function
    lf_lost = lambda g: g[(df_model.loc[g.index]['abs_code'].str.contains(unplanned_codes_str, na=False))].sum()
    lf_lost.__name__ = 'lh_lfunc'

    # Planned Absences
    planned_codes_str = 'VACF|POOL|PAA'
    lf_tap = lambda val: (val.str.contains(planned_codes_str, na=False)).sum()
    lf_tap.__name__ = 'tap_lfunc'

    # Total Home Canvasses
    home_codes_str = 'HOMC|HOMR'
    lf_thc = lambda val: (val.str.contains(home_codes_str, na=False)).sum()
    lf_thc.__name__ = 'thc_lfunc'

    # Total FMLA Absences
    fmla_codes_str = 'FMLA|FMLU'
    lf_fmla = lambda val: (val.str.contains(fmla_codes_str, na=False)).sum()
    lf_fmla.__name__ = 'fmla_lfunc'

    # PEIA Code
    lf_peia = lambda g: g[(df_model.loc[g.index]['abs_code'].str.contains('PEIA', na=False))].count()
    lf_peia.__name__ = 'peia_lfunc'

    # Total TPTs
    lf_tpt = lambda g: g[(df_model.loc[g.index]['ch_actl_hrs'] > 0.0) & (df_model.loc[g.index]['ch_flexwrk_code'] == '2')].count()
    lf_tpt.__name__ = 'tpt_lfunc'

    # AGGREGATION

    # Identify the features for aggregation
    group_cols = ['workdate'] + group_levels

    # aggregation dictionary
    agg_dict = {'cid'               : lf_cid,
                'lost_hrs'          : lf_lost,
                'ch_actl_hrs'       : 'sum',
                'ch_paa_hrs'        : 'sum',
                'absence_unplanned' : 'sum',
                'absence_late'      : 'sum',
                'absence_noshow'    : 'sum',
                'absence_any'       : 'sum',
                'lost_hours_pct'    : 'mean',
                'days_on_team'      : 'mean',
                'abs_code'          : [lf_tap,
                                       lf_thc,
                                       lf_fmla,
                                       lf_peia],
                'ch_flexwrk_code'   : lf_tpt,
                'streak'            : [streak_1,
                                       streak_2,
                                       streak_3,
                                       streak_4_plus],
                'cluster'           : 'sum'
               }

    # Create aggregation frame
    logger.info("Aggregating Frame")
    df_agg = df_model.groupby(group_cols).agg(agg_dict).reset_index()

    # Set column names
    agg_cols0 = [x[0] for x in list(df_agg.columns.values)]
    agg_cols1 = [x[1] for x in list(df_agg.columns.values)]

    # These are the aggregated column names that must match each entry in
    # the aggregation dictionary.

    agg_col_names = ['group_total_cid',
                     'lost_hours',
                     'actual_hours',
                     'paa_hours',
                     'absences_unplanned',
                     'absences_late',
                     'absences_noshow',
                     'absences_any',
                     'mean_absence_pct',
                     'mean_experience',
                     'absences_planned',
                     'home_canvasses',
                     'absences_fmla',
                     'peia_count',
                     'tpt_count',
                     'streak_1',
                     'streak_2',
                     'streak_3',
                     'streak_4_plus',
                     'cluster']

    # The slice changes based on the number of aggregation columns.
    agg_slice = slice(len(group_cols), len(agg_cols1)+1)
    agg_cols = agg_cols0
    agg_cols[agg_slice] = agg_col_names
    df_agg.columns = agg_cols

    # Eliminate any days when there was a possible plant shutdown
    df_agg = df_agg[df_agg['actual_hours'] > 0].reset_index(drop=True)

    # Group and TPT Calculations
    df_agg['group_total'] = df_agg['group_total_cid'] - df_agg['home_canvasses']
    df_agg['tpt_unplanned'] = df_agg['tpt_count'] - df_agg['absences_planned']
    df_agg['tpt_extra'] = df_agg['tpt_unplanned'] - df_agg['absences_unplanned']    
    
    # Rolling Features
    
    logger.info("Creating Rolling Features")
    roll_periods = [5, 12, 20]
    for rp in roll_periods:
        fname = '_'.join([target, 'rolling_sum', str(rp)])
        df_agg[fname] = df_agg.groupby(group_levels)[target].transform(lambda x: x.rolling(rp).sum().fillna(0))
        fname = '_'.join([target, 'rolling_median', str(rp)])
        df_agg[fname] = df_agg.groupby(group_levels)[target].transform(lambda x: x.rolling(rp).median().fillna(0))

    roll_period = 20
    df_agg['actual_hours_rolling_mean_20'] = df_agg.groupby(group_levels)['actual_hours'].transform(lambda x: x.rolling(roll_period).mean().fillna(0))
    df_agg['actual_hours_rolling_median_20'] = df_agg.groupby(group_levels)['actual_hours'].transform(lambda x: x.rolling(roll_period).median().fillna(0))
    df_agg['lost_hours_rolling_mean_20'] = df_agg.groupby(group_levels)['lost_hours'].transform(lambda x: x.rolling(roll_period).mean().fillna(0))
    df_agg['lost_hours_rolling_median_20'] = df_agg.groupby(group_levels)['lost_hours'].transform(lambda x: x.rolling(roll_period).median().fillna(0))

    # Kim-Powell Residual Features
    epsilon = 0.5
    cap_high = 1000.0
    cap_low = -cap_high
    target1 = '_'.join([target, '1'])
    df_agg[target1] = df_agg.groupby(group_levels)[target].transform(lambda x: x.shift().fillna(0))
    for rp in roll_periods:
        fname = '_'.join(['kp_residual', str(rp)])
        fmedian = '_'.join([target, 'rolling_median', str(rp)])
        df_agg[fname] = (df_agg[target] - df_agg[target1]) / (df_agg[fmedian] - df_agg[target1] + epsilon)
        df_agg[fname] = df_agg[fname].replace(np.inf, cap_high)
        df_agg[fname] = df_agg[fname].replace(-np.inf, cap_low)
    df_agg.drop(columns=[target1], inplace=True)

    # Calendrical Features
    logger.info("Creating Calendrical Features")
    df_agg['workdate_dt'] = pd.to_datetime(df_agg['workdate'])
    df_agg['year'] = df_agg['workdate_dt'].dt.year
    df_agg['quarter'] = df_agg['workdate_dt'].dt.quarter
    df_agg['month'] = df_agg['workdate_dt'].dt.month
    df_agg['week'] = df_agg['workdate_dt'].dt.week
    df_agg['day'] = df_agg['workdate_dt'].dt.day
    df_agg['day_of_week'] = df_agg['workdate_dt'].dt.dayofweek
    df_agg['day_of_year'] = df_agg['workdate_dt'].dt.dayofyear
    df_agg['nth_kday'] = df_agg[['day', 'month', 'year']].apply(lambda x: get_nth_kday_of_month(*x), axis=1)
    df_agg['diff'] = df_agg.groupby(group_levels)['workdate_dt'].diff().fillna(0)
    df_agg['next_day_delta'] = df_agg.groupby(group_levels)['diff'].shift(-1).dt.days.fillna(0)
    df_agg.drop(columns=['workdate_dt', 'diff'], inplace=True)

    # Test for Peak SARIMAX for production lines only
    if use_peaks and 'production_line' in group_levels:
        df_agg = merge_peak_table(pipeline_specs, df_agg, peak_table, group_levels)

    # Sequenced Frame Start
    df_seq = df_agg.copy(deep=True)

    # Lag the leading features for prediction
    logger.info("Lagging Features")
    shift_cols = ['group_total_cid',
                  'lost_hours',
                  'actual_hours',
                  'paa_hours',
                  'absences_late',
                  'absences_noshow',
                  'absences_any',
                  'mean_absence_pct',
                  'mean_experience',
                  'absences_planned',
                  'home_canvasses',
                  'absences_fmla',
                  'peia_count',
                  'tpt_count',
                  'streak_1',
                  'streak_2',
                  'streak_3',
                  'streak_4_plus',
                  'cluster',
                  'absences_unplanned_rolling_sum_5',
                  'absences_unplanned_rolling_median_5',
                  'absences_unplanned_rolling_sum_12',
                  'absences_unplanned_rolling_median_12',
                  'absences_unplanned_rolling_sum_20',
                  'absences_unplanned_rolling_median_20',
                  'actual_hours_rolling_mean_20',
                  'actual_hours_rolling_median_20',
                  'lost_hours_rolling_mean_20',
                  'lost_hours_rolling_median_20',
                  'group_total',
                  'tpt_unplanned',
                  'tpt_extra',
                  'kp_residual_5',
                  'kp_residual_12',
                  'kp_residual_20']
    # The grouping is very important here!
    df_seq[shift_cols] = df_seq.groupby(group_levels)[shift_cols].transform('shift')

    logger.info("Model Table: %s\n%s\n" % (df_seq.shape, df_seq.tail(100)))
    return df_agg, df_seq






#
# Function make_sarimax_predictions
#

def make_sarimax_predictions(npreds, pred_date, df_ts, target, ntop, future_dates, exog_cols, p, d, q):
    # Initialize predictions
    predictions = []
    ar_pattern = "^ar.L"
    pred_date_str = pred_date.strftime('%Y-%m-%d')
    # Calculate offset into time series
    cutoff_date = df_ts['workdate'].iloc[-npreds]
    logger.info("Data Cutoff Date: %s, Prediction Date: %s" % (cutoff_date, pred_date_str))
    if cutoff_date >= pred_date_str:
        logger.info("Historical Prediction")
        start_period = np.where(df_ts['workdate'] >= pred_date_str)[0].tolist()[0]
    else:
        logger.info("Future Prediction")
        # construct exogenous frame of length npreds for future dates
        df_ts = generate_future_frame(npreds, df_ts, future_dates, exog_cols, p, d, q)
        start_period = len(df_ts[target].values) - npreds
    end_period = start_period + npreds - 1
    try:
        # Run model
        model = SARIMAX(df_ts[target][:start_period], df_ts[exog_cols][:start_period],
                        order=(p, d, q), simple_differencing=True)
        model_fit = model.fit(method='powell')
        # get predictions
        predictions = model_fit.predict(start_period, end_period, exog=df_ts[exog_cols][start_period:end_period+1])
        # get feature importances
        features_html = model_fit.summary().tables[1].as_html()
        df_feat = pd.read_html(features_html)[0].iloc[1:, :]
        df_feat.columns = ['feature', 'coef', 'std err', 'Z', 'P>|z|', 'ci_low', 'ci_high']
        df_feat = df_feat[df_feat.feature != 'sigma2']
        ar_filter = df_feat['feature'].str.contains(ar_pattern)
        df_feat = df_feat[~ar_filter]
        df_feat['Z'] = df_feat['Z'].astype('float')
        df_feat['Z_abs'] = abs(df_feat['Z'])
        df_feat.sort_values(by=['Z_abs'], ascending=False, inplace=True)
        features_ntop = df_feat['feature'].head(ntop).tolist()
    except:
        logger.info("Could not fit model")
        predictions = np.array([0] * npreds)
        features_ntop = npreds * [['None'] * ntop]
    # return predictions
    return predictions, features_ntop


#
# Function create_prediction_frame
#

def create_prediction_frame(predictions, group_levels, ntop):
    logger.info("Creating Prediction Frame")

    # Create new prediction data frame
    pcols = group_levels + ['workdate', 'predicted', 'top_features']
    dfp_new = pd.DataFrame(predictions, columns=pcols)
    
    # Expand top feature column
    fvalues = dfp_new['top_features'].values.tolist()
    fnames = ['model_feature'+str(i+1) for i in range(ntop)]
    dff = pd.DataFrame(fvalues, columns=fnames)
    
    # Rejoin new prediction frame with features
    dfp_new.drop(columns=['top_features'], inplace=True)
    dfp_new = pd.concat([dfp_new, dff], axis=1)

    logger.info("New Predictions Shape: %s" % (dfp_new.shape,))    
    return dfp_new



#
#  Compression_algorithm from R script at Crewlevel @ T9939VS
#

def  compression_crew_level(pipeline_specs, plant_specs ):
    """
    args    : Arguments for running R script at command line mode
            
    Example : args : rexe_path path_R_script "20200112" "jnap" "2018-01-01"
        
    """
    logger.info("Developing Command for Rscript")
    pred_date = plant_specs['prediction_date'].replace("-",'')
    plant_code = pipeline_specs['plant_id']
    start_date = plant_specs['base_table']['start_date']
    flag = pipeline_specs['test_flag']
    curr_dir = os.getcwd()
    file_path = curr_dir+"\\"+pipeline_specs['compression_r_file']
    #Command for Rscript CMD execution
    if flag:
        flag = "test"
        commands = [pipeline_specs['r_exepath'],file_path]+[pred_date,plant_code , str(start_date), flag]
    else:
        flag = "PROD"
        commands = [pipeline_specs['r_exepath'],file_path]+[pred_date,plant_code , str(start_date),flag ]
    log_command = " ".join(commands)
    logger.info("Commands for R Script : %s" % log_command)
    try :
        logger.info("Calling Rscript from python")
        values = str(subprocess.check_call(commands))
        logger.info("Sucessfully Complted Rscript with response :%s" % values)
        return values
    except:
        
        return 9999

#
# Function to make_compression_predictions @ T9939VS
#
def make_compression_predictions(pipeline_specs,plant_specs, df_model_seq , crew_dates):

    
    # Calling Compression Algorithm in R Script
    crew_compress_predictions = []
    values = compression_crew_level(pipeline_specs, plant_specs )
    logger.info("Compression algorithm returned response : %s" % values)
    group_levels = plant_specs['model']['levels']
    npreds = plant_specs['model']['npreds']
    plant_id = pipeline_specs['plant_id']
    pred_date = pd.to_datetime(plant_specs['prediction_date'])
    target = plant_specs['model']['target']
    flag = pipeline_specs['test_flag']
    project_directory  = pipeline_specs['project_directory']
    # Developing file name to store compression predictions
    pred_date_str = plant_specs['prediction_date'].replace('-','')
    if flag:
        flag = "test"
        table_name = "_".join(["lab_datasci.abs",flag,plant_id,"compression_crew",pred_date_str,"tbl"])
    else:
        table_name = "_".join(["lab_datasci.abs",plant_id,"compression_crew",pred_date_str,"tbl"])
    gcols = ['crew']
    pred_table = str.lower('_'.join(['abs', plant_id, 'crew_compression_prediction_tbl']))
    file_name = '.'.join([pred_table,'csv'])
    logger.info("Created filename : %s" % file_name)
    pred_file = '\\'.join([project_directory, file_name])
    
    
    if int(values) == 0:
        
        logger.info("Connecting Database")
        conn_dl, curs_dl = connect_greenplum(pipeline_specs['datalake'])
        # Query to read compression data
        query = "select * from " + table_name 
        logger.info("Reading Compression Prediction from : %s" % table_name)
        df = create_frame_from_pg(conn_dl, table_name, query)
        agg_dict_compress = {'actual' : 'mean'}
        # close connection
        logger.info("Closing connection")
        conn_dl.commit()
        curs_dl.close()
        logger.info("Aggregating Compression Prediction")
        df_grp_crew = df[df.imp_extrema == 'imp_max'].groupby(gcols, as_index = False).agg(agg_dict_compress)
        df_grp_crew = df_grp_crew.rename(columns = {'actual': 'predicted_compression_crew_level'})   
        df_grp_crew['predicted_compression_crew_level'] = df_grp_crew['predicted_compression_crew_level'].apply(math.ceil)
        # Write out the new crew level compression prediction table
        logger.info("Writing Aggregated Compression Prediction to : %s" % pred_file)
        df_grp_crew.to_csv(pred_file, index=False)
    else:
        logger.info("Reading Aggregated Compression Prediction from Previous Run : %s" % pred_file)
        df_grp_crew = pd.read_csv(pred_file)
        message = "Compression Prediction are from previous run and need investigation"
        pipeline_specs['error_logs'].append(message)
    
    #Convert Prediction to list
    prediction_combinations = df_grp_crew.values.tolist()
    logger.info(prediction_combinations)
    ftop = 'compression'
    for lst in prediction_combinations:
        crew = lst[0]
        predictions = lst[1]
        crew_compress_predictions.extend(zip([crew]*npreds, crew_dates[crew], [predictions]*npreds, [ftop]*npreds))
    logger.info("Creating Compression Future Frame ")
    df_compress_pred = create_prediction_frame(crew_compress_predictions,['crew'], 1 )
    #Run Share Model on Compression Crew Level Prediction
    logger.info("Developing Prediction from Share Model for Compression")
    df_compress_pred_share = make_share_predictions(npreds,
                                   pred_date,
                                   df_model_seq,
                                   target,
                                   df_compress_pred,
                                   group_levels)
    # Rename predicted Columns
    df_compress_pred_share = df_compress_pred_share.rename(columns = {'predicted': 'predicted_compression'})
    df_compress_pred_share['predicted_compression'] = df_compress_pred_share['predicted_compression'].apply(math.ceil)
    # Selection required columns for compression share model 
    cols = ['workdate']+group_levels + ['predicted_compression']
    df_compress_pred_share = df_compress_pred_share.loc[:, cols]
    df_compress_pred_share = pd.merge(df_compress_pred_share,df_grp_crew, how = 'left', left_on = gcols, right_on = gcols)
    logger.info("Completed Compression Predictions")
    return df_compress_pred_share 





#
# Function make_share_predictions
#

def make_share_predictions(npreds, pred_date, df_model_seq, target, df_pred_crew, levels):
    # Calculate shares
    logger.info("Share Model")
    au_target = 'absences_unplanned_rolling_sum_12'
    window_date = (pred_date - datetime.timedelta(90)).strftime('%Y-%m-%d')
    df_au = df_model_seq[df_model_seq['workdate'] >= window_date].groupby(levels).agg({au_target : 'mean'}).reset_index()
    df_pcts = df_au.groupby('crew')[au_target].apply(lambda x: x / float(x.sum()))
    df_au['pct'] = df_pcts
    # Expand crew predictions for production lines
    plines = df_au['production_line'].unique().tolist()
    plines_set = set(df_au[['crew','production_line']].itertuples(index=False, name = None))
    df_pred_crew_new = pd.DataFrame(np.repeat(df_pred_crew.values, len(plines), axis=0))
    df_pred_crew_new.columns = df_pred_crew.columns
    df_pred_crew_new['production_line'] = plines * len(df_pred_crew)
    df_pred_crew_new = df_pred_crew_new[df_pred_crew_new[['crew', 'production_line']].apply(lambda x : True if tuple(x) in plines_set else False , axis=1)]
    # Merge crew predictions with percentages
    df_pred_share = pd.merge(df_pred_crew_new,
                             df_au,
                             left_on=levels,
                             right_on=levels,
                             how='left')
    df_pred_share.rename(columns={'predicted' : 'predicted_crew'}, inplace=True)
    df_pred_share['predicted'] = df_pred_share['predicted_crew'] * df_pred_share['pct']
    df_pred_share.drop(columns=[au_target, 'pct', 'predicted_crew'], inplace=True)
    logger.info("Developed Share Predictions")
    # return predictions
    return df_pred_share


#
# Function create_ensemble
#

def create_ensemble(model_predictions):
    logger.info("Ensemble Model")
    n_models = len(model_predictions)
    if n_models > 1:
        logger.info("Ensemble Model (%d models)" % n_models)
        # average ensemble_predictions to derive final predictions
        df_pred = pd.concat(model_predictions, axis=1)
        df_pred['mean'] = df_pred['predicted'].mean(axis=1)
        df_pred = df_pred.loc[:, ~df_pred.columns.duplicated()]
        df_pred.drop(columns=['predicted'], inplace=True)
        df_pred.rename(columns={'mean' : 'predicted'}, inplace=True)
    else:
        # assign single model predictions to final predictions
        logger.info("Single Model")
        df_pred = model_predictions[0]
    # Round and zero final predictions
    df_pred['predicted'] = df_pred['predicted'].apply(math.ceil)
    df_pred.loc[df_pred['predicted'] < 0, 'predicted'] = 0
    # Return final predictions
    return df_pred







#
# Function make_plant_predictions
#

def make_plant_predictions(pipeline_specs, plant_specs, crew_maps, df_model_seq, df_base):
    logger.info("Plant Predictions")

    # Prediction Specifications
    shift_days = plant_specs['plant']['shift_days']
    models = plant_specs['model']['models']
    npreds = plant_specs['model']['npreds']
    exog_cols = plant_specs['model']['features']
    ntop = plant_specs['model']['top_features']
    pred_date = pd.to_datetime(plant_specs['prediction_date'])
    target = plant_specs['model']['target']
    group_levels = plant_specs['model']['levels']
    production_lines = plant_specs['model']['production_lines']
    crews = plant_specs['model']['crews']
    p = plant_specs['model']['p_arima']
    d = plant_specs['model']['d_arima']
    q = plant_specs['model']['q_arima']
    minimum_rows = 20

    # Log crews and production lines
    logger.info("Crews: %s" % crews)
    plines = list(production_lines.keys())
    logger.info("Production Lines: %s" % plines)

    # Get crew dates
    crew_dates = {}
    for cm in crew_maps:
        crew = cm[0]
        crew_days = cm[1]
        crew_dates[crew] = get_crew_dates(pred_date, npreds, crew_days, shift_days)
    logger.info("\nCrew Dates: \n%s" % crew_dates)

    # Ensemble Predictions
    ensemble_predictions = []
    
    # Create a model table at the crew level
    logger.info("Crew Level Model Table")
    _, df_model_crew_seq = create_model_table(df_base, pipeline_specs, plant_specs, ['crew'])

    # Iterate over crews
    logger.info("Crew Level Predictions")
    crew_predictions = []
    for crew in crews:
        logger.info("Crew %s" % crew)
        # Subset table
        df_sub = df_model_crew_seq.copy(deep=True)
        df_sub = df_sub[df_sub['crew']==crew]
        df_sub = df_sub.iloc[1:].reset_index(drop=True)
        nrows = df_sub.shape[0]
        logger.info("Crew Rows: %d" % nrows)
        # Call SARIMAX
        predictions, ftop = make_sarimax_predictions(npreds,
                                                     pred_date,
                                                     df_sub,
                                                     target,
                                                     ntop,
                                                     crew_dates[crew],
                                                     exog_cols,
                                                     p, d, q)
        # Return Final Predictions
        logger.info("Predictions : %s" % predictions)
        actuals = get_actuals(npreds, df_sub, target, crew_dates[crew])
        logger.info("Actuals     : %s" % actuals)
        crew_predictions.extend(zip([crew]*npreds, crew_dates[crew], predictions, [ftop]*npreds))

    # Store crew predictions in dataframe
    df_pred_crew = create_prediction_frame(crew_predictions, ['crew'], ntop)
    df_pred_crew['predicted'] = df_pred_crew['predicted'].apply(math.ceil)
    
    #
    #Compression Model
    #
    logger.info("Compression Predictions")
    df_pred_compress_share = make_compression_predictions(pipeline_specs,plant_specs, df_model_seq , crew_dates)

    # SHARE Model
    if 'share' in models:
        logger.info("SHARE Model")
        df_pred_share = make_share_predictions(npreds,
                                               pred_date,
                                               df_model_seq,
                                               target,
                                               df_pred_crew,
                                               group_levels)
        # Add SHARE predictions to ensemble
        ensemble_predictions.append(df_pred_share)

    # Production Line Level Predictions
    logger.info("Production Line Level Predictions")

    # SARIMAX Model
    if 'sarimax' in models:
        logger.info("SARIMAX Model")

        # Initialize Predictions and Target
        sarimax_predictions = []

        # Loop through each combination
        for crew, pline in itertools.product(crews, plines):
            logger.info("Crew %s, Production Line: %s " % (crew, pline))
            df_sub = df_model_seq.copy(deep=True)
            df_sub = df_sub[(df_sub['crew']==crew) & (df_sub['production_line']==pline)]
            df_sub = df_sub.iloc[1:].reset_index(drop=True)
            nrows = df_sub.shape[0]
            logger.info("Rows: %d" % nrows)
            if nrows >= minimum_rows:
                predictions, ftop = make_sarimax_predictions(npreds,
                                                             pred_date,
                                                             df_sub,
                                                             target,
                                                             ntop,
                                                             crew_dates[crew],
                                                             exog_cols,
                                                             p, d, q)
                # Return Final Predictions
                logger.info("Predictions : %s" % predictions)
                actuals = get_actuals(npreds, df_sub, target, crew_dates[crew])
                logger.info("Actuals     : %s" % actuals)
                sarimax_predictions.extend(zip([crew]*npreds, [pline]*npreds, crew_dates[crew], predictions, [ftop]*npreds))

        # Store SARIMAX predictions in dataframe
        df_pred_sarimax = create_prediction_frame(sarimax_predictions, group_levels, ntop)

        # Add SARIMAX predictions to ensemble
        ensemble_predictions.append(df_pred_sarimax)

    # Ensemble
    df_pred_all = create_ensemble(ensemble_predictions)

    # Filter out any production lines that are not in the specifications
    df_pred_all = df_pred_all[df_pred_all['production_line'].isin(plines)]
        
    # Return tables and predictions at both levels
    return df_model_seq, df_pred_all, df_model_crew_seq, df_pred_crew, df_pred_compress_share


#
# Function store_predictions
#

def store_predictions(plant_id, pipeline_specs, plant_specs, dfp_new, df_model, levels):
    logger.info("Storing %s Predictions" % plant_id)

    # Get specifications
    project_directory = pipeline_specs['project_directory']
    chrysler_holiday_table = pipeline_specs['holidays']['calendar_us']
    exclude_dates = plant_specs['plant']['exclude_dates']
    target = plant_specs['model']['target']
    band_pct = plant_specs['model']['band_pct']
    departments = plant_specs['model']['departments']
    pline_map = plant_specs['model']['production_lines']
    pred_date_str = plant_specs['prediction_date'].replace('-', '')
    prediction_date = plant_specs['prediction_date']
    inactive_prod_lines_details = plant_specs['model']['inactive_production_lines']


    # Set prediction table name and file
    level_tag = '_'.join(levels)
    pred_table = str.lower('_'.join(['abs', plant_id, level_tag, 'prediction_tbl']))

    file_name = '.'.join([pred_table, 'csv'])
    pred_file = '/'.join([project_directory, file_name])

    # Read in the prediction table, if available.
    # Otherwise, we bootstrap a new table.

    logger.info("Reading Existing Prediction Table %s" % pred_table)    
    try:
        dfp = pd.read_csv(pred_file)
        logger.info("Prediction Table Shape: %s" % (dfp.shape,))
        # Check to avoid duplicated predictions - @ t9939vs
        logger.info("Retrigger Validation Check For Prediction Table")
        dfp = dfp[dfp.workdate < prediction_date].reset_index(drop=True)
        logger.info("Prediction Table Shape: %s" % (dfp.shape,))
        # Store backup prediction file
        backup_table = str.lower('_'.join(['abs', plant_id, level_tag, 'prediction_backup_tbl']))
        logger.info("Storing Backup Prediction Table %s" % backup_table)
        file_name = '.'.join([backup_table, 'csv'])
        pred_backup_file = '/'.join([project_directory, file_name])
        dfp.to_csv(pred_backup_file)
    except:
        logger.info("Could Not Find Prediction Table %s" % pred_table)
        logger.info("Bootstrapping New Prediction Table %s" % pred_table)
        dfp = pd.DataFrame(columns=dfp_new.columns)
    
    # Calculate prediction bands for new predictions
    dfp_new['predicted_high'] = (round((1.0 + band_pct) * dfp_new['predicted'])).astype(int)
    dfp_new['predicted_low'] = (round((1.0 - band_pct) * dfp_new['predicted'])).astype(int)

    # Filter out holiday predictions
    dfp_new = dfp_new[dfp_new['workdate'].isin(chrysler_holiday_table) == False]

    # Write out new predictions to a separate file
    common_cols = ['workdate'] + levels
    pcols = common_cols + ['predicted', 'predicted_high', 'predicted_low']
    dfp_out = dfp_new[pcols]
    pred_new_file = '_'.join([str.lower(plant_id), 'predictions', level_tag, pred_date_str]) + '.csv'
    pred_new_path = '/'.join([project_directory, pred_new_file])
    dfp_out.to_csv(pred_new_path, index=False)
    
    # Create Data Insertion Date @ t9939vs
    logger.info("Developing Data Insertion Datetime ")
    dfp_new['created_date'] = pd.datetime.now().strftime("%Y-%m-%d")
    
    # Append new predictions to the existing table
    logger.info("Storing New Prediction Table %s" % pred_table)
    dfp = pd.concat([dfp, dfp_new])

    # Filter out plant exclusion dates
    expanded_dates = expand_dates(exclude_dates)
    dfp = dfp[dfp['workdate'].isin(expanded_dates) == False]    

    # Drop duplicate predictions
    dfp.drop_duplicates(subset=common_cols, inplace=True)
    
    # Write out the new prediction table
    logger.info("New Prediction Table Shape: %s" % (dfp.shape,))
    dfp.to_csv(pred_file, index=False)

    # Set actual column in model table based on target
    df_model['actual'] = df_model[target]

    # Merge prediction frame with model frame
    logger.info("Merging Prediction Table with Model Table")
    dfpm = pd.merge(dfp, df_model,
                    left_on=common_cols,
                    right_on=common_cols,
                    how='left')
    
    #Filter Inactive Production Pipelines @ T9939VS
    if (inactive_prod_lines_details) and ('production_line' in levels):
        
        
        df_inactive = dfpm[dfpm.production_line.isin(inactive_prod_lines_details.keys())]
        df_active = dfpm[~dfpm.production_line.isin(inactive_prod_lines_details.keys())]
        logger.info("Separated Inactive and Active Production lines")
        lst_inactive_frames = list()
        
        for production_line, dataframe in df_inactive.groupby(by = ['production_line']):
            date_ranges = inactive_prod_lines_details.get(production_line)
            if date_ranges.get('inactive_start_date') and date_ranges.get('inactive_end_date') :
                dataframe = dataframe[(dataframe.workdate < date_ranges.get('inactive_start_date')) | (dataframe.workdate > date_ranges.get('inactive_end_date') )]
            else:
                dataframe = dataframe[(dataframe.workdate < date_ranges.get('inactive_start_date')) ]
            lst_inactive_frames.append(dataframe)
    
        logger.info("Concat Filtered Inactive and Active Production lines ")
        df_inactive = pd.concat(lst_inactive_frames)
        dfpm = pd.concat([df_active, df_inactive])
    

    # Populate final frame with department information
    
    if 'production_line' in common_cols:
        pline_map2 = {}
        for k, v in pline_map.items():
            pline_map2[k] = v[0]
        dfpm['dept_name'] = dfpm['production_line'].map(pline_map2)
        dfpm['dept_name'].fillna('Not Mapped', inplace=True)
        dfpm['dept'] = dfpm['dept_name'].map(departments)
        dfpm['dept'].fillna('0000', inplace=True)
    
    # Create calendrical features
    if target == 'absences_unplanned':
        dfpm = create_calendrical_stats(dfpm, pipeline_specs, plant_specs, levels)
    
    # Sort for the final table
    ascend_array = [True] * len(common_cols)
    dfpm.sort_values(common_cols, ascending=ascend_array, inplace=True)
    logger.info("Plant Table Shape: %s" % (dfpm.shape,))
    
    # Store final plant frame
    table_tag = '_'.join([plant_id, 'plant', level_tag, pred_date_str])
    write_frame_to_pg(dfpm, table_tag, pipeline_specs)