# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 12:27:21 2019

@author: t0272m1
"""


#
# Imports (External and Internal)
#

import datetime
import itertools
import logging
import pandas as pd

from database import connect_greenplum
from database import connect_jdbc
from database import create_frame_from_db2
from database import create_sqlalchemy_engine
from database import sql_or
from database import sql_ro


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function set_production_lines
#

def set_production_lines(df, plant_specs):
    logger.info("Assigning Production Lines")
    # Get departments and production lines from configuration file
    depts = plant_specs['model']['departments']
    plines = plant_specs['model']['production_lines']
    # Set production lines
    df['production_line'] = ' '    
    for k, v in plines.items():
        pline = k
        dept_name = v[0]
        dept_number = int(depts[dept_name])
        teams = v[1]
        if teams:
            logger.info("Setting Production Line %s with Teams %s" % (pline, teams))
            df['production_line'][(df['ch_dept'] == dept_number) & (df['team'].str[1:].isin(teams))] = pline
        else:
            logger.info("Setting Production Line %s with All Teams" % pline)
            df['production_line'][df['ch_dept'] == dept_number] = pline
    # Return base table with assigned production lines
    return df

    
#
# Function create_base_table
#

def create_base_table(pipeline_specs, plant_specs):
    logger.info("Building Base Table")

    # Extract fields from specifications
    plant_code = str(plant_specs['plant']['code'])
    start_date = plant_specs['base_table']['start_date']
    end_date = plant_specs['base_table']['end_date']
    plant_depts = list(plant_specs['model']['departments'].values())
    
    # Connect to JDBC
    conn_hr, curs_hr = connect_jdbc(pipeline_specs['jdbc'])
    logger.info("Connected to JDBC: %s %s", conn_hr, curs_hr)

    # Create SQL Alchemy Engine
    engine_dl = create_sqlalchemy_engine(pipeline_specs['datalake'])
    logger.info("Connected to SQL Alchemy Engine: %s", engine_dl)

    # Connect to Greenplum
    conn_dl, curs_dl = connect_greenplum(pipeline_specs['datalake'])
    logger.info("Connected to Greenplum: %s %s", conn_dl, curs_dl)
    
    # Date Calculations

    # Start date
    start_date_str = start_date.strftime('%Y-%m-%d')
    start_year = start_date.year
    start_month = start_date.month
    start_day = start_date.day

    # End date is usually the last day of data
    if not end_date:
        end_date = datetime.datetime.now()
    end_date_str = end_date.strftime('%Y-%m-%d')
    end_year = end_date.year
    end_month = end_date.month
    end_day = end_date.day

    # Get snapshot from WH_ABS_HIST_TBL
    table = "WH_ABS_HIST_TBL"
    query = "select * from \"AUCERPTG\".\"" + table + "\" where " + sql_ro('ABS_DT', start_date_str, ">=") \
            + " and " + sql_ro('i_corploc', plant_code) + " and " + sql_or('i_dept', plant_depts)
    logger.info("Creating Absentee History Table")
    df_abs = create_frame_from_db2(curs_hr, table, query)
    logger.info("Absentee History Table: %s\n%s\n" % (df_abs.shape, df_abs.head()))

    # Get first absentee record of each employee
    df_abs = df_abs.groupby(['EMPLID', 'ABS_DT']).head(1).reset_index()

    # Get snapshot from WH_CHRYSLER_PERSON_TBL
    table = "WH_CHRYSLER_PERSON_TBL"
    query = "select * from \"AUCERPTG\".\"" + table + "\" where " + sql_ro('ch_corploc', plant_code)
    logger.info("Creating Chrysler Person Table")
    df_person = create_frame_from_db2(curs_hr, table, query)
    logger.info("Chrysler Person Table: %s\n%s\n" % (df_person.shape, df_person.head()))

    # Get snapshot from WH_EMPL_REL_DATE_TBL
    table = "WH_EMPL_REL_DATE_TBL"
    query = "select * from \"AUCERPTG\".\"" + table + "\""
    logger.info("Creating Employee Payroll Dates Table")
    df_empl = create_frame_from_db2(curs_hr, table, query)
    df_empl['PAY_START_DT'] = pd.to_datetime(df_empl['PAY_END_DT']) - datetime.timedelta(days=6)
    logger.info("Employee Payroll Dates Table: %s\n%s\n" % (df_empl.shape, df_empl.head()))

    # Get snapshot from WH_PAYROLL_DETAIL_TBL
    table = "WH_PAYROLL_DETAIL_TBL"
    query = "select * from \"AUCERPTG\".\"" + table + "\" where " + sql_ro('location', plant_code) \
            + " and " + sql_ro('year', str(start_year), ">=") + " and " + sql_ro('rpt_month', str(start_month), ">=")
    logger.info("Creating Employee Payroll Detail Table")
    df_pay = create_frame_from_db2(curs_hr, table, query)
    df_pay = df_pay.merge(df_empl, left_on=['YEAR', 'PAYWEEK'], right_on=['RPT_YEAR', 'PAYWEEK'], how='left')
    logger.info("Employee Payroll Detail Table: %s\n%s\n" % (df_pay.shape, df_pay.head()))

    # Get snapshot from WH_HOURS_WORKED_TBL
    table = "WH_HOURS_WORKED_TBL"
    query = "select * from \"AUCERPTG\".\"" + table + "\" where " + sql_ro('DATE', start_date_str, ">=") \
            + " and " + sql_ro('CH_CORPLOC', plant_code) + " and " + sql_or('ch_dept', plant_depts)
    logger.info("Creating Hours Worked Table")
    df_worked = create_frame_from_db2(curs_hr, table, query)
    df_worked = df_worked.groupby(['EMPLID', 'DATE']).agg(
            {'CH_CORPLOC' : 'last',
             'CH_DEPT' : 'last',
             'CH_SUPV_GRP' : 'last',
             'SHIFT' : 'last',
             'CH_ACTL_HRS' : 'sum',
             'CH_PAA_HRS' : 'sum',
             'HOLIDAY_HRS' : 'sum'})
    df_worked.reset_index(inplace=True)
    logger.info("Hours Worked Table: %s\n%s\n" % (df_worked.shape, df_worked.head()))

    # Get off-role data
    table = "WH_SA_WC_DETAIL_TBL"
    query = "select * from \"AUCERPTG\".\"" + table + "\" where " + sql_ro('LOCN', plant_code) + " and " + sql_ro('YEAR', str(start_year), ">=") + " and " + sql_ro('MONTH', str(start_month), ">=")
    logger.info("Creating Off-Role Table")    
    df_sawc = create_frame_from_db2(curs_hr, table, query)
    logger.info("Off-Role Table: %s\n%s\n" % (df_sawc.shape, df_sawc.head()))

    # Get snapshot from WH_SNAPSHOT_TBL
    table = "WH_SNAPSHOT_TBL"
    query = "select EMPLID, EFFDT, REG_TEMP, FULL_PART_TIME, CH_FLEXWRK_CODE, ZIP from \"AUCERPTG\".\"" + table \
            + "\" where " + sql_ro('EFFDT', start_date_str, ">=") + " and " + sql_ro('CH_CORPLOC', plant_code)
    features_snap = ['EMPLID', 'EFFDT', 'REG_TEMP', 'FULL_PART_TIME', 'CH_FLEXWRK_CODE', 'ZIP']
    logger.info("Creating Snapshot Table")
    df_snap = create_frame_from_db2(curs_hr, table, query, features_snap)
    logger.info("Snapshot Table: %s\n%s\n" % (df_snap.shape, df_snap.head()))

    # Create subsets

    # Pay Subset
    logger.info("Creating Pay Subset Table")
    cols_pay_weekly = ['EMPLID',
                       'PAYWEEK',
                       'STRAIGHT_HRS',
                       'VAL_STRAIGHT',
                       'TIME_HALF_HRS',
                       'DOUBLE_TIME_HRS',
                       'SHIFT_PREMIUM',
                       'EG',
                       'PAY_START_DT',
                       'PAY_END_DT',
                       'PAY_CHECK_DT']
    df_pay_subset = df_pay[cols_pay_weekly]

    # Expand weekly data for daily base table
    df_pay_subset = pd.concat([pd.DataFrame({'EMPLID': row.EMPLID,
                                             'date': pd.date_range(row.PAY_START_DT, row.PAY_END_DT),
                                             'PAYWEEK': row.PAYWEEK,
                                             'STRAIGHT_HRS_BY_PAYWEEK': row.STRAIGHT_HRS,
                                             'VAL_STRAIGHT_BY_PAYWEEK': row.VAL_STRAIGHT,
                                             'TIME_HALF_HRS_BY_PAYWEEK': row.TIME_HALF_HRS,
                                             'DOUBLE_TIME_HRS_BY_PAYWEEK': row.DOUBLE_TIME_HRS,
                                             'SHIFT_PREMIUM_BY_PAYWEEK': row.SHIFT_PREMIUM,
                                             'EG_BY_PAYWEEK': row.EG,
                                             'PAY_CHECK_DT': row.PAY_CHECK_DT})
                               for i, row in df_pay_subset.iterrows()], ignore_index=True)
    df_pay_subset['date'] = df_pay_subset['date'].dt.strftime('%Y-%m-%d')
    df_pay_subset.drop_duplicates(inplace=True)
    logger.info("Pay Subset Table: %s\n%s\n" % (df_pay_subset.shape, df_pay_subset.head()))
    
    # Off-Role Subset
    logger.info("Defining Off-Role Subset")
    cols_or_period = ['EMPLID',
                      'STARTDATE',
                      'ENDDATE',
                      'NUMOFHOURS',
                      'CODE']
    df_sawc_subset = df_sawc[cols_or_period]
    df_sawc_subset = df_sawc_subset.groupby(['EMPLID', 'STARTDATE', 'ENDDATE']).agg({'NUMOFHOURS':'sum', 'CODE':'first'}).reset_index()
    df_sawc_subset.loc[df_sawc_subset['ENDDATE'] == '9999-12-31', 'ENDDATE'] = end_date_str
    # Expand monthly data for daily base table
    df_sawc_subset = pd.concat([pd.DataFrame({'EMPLID': row.EMPLID,
                                              'date': pd.date_range(row.STARTDATE, row.ENDDATE),
                                              'OFFROLE_HOURS_FOR_PERIOD': row.NUMOFHOURS,
                                              'OFFROLE_REASON_CODE_FOR_PERIOD': row.CODE})
                                for i, row in df_sawc_subset.iterrows()], ignore_index=True)
    df_sawc_subset['date'] = df_sawc_subset['date'].dt.strftime('%Y-%m-%d')
    df_sawc_subset.drop_duplicates(inplace=True)
    logger.info("Off-Role Subset Table: %s\n%s\n" % (df_sawc_subset.shape, df_sawc_subset.head()))
    
    # Person Subset
    logger.info("Defining Person Subset")
    df_person_subset = df_person[['EMPLID', 'EMPL_PART_TIME', 'EMPL_TMP', 'CH_D_CORP', 'SEX', 'BIRTHDATE', 'JOB_CLASSIFICATION']]
    logger.info("Person Subset Table: %s\n%s\n" % (df_person_subset.shape, df_person_subset.head()))

    # Absence Subset
    logger.info("Defining Absence Subset")
    df_abs_subset = df_abs[['EMPLID', 'ABS_DT', 'LOST_HRS', 'ABS_CODE', 'C_TYP']]
    logger.info("Absence Subset Table: %s\n%s\n" % (df_abs_subset.shape, df_abs_subset.head()))

    # Hours Worked Subset
    logger.info("Defining Hours Worked Subset")
    df_worked_subset = df_worked[['EMPLID', 'DATE', 'CH_CORPLOC', 'CH_DEPT', 'CH_SUPV_GRP', 'SHIFT', 'CH_ACTL_HRS', 'CH_PAA_HRS', 'HOLIDAY_HRS']]
    logger.info("Hours Worked Subset Table: %s\n%s\n" % (df_worked_subset.shape, df_worked_subset.head()))

    # Generate dates for base table
    logger.info("Generating Dates for Base Table")
    time_range = pd.date_range(start=start_date_str, end=end_date_str)
    dates = time_range.strftime('%Y-%m-%d')

    # Unique CIDs
    cids = df_worked['EMPLID'].unique()
    logger.info("Number of CIDs: %d" % len(cids))

    # Create initial base table
    row_data = list(itertools.product(*(cids, dates)))
    df = pd.DataFrame(row_data, columns=['CID', 'WorkDate'])
    logger.info("Initial Base Table: %s\n%s\n" % (df.shape, df.head()))

    # Merge snapshot history
    logger.info("Merging Snapshot History into Base Table")
    df = df.merge(df_snap, left_on=['CID', 'WorkDate'], right_on=['EMPLID', 'EFFDT'], how='left')

    # We first get the snapshot data and then backfill from the person table where necessary.
    df = df.rename(columns={'REG_TEMP': 'EMPL_TMP', 'FULL_PART_TIME': 'EMPL_PART_TIME'})
    df['EMPL_TMP'] = df['EMPL_TMP'].map({'R':'N', 'T':'Y'})
    df['EMPL_PART_TIME'] = df['EMPL_PART_TIME'].map({'F':'N', 'P':'Y'})

    # Forward-fill the snapshot data
    df['EMPL_PART_TIME'] = df.groupby("CID")['EMPL_PART_TIME'].transform(lambda x: x.fillna(method='ffill'))
    df['EMPL_TMP'] = df.groupby("CID")['EMPL_TMP'].transform(lambda x: x.fillna(method='ffill'))
    df['CH_FLEXWRK_CODE'] = df.groupby("CID")['CH_FLEXWRK_CODE'].transform(lambda x: x.fillna(method='ffill'))
    df['ZIP'] = df.groupby("CID")['ZIP'].transform(lambda x: x.fillna(method='ffill'))
    df['ZIP'].fillna(' ', inplace=True)
    df.drop(columns=['EMPLID', 'EFFDT'], inplace=True, errors='ignore')

    # Merge the absence history
    logger.info("Merging Absence History into Base Table")
    df = df.merge(df_abs_subset, left_on=['CID', 'WorkDate'], right_on=['EMPLID', 'ABS_DT'], how='left')
    # Initialize and/or forward-fill NA values
    df['LOST_HRS'].fillna(0.0, inplace=True)
    df['ABS_CODE'].fillna(' ', inplace=True)
    df['C_TYP'].fillna(' ', inplace=True)
    df.drop(columns=['EMPLID', 'ABS_DT'], inplace=True, errors='ignore')

    # Merge the worked hours subset
    logger.info("Merging Hours Worked into Base Table")
    df = df.merge(df_worked_subset, left_on=['CID', 'WorkDate'], right_on=['EMPLID', 'DATE'], how='left')
    # Fill null hours with default 0.0
    df['CH_ACTL_HRS'].fillna(0.0, inplace=True)
    df['CH_PAA_HRS'].fillna(0.0, inplace=True)
    df['HOLIDAY_HRS'].fillna(0.0, inplace=True)
    df['CH_DEPT'] = df.groupby("CID")['CH_DEPT'].transform(lambda x: x.fillna(method='ffill'))
    df['CH_DEPT'].fillna(0, inplace=True)
    df['CH_SUPV_GRP'] = df.groupby("CID")['CH_SUPV_GRP'].transform(lambda x: x.fillna(method='ffill'))
    df['CH_SUPV_GRP'].fillna(0, inplace=True)
    df['SHIFT'] = df.groupby("CID")['SHIFT'].transform(lambda x: x.fillna(method='ffill'))
    df['SHIFT'].fillna(0, inplace=True)
    df['CH_CORPLOC'] = df.groupby("CID")['CH_CORPLOC'].transform(lambda x: x.fillna(method='ffill'))
    df['CH_CORPLOC'].fillna(0, inplace=True)
    df.drop(columns=['EMPLID', 'DATE'], inplace=True)

    # Merge the person subset
    logger.info("Merging Person Subset into Base Table")
    df = df.merge(df_person_subset, left_on=['CID'], right_on=['EMPLID'], how='left')
    df['EMPL_TMP'] = df['EMPL_TMP_x'].fillna('NA')
    df['EMPL_PART_TIME'] = df['EMPL_PART_TIME_x'].fillna('NA')
    df['CH_FLEXWRK_CODE'].fillna('4', inplace=True)
    df['CH_D_CORP'].fillna(end_date_str, inplace=True)
    drop_cols = ['EMPLID', 'EMPL_TMP_x', 'EMPL_PART_TIME_x', 'EMPL_PART_TIME_y', 'EMPL_TMP_y']
    df.drop(columns=drop_cols, inplace=True)

    # Merge the pay period subset
    logger.info("Merging Pay Period into Base Table")
    df_pay_subset['HRLYPAYRATE'] = df_pay_subset['VAL_STRAIGHT_BY_PAYWEEK'] / df_pay_subset['STRAIGHT_HRS_BY_PAYWEEK']
    df = df.merge(df_pay_subset, left_on=['CID', 'WorkDate'], right_on=['EMPLID', 'date'], how='left')
    df.drop(columns=['EMPLID', 'date', 'VAL_STRAIGHT_BY_PAYWEEK'], inplace=True)

    # Merge the off-role subset
    logger.info("Merging Off-Role into Base Table")
    df = df.merge(df_sawc_subset, left_on=['CID', 'WorkDate'], right_on=['EMPLID', 'date'], how='left')
    df['OFFROLE_HOURS_FOR_PERIOD'].fillna(0.0, inplace=True)
    df['OFFROLE_REASON_CODE_FOR_PERIOD'].fillna(0, inplace=True)
    df.drop(columns=['EMPLID', 'date'], inplace=True)

    # Type Conversion and Column Mapping
    logger.info("Type Conversion and Column Mapping")
    df['CH_DEPT'] = df['CH_DEPT'].astype(int)
    df['CH_SUPV_GRP'] = df['CH_SUPV_GRP'].astype(int)
    df['SHIFT'] = df['SHIFT'].astype(int)
    df['CH_CORPLOC'] = df['CH_CORPLOC'].astype(int)

    # map column names to lower case
    df.columns = map(str.lower, df.columns)

    # Base Table Features
    logger.info("Calculating Age and Tenure")
    df['workdate_dt'] = pd.to_datetime(df['workdate'])
    days_in_year = 365
    # Calculate age
    df['birthdate_dt'] = pd.to_datetime(df['birthdate'])
    df['age'] = (df['workdate_dt'] - df['birthdate_dt']) / (days_in_year * pd.offsets.Day(1))
    df['age'].fillna(0.0, inplace=True)
    # Calculate tenure
    df['ch_d_corp_dt'] = pd.to_datetime(df['ch_d_corp'])
    df['tenure'] = (df['workdate_dt'] - df['ch_d_corp_dt']) / (days_in_year * pd.offsets.Day(1))
    # set tenure to a minimum of 0 when ch_d_corp is not available
    df['tenure'].loc[df['tenure'] < 0.0] = 0.0
    df.drop(columns=['workdate_dt', 'birthdate_dt', 'ch_d_corp_dt'], inplace=True)

    # Drop unnecessary columns from base table
    logger.info("Dropping Columns")
    cols_drop = ['ch_d_corp', 'sex', 'payweek', 'pay_check_dt']
    df.drop(columns=cols_drop, inplace=True)

    # Drop duplicates as a result of the merge
    logger.info("Dropping Duplicates")
    df.drop_duplicates(inplace=True)

    # Return base table
    logger.info("Base Table: %s\n%s\n" % (df.shape, df.sample(100)))
    return df