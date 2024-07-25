# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:28:17 2019

@author: t0272m1
"""


#
# SHAP Crew Map
#
#     Pandas Days {S:6, M:0, T:1, W:2, T:3, F:4, S:5}
#

shap_crew_map = {
    '0': ['A', [0, 1, 2, 3]],
    '3': ['B', [2, 3, 4, 5]],
    '6': ['C', [4, 5, 0, 1]]
}


#
# Function get_shap_crew_map
#

def get_shap_crew_map():
    crew_maps = [shap_crew_map['0'],
                 shap_crew_map['3'],
                 shap_crew_map['6']]
    return crew_maps

    
#
# Function set_shap_crew_map
#

def set_shap_crew_map(df):
    df['crew'] = 'A'
    df['crew'][(df['ch_supv_grp'] > 100) & (df['ch_supv_grp'] < 500)] = 'B'
    df['crew'][df['ch_supv_grp'] > 599] = 'C'
    return df


#
# Function shap_work_day
#

def shap_work_day(row):
    is_work_day = 'N'
    crew_code = row['team'][0]
    try:
        work_days = shap_crew_map[crew_code][1]
        if row['pandas_day_of_week'] in work_days:
            is_work_day = 'Y'
    except:
        is_work_day = 'N'
    return is_work_day
