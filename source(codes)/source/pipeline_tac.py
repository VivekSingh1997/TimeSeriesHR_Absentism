# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:28:17 2019

@author: t0272m1
"""


#
# TAC Crew Map
#
#     Pandas Days {S:6, M:0, T:1, W:2, T:3, F:4, S:5}
#

tac_crew_map = {
    '0': ['AM', [0, 1, 2, 3, 4, 5]],
    '3': ['PM', [0, 1, 2, 3, 4, 5]]
}


#
# Function get_tac_crew_map
#

def get_tac_crew_map():
    crew_maps = [tac_crew_map['0'],
                 tac_crew_map['3']]
    return crew_maps


#
# Function set_tac_crew_map
#

def set_tac_crew_map(df):
    df['crew'] = 'AM'
    df['crew'][(df['ch_supv_grp'] > 100) & (df['ch_supv_grp'] < 500)] = 'PM'
    return df


#
# Function tac_work_day
#

def tac_work_day(row):
    is_work_day = 'N'
    crew_code = row['team'][0]
    try:
        work_days = tac_crew_map[crew_code][1]
        if row['pandas_day_of_week'] in work_days:
            is_work_day = 'Y'
    except:
        is_work_day = 'N'
    return is_work_day
