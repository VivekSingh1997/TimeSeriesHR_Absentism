# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:40:16 2019

@author: t0272m1
"""


#
# Imports (External and Internal)
#

import datetime
from flask import Flask
from flask_apscheduler import APScheduler
import logging
import pandas as pd
import requests

from database import connect_greenplum
from database import create_frame_from_pg
from database import create_sqlalchemy_engine


#
# Configuration Class for Scheduler
#

class Config(object):
    SCHEDULER_API_ENABLED = True


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function write_frame_to_pg
#

def write_frame_to_pg(df, table_name, datalake_specs):
    # extract specifications
    schema = datalake_specs['schema']
    # establish connection
    logger.info("Establishing connection to Greenplum")
    conn_dl, curs_dl = connect_greenplum(datalake_specs)
    # establish SQL Alchemy connection
    engine_dl = create_sqlalchemy_engine(datalake_specs)
    # create table name
    full_table_name = '.'.join([schema, table_name])
    logger.info("Table Name: %s" % full_table_name)
    # drop table if already exists
    logger.info("Dropping table")
    drop_table = 'drop table if exists ' + full_table_name
    curs_dl.execute(drop_table)
    # create empty table
    logger.info("Creating table %s" % full_table_name)
    empty_table = pd.io.sql.get_schema(df, full_table_name, con=engine_dl)
    empty_table = empty_table.replace('"', '')
    logger.info(empty_table)
    curs_dl.execute(empty_table)
    # save the CSV file
    file_name = table_name + '.csv'
    logger.info("Saving CSV file %s" % file_name)
    df.fillna(0, inplace=True)
    df.to_csv(file_name, index=False)
    # create sql for copying table
    SQL_STATEMENT = """
        COPY %s FROM STDIN WITH
            CSV
            HEADER
            DELIMITER AS ','
        """
    # copy file to the table
    logger.info("Copying table from %s" % file_name)
    f = open(file_name)
    curs_dl.copy_expert(sql=SQL_STATEMENT % full_table_name, file=f)
    # execute grants
    logger.info("Executing grants")
    grant = 'grant all on table ' + full_table_name + ' to datasci'
    curs_dl.execute(grant)
    grant = 'grant select on table ' + full_table_name + ' to hrba'
    curs_dl.execute(grant)
    # close connection
    logger.info("Closing connection")
    conn_dl.commit()
    curs_dl.close()
    return
    

# Plant and Other Locations Data

plant_locs = {'BRAP' : ['9126', '43.759406,-79.716659'],
              'BVP'  : ['4015', '42.241561,-88.872995'],
              'CTC'  : ['0000', '42.654639,-83.226746'],
              'JNAP' : ['4012', '42.375292,-82.966222'],
              'SHAP' : ['4025', '42.571145,-83.034768'],
              'TAC'  : ['2459', '41.695379,-83.520427'],
              'WAP'  : ['9103', '42.296940,-82.985780'],
              'WTAP' : ['2452', '42.456030,-83.041058']}

# Flask Scheduler
scheduler = APScheduler()

# Proxies
https_proxy = 'https://t9939vs:Jeepnov@2019@iproxy.appl.chrysler.com:9090'
proxyDict = {"https" : https_proxy}

# DarkSky API
darksky_api_key = 'b5bc908b21f611a8e19dbf1321be24df'

#
# Read in the weather table
#

lake_specs = {'schema': 'lab_datasci',
              'host': 'shbdmdwp001.servers.chrysler.com',
              'port': 5432,
              'user': 'datasci',
              'password': 'datasci_01',
              'database': 'odshawq'}


#
# Scheduled Tasks
#


@scheduler.task('interval', id='DarkSky Forecast', seconds=3600, misfire_grace_time=1800)
def get_darksky_forecast():
    # Read current weather table
    conn_dl, curs_dl = connect_greenplum(lake_specs)
    table_name = "abs_weather_forecast_tbl"
    query = "select * from \"" + lake_specs['schema'] + "\".\"" + table_name + "\""
    logger.info("Query: %s" % query)
    try:
        dfw = create_frame_from_pg(conn_dl, table_name, query)
    except:
        logger.info("Table %s not found" % table_name)
        dfw = pd.DataFrame()
    logger.info("Closing connection")
    conn_dl.commit()
    curs_dl.close()
    # Make DarkSky Request
    request_options = '?exclude=currently,minutely,hourly,flags'
    df_plants = []
    for p in plant_locs:
        logger.info('Getting Weather Forecast for %s' % p)
        lat_lon = plant_locs[p][1]
        darksky_request = 'https://api.darksky.net/forecast/' + darksky_api_key + '/' + lat_lon + request_options
        darksky_response = requests.get(darksky_request, proxies=proxyDict)
        if darksky_response.status_code == 200:
            logger.info("Valid Response %d" % darksky_response.status_code)
            darksky_json = darksky_response.json()
            try:
                logger.info("Getting Forecast for %s" % p)
                weather_data = darksky_json['daily']['data']
                forecast_days = []
                for wd in weather_data:
                    wd = {k.lower(): v for k, v in wd.items()}
                    forecast_days.append(wd)           
                df_plant = pd.DataFrame(forecast_days)
                df_plant['plant_id'] = p
                df_plant['plant_code'] = plant_locs[p][0]
                df_plants.append(df_plant)
            except:
                logger.info("No Weather Forecast for %s" % p)
        else:
            logger.info("Invalid Response %d" % darksky_response.status_code)
    # Merge plant records
    dfp = pd.concat(df_plants)
    dfp['utctime'] = dfp['time']
    dfp['date'] = dfp['utctime'].apply(lambda x: datetime.datetime.utcfromtimestamp(x).strftime('%Y-%m-%d'))
    dfp['time'] = dfp['utctime'].apply(lambda x: datetime.datetime.utcfromtimestamp(x).strftime('%H:%M'))
    # Merge current weather table with updated plant weather
    df = pd.concat([dfw, dfp])
    # Write the table to the data lake
    df.drop_duplicates(inplace=True)
    logger.info("Writing Weather Forecast Table: %s" % (df.shape,))
    write_frame_to_pg(df, table_name, lake_specs)


#
# Run the server
#

if __name__ == '__main__':
    
    # Logging
    output_file = 'darksky_weather_forecast.log'
    logging.basicConfig(format="[%(asctime)s] %(levelname)s\t%(message)s",
                        filename=output_file, filemode='a', level=logging.DEBUG,
                        datefmt='%m/%d/%y %H:%M:%S')
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s",
                                  datefmt='%m/%d/%y %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    # Create Flask instance
    
    app = Flask(__name__)
    app.config.from_object(Config())

    # it is also possible to enable the API directly
    # scheduler.api_enabled = True
    scheduler.init_app(app)
    scheduler.start()

    app.run()