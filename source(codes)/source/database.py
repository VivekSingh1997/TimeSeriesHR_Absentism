# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:37:50 2019

@author: t0272m1
"""


#
# Imports
#

import jaydebeapi as jdb
import logging
import pandas as pd
import pandas.io.sql as psql
import psycopg2
from sqlalchemy import create_engine


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function connect_jdbc
#

def connect_jdbc(specs):
    con = jdb.connect(specs['driver'],
                      specs['server'],
                      [specs['user'], specs['password']],
                      specs['jar_file'])
    curs = con.cursor()
    logger.info("JDBC Connection Created")
    return con, curs


#
# Function create_sqlalchemy_engine
#

def create_sqlalchemy_engine(specs):
    estring =  """postgresql://{user}:{password}@{host}:{port}/{database}
               """.format(host = specs['host'],
                          port = specs['port'],
                          user = specs['user'],
                          password = specs['password'],
                          database = specs['database'])
    engine = create_engine(estring)
    logger.info("SQL Alchemy Engine Created")
    return engine


#
# Function connect_greenplum
#

def connect_greenplum(specs):
    # GPDB -> 'shbdmdwp001.servers.chrysler.com'
    # HAWQ -> 'shbdhdmp002.servers.chrysler.com'
    
    conn =  """
    dbname='{database}' user='{user}' host='{host}' port='{port}' password='{password}'
    """.format(host = specs['host'],
               port = specs['port'],
               user = specs['user'],
               database = specs['database'],
               password = specs['password'])

    conn = psycopg2.connect(conn)
    cur = conn.cursor()
    logger.info("Data Lake Connection Created")

    # set role
    logger.info("Setting role to datasci")
    sql_setrole = 'set role to datasci'
    cur.execute(sql_setrole)

    # five minutes    
    logger.info("Setting timeout value")
    sql_timeout = "set statement_timeout to 86400000;"
    cur.execute(sql_timeout)
    
    return conn, cur


#
# Function create_frame_from_db2
#
 
def create_frame_from_db2(cursor, table, query, features=None):
    logger.info("Accessing Table %s\n" % table)
    qschema = "select name, coltype, length from sysibm.syscolumns where tbname='" + table + "' order by colno"
    cursor.execute(qschema)
    table_schema = cursor.fetchall()
    logger.info("%s Table Schema", table)
    for field in table_schema:
        logger.info(field)
    cursor.execute(query)
    results = cursor.fetchall()
    if features:
        table_columns = features
    else:
        table_columns = [x[0] for x in table_schema]
    df = pd.DataFrame(results, columns=table_columns)
    df = df.apply(pd.to_numeric, errors='ignore')
    logger.info("%s Pandas Schema", table)
    for f in df.columns:
        logger.info("%s, Type: %s" % (f, df[f].dtype))
    return df


#
# Function create_frame_from_pg
#

def create_frame_from_pg(connection, table, query):
    df = psql.read_sql(query, connection)
    df = df.apply(pd.to_numeric, errors='ignore')
    logger.info("\nPandas Schema")
    for f in df.columns:
        logger.info("%s, Type: %s" % (f, df[f].dtype))
    return df


#
# Function write_frame_to_pg
#

def write_frame_to_pg(df, table_tag, pipeline_specs):
    # extract specifications
    schema = pipeline_specs['datalake']['schema']
    project_directory = pipeline_specs['project_directory']
    test_flag = pipeline_specs['test_flag']
    # establish connection
    logger.info("Establishing connection to Greenplum")
    conn_dl, curs_dl = connect_greenplum(pipeline_specs['datalake'])
    # establish SQL Alchemy connection
    engine_dl = create_sqlalchemy_engine(pipeline_specs['datalake'])
    # determine prefix based on test flag
    prefix = 'abs'
    prefix = '_'.join([prefix, 'test']) if test_flag else prefix
    # create table name
    table = str.lower('_'.join([prefix, table_tag, 'tbl']))
    full_table_name = '.'.join([schema, table])
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
    file_name = table + '.csv'
    csv_file = '/'.join([project_directory, file_name])
    logger.info("Saving CSV file %s" % csv_file)
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
    logger.info("Copying table from %s" % csv_file)
    f = open(csv_file)
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


#
# Function show_features
#

def show_features(df, threshold = 200):
    for f in df.columns:
        logger.info("\nFeature: %s" % f)
        vc = df[f].value_counts()
        vc_len = len(vc)
        logger.info("Value Counts: %d" % vc_len)
        if vc_len and vc_len <= threshold:
            uv = df[f].unique()
            logger.info("Unique Values: %s" % uv)
    return


#
# Function sql_ro
#

def sql_ro(field_name, value, cond='='):
    output = field_name + cond + "'" + value + "'"
    return output


#
# Function sql_or
#

def sql_or(field_name, value_list):
    efield = field_name + '='
    vjoin = ' or '.join([efield+"'"+v+"'" for v in value_list])
    output = '(' + vjoin + ')'
    return output