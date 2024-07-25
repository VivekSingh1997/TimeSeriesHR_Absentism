# -*- coding: utf-8 -*-
"""
HR Analytics Pipeline

Created on Tue Oct 23 16:13:01 2018

@author : t0272m1
Module  : __main__

"""


#
# Imports (External and Internal)
#

import argparse
import datetime
import logging
import pandas as pd
import sys
import yaml

from base_table import create_base_table
from base_table import set_production_lines
from database import connect_greenplum
from database import create_frame_from_pg
from database import write_frame_to_pg
from model import create_model_table
from model import make_plant_predictions
from model import store_predictions
from pipeline_bvp import bvp_work_day
from pipeline_bvp import get_bvp_crew_map
from pipeline_bvp import set_bvp_crew_map
from pipeline_jnap import jnap_work_day
from pipeline_jnap import get_jnap_crew_map
from pipeline_jnap import set_jnap_crew_map
from pipeline_shap import shap_work_day
from pipeline_shap import get_shap_crew_map
from pipeline_shap import set_shap_crew_map
from pipeline_tac import tac_work_day
from pipeline_tac import get_tac_crew_map
from pipeline_tac import set_tac_crew_map
from pipeline_wap import wap_work_day
from pipeline_wap import get_wap_crew_map
from pipeline_wap import set_wap_crew_map
from pipeline_wtap import wtap_work_day
from pipeline_wtap import get_wtap_crew_map
from pipeline_wtap import set_wtap_crew_map


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Crew Tables for Each Plant
#

crew_map_get_table = {
    'BVP'  : get_bvp_crew_map,
    'JNAP' : get_jnap_crew_map,
    'SHAP' : get_shap_crew_map,
    'TAC'  : get_tac_crew_map,
    'WAP'  : get_wap_crew_map,
    'WTAP' : get_wtap_crew_map,
    }

crew_map_set_table = {
    'BVP'  : set_bvp_crew_map,
    'JNAP' : set_jnap_crew_map,
    'SHAP' : set_shap_crew_map,
    'TAC'  : set_tac_crew_map,
    'WAP'  : set_wap_crew_map,
    'WTAP' : set_wtap_crew_map,
    }

crew_work_day_table = {
    'BVP'  : bvp_work_day,
    'JNAP' : jnap_work_day,
    'SHAP' : shap_work_day,
    'TAC'  : tac_work_day,
    'WAP'  : wap_work_day,
    'WTAP' : wtap_work_day,
    }


#
# Production Models
#

production_models = [
    'sarimax',
    'share'
    ]

    
#
# Function valid_date
#

def valid_date(date_string):
    r"""Determine whether or not the given string is a valid date.

    Parameters
    ----------

    date_string : str
        An alphanumeric string in the format %Y-%m-%d.

    Returns
    -------
    date_string : str
        The valid date string.

    Raises
    ------
    ValueError
        Not a valid date.

    Examples
    --------

    >>> valid_date('2016-7-1')   # datetime.datetime(2016, 7, 1, 0, 0)
    >>> valid_date('345')        # ValueError: Not a valid date

    """
    try:
        dt_string = datetime.strptime(date_string, "%Y-%m-%d")
        return dt_string
    except:
        message = "Not a valid date: '{0}'.".format(date_string)
        raise argparse.ArgumentTypeError(message)


#
# Function get_pipeline_config
#

def get_pipeline_config(directory):
    r"""Read in the configuration file for the HR Pipeline.

    Parameters
    ----------
    directory : type
        The directory where HR files are stored

    Returns
    -------
    specs : dict
        The parameters for controlling the HR Pipeline.

    Raises
    ------
    ValueError
        Unrecognized value of a ``pipeline.yml`` field.
    """

    logger.info("Pipeline Configuration")

    # Read the configuration file

    full_path = '/'.join([directory, 'pipeline.yml'])
    with open(full_path, 'r') as ymlfile:
        specs = yaml.load(ymlfile, Loader=yaml.FullLoader)

    # Log the configuration parameters

    logger.info('PIPELINE CONFIGURATION:')
    logger.info('datalake:schema          = %s', specs['datalake']['schema'])
    logger.info('datalake:host            = %s', specs['datalake']['host'])
    logger.info('datalake:port            = %s', specs['datalake']['port'])
    logger.info('datalake:user            = %s', specs['datalake']['user'])
    logger.info('datalake:password        = %s', specs['datalake']['password'])
    logger.info('datalake:database        = %s', specs['datalake']['database'])
    logger.info('jdbc:driver              = %s', specs['jdbc']['driver'])
    logger.info('jdbc:server              = %s', specs['jdbc']['server'])
    logger.info('jdbc:user                = %s', specs['jdbc']['user'])
    logger.info('jdbc:password            = %s', specs['jdbc']['password'])
    logger.info('jdbc:jar_file            = %s', specs['jdbc']['jar_file'])
    logger.info('holidays:calendar_us     = %s', specs['holidays']['calendar_us'])
    logger.info('holidays:calendar_canada = %s', specs['holidays']['calendar_canada'])
    
    # Specifications for the HR Pipeline
    return specs


#
# Function get_plant_config
#

def get_plant_config(specs):
    r"""Read in the plant configuration file.

    Parameters
    ----------
    specs : dict
        The specifications for controlling the HR Pipeline.

    Returns
    -------
    plant_specs : dict
        The parameters for generating the plant base and model table.

    Raises
    ------
    ValueError
        Unrecognized value of a ``plant.yml`` field.
    """

    logger.info("Plant Configuration")
    
    # Extract fields from pipeline specifications

    plant_id = specs['plant_id']
    project_directory = specs['project_directory']

    # Read the configuration file

    file_name = '.'.join(['config_' + plant_id, 'yml'])
    full_path = '/'.join([project_directory, file_name])
    with open(full_path, 'r') as ymlfile:
        plant_specs = yaml.load(ymlfile, Loader=yaml.FullLoader)
        
    # Check models
    
    for m in plant_specs['model']['models']:
        if m not in production_models:
            logger.info("Invalid Model: %s" % m)
            sys.exit("Halting Production Pipeline")

    # Log the configuration parameters

    logger.info('PLANT CONFIGURATION:')
    logger.info('plant:code                              = %s', plant_specs['plant']['code'])
    logger.info('plant:latitude                          = %f', plant_specs['plant']['latitude'])
    logger.info('plant:longitude                         = %f', plant_specs['plant']['longitude'])
    logger.info('plant:market_id                         = %d', plant_specs['plant']['market_id'])
    logger.info('plant:shift_days                        = %d', plant_specs['plant']['shift_days'])
    logger.info('plant:shift_hours                       = %d', plant_specs['plant']['shift_hours'])
    logger.info('plant:absence_codes                     = %s', plant_specs['plant']['absence_codes'])
    logger.info('plant:exclude_dates                     = %s', plant_specs['plant']['exclude_dates'])
    logger.info('base_table:start_date                   = %s', plant_specs['base_table']['start_date'])
    logger.info('base_table:end_date                     = %s', plant_specs['base_table']['end_date'])
    logger.info('base_table:use_table                    = %s', plant_specs['base_table']['use_table'])
    logger.info('base_table:use_table_date               = %s', plant_specs['base_table']['use_table_date'])
    logger.info('base_table:write_table                  = %r', plant_specs['base_table']['write_table'])
    logger.info('model:models                            = %s', plant_specs['model']['models'])
    logger.info('model:target                            = %s', plant_specs['model']['target'])
    logger.info('model:levels                            = %s', plant_specs['model']['levels'])
    logger.info('model:crews                             = %s', plant_specs['model']['crews'])
    logger.info('model:departments                       = %s', plant_specs['model']['departments'])
    logger.info('model:production_lines                  = %s', plant_specs['model']['production_lines'])
    logger.info('model:npreds                            = %d', plant_specs['model']['npreds'])
    logger.info('model:p_arima                           = %d', plant_specs['model']['p_arima'])
    logger.info('model:d_arima                           = %d', plant_specs['model']['d_arima'])
    logger.info('model:q_arima                           = %d', plant_specs['model']['q_arima'])
    logger.info('model:features                          = %s', plant_specs['model']['features'])
    logger.info('model:top_features                      = %d', plant_specs['model']['top_features'])
    logger.info('model:band_pct                          = %f', plant_specs['model']['band_pct'])
    logger.info('model:write_table                       = %r', plant_specs['model']['write_table'])
    logger.info('model:use_peaks                         = %r', plant_specs['model']['use_peaks'])
    logger.info('model:peak_table                        = %s', plant_specs['model']['peak_table'])
    logger.info('model:inactive_productionlines          = %s', plant_specs['model']['inactive_production_lines'])
    # Specifications to create the model
    return plant_specs



#
# Function pipeline_main
#

def pipeline_main(plant_id, project_directory, pipeline_specs, plant_specs):
    logger.info("%s Pipeline", plant_id)

    # Extract specs
    project_directory = pipeline_specs['project_directory']
    inactive_prod_lines_details = plant_specs['model']['inactive_production_lines']
    test_flag = pipeline_specs['test_flag']
    chrysler_holiday_table = pipeline_specs['holidays']['calendar_us']
    plant_code = plant_specs['plant']['code']
    use_table = plant_specs['base_table']['use_table']
    try:
        use_table_date_str = plant_specs['base_table']['use_table_date'].strftime('%Y-%m-%d').replace('-', '')
    except:
        use_table_date_str = None
    write_base_table = plant_specs['base_table']['write_table']
    group_cols = plant_specs['model']['levels']
    write_model_table = plant_specs['model']['write_table']
    pred_date_str = plant_specs['prediction_date'].replace('-', '')
    
    # Get crew maps
    crew_maps = crew_map_get_table[plant_id]()
    
    # Base Table

    if use_table or use_table_date_str:
        # We can either use our table from HRDW or integrate the table
        # from Data Engineering to avoid building a base table every time.
        logger.info("Using Existing Base Table")
        if use_table_date_str:
            prefix = 'abs'
            prefix = '_'.join([prefix, 'test']) if test_flag else prefix            
            base_table_str = str.lower('_'.join([prefix, plant_id, 'base', use_table_date_str, 'tbl']))
            file_name = '.'.join([base_table_str, 'csv'])
            base_table_file = '/'.join([project_directory, file_name])
            logger.info("Reading Base Table File: %s" % base_table_file)
            df = pd.read_csv(base_table_file)
        else:
            conn_dl, curs_dl = connect_greenplum(pipeline_specs['datalake'])
            query = "select * from " + use_table + " where ch_corploc=" + str(plant_code)
            logger.info("Query: %s" % query)
            df = create_frame_from_pg(conn_dl, use_table, query)
            logger.info("Closing connection")
            conn_dl.commit()
            curs_dl.close()         
        logger.info("Base Table Shape: %s" % (df.shape,))        
    else:
        # Create Base Table
        df = create_base_table(pipeline_specs, plant_specs)
            
    # Base Table Additions

    # Define Teams
    # Forward-pad single and double-digit supervisor groups to triple digits.
    
    logger.info("Defining Teams")
    df['team'] = df['ch_supv_grp'].astype(float).astype(int).astype(str)
    df['team'] = df['team'].str.zfill(3)

    # Assign whether or not this is a work day
    
    logger.info("Setting Work Days")
    df['pandas_day_of_week'] = pd.to_datetime(df['workdate']).dt.dayofweek
    df['is_work_day'] = df.apply(crew_work_day_table[plant_id], axis=1)
    df['is_work_day'][df['workdate'].isin(chrysler_holiday_table)] = 'N'
    df.drop(['pandas_day_of_week'], axis=1, inplace=True)

    # Set crew map
    logger.info("Defining Crew Maps")
    df = crew_map_set_table[plant_id](df)

    # Set Production Lines
    df = set_production_lines(df, plant_specs)

    # Store Base Table
    if write_base_table:
        table_tag = '_'.join([plant_id, 'base', pred_date_str])
        write_frame_to_pg(df, table_tag, pipeline_specs)

    # Create Model Table
    df_model, df_model_seq = create_model_table(df, pipeline_specs, plant_specs)

    # Store Model Table
    if write_model_table:
        table_tag = '_'.join([plant_id, 'model', pred_date_str])
        write_frame_to_pg(df_model, table_tag, pipeline_specs)
        table_tag = '_'.join([plant_id, 'model_seq', pred_date_str])
        write_frame_to_pg(df_model_seq, table_tag, pipeline_specs)
        
        
    
    # Make Predictions @ t9939vs - Switch condition to incorporate active and inactive production lines
    
    if inactive_prod_lines_details:
        logger.info("Developing Predictions Without Inactive Production lines")
        df_model_seq_temp, all_predictions, df_model_crew_seq, crew_predictions, df_compress_pred_share = \
            make_plant_predictions(pipeline_specs,
                                   plant_specs,
                                   crew_maps,
                                   df_model_seq[~df_model_seq.production_line.isin(inactive_prod_lines_details.keys())],
                                   df[~df.production_line.isin(inactive_prod_lines_details.keys())])
    else:
        logger.info("No Inactive Production lines")
        df_model_seq, all_predictions, df_model_crew_seq, crew_predictions , df_compress_pred_share= \
        make_plant_predictions(pipeline_specs,
                               plant_specs,
                               crew_maps,
                               df_model_seq,
                               df)
    logger.info("Plant Predictions Completed")
    # Joining prediction from compression with all predictions @ T9939VS
    all_predictions = pd.merge(all_predictions, df_compress_pred_share, how = 'left', left_on =['workdate']+group_cols, right_on = ['workdate']+group_cols)
    # Store Predictions
    store_predictions(plant_id, pipeline_specs, plant_specs, all_predictions, df_model_seq, group_cols)
    store_predictions(plant_id, pipeline_specs, plant_specs, crew_predictions, df_model_crew_seq, ['crew'])
    logger.info("Saving Predictions Completed ")
    return

#
# Function main
#

def main(args=None):

    r"""Main Pipeline
    
    Notes
    -----
    
    (1) Parse the command line argments.
    (2) Initialize logging.
    (3) Read the pipeline configuration.
    (4) Call the relevant plant pipeline.
    """

    # Argument Parsing

    parser = argparse.ArgumentParser(description="HR Pipeline Parser")
    parser.add_argument('--plant', dest='plant_id',
                        help="plant is a unique 4-letter identifier",
                        required=True)
    parser.add_argument('--pdate', dest='prediction_date',
                        help="date in yyyy-mm-dd format (usually the following Sunday)",
                        required=True)
    parser.add_argument('--pdir', dest='project_directory',
                        help="location of project input and output files",
                        required=True)
    parser.add_argument('--test', dest='test_flag',
                        help="test changes to the production pipeline",
                        action='store_true',
                        required=False)
    args = parser.parse_args()
    
    # Extract arguments into variables
    
    plant_id = args.plant_id
    prediction_date = args.prediction_date
    project_directory = args.project_directory
    test_flag = args.test_flag

    # Logging

    # @t8828FA change log specific to each plant and create a folder
    output_file = "/".join([project_directory,'log', plant_id+'_'+prediction_date+'.log'])
    
    
    #@ T8828FA change type of file mode to create or over write 
    logging.basicConfig(format="[%(asctime)s] %(levelname)s\t%(message)s",
                        filename=output_file, filemode='w', level=logging.DEBUG,
                        datefmt='%m/%d/%y %H:%M:%S')
    
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s",
                                  datefmt='%m/%d/%y %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    # Start the pipeline

    logger.info('*'*80)
    logger.info("HR Pipeline Start for Plant %s", plant_id)
    logger.info('*'*80)

    # Log startup parameters

    logger.info('plant_id          = %s', plant_id)
    logger.info('prediction_date   = %s', prediction_date)
    logger.info('project_directory = %s', project_directory)
    logger.info('test_flag         = %r', test_flag)

    # Read pipeline configuration file

    pipeline_specs = get_pipeline_config(project_directory)
    pipeline_specs['plant_id'] = plant_id
    pipeline_specs['project_directory'] = project_directory
    pipeline_specs['test_flag'] = test_flag
    pipeline_specs['error_logs'] = []
    # Read plant configuration file
    plant_specs = get_plant_config(pipeline_specs)
    plant_specs['prediction_date'] = prediction_date

    # Call the plant pipeline

    logger.info("Calling Pipeline for %s", plant_id)
    pipeline_main(plant_id, project_directory, pipeline_specs, plant_specs)
    
    #Writing Comments
    logger.info('*'*80)
    err_lst = pipeline_specs['error_logs']
    err_string = "\n".join(err_lst)
    logger.info("Comments")
    logger.info(err_string)
    logger.info('*'*80)
    
    # Complete the pipeline

    logger.info('*'*80)
    logger.info("HR Pipeline End")
    logger.info('*'*80)


#
# MAIN PROGRAM
#

if __name__ == "__main__":
    main()
