## Data Eng Processing:
#   - Get rid of duplicate columns
#   - Add some feature columns

# features to get:
# - normalized_visits_by_state_scaling
# - month
# - week of the month
# - usf_div_nbr

# target: sum

# dataset date range: 01/02 - 03/06


import argparse
import requests
import tempfile

import pandas as pd
import numpy as np
import sys
import os
import io
import subprocess
import time
from time import gmtime, strftime, sleep
from pathlib import Path
import logging 
import platform
import ast


# double check pythnon version
print("python version:     ", platform.python_version())



# ---------------     install extras     ---------------
# An alternative to this would be to create an image with the dependencies already installed


# Look at directory structure to verify:
print([x[0] for x in os.walk("/opt/ml/processing")])
print([x for x in os.walk("/opt/ml/processing")])

print(f"working directory:     {os.getcwd()}")
print(f"system path:     {sys.path}")
print(f"current file:    {__file__}")
print()

# install requirements from requirements .txt 
# Note: Python 3.8 required (only for Snowpark)

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-r",
    "/opt/ml/processing/pipeline_reqs/01_data_eng_reqs.txt",
])

# Install deps from requirements.txt
import yaml
import sagemaker
from sagemaker.session import Session
from sagemaker import get_execution_role
from sagemaker.feature_store.feature_group import FeatureGroup, IngestionManagerPandas
from sagemaker.feature_store.feature_definition import FeatureDefinition, FeatureTypeEnum
from sagemaker.feature_store.inputs import DataCatalogConfig
import boto3
from botocore.exceptions import ClientError

# Needed for snowflake data source
from snowflake.snowpark import Session as sf_session
from snowflake.snowpark.types import IntegerType, StringType, StructType, StructField, BooleanType
from snowflake.snowpark.functions import col, when, replace, sproc
import snowflake.snowpark as sp
from cryptography.fernet import Fernet
import snowflake.connector



# Import feature store creation module
# from .opt.ml.processing.feature_store.create_feature_store import FeatureStoreGenerator
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname("/opt/ml/processing/feature_store"))
sys.path.append("/opt/ml/processing/feature_store")
print(f"system path:     {sys.path}")

from create_feature_store import FeatureStoreGenerator



# Selected Features -> Move to config file
# records added time required for feature store
features = ['normalized_visits_by_state_scaling', 'month', 'week_of_month', 
            'usf_div_nbr', 'RecordsAddedTime']

target = 'sum'





#  -----------------------     get credentials for Snowflake from Secrets Manager      ----------------------- 

    
def get_snowflake_creds(secret_name: str) -> dict:

    secret_name = secret_name
    region_name = "us-east-2"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    # Decrypts secret using the associated KMS key.
    # secret is return as a string, want a dict
    secret = ast.literal_eval(get_secret_value_response['SecretString'])
    
    return secret
    
    



# -----------------------     Feature Processing Functions      -----------------------

# Doing any feature generation work

def generate_foot_traffic_features(foot_traffic_df:pd.DataFrame, features:list, target:str) -> pd.DataFrame:
    
    features.append(target)
    
    foot_traffic_df['month'] = pd.to_datetime(foot_traffic_df['purchase_week']).dt.month
    foot_traffic_df['week_of_year'] = pd.to_datetime(foot_traffic_df['purchase_week']).dt.weekofyear
    foot_traffic_df['week_of_month'] = foot_traffic_df['week_of_year'] - (4 * (foot_traffic_df['month'] - 1))
    
    current_time_sec = int(round(time.time()))
    foot_traffic_df['RecordsAddedTime'] = current_time_sec
    foot_traffic_df['RecordsAddedTime'] = foot_traffic_df['RecordsAddedTime'].astype('float64')
    
    feature_df = foot_traffic_df[features]
    
    feature_df = feature_df.reset_index()
    
    return feature_df













# -----------------------    Functions For Sagemaker Feature Store     -------------------------  

def feature_filter(model_inupt: pd.DataFrame, features:list, target:str) -> pd.DataFrame:
    X_y = model_inupt[features+[target]]
    return X_y

    
    
    
def save_to_sagemaker_feature_store(model_input:pd.DataFrame,  
                                                 feature_group_name:str,
                                                 feature_bucket:str,
                                                 feature_bucket_prefix:str,
                                                 feature_record_identifier_feature_name:str,
                                                 feature_event_time_feature_name:str) -> None:
    
    feat_store_generator = FeatureStoreGenerator(
                                                 feature_group_name,
                                                 feature_bucket,
                                                 feature_bucket_prefix,
                                                 feature_record_identifier_feature_name,
                                                 feature_event_time_feature_name,
                                                 model_input
                                                )
    
    feat_group = feat_store_generator.create_feature_group()
              
    group_exists, example_record = feat_store_generator.test_feat_group()
    
    feat_store_generator.load_feature_group()
    
    return
    
    
    

    
    
    
    
# -----------------------     Functions For Snowflake  Feature Store     -----------------------  

# need to update from pointing at the conf dict

def create_sf_session_from_secret(secret_name:str):
    
    sf_conn_params = get_snowflake_creds(secret_name)
    
    session =  sf_session.builder.configs(sf_conn_params).create() 
    
    return session


def create_sf_conn_from_secret(secret_name:str):
    
    sf_conn_params = get_snowflake_creds(secret_name)
    conn = snowflake.connector.connect(**sf_conn_params)
    
    return conn


def create_session_from_conf(sf_conf_path:str, key:str):
    #ex: 
    # sf_conf_path = '/opt/ml/processing/credentials/credentials.yml'
    # key = b'IzMGv8_9vq1ui6wMxCWcPdXZDpoDXDmG-6arzRvPVtk='
    
    # could potentially encode then read connection parameters
    # using AWS secrets will be prefered though
    
    sf_conf = yaml.safe_load(Path(sf_conf_path).read_text())

    print("encrypted token:    ", sf_conf['snowflake_dev']['password'])
    print("encrypted token type:    ", type(sf_conf['snowflake_dev']['password']))

    f = Fernet(key)
    sf_pswrd = str(f.decrypt(bytes(sf_conf['snowflake_dev']['password'], 'utf-8')), 'utf-8')
    sf_conf['snowflake_dev'].update({"password": sf_pswrd})
    
    session =  sf_session.builder.configs(sf_conf['snowflake_dev']).create() 
    
    return session



def populate_sf_feature_store(conf:dict, model_input_df:pd.DataFrame, session, store_exists:bool):
    # create feature store if not exists
    
    database = conf['feature_store']['snowflake']['database']
    schema = conf['feature_store']['snowflake']['schema']
    table = conf['feature_store']['snowflake']['table']
    id_col = conf['feature_store']['snowflake']['record_identifier_feature_name']
    
    if store_exists:
        # If feature store exists:
        # add data based on new 'flight_ids'
        print("feature store exists: adding new records")
        feature_store_query = f"""select "{id_col}" from "{database}"."{schema}"."{table}" """
        feature_store_df = session.sql(feature_store_query).to_pandas()
        unique_feat_store_ids = feature_store_df[id_col].unique()
        
        new_records_df = model_input_df[~model_input_df[id_col].isin(unique_feat_store_ids)]
        
        print(f"inserting {len(new_records_df)} new records")
        
        # don't try to insert if no new records
        if len(new_records_df) == 0:
            print("no records to insert")
            pass
        else:
            session.write_pandas(new_records_df, table, overwrite=False)
        
        
        
    else:
        # Else add all of the new data + create feature store
        print("feature store does not exist: creating new feature store")
        session.write_pandas(model_input_df, table, auto_create_table=True, overwrite=True)
        print("created table")
    
    
    return
    
    
    
def save_to_snowflake_feature_store(conf:dict, model_input:pd.DataFrame, session):
    # DF -> Snowflake Table (CDC enabled)
    
    # check if feat store exists
    print("saving to snowflake")
    
    database = conf['feature_store']['snowflake']['database']
    schema = conf['feature_store']['snowflake']['schema']
    table = conf['feature_store']['snowflake']['table']

    print("querying information schema")
    table_exists_query = f"""select * from {database}.INFORMATION_SCHEMA.TABLES where table_schema = '{schema}' and table_name = '{table}' """
    available_tables_df = session.sql(table_exists_query).to_pandas()
    
    # set session info:
    session.sql(f""" use database {database} """).collect()
    session.sql(f""" use schema {schema} """).collect()

    if len(available_tables_df) == 0:
        store_exists = False
        
    if len(available_tables_df) == 1:
        store_exists = True
        
    
    populate_sf_feature_store(conf, model_input, session, store_exists)
    
    
    # add validation later -> check if feature store schema == schema of the snowflake table (raise error)
    
    return
    
    
    

    
    
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    
    # this is where the files get added to in the processing job container
    base_dir = "/opt/ml/processing"
    
    
    
    # -----------------------       load logic        ------------------------
    print("loading test data")
    
    data_source = "s3"
    feature_store = "sagemaker"
    
    # vars for feature store
    # note - current dataset doesn't have a unique index, so using pandas index as identifier
    
    feature_group_name = 'foot-traffic-test-feature-group-processed'
    feature_bucket = 'sagemaker-us-east-2-644944822023'
    feature_bucket_prefix = 'FootTrafficTestTransform/feature_store/'
    feature_catalog = 'AwsDataCatalog'
    feature_database = 'sagemaker_featurestore'
    feature_table_name = 'foot-traffic-test-feature-group'
    feature_record_identifier_feature_name ='index'
    feature_event_time_feature_name = 'RecordsAddedTime'

    
    
    print(f"data source:  {data_source}")
    print(f"feature store: {feature_store}")
    
    
    if data_source == 's3':
        foot_traffic_df = pd.read_csv(f"{base_dir}/pipeline_input/ft_cases_per_week.csv")

        
    # set up in Snowflake version
    elif data_source == 'snowflake':
        
        # create connection: 
        # session suddenly stopped working, eventhough no changes to code.... 
        # (exact same code works in diff environment, so somehow env is causing issues)
        session = create_sf_session_from_secret("hakkoda_snowflake_user")
        #  use conn instead to see if that works
        conn = create_sf_conn_from_secret("hakkoda_snowflake_user")
        
        base_schema = 'DEMO_DB.MLOPS'
        companies_sql = f""" select * from {base_schema}.companies """
        print(companies_sql)
        #companies = session.sql(companies_sql).to_pandas()
        companies = pd.read_sql(companies_sql, conn)
        companies.columns = [col.lower() for col in companies.columns.tolist()]
        
        shuttles_sql = f""" select * from {base_schema}.shuttles """
        #shuttles = session.sql(shuttles_sql).to_pandas()
        shuttles = pd.read_sql(shuttles_sql, conn)
        shuttles.columns = [col.lower() for col in shuttles.columns.tolist()]
        
        reviews_sql = f""" select * from {base_schema}.reviews """
        #reviews = session.sql(reviews_sql).to_pandas()
        reviews = pd.read_sql(reviews_sql, conn)
        reviews.columns = [col.lower() for col in reviews.columns.tolist()]
        
    else:
        raise ValueError(f"""Accepted data sources are: 's3' and 'snowflake'. Provided data source specification: {data_source}""")
    


    # -----------------------    transform logic    --------------------------
    print("transforming data")

    model_input = generate_foot_traffic_features(foot_traffic_df, features, target)
    
    
    
    
    
    
    
    
    
    
    
    # --------------------    Save Data to Feature Store  --------------------
    if feature_store == 'sagemaker':
        save_to_sagemaker_feature_store(model_input, 
                                        feature_group_name,
                                        feature_bucket, 
                                        feature_bucket_prefix, 
                                         feature_record_identifier_feature_name,
                                         feature_event_time_feature_name
                                        )
        
    elif feature_store == 'snowflake':
        # create connection: 
        session = create_sf_session_from_secret("hakkoda_snowflake_user")
        
        save_to_snowflake_feature_store(conf, model_input, session)
        
    else:
        raise ValueError(f"""Accepted feature stores are: 'sagemaker' and 'snowflake'. Provided data source specification: {feature_store}""")

    
    model_input_filtered = feature_filter(model_input, features, target)
    
    print(model_input_filtered.head())
    
    # Save a backup to S3
    model_input_filtered.to_csv(f"{base_dir}/processed/model_input.csv", index=False)
    
    
    
    