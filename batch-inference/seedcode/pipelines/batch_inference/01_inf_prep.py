# processing job for getting data ready for Batch transform job from Feature Store

import argparse
import os
import requests
import tempfile
from typing import Tuple
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import io
import subprocess
import time
from time import gmtime, strftime, sleep
from pathlib import Path
import logging 
import ast


# install extras

#  looking at the files in the container for troubleshooting:
# print([x[0] for x in os.walk("/opt/ml/processing")])
# print([x for x in os.walk("/opt/ml/processing")])

# print(f"working directory:     {os.getcwd()}")
# print(f"system path:     {sys.path}")
# print(f"current file:    {__file__}")
# print()


# Intall additional dependencies:
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-r",
    "/opt/ml/processing/pipeline_reqs/01_inf_prep_reqs.txt",
])



# need to install deps before sagemaker can be imported
import sagemaker
from sagemaker.session import Session
from sagemaker import get_execution_role
from sagemaker.feature_store.feature_group import FeatureGroup, IngestionManagerPandas
from sagemaker.feature_store.dataset_builder import DatasetBuilder
from sagemaker.feature_store.feature_definition import FeatureDefinition, FeatureTypeEnum
from sagemaker.feature_store.inputs import DataCatalogConfig
import boto3

# Needed for snowflake data source
from snowflake.snowpark import Session
from snowflake.snowpark.types import IntegerType, StringType, StructType, StructField, BooleanType
from snowflake.snowpark.functions import col, when, replace, sproc
import snowflake.snowpark as sp
from cryptography.fernet import Fernet


# make sure sagemaker session is available for reading feature store
os.environ["AWS_DEFAULT_REGION"] = "us-east-2"
sagemaker_session = sagemaker.Session(boto3.session.Session())
region = sagemaker_session.boto_region_name
s3_bucket_name = sagemaker_session.default_bucket()




# --------------------------    Sagemaker Feature Store    ---------------------------------

def read_from_sagemaker_feature_store(feat_group_name:str, selected_features:list) -> pd.DataFrame:
    """
    Read from the feature store using select query (athena)
    Athena query requires S3 bucket to hold results
    """
    processed_feat_group = FeatureGroup(name=feat_group_name, sagemaker_session=sagemaker_session)
    default_s3_bucket_name = sagemaker_session.default_bucket()

    feat_query = processed_feat_group.athena_query()

    feat_table = feat_query.table_name
    feat_db = feat_query.database
    feat_catalog = feat_query.catalog
    

    query_string = f"""SELECT * FROM "{feat_db}"."{feat_table}";"""


    dataset = pd.DataFrame()

    feat_query.run(query_string=query_string, output_location='s3://'+default_s3_bucket_name+'/query_results/')
    feat_query.wait()
    
    dataset = feat_query.as_dataframe()
    
    print(f"sample of dataset:    {dataset.head()}")
    print(f"dataset columns:  {dataset.columns}")
    
    dataset = dataset[selected_features]
    
    return dataset


# --------------------------      Snowflake Feature Store    -------------------------- 
# go back and update conf

def _get_snowflake_creds(secret_name: str) -> dict:

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


def _create_sf_session_from_secret(secret_name:str):
    
    sf_conn_params = _get_snowflake_creds(secret_name)
    
    session =  Session.builder.configs(sf_conn_params).create() 
    
    return session



def read_from_snowflake_feature_store(secret_name:str, conf:dict,  selected_features:list) -> pd.DataFrame:
    
    database = conf['feature_store']['snowflake']['database']
    schema = conf['feature_store']['snowflake']['schema']
    table = conf['feature_store']['snowflake']['table']
    
    session = _create_sf_session_from_secret(secret_name)
    
    feature_store_query = f"""select * from "{database}"."{schema}"."{table}" """
    dataset = session.sql(feature_store_query).to_pandas()
    
    print(f"sample of dataset:    {dataset.head()}")
    
    dataset = dataset[selected_features]
    
    return dataset
    
    
    




# --------------------------     Transform logic to get most recent data for inference     -------------------------- 

def get_new_data(data: pd.DataFrame, features:list) -> pd.DataFrame:
    
    """Get the most recent data from feature store for batch inference
    
    Args:
        data: Data containing features and target 
        features: features names for training
    Returns:
        dataframe with data for inference
    """
    
    # Could get more advanced here with looking at insert times
    
    data_batch_inf = data[data['month'] == 3]
    
    
    batch_inf = data_batch_inf[features]
    
    
    return batch_inf
    




if __name__ == "__main__":
    
    base_dir = "/opt/ml/processing"
    feature_store = "sagemaker"
    feature_group_name = 'foot-traffic-test-feature-group-processed'
    
    # snowflake only:
    secrets_name = "hakkoda_snowflake_user"

    selected_features = ['normalized_visits_by_state_scaling',
                         'month', 'week_of_month', 
                         'usf_div_nbr', 'recordsaddedtime', 'sum']

    
    print("loading test data")
    

    # load logic from feature store
    if feature_store == 'sagemaker':
        
        print("using data from Sagemaker feature store")
        data = read_from_sagemaker_feature_store(feature_group_name, selected_features)
        
    elif feature_store == 'snowflake':

        print("using data from Snowflake feature store")
        secret_name = conf['project']['secrets']['snowflake']
        data = read_from_snowflake_feature_store(secret_name, conf,  selected_features)
        
    else:
        raise ValueError(f"""Accepted feature stores are: 'sagemaker' and 'snowflake'. Provided data source specification: {feature_store}""")

    
    
    print("loaded data: ", data.head())
    
    
    print("transforming data")
    # transform logic
    batch_inf = get_new_data(data, ['normalized_visits_by_state_scaling',
                                                 'month', 'week_of_month', 
                                                 'usf_div_nbr'])
    
    # logic for baseline + batch transform jobs
    
    
    
    # would use logging later - 
    print(batch_inf.head())
    
    # save logic
    batch_inf.to_csv(f"{base_dir}/batch_inf/batch_inf_no_target.csv", header = False, index=False)
    
    
    



