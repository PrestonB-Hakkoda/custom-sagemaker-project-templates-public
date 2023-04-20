# Need to create the feature store if it doesn't exist

import sagemaker
import boto3
import sys
import logging
import pandas as pd
import numpy as np
import io
import os
import time
from time import gmtime, strftime, sleep
import yaml
from pathlib import Path
import logging 
import boto3

from sagemaker.session import Session
from sagemaker import get_execution_role

from sagemaker.feature_store.feature_group import FeatureGroup, IngestionManagerPandas
from sagemaker.feature_store.feature_definition import FeatureDefinition, FeatureTypeEnum
from sagemaker.feature_store.inputs import DataCatalogConfig



print("file loaded:    ",__file__)

logger = logging.getLogger(__name__)


class FeatureStoreGenerator():
    def __init__(self, 
                 feature_group_name:str,
                 feature_bucket: str,
                 feature_bucket_prefix:str,
                 feature_record_identifier_feature_name:str,
                 feature_event_time_feature_name:str,
                 df:pd.DataFrame):
        
        """
        Creates a feature store based on the feature store config file
        (wont overwrite existing feature store)
        """
        
        os.environ["AWS_DEFAULT_REGION"] = "us-east-2"
        
        #iam = boto3.client('iam')
        # we can't get the sagemaker execution role when executing from codebuild because a different role executes
        #self.role = iam.get_role(RoleName='AmazonSageMaker-ExecutionRole-20230109T000000')['Role']['Arn']
        #self.role = iam.get_role(RoleName="AmazonSageMakerServiceCatalogProductsCodeBuildRole/Sagemaker")['Role']['Arn']
        
        # trying to manually add role:
        # did i even need this lol?
        self.role = "arn:aws:iam::644944822023:role/AmazonSageMaker-ExecutionRole-20230109T000000"
        
        # only works in sagemaker notebooks
        #self.role = get_execution_role()
        #self.sagemaker_session = sagemaker.Session()
        
        self.sagemaker_session = sagemaker.Session(boto3.session.Session())
        self.region = self.sagemaker_session.boto_region_name
        self.s3_bucket_name = self.sagemaker_session.default_bucket()
        self.feature_group_name = feature_group_name
        self.feature_bucket = feature_bucket
        self.feature_bucket_prefix = feature_bucket_prefix
        self.feature_record_identifier_feature_name = feature_record_identifier_feature_name
        self.feature_event_time_feature_name = feature_event_time_feature_name
        
        
        self.df = df
        
        
        

        
    def check_for_existing_feat_stores(self) -> bool:
        feature_groups_dict = self.sagemaker_session.boto_session.client(
                                "sagemaker", region_name=self.region
                                ).list_feature_groups() 

        feat_group_names = [group['FeatureGroupName'] for group in feature_groups_dict['FeatureGroupSummaries']]

        if self.feature_group_name in feat_group_names:
            group_exists = True
            
            logger.info(f"Feature Group Already Exists: {self.feature_group_name}")

        else:
            group_exists = False

        return group_exists

    
    

    def create_feature_group(self):
        
        group_exists = self.check_for_existing_feat_stores()
        
        if not group_exists:
            s3_uri = f"s3://{self.feature_bucket}/{self.feature_bucket_prefix}"

            # data_catalog_config = DataCatalogConfig(table_name = self.conf['table_info']['table_name'],
            #                                         catalog = self.conf['data_location_info']['catalog'],
            #                                         database = self.conf['data_location_info']['database'])


            feat_group = FeatureGroup(name = self.feature_group_name,
                                       sagemaker_session = self.sagemaker_session)


            # defining the feature group as the processed dataframe data
            feat_group.load_feature_definitions(data_frame=self.df)

            feat_group.create(
            s3_uri = s3_uri,
            record_identifier_name = self.feature_record_identifier_feature_name,
            event_time_feature_name = self.feature_event_time_feature_name,
            role_arn = self.role,
            enable_online_store = True,
            disable_glue_table_creation = False,
            data_catalog_config = None
            #data_catalog_config = data_catalog_config,
            #disable_glue_table_creation = True
            )
            
        else:
            
            feat_group = FeatureGroup(name=self.feature_group_name,
                                      sagemaker_session=self.sagemaker_session)
            
        
        self.feat_group = feat_group

        return feat_group

        
        
        
    @staticmethod
    def check_feature_group_status(feature_group):
        status = feature_group.describe().get("FeatureGroupStatus")
        while status == "Creating":
            print(f"Waiting for feature group {feature_group.name} to be created.....")
            time.sleep(5)
            status = feature_group.describe().get("FeatureGroupStatus")
        print(f"FeatureGroup {feature_group.name} status: {status}")
    
    
    


    def test_feat_group(self) -> (bool, dict):
        
        # feature store needs to be created before tests can be ran:
        FeatureStoreGenerator.check_feature_group_status(self.feat_group)
        
        
        group_exists = self.check_for_existing_feat_stores()
        
        # if feature group has to be created:
        if group_exists:
            logger.info("Feature Group Created")
        
        
        exmaple_id = str(self.df[self.feature_record_identifier_feature_name].iloc[0])
    
        featurestore_runtime = self.sagemaker_session.boto_session.client(service_name='sagemaker-featurestore-runtime', region_name=self.region)
        
        example_record = featurestore_runtime.get_record(FeatureGroupName=self.feature_group_name, RecordIdentifierValueAsString=exmaple_id)
        
        logger.info(f"example record:    {example_record}")
        
        
        return group_exists, example_record
    
    
    
    
    
    def get_current_records(self) -> pd.DataFrame:
        
        default_s3_bucket_name = self.sagemaker_session.default_bucket()

        feat_query = self.feat_group.athena_query()

        # get db and table for feature group
        feat_table = feat_query.table_name
        feat_db = feat_query.database
        
        query_string = f""" SELECT * FROM "{feat_db}"."{feat_table}"; """
        
        dataset = pd.DataFrame()

        feat_query.run(query_string=query_string, output_location='s3://'+default_s3_bucket_name+'/query_results/')
        feat_query.wait()
        
        dataset = feat_query.as_dataframe()
        
        print(f"Existing records df:   {dataset.head()}")
        
        return dataset
    
    
    
    
    
    def load_feature_group(self,  max_workers:int = 3, wait:bool = True):
        
        # check current records against new records
        # For this case we won't reload any records already in the feature store (ex: maybe append only records)
        feature_store_records_df = self.get_current_records()
        
        id_col = self.feature_record_identifier_feature_name
        record_ids = feature_store_records_df[id_col].values

        new_records_df = self.df[~self.df[id_col].isin(record_ids)]
        
        print(f"New records df:   {new_records_df.head()}")
        print(f"Num records to add:     {len(new_records_df)}")
        
        ingestion_manager = self.feat_group.ingest(data_frame=new_records_df, max_workers=max_workers, wait=wait)
        
        print(f"Num records to added:     {len(new_records_df)}")
        
        return ingestion_manager
        
        
        

              
              
              
    
if __name__ == "__main__":
    
    # testing creation:
    feature_group_name = 'foot-traffic-test-feature-group-processed'
    feature_bucket = 'sagemaker-us-east-2-644944822023'
    feature_bucket_prefix = 'FootTrafficTestTransform/feature_store/'
    feature_record_identifier_feature_name ='index'
    feature_event_time_feature_name = 'RecordsAddedTime'
    

    example_record = {
                        'index': [500000],
                        'normalized_visits_by_state_scaling': [200],
                        'month': [3],
                        'week_of_month': [1],
                        'usf_div_nbr': [2000],
                        'sum': [50.2],
                        'RecordsAddedTime': [1677000326.0]
                    }
    
    df = pd.DataFrame(example_record)
    
    feat_store_generator = FeatureStoreGenerator(feature_group_name,
                                                feature_bucket,
                                                feature_bucket_prefix,
                                                feature_record_identifier_feature_name,
                                                feature_event_time_feature_name,
                                                df)
    
    feat_group = feat_store_generator.create_feature_group()
              
    group_exists, example_record = feat_store_generator.test_feat_group()
    
    feat_store_generator.load_feature_group()
