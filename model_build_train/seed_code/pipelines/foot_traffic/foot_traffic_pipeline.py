"""Example workflow pipeline script for foot traffic pipeline.


                  baseline conditional -> (calculate baseline or skip baseline calc)  
                  
                                                                               . register model
                                                                               . create model       --> batch transform
                                                                               . data quality check --> batch transform
                                                                              .
    processing -> test prep ------------> training -> eval -> mse condition  .
                                                                              .
                                                                               . stop

    
    processing           - processes and loads processed data to feature store
    test prep            - splits data for training 
    baseline conditional - calculates baseline for data quality monitoring
    training             - training step
    eval                 - model evaluation
    mse condition        - check the mse against a specified threshold
    register model       - Save model to the model registry
    create model         - Create the model based on the training job
    data quality check   - Check the data quality against the baseline metrics
    batch transform      - Test a batch transform job
    


Implements a get_pipeline(**kwargs) method.


I'd really like to break this code up more, but a lot of it is interconnected.
It may make things more confusing if various peices were in different files because of the interconnectivity. 

"""



import sagemaker
import boto3
import sagemaker.amazon.common as smac
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.transformer import Transformer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.serializers import CSVSerializer

import pandas as pd
import numpy as np

import os
import json
import re
#import yaml
from pathlib import Path
import time


# For Pipeline Parameters:
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
    ParameterFloat,
)


# For Processors
from sagemaker.sklearn.processing import SKLearnProcessor # just using since its easy
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ScriptProcessor

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, TuningStep
from sagemaker.workflow.properties import PropertyFile
from sagemaker.tuner import ContinuousParameter, HyperparameterTuner, CategoricalParameter



# For Model Handling Steps
from sagemaker.model import Model
from sagemaker.inputs import CreateModelInput
from sagemaker.workflow.model_step import ModelStep
from sagemaker.inputs import TransformInput
from sagemaker.workflow.steps import TransformStep
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.monitor_batch_transform_step import MonitorBatchTransformStep

from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.quality_check_step import DataQualityCheckConfig



# For Conditional Steps
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import Join

from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo, ConditionEquals
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet


# For the pipeline
from sagemaker.workflow.pipeline import Pipeline



BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# move this
#base_uri = f"s3://{default_bucket}/foot_traffic_test"

# --------------------------------------------------------------------------------------------------
# -----------------------     Helper functions for Sagemaker Connections    ------------------------
# --------------------------------------------------------------------------------------------------


def _get_sagemaker_client(region):
    """Gets the sagemaker client.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker_client

    

def _get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def _get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def _get_pipeline_custom_tags(new_tags, region, sagemaker_project_name=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.describe_project(ProjectName=sagemaker_project_name)
        sagemaker_project_arn = response["ProjectArn"]
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags








# --------------------------------------------------------------------------------------------------
# -----------------------     Helper function for Processing Instances     -------------------------
# --------------------------------------------------------------------------------------------------


def _construct_processors(pipeline_session,
                          sagemaker_session,
                          role,
                          estimator_output_path:str = "s3://mlops-sagemaker-testing-usf/FootTrafficTrain",
                          test_eng: bool = False,
                          test_prep:bool = False,
                          test_eval:bool = False,
                          test_baseline:bool = False,
                         ) -> tuple:
    """defines and gets the processor objects 
    
    If the sagemaker session is applied in stead of the pipeline_session,
    the job will be kicked off instead of being defined in the pipeline.
    This can be used for testing purposes to run single processing jobs. 
    
    Args:
        pipeline_session:  pipeline session (helps create pipeline)
        sagemaker_session: sagemaker session (needed for testing)
        role:              sagemaker execution role
        
        
    Returns:
        A tuple of all processors defined
        (in this case - data_eng_processor, train_prep_processor,
        xgb_train, eval_processor, baseline_processor)
    
    """
    # -----------------    Data Eng/Prepprocessing Job    ----------------

    eng_processor_conf = {
            'role': role,
            'instance_type': "ml.m5.large", # ml.m5.large are typically the fastest instances to launch
            'instance_count': 1,
            "base_job_name": "data-eng-processing",
            "framework_version": "1.0-1", # 0.23-1, 1.0-1
            "sagemaker_session": pipeline_session
    }

    if test_eng:
        eng_processor_conf.update({"sagemaker_session": sagemaker_session})


    # conf['data_eng']['processor'].update({"sagemaker_session":pipeline_session}) # <-  this is an example of using a config file instead of inline code
    # moving the config to a yaml file can help clean up the clutter in the code

    data_eng_processor = SKLearnProcessor(**eng_processor_conf)



    # Processing job - to create the training datasets
    # This could potentially be done with args to the training job instead to save some time
    # However, if the feature store will be used for other models with different processing techniques (ex: encoding teqniques) you'll need this job
    # Also, I like how this is more explicit

    train_prep_processor_conf = {
            "role": role,
            "instance_type": "ml.m5.large",
            "instance_count": 1,
            "base_job_name": "training-prep-processing",
            "framework_version": "1.0-1", # 0.23-1, 1.0-1
            "sagemaker_session": pipeline_session
    }

    if test_prep:
        train_prep_processor_conf.update({"sagemaker_session": sagemaker_session})


    train_prep_processor = SKLearnProcessor(**train_prep_processor_conf)



    # Training job ("Estimator")
    # I'm using xgboost instead of the linear estimator
    # Typically xgboost provides superior performance for regression and classification tasks

    estimator_image_conf = {
            "framework": "xgboost",
            "region": "us-east-2",
            "version": "1.0-1",
            "py_version": "py3",
            "instance_type": "ml.m5.large"
    }

    image_uri = sagemaker.image_uris.retrieve(**estimator_image_conf)


    estimator_conf = {
        "output_path": estimator_output_path,
        "instance_type": "ml.m5.large",
        "instance_count": 1,
        "role": role,
        "image_uri": image_uri,
        "sagemaker_session": pipeline_session
    }


    xgb_train = Estimator(**estimator_conf)

    # Hyperarameters must be supplied
    estimator_hyper_params_conf = {
            "eval_metric": "rmse",
            "objective": "reg:squarederror",
            "num_round": 50,
            "max_depth": 5,
            "eta": 0.2,
            "gamma": 4,
            "min_child_weight": 6,
            "subsample": 0.7
    }

    xgb_train.set_hyperparameters(**estimator_hyper_params_conf)



    # Eval Job
    # Finally create the processing job for evaluating the model results
    # The results of this step can be used to conditionally register the model

    eval_processor_conf = {
            "role": role,
            "instance_type": "ml.m5.large",
            "instance_count": 1,
            "base_job_name": "eval-processing",
            "command": ["python3"],
            "sagemaker_session":pipeline_session,
            "image_uri": image_uri
    }

    if test_eval:
        eval_processor_conf.update({"sagemaker_session": sagemaker_session})

    eval_processor = ScriptProcessor(**eval_processor_conf)




    # Baseline Job
    # The baseline job does not need to run every time the pipeline runs
    # The results will be used to check the inputs to the batch transform jobs

    baseline_conf = {
            "role": role,
            "instance_type": "ml.m5.large",
            "instance_count": 1,
            "base_job_name": "baseline-compute",
            "command": ["python3"],
            "sagemaker_session":pipeline_session,
            "image_uri": image_uri
    }

    if test_baseline:
        baseline_conf.update({"sagemaker_session": sagemaker_session})


    baseline_processor = ScriptProcessor(**baseline_conf)
    
    
    
    
    
    return data_eng_processor, train_prep_processor, xgb_train, eval_processor, baseline_processor







# --------------------------------------------------------------------------------------------------
# --------------------------     Helper function for Pipeline Steps    -----------------------------
# --------------------------------------------------------------------------------------------------


def _create_pipeline_steps(
                           data_eng_processor,
                           train_prep_processor,
                           xgb_train,
                           eval_processor,
                           baseline_processor,
                           evaluation_report,
                           default_bucket,
                           test_eval:bool = False,
                           model_prefix = "FootTrafficTrain",
                           code_location_process = "02_foot-traffic-test-pipeline-extras/jobs/01_data_eng_s3.py",
                           code_location_train = "02_foot-traffic-test-pipeline-extras/jobs/02_training_prep.py",
                           eval_code_location = "02_foot-traffic-test-pipeline-extras/jobs/03_eval.py",
                           baseline_code_location = "02_foot-traffic-test-pipeline-extras/jobs/04_baseline_monitoring.py",
                           input_dataset_s3 = 's3://mlops-sagemaker-testing-usf/foot_traffic_test/data/pipeline_input/ft_cases_per_week_fake.csv',
                           intermediate_dataset_prefix = "foot-traffic",
                            ) -> tuple:
    """Define the core pipeline steps
    
    This consists of specifying a lot of i/o paths for the processors. 
    Some of this can be parameterized later
    *Note that the processing code needs to line up with these paths
    
    Args:
        1. Processing objects from the previous step
        2. Code location for each step
    
    Returns:
        A tuple of pipeline steps
        (step_process, step_prep, step_train, step_eval, step_baseline)
        
        
    """
    
    # ------------------------        Data Eng Step       ----------------------------

    # The data input will be the S3 location of the 
    # The "destination" is the location in the job container where the data will be loaded to

    # *note the file names and pathing should correspond with the file names + paths in the jobs

    data_eng_inputs_list = [
        {
            "foot_traffic": {
                "source": input_dataset_s3,
                "destination": "/opt/ml/processing/pipeline_input"
            }
        },
        {
            "data_eng_reqs": {
                "source": 's3://mlops-sagemaker-testing-usf/foot-traffic/requirements/01_data_eng_reqs.txt',
                "destination": "/opt/ml/processing/pipeline_reqs"
            }
        },
        {
            "feature_store_script": {
                "source": 's3://mlops-sagemaker-testing-usf/foot-traffic/requirements/create_feature_store.py',
                "destination": "/opt/ml/processing/feature_store"
            }
        }
    ]

    # Output will be saved to the feature store
    # The output added here is just for example

    # we need to include the requirements file to install any additional specified requirements
    # we need to include the feature store script to conditionally create it if it doesn't exist

    data_eng_output_list = [ 
        {
            "processed_foot_traffic": {
                "output_name": "model_input",
                "source": "/opt/ml/processing/processed",
                "destination": f's3://mlops-sagemaker-testing-usf/{intermediate_dataset_prefix}/data/processed/model_input.csv'
            }
        }
    ]

    data_eng_inputs =  [ ProcessingInput(**list(item.values())[0]) for item in data_eng_inputs_list ]
    data_eng_outputs = [ ProcessingOutput(**list(item.values())[0]) for item in data_eng_output_list ]
    #print(data_eng_inputs)

    data_eng_args = data_eng_processor.run(
            inputs = data_eng_inputs,
            outputs = data_eng_outputs,
            code = code_location_process
    )
        

    step_process = ProcessingStep(name="FootTrafficProcessing", step_args=data_eng_args)







    # ------------------------        Training Prep Step      ----------------------------


    train_prep_inputs_list = [
        # I'm passing this model input only in case I need it for validation purposes
        # In the training prep script the input data is taken from the feature store and not this file
        {
            "data_input": {
                "source": f's3://mlops-sagemaker-testing-usf/{intermediate_dataset_prefix}/data/processed/model_input.csv',
                "destination": "/opt/ml/processing/data"
            }
        },
        {
            "train_prep_reqs": {
                "source": 's3://mlops-sagemaker-testing-usf/foot-traffic/requirements/02_training_prep_reqs.txt',
                "destination": "/opt/ml/processing/pipeline_reqs"
            }
        },
    ]
    train_prep_outputs_list = [
        {
            "train": {
                "output_name": "train",
                "source": "/opt/ml/processing/train",
                "destination": f's3://mlops-sagemaker-testing-usf/{intermediate_dataset_prefix}/data/train/train.csv'}
        },
        {
            "test": {
                "output_name": "test",
                "source": "/opt/ml/processing/test",
                "destination": f's3://mlops-sagemaker-testing-usf/{intermediate_dataset_prefix}/data/test/test.csv'}
        },
        {
            "val": {
                "output_name": "val",
                "source": "/opt/ml/processing/val",
                "destination": f's3://mlops-sagemaker-testing-usf/{intermediate_dataset_prefix}/data/val/val.csv'}
        },

        # We also need the data in a certain format to create the baseline
        # This format should essentially be the same as the batch transform input, but with headers
        {
            "baseline_input": {
                "output_name": "baseline",
                "destination": f's3://mlops-sagemaker-testing-usf/{intermediate_dataset_prefix}/data/baseline/batch_inf_no_target_with_header.csv',
                "source": "/opt/ml/processing/baseline"
            }
        },
        # could also filter out the target column with some args
        # However, I like having the headers on the data as long as possible, and in some cases only data without headers can be passed to the Estimator
        {
            "batch_transform_input": {
                "output_name": "batch_inf",
                "destination": f's3://mlops-sagemaker-testing-usf/{intermediate_dataset_prefix}/data/batch_inf/batch_inf_no_target.csv',
                "source": "/opt/ml/processing/batch_inf"
            }
        },
    ]

    

    train_prep_inputs = [ ProcessingInput(**list(item.values())[0]) for item in train_prep_inputs_list ]
    train_prep_outputs = [ ProcessingOutput(**list(item.values())[0]) for item in train_prep_outputs_list ]

    train_prep_args = train_prep_processor.run(
        inputs = train_prep_inputs,
        outputs = train_prep_outputs,
        code = code_location_train
    )


    step_prep = ProcessingStep(name="FootTrafficPrep", step_args=train_prep_args, depends_on = ["FootTrafficProcessing"])







    # ------------------------        Tunning/Training Step      ----------------------------

    # good resource for hyperparameter turning:
    # https://github.com/aws/amazon-sagemaker-examples/blob/main/sagemaker-pipelines/tabular/tuning-step/sagemaker-pipelines-tuning-step.ipynb


    metric_definitions = [
        {
            "Name": "mse",
            "Regex": "mse: ([0-9\\.]+)",
        }
    ]

    objective_metric_name = "validation:rmse"
    objective_type = "Minimize"

    hyperparameter_ranges = {
        "alpha": ContinuousParameter(0.01, 10, scaling_type="Logarithmic"),
        "lambda": ContinuousParameter(0.01, 10, scaling_type="Logarithmic"),
    }

    tuner = HyperparameterTuner(
        xgb_train,
        objective_metric_name,
        hyperparameter_ranges,
        metric_definitions,
        max_jobs=3,
        max_parallel_jobs=3,
        objective_type=objective_type,
        strategy="Random"
    )




    training_input_train =  {"s3_data": f's3://mlops-sagemaker-testing-usf/{intermediate_dataset_prefix}/data/train/train.csv',
                             "content_type": "text/csv"}

    training_input_val =    {"s3_data": f's3://mlops-sagemaker-testing-usf/{intermediate_dataset_prefix}/data/val/val.csv',
                             "content_type": "text/csv"}

    train_args = tuner.fit(
        inputs = {
            "train": TrainingInput(**training_input_train),
            "validation": TrainingInput(**training_input_val)
        }
    )


    tuning_step_conf = {
        "name": "FootTrafficTraining",
        "step_args": train_args,
        "depends_on": ["FootTrafficPrep"]
    }

    step_train = TuningStep(**tuning_step_conf)






    # ---------------------------       Eval Step       --------------------------------
    eval_inputs_list = [
        {
            "test_input": {
                "source": f's3://mlops-sagemaker-testing-usf/{intermediate_dataset_prefix}/data/test/test.csv',
                "destination": "/opt/ml/processing/test"}
        },
        {
            "eval_reqs": {
                "source": 's3://mlops-sagemaker-testing-usf/foot-traffic/requirements/03_eval_reqs.txt',
                "destination": "/opt/ml/processing/pipeline_reqs"
            }
        },
    ]    

    # output should be in json format
    eval_outputs_list = [
        {
            "output_score": {
                "output_name": "evaluation",
                "source": "/opt/ml/processing/evaluation",
                "destination": f's3://mlops-sagemaker-testing-usf/{intermediate_dataset_prefix}/data/evaluation/evaluation.json'}

        },
        {
            "metrics_chart": {
                "output_name": "eval_chart",
                "source": "/opt/ml/processing/chart/preds",
                "destination": f's3://mlops-sagemaker-testing-usf/{intermediate_dataset_prefix}/data/charts/preds/preds_plot.png'
            }
        },
        {
            "foot_traffic_chart": {
                "output_name": "traffic_chart",
                "source": "/opt/ml/processing/chart/traffic",
                "destination": f's3://mlops-sagemaker-testing-usf/{intermediate_dataset_prefix}/data/charts/traffic/traffic_plot.png'
            }
        }
    ]


    eval_inputs = [ ProcessingInput(**list(item.values())[0]) for item in eval_inputs_list ]



    eval_inputs = eval_inputs + [ProcessingInput(source = step_train.get_top_model_s3_uri(top_k=0,
                                                                                          s3_bucket=default_bucket,
                                                                                          prefix=model_prefix),
                                                                             destination = "/opt/ml/processing/model")]


    # we won't have the training uri for testing, so we need to use the uri of an older model
    # Update this path if testing!
    if test_eval:  
        eval_inputs = [ ProcessingInput(**list(item.values())[0]) for item in eval_inputs_list ]
        eval_inputs = eval_inputs + [ProcessingInput(source = "s3://mlops-sagemaker-testing-usf/FootTrafficTestTrain/p0ceb0gayvc7-FootTra-e4x3M4xyee-003-174f95b6/output/model.tar.gz",
                                                                             destination = "/opt/ml/processing/model")]

    eval_outputs = [ ProcessingOutput(**list(item.values())[0]) for item in eval_outputs_list ]


    eval_args = eval_processor.run(
        inputs = eval_inputs,
        outputs = eval_outputs,
        code = eval_code_location,
    )


    step_eval = ProcessingStep(
        name="FootTrafficEval",
        step_args=eval_args,
        property_files=[evaluation_report],
    )






    # ---------------------------      Baseline Step       --------------------------------

    # This step creates the baseline which the model monitor will run against before the batch transform job



    baseline_inputs_list = [
        {
            "baseline_input": {
                "source": f's3://mlops-sagemaker-testing-usf/{intermediate_dataset_prefix}/data/baseline/batch_inf_no_target_with_header.csv',
                "destination": "/opt/ml/processing/baseline_input"
            }
        },
        {
            "baseline_reqs": {
                "source": 's3://mlops-sagemaker-testing-usf/foot-traffic/requirements/04_baseline_monitor_reqs.txt',
                "destination": "/opt/ml/processing/pipeline_reqs"
            }
        },
    ]

    baseline_outputs_list = []

    baseline_inputs = [ ProcessingInput(**list(item.values())[0]) for item in baseline_inputs_list ]
    baseline_outputs = []

    baseline_args = baseline_processor.run(
        inputs = baseline_inputs,
        outputs = baseline_outputs,
        code = baseline_code_location
    )

    step_baseline = ProcessingStep(
        name = "FootTrafficBaselineCompute", 
        step_args = baseline_args,
        depends_on = ["FootTrafficPrep"],
    )

    return step_process, step_prep, step_train, step_eval, step_baseline
    
    

    
    
    
    
    
    
def _create_extra_processing_steps(default_bucket,
                                  pipeline_session,
                                  role,
                                  model_approval_status,
                                  transform_input_param,
                                  step_train,
                                  intermediate_dataset_prefix = "foot-traffic",
                                  model_prefix = "FootTrafficTrain",
                                  model_package_group_name = "foot-traffic-model-group",
                                  ):
    
    """Define the additional pipeline steps
    
    These are separated because the need some specific inputs and other objects created
    
    Args:
        model_prefix: 
        pipeline_session:
        role:
        intermediate_dataset_prefix:
        model_package_group_name:
        model_approval_status:
    
    Returns:
        A tuple of pipeline steps
        (step_create_model, step_register, transform_and_monitor_step)
         
    """
    
    
    # ------------    Saving Model Object   ----------------

    # we need to get the model from the training job, 
    # so it can be used in the transform step
    
    estimator_image_conf = {
            "framework": "xgboost",
            "region": "us-east-2",
            "version": "1.0-1",
            "py_version": "py3",
            "instance_type": "ml.m5.large"
    }

    image_uri = sagemaker.image_uris.retrieve(**estimator_image_conf)

    
    model = Model(
        image_uri=image_uri,

        # here we are just getting the best model from the tuning/training job
        model_data = step_train.get_top_model_s3_uri(top_k=0, s3_bucket=default_bucket, prefix=model_prefix),
        sagemaker_session=pipeline_session,
        name = 'foot_traffic_regressor',
        role=role,
    )
    

    step_create_model = ModelStep(
        name="FootTrafficModel",
        step_args=model.create(instance_type="ml.m5.large", accelerator_type="ml.eia1.medium", ),
    )





    # ----------------------           Register Model         ----------------------------

    # This step registers the model to the model package group
    # When constructing the pipeline we can make this step conditional based on a MSE score

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=f's3://mlops-sagemaker-testing-usf/{intermediate_dataset_prefix}/data/evaluation/evaluation.json',
            content_type="application/json",
        )
    )

    register_args = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )
    step_register = ModelStep(name="FootTrafficRegisterModel", step_args=register_args)






    # --------------   Batch Transformer   -------------------

    # the batch transformer performs the batch inference
    # This is added in this pipeline to make sure it works, 
    # but the batch transform on prod data would be handled by another pipeline in a CI/CD process

    # There are some limitations with the batch transform job - 
    # there may be formatting considerations 
    # The max payload size also has a suprising low limit, so I'm using the "SingleRecord" strategy


    transformer = Transformer(
        model_name=step_create_model.properties.ModelName,
        instance_type="ml.m5.large",
        instance_count=1,
        output_path=f"s3://{default_bucket}/FootTrafficTransform",
        max_payload=90,
        assemble_with="Line",
        strategy="SingleRecord",
        sagemaker_session=pipeline_session
    )


    batch_data = ParameterString(
        name="BatchData",
        default_value=f's3://mlops-sagemaker-testing-usf/{intermediate_dataset_prefix}/data/test/test.csv',
    )






    # -----------------          Model monitoring   ------------------------ 

    # The model monitoring step happens right before the batch transform step
    # by having the monitoring step right befroe the batch transform step, 
    # we are able to validate the data going directly into the model

    # Further data quality checks could be added further upstream in the pipeline

    # This step also relies on the baseline calculation job being completed



    transform_arg = transformer.transform(
        transform_input_param,
        content_type='text/csv'
    )

    job_config = CheckJobConfig(role=role)
    data_quality_config = DataQualityCheckConfig(
        baseline_dataset=f's3://mlops-sagemaker-testing-usf/{intermediate_dataset_prefix}/data/baseline/batch_inf_no_target_with_header.csv',
        dataset_format=DatasetFormat.csv(header=False),
        output_s3_uri=f's3://mlops-sagemaker-testing-usf/{intermediate_dataset_prefix}/data_quality/reports',
    )


    transform_and_monitor_step = MonitorBatchTransformStep(
        name="MonitorFootTrafficDataQuality",
        transform_step_args=transform_arg,
        monitor_configuration=data_quality_config,
        check_job_configuration=job_config,
        # since this is for data quality monitoring,
        # you could choose to run the monitoring job before the batch inference.
        monitor_before_transform=True,
        # if violation is detected in the monitoring, you can skip it and continue running batch transform
        fail_on_violation=False,
        supplied_baseline_statistics=f's3://mlops-sagemaker-testing-usf/{intermediate_dataset_prefix}/data_quality/results/statistics.json',
        supplied_baseline_constraints=f's3://mlops-sagemaker-testing-usf/{intermediate_dataset_prefix}/data_quality/results/constraints.json',
    )
    
    

    return step_create_model, step_register, transform_and_monitor_step







def _create_conditional_steps(mse_threshold,
                             step_eval,
                             step_register,
                             step_create_model,
                             transform_and_monitor_step,
                             step_baseline,
                             evaluation_report,
                              baseline_compute_status,
                             ):
    
    """Define the additional pipeline steps
    
    These are separated because the need some specific inputs and other objects created
    
    Args:
        
    
    Returns:
        A tuple of conditional steps
         
    """
    # ------------------   Register Model Condition Step   ----------------------

    # If the eval job does not produce a MSE score above the threshold, then don't register the model
    # Otherwise register the model



    # The fail step for if the MSE does not pass the threshold

    step_fail = FailStep(
        name="FootTrafficMSEFail",
        error_message=Join(on=" ", values=["Execution failed due to MSE >", mse_threshold]),
    )



    cond_lte = ConditionLessThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="regression_metrics.mse.value",
        ),
        right=mse_threshold,
    )

    
    step_cond = ConditionStep(
        name="FootTrafficTestMSECond",
        conditions=[cond_lte],
    #    if_steps=[step_register, step_create_model, step_transform],
        if_steps = [step_register, step_create_model, transform_and_monitor_step],
        else_steps=[step_fail],
    )


    # ------------------   Calculate Baseline Condition Step   ----------------------


    # parameter that can be manually set to trigger the baseline job:
    # Run the baseline job if set to true


    # Technically not a "fail" step, but just a dummy step to indicated that the baseline compute was skipped
    # do I even need this fail step?
    step_skip_baseline = FailStep(
        name="FootTrafficTestBaselineSkip",
        error_message=Join(on=" ", values=["Baseline computing was skipped with status: ", baseline_compute_status]),
    )

    cond_baseline = ConditionEquals(baseline_compute_status, "True")


    baseline_calc_step_cond = ConditionStep(
        name = "FootTrafficTestBaselineCond",
        conditions = [cond_baseline],
        if_steps = [step_baseline],
        #else_steps = [step_skip_baseline]
        else_steps = []
    )

    return step_cond, baseline_calc_step_cond






















# --------------------------------------------------------------------------------------------------
# -----------------------         Construct the pipeline definition         ------------------------
# --------------------------------------------------------------------------------------------------




def get_pipeline(
    region,
    sagemaker_project_name=None,
    role=None,
    default_bucket=None,
    model_package_group_name="FootTrafficModelPackageGroup",
    pipeline_name="FootTrafficModelBuildPipeline", 
    base_job_prefix="sagemaker/foot-traffic", # place to upload training files within the bucket
    estimator_output_path = "s3://mlops-sagemaker-testing-usf/FootTrafficTrain",
    model_prefix = "FootTrafficTrain",
    
    test_eng = False,
    test_prep = False,
    test_eval = False,
    test_baseline = False,
    
    baseline_compute_default = "False",
    ):
    """Gets a SageMaker ML Pipeline instance working with on foot traffic data.

    Args:
        region:                      AWS region to create and run the pipeline.
        role:                        IAM role to create and run steps and pipeline.
        default_bucket:              the bucket to use for storing the artifacts
        model_package_group_name:    Name of the model registry
        pipeline_name:               Name of the pipeline
        base_job_prefix:             Prefix for created jobs
        estimator_output_path:       Where the model (estimator) should be saved
        model_prefix:                Prefix for created model
        test_eng:                    Test data eng job (doesn't create pipeline)
        test_prep:                   Test data prep job (doesn't create pipeline)
        test_eval:                   Test eval job (doesn't create pipeline)
        test_baseline:               Test baseline job (doesn't create pipeline)
        baseline_compute_default:    Compute the baseline

    Returns:
        an instance of a pipeline/the pipeline definition
    """
        
    # Init the connections
    
    
    sagemaker_session = _get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    pipeline_session = _get_pipeline_session(region, default_bucket)
    
    
    
    
    # ---------------  parameters for pipeline execution  ----------------
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    
    # not used
    instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.m5.xlarge")

    # This dictates the default approval status for the to be deployed
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )

    # this threshold will determine if the model is saved to the registry
    # setting super high just for testing
    mse_threshold = ParameterFloat(name="MseThreshold", default_value=500000.0) 
    
    
    evaluation_report = PropertyFile(
        name="EvaluationReport", output_name="evaluation", path="evaluation.json"
    )
    
    transform_input_param = ParameterString(
        name="transform_input",
        default_value=f's3://mlops-sagemaker-testing-usf/foot-traffic/data/batch_inf/batch_inf_no_target.csv',
    )
    
    
    baseline_compute_status = ParameterString(
        name="ComputeBaseline", default_value=baseline_compute_default
    )

    
    
    
    # -------------------------     Defining Processors (containers)      -------------------------


    # modify the _construct_processors function based on processors needed
    # this is basically evertyhign that needs an instance to be ran
    data_eng_processor, train_prep_processor, xgb_train, eval_processor, baseline_processor = _construct_processors(
                          pipeline_session,
                          sagemaker_session,
                          role,
                          estimator_output_path = estimator_output_path,
                          test_eng = test_eng,
                          test_prep = test_prep,
                          test_eval = test_eval,
                          test_baseline = test_baseline,
                         ) 
    
    
    
    
    
    # --------------------      Create the pipeline/workflow Steps   ---------------------------
    
    
    # running the steps should trigger any processing job tests as well
    step_process, step_prep, step_train, step_eval, step_baseline = _create_pipeline_steps(
                           data_eng_processor,
                           train_prep_processor,
                           xgb_train,
                           eval_processor,
                           baseline_processor,
                           evaluation_report,
                           default_bucket,
                           test_eval = test_eval, # need to update one extra thing for the steps
                           model_prefix = model_prefix,
                           code_location_process = os.path.join(BASE_DIR, "01_data_eng_s3.py"),
                           code_location_train = os.path.join(BASE_DIR, "02_training_prep.py"),
                           eval_code_location = os.path.join(BASE_DIR, "03_eval.py"),
                           baseline_code_location = os.path.join(BASE_DIR, "04_baseline_monitoring.py"),
                           input_dataset_s3 = 's3://mlops-sagemaker-testing-usf/foot_traffic_test/data/pipeline_input/ft_cases_per_week_fake.csv',
                           intermediate_dataset_prefix = "foot-traffic"
                            )
    
    
    
    
    # --------------------      Create the pipeline/workflow Steps   ---------------------------
    
    step_create_model, step_register, transform_and_monitor_step = _create_extra_processing_steps(
                                  default_bucket,
                                  pipeline_session,
                                  role,
                                  model_approval_status,
                                  transform_input_param,
                                  step_train,
                                  intermediate_dataset_prefix = "foot-traffic",
                                  model_prefix = model_prefix,
                                  model_package_group_name = model_package_group_name
                                  )
    
    
    # --------------------      Create the pipeline/workflow Steps   ---------------------------    
    
    
    
    step_cond, baseline_calc_step_cond = _create_conditional_steps(
                             mse_threshold,
                             step_eval,
                             step_register,
                             step_create_model,
                             transform_and_monitor_step,
                             step_baseline,
                             evaluation_report,
                             baseline_compute_status,
                             )
    
    
    
    # -----------------------------      Define the pipeline    -----------------------------------
    

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_count,
            instance_type,
            model_approval_status,
            mse_threshold,
            transform_input_param,
            baseline_compute_status,
        ],
        steps=[step_process, step_prep, baseline_calc_step_cond, step_train, step_eval, step_cond],
    )
    
    
    return pipeline




# --------------------------   You could just call a job instead of creating a pipeline for loading -------------------------
# Actually I'm thinking this makes more sense just to include this with the batch deploy pipelines
def get_pipeline_load_only(    
    region,
    role=None,
    default_bucket=None,
    model_package_group_name="FootTrafficModelPackageGroup",
    pipeline_name="FootTrafficModelBuildPipeline", 
    base_job_prefix="sagemaker/foot-traffic", # place to upload training files within the bucket
    estimator_output_path = "s3://mlops-sagemaker-testing-usf/FootTrafficTrain",
    model_prefix = "FootTrafficTrain",
    
    test_eng = False,
    
    ):
    """
    Create a "pipeline" for loading the feature store only
    
    This can be used to load the feature store on a specified interval (with eventbridge)
    
    
    """
    
    
    
    
    return