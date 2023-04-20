import os

import boto3
import sagemaker
from sagemaker import model
import sagemaker.session
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import (
    TransformStep, 
    Transformer, 
    TransformInput
)

from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, TuningStep

from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.quality_check_step import DataQualityCheckConfig
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker.workflow.monitor_batch_transform_step import MonitorBatchTransformStep
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo, ConditionEquals
from sagemaker.workflow.condition_step import ConditionStep


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
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

def _get_session(region, default_bucket=None):
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

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags



def get_pipeline(
    region="us-east-2",
    role=None,
    default_bucket=None,
    pipeline_name="FootTrafficBatchInference",
    base_job_prefix="foot-traffic",
    intermediate_dataset_prefix = "foot-traffic",
    test_inf_prep = False
):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

    Args:
        region: AWS region to create and run the pipeline.
        model_name: Name of the SageMaker Model to deploy

    Returns:
        an instance of a pipeline
    """
    
    # make sure this is added to args
    if default_bucket is None:
        default_bucket = "sagemaker-us-east-2-644944822023"

    sagemaker_session = _get_session(region, default_bucket)
    if role is None:
        #role = sagemaker.session.get_execution_role(sagemaker_session)
        role = "arn:aws:iam::644944822023:role/service-role/AmazonSageMakerServiceCatalogProductsExecutionRole"

    pipeline_session = _get_pipeline_session(region, default_bucket)

    #### PARAMETERS
    model_name = ParameterString(name = "ModelName", default_value='${ModelName}')
    #model_name = ParameterString(name = "ModelName", default_value="test")
    #batch_inference_instance_count = ParameterInteger(name = "BatchInstanceCount", default_value=1)
    #batch_inference_instance_type = ParameterString(name = "BatchInstanceType", default_value='ml.m5.xlarge')
    #input_path = ParameterString(name = "InputPath", default_value=f"s3://sagemaker-servicecatalog-seedcode-{region}/dataset/abalone-dataset.csv")
    #output_path = ParameterString(name = "OutputPath", default_value = f"s3://{default_bucket}/FootTrafficTransform")
    #transform_input_param = ParameterString(
    #    name="transform_input",
    #    default_value=f's3://sagemaker-us-east-2-644944822023/{intermediate_dataset_prefix}/data/batch_inf/batch_inf_no_target.csv',
    #)

    
    # ------------------------------------------------------------------------------------------
    # ----------------------------------   SAGEMAKER CONSTRUCTS  -------------------------------
    # ------------------------------------------------------------------------------------------
    
    
    inf_prep_processor_conf = {
            "role": role,
            "instance_type": "ml.m5.large",
            "instance_count": 1,
            "base_job_name": "training-prep-processing",
            "framework_version": "1.0-1", # 0.23-1, 1.0-1
            "sagemaker_session": pipeline_session
    }

    if test_inf_prep:
        inf_prep_processor_conf.update({"sagemaker_session": sagemaker_session})


    inf_prep_processor = SKLearnProcessor(**inf_prep_processor_conf)

    
    
    
    
    # Transformer
    
    transform = Transformer(
        model_name=model_name,
        instance_count=1,
        instance_type="ml.m5.large",
        #output_path=output_path,
        output_path = f"s3://{default_bucket}/FootTrafficTransform",
        base_transform_job_name=f"{base_job_prefix}/batch-transform-job",
        max_payload=90,
        assemble_with="Line",
        strategy="SingleRecord",
        accept='text/csv',
        sagemaker_session=pipeline_session
    )
    
    
    
    
    
    
    # ------------------------------------------------------------------------------------------
    # -------------------------------------      STEPS     ------------------------------------- 
    # ------------------------------------------------------------------------------------------

    
    inf_prep_inputs_list = [
        {
            "inf_prep_reqs": {
                "source": 's3://sagemaker-us-east-2-644944822023/foot-traffic/requirements/01_inf_prep_reqs.txt',
                "destination": "/opt/ml/processing/pipeline_reqs"
            }
        },
    ]
    inf_prep_outputs_list = [
        # could also filter out the target column with some args
        # However, I like having the headers on the data as long as possible, and in some cases only data without headers can be passed to the Estimator
        {
            "batch_transform_input": {
                "output_name": "batch_inf",
                "destination": f's3://sagemaker-us-east-2-644944822023/{intermediate_dataset_prefix}/data/batch_inf/batch_inf_no_target.csv',
                "source": "/opt/ml/processing/batch_inf"
            }
        },
    ]

    
    
    code_location_inf = os.path.join(BASE_DIR, "01_inf_prep.py")
    
    inf_prep_inputs = [ ProcessingInput(**list(item.values())[0]) for item in inf_prep_inputs_list ]
    inf_prep_outputs = [ ProcessingOutput(**list(item.values())[0]) for item in inf_prep_outputs_list ]

    inf_prep_args = inf_prep_processor.run(
        inputs = inf_prep_inputs,
        outputs = inf_prep_outputs,
        code = code_location_inf
    )


    step_prep_inf = ProcessingStep(name="FootTrafficInfPrep", step_args=inf_prep_args)



    
    # ------     Transformer     --------
    
        
    transform_arg = transform.transform(
        #transform_input_param,
        f's3://sagemaker-us-east-2-644944822023/{intermediate_dataset_prefix}/data/batch_inf/batch_inf_no_target.csv',
        content_type='text/csv'
    )

    job_config = CheckJobConfig(role=role)
    
    data_quality_config = DataQualityCheckConfig(
        baseline_dataset=f's3://sagemaker-us-east-2-644944822023/{intermediate_dataset_prefix}/data/baseline/batch_inf_no_target_with_header.csv',
        dataset_format=DatasetFormat.csv(header=False),
        output_s3_uri=f's3://sagemaker-us-east-2-644944822023/{intermediate_dataset_prefix}/data_quality/reports',
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
        supplied_baseline_statistics=f's3://sagemaker-us-east-2-644944822023/{intermediate_dataset_prefix}/data_quality/results/statistics.json',
        supplied_baseline_constraints=f's3://sagemaker-us-east-2-644944822023/{intermediate_dataset_prefix}/data_quality/results/constraints.json',
    )
    
    
    
    
    
    

    # old
    # transform_step = TransformStep(
    #     name='BatchInferenceStep',
    #     transformer=transform,
    #     inputs=TransformInput(data=input_path, content_type='text/csv')
    # )

    
    
    
    #### PIPELINE
    
    # should be able to set up dependencies this way (below), but you can't =(
    #transform_and_monitor_step.add_depends_on([step_prep_inf])
    # the transform_and_monitor_step is actually just helper that combines other steps and not fully featured
    # see docs - https://github.com/aws/sagemaker-python-sdk/blob/40dd06aa7fdf474ca8219c198631dc0dffbd605c/src/sagemaker/workflow/monitor_batch_transform_step.py
    
    
    # hackey workaround:
    baseline_compute_status = ParameterString(
        name="CheckMetricsAndTransform", default_value="True"
    )
        
    force_dependency_cond = ConditionEquals(baseline_compute_status, "True")


    transform_dependency_step_cond = ConditionStep(
        name = "TransformDependencyWorkaround",
        depends_on = [step_prep_inf],
        conditions = [force_dependency_cond],
        if_steps = [transform_and_monitor_step],
        else_steps = []
    )
    
    
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[model_name, 
                    baseline_compute_status,
               #     batch_inference_instance_count,
               #     batch_inference_instance_type,
               #     input_path,
               #     output_path,
               #     transform_input_param
            ],
        steps=[step_prep_inf,
               transform_dependency_step_cond],
        sagemaker_session=sagemaker_session
    )
    
    
    
    
    return pipeline

    
