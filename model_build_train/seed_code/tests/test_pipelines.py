import pytest


# pipeline defaults:
default_pipeline_params = {
    "region": "us-east-2",
    "sagemaker_project_name": "foot-traffic-build-train-sagemaker-foot-traffic-build-train",
    "role":"arn:aws:iam::644944822023:role/AmazonSageMaker-ExecutionRole-20230109T000000",
    "default_bucket":"sagemaker-us-east-2-644944822023",
    "model_package_group_name": "foot-traffic-build-train-sagemaker-foot-traffic-build-train",
    "pipeline_name":"foot-traffic-build-train-sagemaker-foot-traffic-build-train",
    "base_job_prefix":"sagemaker/foot-traffic", # place to upload training files within the bucket
    "estimator_output_path": "s3://sagemaker-us-east-2-644944822023/FootTrafficTrain",
    "model_prefix": "FootTrafficTrain",
    "test_eng": False,
    "test_prep": False,
    "test_eval": False,
    "test_baseline": False,
    "baseline_compute_default": "False"
}



def test_pipelines_importable():
    import pipelines  # noqa: F401

    
    

def test_pipeline_definition():
    
    from pipelines.foot_traffic.foot_traffic_pipeline import get_pipeline
    
    
    try:
        pipeline = get_pipeline(**default_pipeline_params)
        
        
    except Exception as e:
        print("error creating the pipeline definition")
        print(e)
    

    if pipeline:
        from sagemaker.workflow.pipeline import Pipeline

    assert isinstance(pipeline, Pipeline)
    
    return
            

            
def test_data_eng_job():
    
    from pipelines.foot_traffic.foot_traffic_pipeline import get_pipeline
    
    try:
        test_eng_params = default_pipeline_params
        test_eng_params.update({"test_eng": True})
        get_pipeline(**test_eng_params)
    
    except Exception as e:
        print("error testing data engineering job")
        print(e)

    return
    
    
def test_prep_job():
    from pipelines.foot_traffic.foot_traffic_pipeline import get_pipeline
    
    try:
        test_prep_params = default_pipeline_params
        test_prep_params.update({"test_prep": True})
        get_pipeline(**test_prep_params)
    
    except Exception as e:
        print("error testing data Prep job")
        print(e)

    return


def test_eval_job():
    
    from pipelines.foot_traffic.foot_traffic_pipeline import get_pipeline
    
    try:
        test_eval_params = default_pipeline_params
        test_eval_params.update({"test_eval": True})
        get_pipeline(**test_eval_params)
    
    except Exception as e:
        print("error testing model eval job")
        print(e)

    return


def test_baseline_job():
    
    from pipelines.foot_traffic.foot_traffic_pipeline import get_pipeline
    
    try:
        test_baseline_params = default_pipeline_params
        test_baseline_params.update({"test_baseline": True})
        get_pipeline(**test_baseline_params)
    
    except Exception as e:
        print("error testing baseline job")
        print(e)

    return

    
    
    