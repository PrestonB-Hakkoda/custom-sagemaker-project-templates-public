## Layout of the SageMaker ModelBuild Project Template

The template provides a starting point for bringing your SageMaker Pipeline development to production.

```
|-- codebuild-buildspec.yml
|-- CONTRIBUTING.md
|-- pipelines
|   |-- foot_traffic
|   |   |-- __init__.py
|   |   |-- 01_data_eng_s3.py 
|   |   |-- 01_data_eng_snowflake.py (todo)
|   |   |-- 02_training_prep.py
|   |   |-- 03_eval.py
|   |   |-- 04_baseline_monitoring.py
|   |   |-- foot_traffic_pipeline.py
|   |   |-- requirements
|   |   |   |-- 01_data_eng_reqs.txt
|   |   |   |-- 02_training_prep_reqs.txt
|   |   |   |-- 03_eval_reqs.txt
|   |   |   |-- 04_baseline_monitor_reqs.txt
|   |   |   |-- create_feature_store.py
|   |-- get_pipeline_definition.py
|   |-- __init__.py
|   |-- run_pipeline.py
|   |-- _utils.py
|   `-- __version__.py
|-- README.md
|-- sagemaker-pipelines-project.ipynb
|-- setup.cfg
|-- setup.py
|-- tests
|   |-- test_pipelines.py
|   |-- __init__.py
|   |-- test_build_cmd.txt
`-- tox.ini
```

## Start here
This is a sample code repository that demonstrates how you can organize your code for an ML business solution. This code repository is created as part of creating a Project in SageMaker. 

In this example, we are solving the abalone age prediction problem using the abalone dataset (see below for more on the dataset). The following section provides an overview of how the code is organized and what you need to modify. In particular, `pipelines/pipelines.py` contains the core of the business logic for this problem. It has the code to express the ML steps involved in generating an ML model. You will also find the code for that supports preprocessing and evaluation steps in `preprocess.py` and `evaluate.py` files respectively.

Once you understand the code structure described below, you can inspect the code and you can start customizing it for your own business case. This is only sample code, and you own this repository for your business use case. Please go ahead, modify the files, commit them and see the changes kick off the SageMaker pipelines in the CICD system.

You can also use the `sagemaker-pipelines-project.ipynb` notebook to experiment from SageMaker Studio before you are ready to checkin your code.

A description of some of the artifacts is provided below:
<br/><br/>
Your codebuild execution instructions. This file contains the instructions needed to kick off an execution of the SageMaker Pipeline in the CICD system (via CodePipeline). You will see that this file has the fields definined for naming the Pipeline, ModelPackageGroup etc. You can customize them as required.

```
|-- codebuild-buildspec.yml
```

<br/><br/>
Your pipeline artifacts, which includes a pipeline module defining the required `get_pipeline` method that returns an instance of a SageMaker pipeline, a preprocessing script that is used in feature engineering, and a model evaluation script to measure the Mean Squared Error of the model that's trained by the pipeline. This is the core business logic, and if you want to create your own folder, you can do so, and implement the `get_pipeline` interface as illustrated here.

```
|-- pipelines
|   |-- foot_traffic
|   |   |-- __init__.py
|   |   |-- 01_data_eng_s3.py 
|   |   |-- 01_data_eng_snowflake.py (todo)
|   |   |-- 02_training_prep.py
|   |   |-- 03_eval.py
|   |   |-- 04_baseline_monitoring.py
|   |   |-- foot_traffic_pipeline.py

```

<br/><br/>
Some processing steps require additional dependiencies. These dependencies can be found in the `pipelines/foot_traffic/requirements` folder. The `*_reqs.txt` files correspond to their respective processing job scripts, and the requirements will be installed at the start of the script. The `create_feature_store.py` function is a helper for initially creating the feature store if it does not exsist. 


```
|-- pipelines
|   |-- foot_traffic
|   |   |-- requirements
|   |   |   |-- 01_data_eng_reqs.txt
|   |   |   |-- 02_training_prep_reqs.txt
|   |   |   |-- 03_eval_reqs.txt
|   |   |   |-- 04_baseline_monitor_reqs.txt
|   |   |   |-- create_feature_store.py
```


<br/><br/>
Utility modules for getting pipeline definition jsons and running pipelines (you do not typically need to modify these):

```
|-- pipelines
|   |-- get_pipeline_definition.py
|   |-- __init__.py
|   |-- run_pipeline.py
|   |-- _utils.py
|   `-- __version__.py
```

<br/><br/>
Python package artifacts:
```
|-- setup.cfg
|-- setup.py
```
<br/><br/>
A stubbed testing module for testing your pipeline as you develop:
```
|-- tests
|   `-- test_pipelines.py
```
<br/><br/>
The `tox` testing framework configuration:
```
`-- tox.ini
```

## Dataset for the Foot Traffic Pipeline

The foot traffic dataset contains the traffic to certain warehouses and the cases of product sold over a several month timeframe. For practical use cases this dataset should be encriched with data from other sources. 