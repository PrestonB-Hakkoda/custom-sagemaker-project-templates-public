# one off baselining step

import os
import tempfile
from typing import Tuple
import sys
import io
import subprocess
import time
from time import gmtime, strftime, sleep
from pathlib import Path
import logging 
import ast



# install requirements from requirements .txt 
# Note: Python 3.8 required (only for Snowpark)

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-r",
    "/opt/ml/processing/pipeline_reqs/04_baseline_monitor_reqs.txt",
])

import sagemaker
from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.quality_check_step import DataQualityCheckConfig





# ---------------     install extras     ---------------
# An alternative to this would be to create an image with the dependencies already installed




if __name__ == "__main__":
    
    os.environ["AWS_DEFAULT_REGION"] = "us-east-2"
    
    role = sagemaker.get_execution_role()
    
    my_default_monitor = DefaultModelMonitor(
        role=role,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        volume_size_in_gb=20,
        max_runtime_in_seconds=3600,
    )

    my_default_monitor.suggest_baseline(
        baseline_dataset='s3://sagemaker-us-east-2-644944822023/foot_traffic_test/data/baseline/batch_inf_no_target_with_header.csv', 
        dataset_format=DatasetFormat.csv(header=True),
        output_s3_uri='s3://sagemaker-us-east-2-644944822023/foot_traffic_test/data_quality/results',
        wait=True,
        logs=False,
    )

