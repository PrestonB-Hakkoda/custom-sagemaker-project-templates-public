# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Provides utilities for SageMaker Pipeline CLI."""
from __future__ import absolute_import

import ast
import os
import sagemaker


def get_pipeline_driver(module_name, passed_args=None):
    """Gets the driver for generating your pipeline definition.

    Pipeline modules must define a get_pipeline() module-level method.

    Args:
        module_name: The module name of your pipeline.
        passed_args: Optional passed arguments that your pipeline may be templated by.

    Returns:
        The SageMaker Workflow pipeline.
    """
    _imports = __import__(module_name, fromlist=["get_pipeline"])
    kwargs = convert_struct(passed_args)
    return _imports.get_pipeline(**kwargs)


def convert_struct(str_struct=None):
    return ast.literal_eval(str_struct) if str_struct else {}

def get_pipeline_custom_tags(module_name, args, tags):
    """Gets the custom tags for pipeline

    Returns:
        Custom tags to be added to the pipeline
    """
    try:
        _imports = __import__(module_name, fromlist=["get_pipeline_custom_tags"])
        kwargs = convert_struct(args)
        return _imports.get_pipeline_custom_tags(tags, kwargs['region'], kwargs['sagemaker_project_arn'])
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return tags



# added this function to help upload requirements to S3
# requirements are added to S3 so they can be read by the processing jobs
def _upload_helper(local_path, base_uri):
    file_uri =  sagemaker.s3.S3Uploader.upload(
    local_path = local_path,
    desired_s3_uri = base_uri
    )
    return file_uri


def upload_reqs(default_bucket):
    
    print(default_bucket)
    
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    
    
    print("reqs base dir: ", BASE_DIR)
    BASE_DIR_REQ = os.path.join(BASE_DIR, "batch_inference/requirements")
    
    inf_req = os.path.join(BASE_DIR_REQ, "01_inf_prep_reqs.txt")

    s3_base_uri = f"s3://{default_bucket}/foot-traffic/requirements"
    
    reqs_paths = [inf_req]
    
    print(f"Reqs upload path:   {s3_base_uri}")
    
    for req_path in reqs_paths:
        _upload_helper(req_path, s3_base_uri)
    
    
    return