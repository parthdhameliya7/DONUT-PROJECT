from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from datetime import datetime
import pandas as pd
import glob
import os
import json
from ast import literal_eval
from PIL import Image
from google.cloud import storage
from gcsfs import GCSFileSystem
from gcs_utils import *
import fnmatch
import subprocess
import logging
import sys
import io

os.environ["AIRFLOW_CONN_GOOGLE_CLOUD_DEFAULT"] = 'google-cloud-platform://'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/opt/airflow/config/advance-anvil-425519-u2-b2800cb795f5.json'

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 6, 25),
    'retries': 1,
}

dag = DAG(
    'dvc_push_1',
    default_args=default_args,
    description='DAG to push data from local to GCS bucket.',
    schedule_interval=None,
)

def push_data_to_dvc(**kwargs):
    logger = logging.getLogger("airflow.task")
    try:
        # Run the DVC add command
        result = subprocess.run(['dvc', 'add', 'data1'], capture_output=True, text=True)
        logger.info(f"DVC add result: {result.stdout}")

        if result.returncode == 0:
            logger.info("DVC add successful.")
            print("DVC add successful.")
            
            git_add_result = subprocess.run(['git', 'add', 'data1.dvc'], capture_output=True, text=True)
            logger.info(f"Git add result: {git_add_result.stdout}")
            print("Git add successful.")
            
            if git_add_result.returncode == 0:
                logger.info("Git add successful.")
                
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                git_commit_result = subprocess.run(['git', 'commit', '-m', f'Added data file and tracked by git at {current_time}'], capture_output=True, text=True)
                logger.info(f"Git commit result: {git_commit_result.stdout}")
                
                if git_commit_result.returncode == 0:
                    logger.info("Git commit successful.")
                    print("Git commit successful..")
                    
                    dvc_push_result = subprocess.run(['dvc', 'push'], capture_output=True, text=True)
                    logger.info(f"DVC push result: {dvc_push_result.stdout}")
                    
                    if dvc_push_result.returncode == 0:
                        logger.info("DVC push successful.")
                        print("DVC push successful.")
                    else:
                        logger.error("DVC push failed.")
                        print("DVC push failed.")
                        logger.error(dvc_push_result.stderr)
                        raise Exception("DVC push failed.")
                else:
                    logger.error("Git commit failed.")
                    logger.error(git_commit_result.stderr)
                    raise Exception("Git commit failed.")
            else:
                logger.error("Git add failed.")
                logger.error(git_add_result.stderr)
                raise Exception("Git add failed.")
        else:
            logger.error("DVC add failed.")
            logger.error(result.stderr)
            raise Exception("DVC add failed.")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

push_data_dvc_task = PythonOperator(
    task_id='push_data_dvc',
    python_callable=push_data_to_dvc,
    provide_context=True,
    dag=dag,
)


push_data_dvc_task