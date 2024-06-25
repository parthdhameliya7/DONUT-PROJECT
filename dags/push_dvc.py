from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import glob
import os
import json
from ast import literal_eval
from PIL import Image
from google.cloud import storage
from gcsfs import GCSFileSystem
from gcs_utils import list_files_with_pattern, download_from_gcs
import fnmatch
import subprocess
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

def push_data_to_dvc():
    try:
        # Run the DVC add command
        result = subprocess.run(['dvc', 'add', 'data1'], capture_output=True, text=True)
        
        # Check if the DVC add command was successful
        if result.returncode == 0:
            print("DVC add successful.")
            print(result.stdout)

            # Track the added data with Git
            git_add_result = subprocess.run(['git', 'add', 'data1.dvc'], capture_output=True, text=True)
            if git_add_result.returncode == 0:
                print("Git add successful.")
                print(git_add_result.stdout)
                
                # Commit the changes
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                git_commit_result = subprocess.run(['git', 'commit', '-m', f'Added data file and tracked by git at {current_time}'], capture_output=True, text=True)
                if git_commit_result.returncode == 0:
                    print("Git commit successful.")
                    print(git_commit_result.stdout)
                    
                    # Run the DVC push command
                    dvc_push_result = subprocess.run(['dvc', 'push'], capture_output=True, text=True)
                    if dvc_push_result.returncode == 0:
                        print("DVC push successful.")
                        print(dvc_push_result.stdout)
                    else:
                        print("DVC push failed.")
                        print(dvc_push_result.stderr)
                else:
                    print("Git commit failed.")
                    print(git_commit_result.stderr)
            else:
                print("Git add failed.")
                print(git_add_result.stderr)
        else:
            print("DVC add failed.")
            print(result.stderr)

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


push_data_dvc_task = PythonOperator(
    task_id='push_data_dvc',
    python_callable=push_data_to_dvc,
    provide_context=True,
    dag=dag,
)

push_data_dvc_task