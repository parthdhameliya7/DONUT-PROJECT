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
    'dvc_pull_1',
    default_args=default_args,
    description='DAG for data pulling and preprocessing.',
    schedule_interval=None,
)


def pull_data_from_dvc():
    try:
        # Check if data1 directory exists and remove it if it does
        if os.path.exists('/opt/airflow/data1'):
            print("Removing existing data1 directory.")
            result = subprocess.run(['rm', '-rf', '/opt/airflow/data1'], capture_output=True, text=True)
            if result.returncode == 0:
                print("data1 directory removed successfully.")
            else:
                print("Failed to remove data1 directory.")
                print(result.stderr)

        # Run the DVC pull command
        result = subprocess.run(['dvc', 'pull'], capture_output=True, text=True)
        
        # Check if the DVC pull command was successful
        if result.returncode == 0:
            print("DVC pull successful.")
            print(result.stdout)

            # Track the pulled data files with Git
            git_add_result = subprocess.run(['git', 'add', '-f', 'data1'], capture_output=True, text=True)
            if git_add_result.returncode == 0:
                print("Git add successful.")
                print(git_add_result.stdout)
                
                # Commit the changes
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                git_commit_result = subprocess.run(['git', 'commit', '-m', f'Pulled data from DVC and added to Git at {current_time}'], capture_output=True, text=True)
                if git_commit_result.returncode == 0:
                    print("Git commit successful.")
                    print(git_commit_result.stdout)
                else:
                    print("Git commit failed.")
                    print(git_commit_result.stderr)
            else:
                print("Git add failed.")
                print(git_add_result.stderr)
        else:
            print("DVC pull failed.")
            print(result.stderr)

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)    

 
def load_df(file_path: str):
    return pd.read_parquet(file_path)
 
def concat_data(data_list):
    return pd.concat(data_list, axis=0).reset_index().drop(['index'], axis=1)
 
def get_image(byte_string):
    image = Image.open(io.BytesIO(byte_string))
    return image
 
def get_jsons(target_string):
    target_json = literal_eval(target_string)['gt_parse']
    return target_json
 
def generate_filename(counter, directory, whset, time, extension):
    current_file = f'{directory}/cord-v2-{whset}-{counter}-{time}.{extension}'
    return current_file
 
def create_data_folders(directory):
    for directs in directory:
        os.makedirs(directs, exist_ok=True)

 
def save_images_to_directory(df, directory, whset, time):
    filepaths = []
    idx = 0
    for i in range(len(df)):
        row = df.iloc[i]
        byte_string = row.image['bytes']
        image = get_image(byte_string)
        filename = generate_filename(idx, directory, whset, time, '.png')
        filepaths.append(filename)
        image.save(filename, "PNG")
        idx += 1
    return filepaths
 
def save_jsons_to_directory(df, directory, whset, time):
    filepaths = []
    idx = 0
    for i in range(len(df)):
        row = df.iloc[i]
        target_strings = row.ground_truth
        target_json = get_jsons(target_strings)
        filename = generate_filename(idx, directory, whset, time, '.json')
        with open(filename, 'w') as f:
            json.dump(target_json, f)
        filepaths.append(filename)
        idx += 1
    return filepaths
 
def save_csvs_to_directory(df, filename, index=False):
    df.to_csv(filename, index=index)
    return None
 
def get_dataframe(image_filepath, json_filepaths):
    dataframe_dict = {
        'image_filepath': image_filepath,
        'json_filepaths': json_filepaths,
    }
    df = pd.DataFrame(dataframe_dict)
    return df
 
 
def get_parquet_file_path(**kwargs):
    base_dir = 'data1' # attach folder name where data will pull
    train_path = glob.glob(os.path.join(base_dir, 'train*.parquet'))
    valid_path = glob.glob(os.path.join(base_dir, 'valid*.parquet'))
    test_path = glob.glob(os.path.join(base_dir, 'test*.parquet'))
    kwargs['ti'].xcom_push(key='train_path', value=train_path)
    kwargs['ti'].xcom_push(key='valid_path', value=valid_path)
    kwargs['ti'].xcom_push(key='test_path', value=test_path)
    # add more line for push
    print(test_path)
    return train_path, valid_path, test_path  #

def load_and_concat_data(**kwargs):
    ti = kwargs['ti']
    # add more line for pull
    train_path = ti.xcom_pull(key='train_path', task_ids='read_parquet_file_path')
    valid_path = ti.xcom_pull(key='valid_path', task_ids='read_parquet_file_path')
    test_path = ti.xcom_pull(key='test_path', task_ids='read_parquet_file_path')
    train_df = concat_data([load_df(i) for i in train_path]).iloc[:2]
    valid_df = concat_data([load_df(i) for i in valid_path]).iloc[:2]
    test_df = concat_data([load_df(i) for i in test_path]).iloc[:2]
    kwargs['ti'].xcom_push(key='train_df', value=train_df)
    kwargs['ti'].xcom_push(key='valid_df', value=valid_df)
    kwargs['ti'].xcom_push(key='test_df', value=test_df)
 
def create_directories(**kwargs):  # Data Directories are created and stored in to airflow-worker-1 this container :
    time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    IMG_DIRECTS = [
        f'/opt/airflow/images/train_images_{time}',
        f'/opt/airflow/images/valid_images_{time}',
        f'/opt/airflow/images/test_images_{time}',
    ]
    JSON_DIRECTS = [
        f'/opt/airflow/jsons/train_json_{time}',
        f'/opt/airflow/jsons/valid_json_{time}',
        f'/opt/airflow/jsons/test_json_{time}',
    ]
    CSVS = [
        f'/opt/airflow/csvs',
    ]
    create_data_folders(IMG_DIRECTS)
    create_data_folders(JSON_DIRECTS)
    create_data_folders(CSVS)
    kwargs['ti'].xcom_push(key='time', value=time)
    kwargs['ti'].xcom_push(key='IMG_DIRECTS', value=IMG_DIRECTS)
    kwargs['ti'].xcom_push(key='JSON_DIRECTS', value=JSON_DIRECTS)
    kwargs['ti'].xcom_push(key='CSVS', value=CSVS)
 
def save_images(**kwargs):
    ti = kwargs['ti']
    time = ti.xcom_pull(key='time', task_ids='create_directories')
    IMG_DIRECTS = ti.xcom_pull(key='IMG_DIRECTS', task_ids='create_directories')
    train_df = ti.xcom_pull(key='train_df', task_ids='load_and_concat_data')
    valid_df = ti.xcom_pull(key='valid_df', task_ids='load_and_concat_data')
    test_df = ti.xcom_pull(key='test_df', task_ids='load_and_concat_data')
 
    train_filepaths = save_images_to_directory(train_df, IMG_DIRECTS[0], 'train', time)
    valid_filepaths = save_images_to_directory(valid_df, IMG_DIRECTS[1], 'valid', time)
    test_filepaths = save_images_to_directory(test_df, IMG_DIRECTS[2], 'test', time)
 
    kwargs['ti'].xcom_push(key='train_image_filepaths', value=train_filepaths)
    kwargs['ti'].xcom_push(key='valid_image_filepaths', value=valid_filepaths)
    kwargs['ti'].xcom_push(key='test_image_filepaths', value=test_filepaths)
 
def extract_ground_truth_strings(**kwargs):
    ti = kwargs['ti']
    train_df = ti.xcom_pull(key='train_df', task_ids='load_and_concat_data')
    valid_df = ti.xcom_pull(key='valid_df', task_ids='load_and_concat_data')
    test_df = ti.xcom_pull(key='test_df', task_ids='load_and_concat_data')
 
    train_gt_strings = train_df.ground_truth.tolist()
    valid_gt_strings = valid_df.ground_truth.tolist()
    test_gt_strings = test_df.ground_truth.tolist()
 
    kwargs['ti'].xcom_push(key='train_gt_strings', value=train_gt_strings)
    kwargs['ti'].xcom_push(key='valid_gt_strings', value=valid_gt_strings)
    kwargs['ti'].xcom_push(key='test_gt_strings', value=test_gt_strings)
 
def convert_strings_to_json(**kwargs):
    ti = kwargs['ti']
    train_gt_strings = ti.xcom_pull(key='train_gt_strings', task_ids='extract_ground_truth_strings')
    valid_gt_strings = ti.xcom_pull(key='valid_gt_strings', task_ids='extract_ground_truth_strings')
    test_gt_strings = ti.xcom_pull(key='test_gt_strings', task_ids='extract_ground_truth_strings')
 
    train_jsons = [get_jsons(string) for string in train_gt_strings]
    valid_jsons = [get_jsons(string) for string in valid_gt_strings]
    test_jsons = [get_jsons(string) for string in test_gt_strings]
 
    kwargs['ti'].xcom_push(key='train_jsons', value=train_jsons)
    kwargs['ti'].xcom_push(key='valid_jsons', value=valid_jsons)
    kwargs['ti'].xcom_push(key='test_jsons', value=test_jsons)
 
def save_jsons(**kwargs):
    ti = kwargs['ti']
    time = ti.xcom_pull(key='time', task_ids='create_directories')
    JSON_DIRECTS = ti.xcom_pull(key='JSON_DIRECTS', task_ids='create_directories')
 
    train_jsons = ti.xcom_pull(key='train_jsons', task_ids='convert_strings_to_json')
    valid_jsons = ti.xcom_pull(key='valid_jsons', task_ids='convert_strings_to_json')
    test_jsons = ti.xcom_pull(key='test_jsons', task_ids='convert_strings_to_json')
 
    train_filepaths = []
    valid_filepaths = []
    test_filepaths = []
 
    for idx, json_data in enumerate(train_jsons):
        filename = generate_filename(idx, JSON_DIRECTS[0], 'train', time, '.json')
        with open(filename, 'w') as f:
            json.dump(json_data, f)
        train_filepaths.append(filename)
 
    for idx, json_data in enumerate(valid_jsons):
        filename = generate_filename(idx, JSON_DIRECTS[1], 'valid', time, '.json')
        with open(filename, 'w') as f:
            json.dump(json_data, f)
        valid_filepaths.append(filename)
 
    for idx, json_data in enumerate(test_jsons):
        filename = generate_filename(idx, JSON_DIRECTS[2], 'test', time, '.json')
        with open(filename, 'w') as f:
            json.dump(json_data, f)
        test_filepaths.append(filename)
 
    kwargs['ti'].xcom_push(key='train_json_filepaths', value=train_filepaths)
    kwargs['ti'].xcom_push(key='valid_json_filepaths', value=valid_filepaths)
    kwargs['ti'].xcom_push(key='test_json_filepaths', value=test_filepaths)
 
def create_csvs(**kwargs):

    ti = kwargs['ti']
    time = ti.xcom_pull(key='time', task_ids='create_directories')
    train_image_filepaths = ti.xcom_pull(key='train_image_filepaths', task_ids='save_images')
    valid_image_filepaths = ti.xcom_pull(key='valid_image_filepaths', task_ids='save_images')
    test_image_filepaths = ti.xcom_pull(key='test_image_filepaths', task_ids='save_images')
    train_json_filepaths = ti.xcom_pull(key='train_json_filepaths', task_ids='save_jsons')
    valid_json_filepaths = ti.xcom_pull(key='valid_json_filepaths', task_ids='save_jsons')
    test_json_filepaths = ti.xcom_pull(key='test_json_filepaths', task_ids='save_jsons')
    CSVS = ti.xcom_pull(key='CSVS', task_ids='create_directories')
 
    all_image_filepaths = train_image_filepaths + valid_image_filepaths + test_image_filepaths #
    all_json_filepaths = train_json_filepaths + valid_json_filepaths + test_json_filepaths  #
 
    df = get_dataframe(all_image_filepaths, all_json_filepaths)
    filename = f'{CSVS[0]}/dataset_{time}.csv' 
    save_csvs_to_directory(df, filename)


pull_data_dvc_task = PythonOperator(
    task_id='pull_data_dvc',
    python_callable=pull_data_from_dvc,
    provide_context=True,
    dag=dag,
)

read_parquet_file_path_task = PythonOperator(
    task_id='read_parquet_file_path',
    python_callable=get_parquet_file_path,
    provide_context=True,
    dag=dag,
)
 
load_and_concat_data_task = PythonOperator(
    task_id='load_and_concat_data',
    python_callable=load_and_concat_data,
    provide_context=True,
    dag=dag,
)
 
create_directories_task = PythonOperator(
    task_id='create_directories',
    python_callable=create_directories,
    provide_context=True,
    dag=dag,
)
 
save_images_task = PythonOperator(
    task_id='save_images',
    python_callable=save_images,
    provide_context=True,
    dag=dag,
)
 
extract_ground_truth_strings_task = PythonOperator(
    task_id='extract_ground_truth_strings',
    python_callable=extract_ground_truth_strings,
    provide_context=True,
    dag=dag,
)
 
convert_strings_to_json_task = PythonOperator(
    task_id='convert_strings_to_json',
    python_callable=convert_strings_to_json,
    provide_context=True,
    dag=dag,
)
 
save_jsons_task = PythonOperator(
    task_id='save_jsons',
    python_callable=save_jsons,
    provide_context=True,
    dag=dag,
)
 
create_csvs_task = PythonOperator(
    task_id='create_csvs',
    python_callable=create_csvs,
    provide_context=True,
    dag=dag,
)

 
pull_data_dvc_task >> read_parquet_file_path_task >> load_and_concat_data_task >> create_directories_task
create_directories_task >> save_images_task
 
create_directories_task >> extract_ground_truth_strings_task
extract_ground_truth_strings_task >> convert_strings_to_json_task
convert_strings_to_json_task >> save_jsons_task
 
[save_images_task, save_jsons_task] >> create_csvs_task