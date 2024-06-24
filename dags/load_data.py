from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from ast import literal_eval
from PIL import Image

import pandas as pd
import glob
import os
import json
import sys
import io
import subprocess
import shutil

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime.now(),
    'retries': 1,
}

dag = DAG(
    'Donut_data_pipeline2',
    default_args=default_args,
    description='A simple data processing DAG',
    schedule_interval=None,
)

def pull_data_from_dvc():
    try:
        result = subprocess.run(['dvc', 'pull'], capture_output=True, text=True)
        if result.returncode == 0:
            print("DVC pull successful.")
            print(result.stdout)
        else:
            print("DVC pull failed.")
            print(result.stderr)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

def load_and_concat_data():
    train_path = glob.glob('data/train*.parquet')
    valid_path = glob.glob('data/valid*.parquet')
    test_path = glob.glob('data/test*.parquet')
    
    def load_df(file_path: str):
        return pd.read_parquet(file_path)
    
    def concat_data(data_list):
        return pd.concat(data_list, axis=0).reset_index(drop=True)
    
    train_df = concat_data([load_df(i) for i in train_path])
    valid_df = concat_data([load_df(i) for i in valid_path])
    test_df = concat_data([load_df(i) for i in test_path])
    
    train_df.to_parquet('data/train_df.parquet')
    valid_df.to_parquet('data/valid_df.parquet')
    test_df.to_parquet('data/test_df.parquet')

def create_directories():
    time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    IMG_DIRECTS = [
        f'images/train_images_{time}',
        f'images/valid_images_{time}',
        f'images/test_images_{time}',
    ]
    JSON_DIRECTS = [
        f'jsons/train_json_{time}',
        f'jsons/valid_json_{time}',
        f'jsons/test_json_{time}',
    ]
    CSVs = [f'csvs']
    for directory in IMG_DIRECTS + JSON_DIRECTS + CSVs:
        os.makedirs(directory, exist_ok=True)
    
    with open('data/directories.json', 'w') as f:
        json.dump({'time': time, 'IMG_DIRECTS': IMG_DIRECTS, 'JSON_DIRECTS': JSON_DIRECTS, 'CSVs': CSVs}, f)

def save_images():
    def get_image(byte_string):
        image = Image.open(io.BytesIO(byte_string))
        return image

    def generate_filename(counter, directory, whset, time, extension):
        current_file = f'{directory}/cord-v2-{whset}-{counter}-{time}.{extension}'
        return current_file

    def save_images_to_directory(df, directory, whset, time):
        filepaths = []
        for idx, row in df.iterrows():
            byte_string = row.image['bytes']
            image = get_image(byte_string)
            filename = generate_filename(idx, directory, whset, time, '.png')
            filepaths.append(filename)
            image.save(filename, "PNG")
        return filepaths

    train_df = pd.read_parquet('data/train_df.parquet')
    valid_df = pd.read_parquet('data/valid_df.parquet')
    test_df = pd.read_parquet('data/test_df.parquet')

    with open('data/directories.json', 'r') as f:
        directories = json.load(f)

    IMG_DIRECTS = directories['IMG_DIRECTS']
    time = directories['time']

    train_filepaths = save_images_to_directory(train_df, IMG_DIRECTS[0], 'train', time)
    valid_filepaths = save_images_to_directory(valid_df, IMG_DIRECTS[1], 'valid', time)
    test_filepaths = save_images_to_directory(test_df, IMG_DIRECTS[2], 'test', time)

    with open('data/image_filepaths.json', 'w') as f:
        json.dump({'train': train_filepaths, 'valid': valid_filepaths, 'test': test_filepaths}, f)

def save_jsons():
    def get_jsons(target_string):
        target_json = literal_eval(target_string)['gt_parse']
        return target_json

    def generate_filename(counter, directory, whset, time, extension):
        current_file = f'{directory}/cord-v2-{whset}-{counter}-{time}.{extension}'
        return current_file

    def save_jsons_to_directory(df, directory, whset, time):
        filepaths = []
        for idx, row in df.iterrows():
            target_strings = row.ground_truth
            target_json = get_jsons(target_strings)
            filename = generate_filename(idx, directory, whset, time, '.json')
            with open(filename, 'w') as f:
                json.dump(target_json, f)
            filepaths.append(filename)
        return filepaths

    train_df = pd.read_parquet('data/train_df.parquet')
    valid_df = pd.read_parquet('data/valid_df.parquet')
    test_df = pd.read_parquet('data/test_df.parquet')

    with open('data/directories.json', 'r') as f:
        directories = json.load(f)

    JSON_DIRECTS = directories['JSON_DIRECTS']
    time = directories['time']

    train_filepaths = save_jsons_to_directory(train_df, JSON_DIRECTS[0], 'train', time)
    valid_filepaths = save_jsons_to_directory(valid_df, JSON_DIRECTS[1], 'valid', time)
    test_filepaths = save_jsons_to_directory(test_df, JSON_DIRECTS[2], 'test', time)

    with open('data/json_filepaths.json', 'w') as f:
        json.dump({'train': train_filepaths, 'valid': valid_filepaths, 'test': test_filepaths}, f)

def create_csvs():
    def get_dataframe(image_filepath, json_filepaths):
        dataframe_dict = {
            'image_filepath': image_filepath,
            'json_filepaths': json_filepaths,
        }
        df = pd.DataFrame(dataframe_dict)
        return df

    def save_csvs_to_directory(df, filename, index=False):
        df.to_csv(filename, index=index)

    with open('data/image_filepaths.json', 'r') as f:
        image_filepaths = json.load(f)
    with open('data/json_filepaths.json', 'r') as f:
        json_filepaths = json.load(f)
    with open('data/directories.json', 'r') as f:
        directories = json.load(f)

    CSVs = directories['CSVs']
    time = directories['time']

    train_df = get_dataframe(image_filepaths['train'], json_filepaths['train'])
    valid_df = get_dataframe(image_filepaths['valid'], json_filepaths['valid'])
    test_df = get_dataframe(image_filepaths['test'], json_filepaths['test'])

    train_filename = f'{CSVs[0]}/train_dataset_{time}.csv'
    valid_filename = f'{CSVs[0]}/valid_dataset_{time}.csv'
    test_filename = f'{CSVs[0]}/test_dataset_{time}.csv'

    save_csvs_to_directory(train_df, train_filename)
    save_csvs_to_directory(valid_df, valid_filename)
    save_csvs_to_directory(test_df, test_filename)

def delete_data_folder():
    try:
        shutil.rmtree('data')
        print("Data folder deleted successfully.")
    except Exception as e:
        print(f"An error occurred while deleting the data folder: {e}")
        sys.exit(1)

pull_data_from_dvc_task = PythonOperator(
    task_id='pull_data_from_dvc',
    python_callable=pull_data_from_dvc,
    dag=dag,
)

load_and_concat_data_task = PythonOperator(
    task_id='load_and_concat_data',
    python_callable=load_and_concat_data,
    dag=dag,
)

create_directories_task = PythonOperator(
    task_id='create_directories',
    python_callable=create_directories,
    dag=dag,
)

save_images_task = PythonOperator(
    task_id='save_images',
    python_callable=save_images,
    dag=dag,
)

save_jsons_task = PythonOperator(
    task_id='save_jsons',
    python_callable=save_jsons,
    dag=dag,
)

create_csvs_task = PythonOperator(
    task_id='create_csvs',
    python_callable=create_csvs,
    dag=dag,
)

delete_data_folder_task = PythonOperator(
    task_id='delete_data_folder',
    python_callable=delete_data_folder,
    dag=dag,
)

# Define the task dependencies
pull_data_from_dvc_task >> load_and_concat_data_task
load_and_concat_data_task >> create_directories_task
create_directories_task >> save_images_task
create_directories_task >> save_jsons_task
save_images_task >> create_csvs_task
save_jsons_task >> create_csvs_task
create_csvs_task >> delete_data_folder_task