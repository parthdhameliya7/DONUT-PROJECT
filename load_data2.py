import pandas as pd 
import glob
import random
from tqdm import tqdm 
import os
from datetime import datetime, timedelta
import json

from ast import literal_eval

from PIL import Image 
import io
from typing import Optional, Any

import os
print(f"Directory information : {os.getcwd()}")


from airflow import DAG 
from airflow.operators.python import PythonOperator 
from airflow.operators.bash import BashOperator 


from airflow import DAG

from airflow.operators.bash import BashOperator 

from airflow.operators.python import PythonOperator 

from datetime import datetime, timedelta

create_new_dirs = True
train_path = glob.glob('data/train*.parquet')
valid_path = glob.glob('data/valid*.parquet')
test_path = glob.glob('data/test*.parquet')

def load_df(file_path: str = None) -> Optional[pd.DataFrame]:
    df = pd.read_parquet(file_path)
    return df

def concat_data(data_list: list = None) -> Optional[pd.DataFrame]:
    return pd.concat(data_list, axis=0).reset_index().drop(['index'], axis=1)

def get_image(byte_string: str = None) -> Optional[Any]:
    image = Image.open(io.BytesIO(byte_string))
    return image

def get_jsons(target_string: str = None) -> Optional[str]:
    target_json = literal_eval(target_string)['gt_parse']
    return target_json

def generate_filename(counter: int = None, directory: str = None, whset: str = None, time: str = None, extension: str = None) -> str:
    current_file = f'{directory}/cord-v2-{whset}-{counter}-{time}.{extension}'
    return current_file

def create_data_folders(directory: list = None) -> None:
    [os.makedirs(directs, exist_ok=True) for directs in directory]
    print(os.getcwd())

def save_images_to_directory(df: pd.DataFrame = None, directory: str = None, whset: str = None, time: str = None) -> Optional[list]:
    filepaths = []
    idx = 0
    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        byte_string = row.image
        byte_string = byte_string['bytes'] 
        image = get_image(byte_string)
        filename = generate_filename(idx, directory, whset, time, '.png')
        filepaths.append(filename)
        image.save(filename, "PNG")
        idx += 1
    return filepaths

def save_jsons_to_directory(df: pd.DataFrame = None, directory: str = None, whset: str = None, time: str = None) -> Optional[list]:
    filepaths = []
    idx = 0
    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        target_strings = row.ground_truth
        target_json = get_jsons(target_strings)
        filename = generate_filename(idx, directory, whset, time, '.json')
        with open(filename, 'w') as f:
            json.dump(target_json, f)
        filepaths.append(filename)
        idx += 1
    return filepaths

def get_csv(image_filepath: list = None, json_filepaths: list = None) -> Optional[pd.DataFrame]:
    dataframe_dict = {
        'image_filepath': image_filepath,
        'json_filepaths': json_filepaths,
    }
    df = pd.DataFrame(dataframe_dict)
    return df

def load_and_concat_data():
    print(os.getcwd())
    global train_df, valid_df, test_df
    train_df = concat_data([load_df(i) for i in train_path])
    valid_df = concat_data([load_df(i) for i in valid_path])
    test_df = concat_data([load_df(i) for i in test_path])

def create_directories():
    global time, IMG_DIRECTS, JSON_DIRECTS
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

    create_data_folders(IMG_DIRECTS)
    create_data_folders(JSON_DIRECTS)

def process_train_images():
    global train_image_filepaths
    train_params = {'df': train_df, 'directory': IMG_DIRECTS[0], 'whset': 'train', 'time': time}
    train_image_filepaths = save_images_to_directory(**train_params)

def process_train_jsons():
    global train_json_filepaths
    train_params = {'df': train_df, 'directory': IMG_DIRECTS[0], 'whset': 'train', 'time': time}
    train_json_filepaths = save_jsons_to_directory(**train_params)

def process_valid_images():
    global valid_image_filepaths
    valid_params = {'df': valid_df, 'directory': IMG_DIRECTS[1], 'whset': 'valid', 'time': time}
    valid_image_filepaths = save_images_to_directory(**valid_params)

def process_valid_jsons():
    global valid_json_filepaths
    valid_params = {'df': valid_df, 'directory': IMG_DIRECTS[1], 'whset': 'valid', 'time': time}
    valid_json_filepaths = save_jsons_to_directory(**valid_params)

def process_test_images():
    global test_image_filepaths
    test_params = {'df': test_df, 'directory': IMG_DIRECTS[2], 'whset': 'test', 'time': time}
    test_image_filepaths = save_images_to_directory(**test_params)

def process_test_jsons():
    global test_json_filepaths
    test_params = {'df': test_df, 'directory': IMG_DIRECTS[2], 'whset': 'test', 'time': time}
    test_json_filepaths = save_jsons_to_directory(**test_params)

def save_csv_files():
    train_csv = get_csv(train_image_filepaths, train_json_filepaths)
    valid_csv = get_csv(valid_image_filepaths, valid_json_filepaths)
    test_csv = get_csv(test_image_filepaths, test_json_filepaths)

    train_csv.to_csv(f'csv/train_{time}.csv', index=False)
    valid_csv.to_csv(f'csv/valid_{time}.csv', index=False)
    test_csv.to_csv(f'csv/test_{time}.csv', index=False)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 6, 10),
    'retries': 3,
}

with DAG(
    dag_id='First_dag_2',
    default_args=default_args,
    description='An Airflow DAG to process images and JSONs',
    schedule_interval='@daily',
    catchup=False,
) as dag:

    t1 = PythonOperator(
        task_id='load_and_concat_data',
        python_callable=load_and_concat_data,
        dag = dag,
    )

    t2 = PythonOperator(
        task_id='create_directories',
        python_callable=create_directories,
        dag = dag,
    )

    t3 = PythonOperator(
        task_id='process_train_images',
        python_callable=process_train_images,
        dag = dag,
    )

    # t4 = PythonOperator(
    #     task_id='process_train_jsons', 
    #     python_callable=process_train_jsons
    # )

    # t5 = PythonOperator(
    #     task_id='process_valid_images',
    #     python_callable=process_valid_images
    # )

    # t6 = PythonOperator(
    #     task_id='process_valid_jsons',
    #     python_callable=process_valid_jsons
    # )

    # t7 = PythonOperator(
    #     task_id='process_test_images',
    #     python_callable=process_test_images
    # )

    # t8 = PythonOperator(
    #     task_id='process_test_jsons',
    #     python_callable=process_test_jsons
    # )

    # t9 = PythonOperator(
    #     task_id='save_csv_files',
    #     python_callable=save_csv_files
    # )

    # t1 >> t2 >> [t3, t4, t5, t6, t7, t8] >> t9
    t1 >> t2 >> t3
    if __name__ == "__main__":
        dag.cli()