import fnmatch
import os
from gcsfs import GCSFileSystem
from google.cloud import storage
# Initialize a storage client
def list_files_with_pattern(bucket_name, pattern):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blobs = bucket.list_blobs()
    return [blob.name for blob in blobs if fnmatch.fnmatch(blob.name, pattern)]


def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    # Create destination directory if it doesn't exist
    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
    
    # Download the blob to the destination file
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} from bucket {bucket_name} to {destination_file_name}")