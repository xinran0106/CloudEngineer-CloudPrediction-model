import logging
from typing import List
from pathlib import Path
import boto3
from botocore.exceptions import ClientError


def upload_files_recursive(s3_client: boto3.client, base_dir: Path, bucket_name: str) -> List[str]:
    """Recursively uploads all files in a directory to an S3 bucket.

    Args:
        s3_client (boto3.client): A Boto3 client for interacting with S3.
        base_dir (Path): The directory containing the files to upload.
        bucket_name (str): The name of the S3 bucket to upload the files to.

    Returns:
        List of S3 URIs for each file that was uploaded.
    """
    s3_uris = []
    for file_path in base_dir.rglob('*'):
        if file_path.is_file():
            # Get the relative path of the file with respect to the base directory
            relative_path = file_path.relative_to(base_dir)

            # Upload the file to S3
            object_key = str(relative_path).replace('\\', '/')
            s3_client.upload_file(str(file_path), bucket_name, object_key)

            # Append the S3 URI to the list of uploaded files
            s3_uris.append(f's3://{bucket_name}/{object_key}')

    return s3_uris

def upload_artifacts(artifacts: Path, config: dict) -> List[str]:
    """Uploads all the artifacts in the specified directory to an S3 bucket.

    Args:
        artifacts (Path): The directory containing all the artifacts to upload.
        config (dict): A dictionary containing the configuration details for uploading to S3.

    Returns:
        List of S3 URIs for each file that was uploaded.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Set the log level for the s3transfer logger to suppress DEBUG messages
    s3transfer_logger = logging.getLogger('s3transfer')
    s3transfer_logger.setLevel(logging.WARNING)

    # Create a Boto3 session and client
    session = boto3.Session()
    s3_client = session.client('s3')

    # Get the S3 bucket name from the config
    bucket_name = config['bucket_name']
    try:
        # Call the function to upload all files in the directory and its subdirectories
        s3_uris = upload_files_recursive(s3_client, artifacts, bucket_name)
    except ClientError as e:
        logging.error('Error occurred during S3 upload: %s', e)
        raise

    # Log the S3 URIs
    logging.info('Uploaded artifacts to S3 successfully')
    return s3_uris
