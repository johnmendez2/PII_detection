import boto3
from botocore.exceptions import ClientError
import os

AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')

def upload_to_s3(user_id, file_path):
    """
    Upload a file to an S3 bucket

    :param user_id: User-specific identifier
    :param file_path: Local file path to upload
    :return: Message indicating success or error
    """
    user_specific_path = f"{user_id}/{os.path.basename(file_path)}"

    # Initialize a session using AWS credentials
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name="eu-north-1",
        config=boto3.session.Config(signature_version='s3v4', s3={'signature_version': 's3v4', 'use_accelerate_endpoint': False})
    )

    try:
        response = s3_client.upload_file(file_path, 'pii-detection-ai-marketplace-app', user_specific_path)
        return "File successfully uploaded"
    except ClientError as e:
        print(e)
        return str(e)

# Example usage
# file_path = 'C:\\Users\\USER\\Desktop\\PII_DETECTION\\GITHUB_PII_DETECTION\\marketplaceuser\\modified_files.zip'
# user_id = 'marketplaceuser'
# result = upload_to_s3(user_id, file_path)
# print(result)
