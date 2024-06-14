import boto3
from botocore.exceptions import NoCredentialsError
import os
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')

def generate_presigned_delete_for_folder(folder_path):
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name="eu-north-1",
        config=boto3.session.Config(signature_version='s3v4', s3={'signature_version': 's3v4', 'use_accelerate_endpoint': False})
    )
    try:
        # Initialize an empty dictionary to store file paths and their pre-signed delete URLs
        delete_urls_dict = {}
        
        # List all objects in the folder
        response = s3_client.list_objects_v2(Bucket='pii-detection-ai-marketplace-app', Prefix=folder_path)
        
        if 'Contents' in response:
            # Iterate over each object and generate a pre-signed delete URL
            for obj in response['Contents']:
                delete_url = generate_presigned_delete(obj['Key'])
                # Store the file path and its pre-signed delete URL in the dictionary
                delete_urls_dict[obj['Key']] = delete_url
            print(delete_urls_dict)
            return delete_urls_dict
        else:
            print("No objects found in the specified folder.")
    except NoCredentialsError:
        print("No AWS credentials found")

def generate_presigned_delete(key):
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name="eu-north-1",
        config=boto3.session.Config(signature_version='s3v4', s3={'signature_version': 's3v4', 'use_accelerate_endpoint': False})
    )
    try:
        # Generate pre-signed URL for deletion
        response = s3_client.generate_presigned_url('delete_object',
                                                     Params={'Bucket': 'pii-detection-ai-marketplace-app', 'Key': key})
        return response
    except NoCredentialsError:
        print("No AWS credentials found")
        return None

# Example usage
# folder_path = "marketplaceuser/"  # Replace with your actual folder path
# generate_presigned_delete_for_folder(folder_path)
