import boto3
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv
import os
# Load the.env file
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
def create_presigned_url(object_key, expiration=3600):
    """Generate a pre-signed URL to share an S3 object within a specified subfolder"""
    
    # Create an S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name="eu-north-1",
        config=boto3.session.Config(signature_version='s3v4', s3={'signature_version': 's3v4', 'use_accelerate_endpoint': False})
    )
    
    try:
        # Generate the pre-signed URL
        response = s3_client.generate_presigned_url('put_object',
                                                    Params={'Bucket': 'pii-detection-ai-marketplace-app',
                                                            'Key': object_key},
                                                    ExpiresIn=expiration)
    except NoCredentialsError:
        print("No AWS credentials found")
        return None
    
    return response

# # Example usage
# object_key = 'sk-1ijqjadaldl100lkdlal0/utils/abcd.txt'
# url = create_presigned_url(object_key)
# print(url)
