import boto3
from botocore.exceptions import ClientError
import os

AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')

def generate_presigned_url(user_id, expiration=3600):
    """
    Generate a pre-signed URL for downloading a file from an S3 bucket

    :param user_id: Identifier for the user-specific path
    :param expiration: Time in seconds for the URL to expire. Defaults to 1 hour (3600 seconds).
    :return: Pre-signed URL as a string. If error, returns None.
    """
    # Initialize a session using AWS credentials
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,  # Corrected
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,  # Corrected
        region_name="eu-north-1",
        config=boto3.session.Config(signature_version='s3v4', s3={'signature_version': 's3v4', 'use_accelerate_endpoint': False})
    )
    user_specific_path = f"{user_id}/modified_files.zip"
    try:
        # Generate the pre-signed URL
        response = s3_client.generate_presigned_url('get_object',
                                                    Params={'Bucket': 'pii-detection-ai-marketplace-app',
                                                            'Key': user_specific_path},
                                                    ExpiresIn=expiration)
        return response
    except ClientError as e:
        print(e)
        return None

# Example usage
# user_id = 'marketplaceuser'  # Replace with the actual user ID
# url = generate_presigned_url(user_id)
# if url:
#     print(f"Pre-signed URL: {url}")
# else:
#     print("Failed to generate pre-signed URL.")
