import boto3
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv
import os
# Load the.env file
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
def chosen_files(user_id, file_paths):
    # Initialize an empty string to hold the final output
    output_string = ""

    # Create an S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name="eu-north-1",
        config=boto3.session.Config(signature_version='s3v4', s3={'signature_version': 's3v4', 'use_accelerate_endpoint': False})
    )

    # Construct the user-specific path within the S3 bucket
    user_specific_path = f"{user_id}/"

    # Iterate over each file path in the array
    for path in file_paths:
        try:
            # List objects in the bucket that match the user-specific path combined with the file path
            response = s3_client.list_objects_v2(Bucket='pii-detection-ai-marketplace-app', Prefix=user_specific_path + path)
            
            # Check if any objects were returned
            if 'Contents' in response:
                for obj in response['Contents']:
                    # Construct the key for the object
                    full_key = obj['Key']
                    
                    # Read the contents of the object
                    obj_content = s3_client.get_object(Bucket='pii-detection-ai-marketplace-app', Key=full_key)['Body'].read().decode('utf-8')
                    
                    # Append the formatted string to the output string
                    output_string += f"{full_key}:\n{obj_content}\n\n"
            else:
                output_string += f"No files found matching {path}\n\n"

        except NoCredentialsError:
            output_string += "Credentials not available\n\n"

    # Return the final output string
    return output_string