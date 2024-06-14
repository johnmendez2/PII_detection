import boto3
import os

def download_s3_files(user_id):
    # Initialize a session using Amazon S3
    s3 = boto3.client('s3')
    
    # List objects in the specified bucket and folder
    paginator = s3.get_paginator('list_objects_v2')
    user_specific_path = f"{user_id}/"
    for page in paginator.paginate(Bucket='pii-detection-ai-marketplace-app', Prefix=user_specific_path):
        for obj in page.get('Contents', []):
            
            # Strip the folder path from the key to get just the filename
            filename_only = obj['Key'].split('/')[-1]  # Adjust based on your OS if necessary
            
            # Extract the file extension
            _, file_extension = os.path.splitext(filename_only)
            
            # Only proceed if the file is a.pdf or.docx
            if file_extension.lower() in ['.pdf', '.docx']:
                # Define the destination key
                local_file = os.path.join(user_id, filename_only)
                
                try:
                    # Ensure the directory exists
                    os.makedirs(os.path.dirname(local_file), exist_ok=True)
                    
                    # Download the file
                    s3.download_file('pii-detection-ai-marketplace-app', obj['Key'], local_file)
                    print(f"Downloaded {obj['Key']} to {local_file}")
                    
                except Exception as e:
                    print(f"Failed to download {obj['Key']}: {e}")

# Example usage
# user_id = 'marketplaceuser'

# download_s3_files( user_id)
