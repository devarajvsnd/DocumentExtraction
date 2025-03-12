import os
from googleapiclient.discovery import build
from google.oauth2 import service_account

# Load images from local folder
def load_local_images(folder_path):
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith((".png", ".jpg", ".jpeg"))]

# Load images from Google Drive (if needed)
def load_drive_images(drive_folder_id, credentials_path):
    creds = service_account.Credentials.from_service_account_file(credentials_path)
    service = build('drive', 'v3', credentials=creds)
    
    results = service.files().list(q=f"'{drive_folder_id}' in parents", fields="files(id, name)").execute()
    files = results.get('files', [])

    return [(file["id"], file["name"]) for file in files]
