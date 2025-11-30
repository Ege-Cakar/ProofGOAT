import os
import dropbox
from dropbox.files import FileMetadata, FolderMetadata

# Credentials provided by user
ACCESS_TOKEN = ""

# Folder to download from (relative to root)
DROPBOX_FOLDER = "/2840"
# Local directory to save to
LOCAL_OUTPUT_DIR = "outputs"

def download_folder(dbx, dbx_folder, local_folder):
    """Download all files from a Dropbox folder to a local folder."""
    if not os.path.exists(local_folder):
        os.makedirs(local_folder)

    print(f"Listing files in {dbx_folder}...")
    try:
        result = dbx.files_list_folder(dbx_folder)
    except dropbox.exceptions.ApiError as e:
        print(f"Error listing folder: {e}")
        return

    while True:
        for entry in result.entries:
            if isinstance(entry, FileMetadata):
                local_path = os.path.join(local_folder, entry.name)
                print(f"Downloading {entry.name} to {local_path}...")
                try:
                    dbx.files_download_to_file(local_path, entry.path_lower)
                except Exception as e:
                    print(f"Error downloading {entry.name}: {e}")
            elif isinstance(entry, FolderMetadata):
                # Recursive download if needed, but assuming flat structure for embeddings
                print(f"Skipping subfolder {entry.name} (not implemented)")
        
        if not result.has_more:
            break
        result = dbx.files_list_folder_continue(result.cursor)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", help="Dropbox access token", default=ACCESS_TOKEN)
    args = parser.parse_args()

    token = args.token
    
    print("Connecting to Dropbox...")
    try:
        dbx = dropbox.Dropbox(token)
        # Check user
        account = dbx.users_get_current_account()
        print(f"Connected as: {account.name.display_name}")
    except dropbox.exceptions.AuthError as e:
        print(f"Authentication failed: {e}")
        print("\n" + "="*60)
        print("ERROR: The provided access token is expired or invalid.")
        print("Please generate a new access token:")
        print("1. Go to https://www.dropbox.com/developers/apps")
        print("2. Select your app")
        print("3. Generate a new 'Generated access token'")
        print("4. Run this script with: python -m scripts.download_data --token YOUR_NEW_TOKEN")
        print("="*60 + "\n")
        return
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    print(f"Downloading data from {DROPBOX_FOLDER} to {LOCAL_OUTPUT_DIR}...")
    download_folder(dbx, DROPBOX_FOLDER, LOCAL_OUTPUT_DIR)
    print("Download complete.")

if __name__ == "__main__":
    main()