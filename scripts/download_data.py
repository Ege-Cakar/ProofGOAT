import os
import dropbox
from dropbox.files import FileMetadata, FolderMetadata

# Credentials provided by user
ACCESS_TOKEN = "sl.u.AGFyaIBG6XIWb-1iFX0lp8z5Y7Ax_6ihjaOcjqefvzuYvb24zQ4_Viv2TUVPbEWwcMhK0OTP9RabagwPNCurBml_24QmLJWsKsaRbAbH8QHxbhhyC9iUR4EqqY5WBY50nf7-94qP2NSWtYaCmGg8zVXeQIWq46g8bwU43i73-QOniCDfeM6yu12MIlKB3LJOhu6SROILL7Kt2-5NMGlTmABKc2-PY-ZHRh2KTOPbm2Mv1vpwqW5LY3HSfNBnVS8We0PC2MY3OfclSp1L5qKMpOJpLsZiqjewNqhwI41efVVYoyJ8WC_l6UCze2KNFwUIKfQhh1ikIwUMltx38PkzOyqvFcChyD6bAnBqcCeb1NN_-MR_lDN8UbusACNlyfp3fzsJ64DQ1naXBuL4tgd16dIVhc4pDxcisLInVzcIyZ82HfKUjqW_l6szRVaQobiW3dcDoKUPaUSE_teVyzWOBI5DWRZaE5_7RHTg9UQqGU79YvoHLGQRbclMHBceUelSY2eKMdQvK7bBGs2ZGceJZ0APkpVemCRHeBcWBGxPrVgsnLk0oQj0dorgh8UQEyuG7pOqjYYaOlx7zKrwYQ4Nmg_nxdv35x1cFvE582Ii7EWaoFkl2y2tHkA69r3NgjpYiiMu6qKazmFL8BrLAbTLgjvbB695roFSEXmwW2MsIagPGI3Ht7OKdPiDvILqvUwjhqYi0n_nG7giBnLYsDc1P6LOj6dzBXARyEPmkwps-G0HunS_5Y4tqfAuWQeT01teE5seurQpTMONW6dcG8WapaRhDN32uYWBVvAcD9IhlUdZIpjVBd8N2y1u5j080jxxLlc6_sO21SsJ_FhCpLIeldZtaQoRDgtTWG2fNnLhyCg78Dgh686jR-suFRgYEXZjuWVuBhqXNWoK3Xtr0ycbYoUeTTO1RO4JZ7327uWXtDsjzW0xMPpPZPr_zsKJnrEBVikMJLz8FRsUsP_FMU1zaXkANVi3KyqMQVxRkbbeYVoVjo9BkWlxoMtIiv8C1x05uwf2LBR0xbc7uIyhAPuHPHXiNtts640N0e9kfIqm-6HPLPtTL-chckN5f1GUxp2hacwqCgVZkezTOKnc2dk3wpZd6Hdt4KzPXY7L8jwPcdxI1lFLRbd5Lb4YxnV4ngdcBH5BthVAfRxaBR6rUzFbRDDUUQoZSYebrzVjQI1tLryX5DhAJLw95jxyYdN0DSpR5_nfrbkmfu6GYWtHJUf3CPDLEFGiDNEWUsIFau0S8VKzMfAKvOrXQw_IrsVCCuBW3Vp72WE1TYp5oJX-3rWCIwc6"

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
    print("Connecting to Dropbox...")
    try:
        dbx = dropbox.Dropbox(ACCESS_TOKEN)
        # Check user
        account = dbx.users_get_current_account()
        print(f"Connected as: {account.name.display_name}")
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    print(f"Downloading data from {DROPBOX_FOLDER} to {LOCAL_OUTPUT_DIR}...")
    download_folder(dbx, DROPBOX_FOLDER, LOCAL_OUTPUT_DIR)
    print("Download complete.")

if __name__ == "__main__":
    main()
