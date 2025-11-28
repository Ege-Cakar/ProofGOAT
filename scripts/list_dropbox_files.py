import argparse
import dropbox
from dropbox.exceptions import AuthError

def list_files(token, folder_path):
    print(f"Connecting to Dropbox with provided token...")
    try:
        dbx = dropbox.Dropbox(token)
        account = dbx.users_get_current_account()
        print(f"Connected as: {account.name.display_name}")
    except AuthError as e:
        print(f"ERROR: Authentication failed. Check your token. {e}")
        return

    print(f"\nListing files in '{folder_path}'...")
    try:
        result = dbx.files_list_folder(folder_path)
    except Exception as e:
        print(f"ERROR: Could not list folder '{folder_path}': {e}")
        return

    files_found = []
    while True:
        for entry in result.entries:
            print(f"- {entry.name} ({getattr(entry, 'size', 'N/A')} bytes)")
            files_found.append(entry.name)
        
        if not result.has_more:
            break
        result = dbx.files_list_folder_continue(result.cursor)
    
    print(f"\nTotal files found: {len(files_found)}")
    
    # Check for specific kimina files
    print("\nChecking for Kimina files:")
    kimina_nl = "kimina17_all_nl_embeddings.parquet"
    kimina_lean = "kimina17_all_lean_embeddings.parquet"
    
    if kimina_nl in files_found:
        print(f"[OK] Found {kimina_nl}")
    else:
        print(f"[MISSING] Could not find {kimina_nl}")
        
    if kimina_lean in files_found:
        print(f"[OK] Found {kimina_lean}")
    else:
        print(f"[MISSING] Could not find {kimina_lean}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="Dropbox access token")
    parser.add_argument("--folder", default="/2840", help="Dropbox folder path")
    args = parser.parse_args()

    list_files(args.token, args.folder)
