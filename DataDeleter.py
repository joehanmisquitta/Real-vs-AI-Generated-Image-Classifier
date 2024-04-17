import os
from tqdm import tqdm

def delete_folder(path):
    """Delete a folder and all its subfolders recursively."""
    try:
        # Count the total number of files and subdirectories to delete
        total_files = 0
        total_dirs = 0
        for root, dirs, files in os.walk(path):
            total_files += len(files)
            total_dirs += len(dirs)

        # Initialize tqdm progress bar
        with tqdm(total=total_files + total_dirs, desc="Deleting", unit="item") as pbar:
            # Remove all files and subdirectories in the folder
            for root, dirs, files in os.walk(path, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                    pbar.update(1)  # Update progress bar
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
                    pbar.update(1)  # Update progress bar
            
            # Remove the main folder itself
            os.rmdir(path)
            pbar.update(1)  # Update progress bar
        print(f"Folder '{path}' and all its contents have been deleted successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
folder_path = "OldData"  # Replace with the path to the folder you want to delete
delete_folder(folder_path)
